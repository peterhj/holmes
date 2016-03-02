//use array_util::{array_argmax};
use board::{Board, Action};
use search::parallel_policies::{
  SearchPolicyWorker,
  PriorPolicy, DiffPriorPolicy,
  GradAccumMode, GradSyncMode,
};
use search::parallel_policies::convnet::{
  ConvnetPolicyWorker,
  ConvnetPriorPolicy,
};
use search::parallel_trace::{
  TreeBatchWorker, RolloutTrajWorker,
  SearchTrace, SearchTraceBatch, ReconNode,
  ParallelSearchReconWorker,
};
use search::parallel_tree::{
  TreePolicyConfig,
  MonteCarloSearchConfig,
  SharedTree,
  ParallelMonteCarloSearchServer,
  SearchWorkerConfig,
  SearchWorkerCommand,
};
use txnstate::{TxnState};
use txnstate::extras::{TxnStateNodeData};

use gsl::functions::{exp_e10, beta, log_beta, norm_inc_beta, psi, beta_pdf, beta_cdf};
use gsl::integration::{IntegrationWorkspace, Integrand};

use libc::{c_void};
use std::cell::{RefCell};
use std::iter::{repeat};
use std::mem::{transmute};
use std::rc::{Rc};
use std::sync::{Arc, RwLock};
use std::sync::atomic::{Ordering};

const ABS_TOL:          f64 = 0.0;
const REL_TOL:          f64 = 1.0e-6;
const MAX_INTERVALS:    usize = 1024;
const WORKSPACE_LEN:    usize = 1024;

fn omega_likelihood_fun(u: f64, d: &OmegaNodeSharedData) -> f64 {
  // XXX(20160222): This computes the $ H(u;s_j,a_j) * p(u;s_j,a_j) $ term
  // in the integrand of the decision likelihood function.
  let diag_j = d.diag_rank;
  let diag_e1 = d.net_succs[diag_j];
  let diag_e2 = d.net_fails[diag_j];
  let mut x = beta_pdf(u, diag_e1, diag_e2);
  for j in 0 .. d.horizon {
    if j != d.diag_rank {
      let e1 = d.net_succs[j];
      let e2 = d.net_fails[j];
      x *= beta_cdf(u, e1, e2);
    }
  }
  x
}

extern "C" fn omega_likelihood_extfun(u: f64, unsafe_data: *mut c_void) -> f64 {
  let data: &OmegaLikelihoodData = unsafe { transmute(unsafe_data) };
  omega_likelihood_fun(u, &*data.shared_data.borrow())
}

extern "C" fn omega_diagonal_g_factor_extfun(u: f64, unsafe_data: *mut c_void) -> f64 {
  // XXX(20160222): This computes the integrand of the "diagonal" G-factor:
  //
  //    $ eta_Q * g(u;s_j,a_j) * H(u;s_j,a_j) * p(u;s_j,a_j) $
  //
  // where:
  //
  //    $ g(u;s_j,a_j) := log(u / (1 - u)) - (psi(e1) - psi(e2)) $
  //
  let data: &OmegaDiagonalGFactorData = unsafe { transmute(unsafe_data) };
  let d = &*data.shared_data.borrow();
  let diag_j = d.diag_rank;
  let e1 = d.net_succs[diag_j];
  let e2 = d.net_fails[diag_j];
  let mut x = (u / (1.0 - u)).ln();
  x -= psi(e1) - psi(e2);
  x *= omega_likelihood_fun(u, d);
  x *= d.prior_equiv;
  x
}

extern "C" fn omega_cross_g_factor_inner_extfun(t: f64, unsafe_data: *mut c_void) -> f64 {
  // XXX(20160222): This computes the inner integrand of the "cross" G-factor:
  //
  //    $ log(t / (1 - t)) * t ^ (e1-1) * (1 - t) ^ (e2-1) $
  //
  let data: &OmegaCrossGFactorInnerData = unsafe { transmute(unsafe_data) };
  let d = &*data.shared_data.borrow();
  let j = data.action_rank;
  let bias = data.bias;
  let e1 = d.net_succs[j];
  let e2 = d.net_fails[j];
  let mut x = exp_e10(bias + (e1 - 1.0) * t.ln() + (e2 - 1.0) * (1.0 - t).ln()).value();
  x *= (t / (1.0 - t)).ln();
  x
}

extern "C" fn omega_cross_g_factor_outer_extfun(u: f64, unsafe_data: *mut c_void) -> f64 {
  // XXX(20160222): This computes the outer integrand of the "cross" G-factor:
  //
  //    $ eta_Q * (I(u;e1,e2) / B(u;e1,e2) - (psi(e1) - psi(e2))) * H(u) * p(u) $
  //
  let mut data: &mut OmegaCrossGFactorOuterData = unsafe { transmute(unsafe_data) };
  let &mut OmegaCrossGFactorOuterData{
    action_rank,
    ref shared_data,
    ref mut inner_integrand,
    ref mut workspace} = data;
  let d = &*shared_data.borrow();
  let j = action_rank;
  let e1 = d.net_succs[j];
  let e2 = d.net_fails[j];
  let z = log_beta(e1, e2);
  inner_integrand.data.bias = -z;
  let (mut x, _) = inner_integrand.integrate_qags(
      0.0, u,
      ABS_TOL, REL_TOL, MAX_INTERVALS,
      workspace,
  );
  x /= norm_inc_beta(u, e1, e2);
  x -= psi(e1) - psi(e2);
  x *= omega_likelihood_fun(u, d);
  x *= d.prior_equiv;
  x
}

pub struct OmegaNodeSharedData {
  pub horizon:      usize,
  pub prior_equiv:  f64,
  pub diag_rank:    usize,
  pub net_succs:    Vec<f64>,
  pub net_fails:    Vec<f64>,
}

impl OmegaNodeSharedData {
  pub fn new() -> OmegaNodeSharedData {
    OmegaNodeSharedData{
      horizon:      0,
      prior_equiv:  0.0,
      diag_rank:    0,
      net_succs:    repeat(0.0).take(Board::SIZE).collect(),
      net_fails:    repeat(0.0).take(Board::SIZE).collect(),
    }
  }

  pub fn reset(&mut self, tree_cfg: TreePolicyConfig, chosen_action_rank: usize, horizon: usize, node: &ReconNode) {
    self.horizon = horizon;
    self.prior_equiv = tree_cfg.prior_equiv as f64;
    self.diag_rank = chosen_action_rank;
    for j in 0 .. horizon {
      let succs_j = node.succ_counts[j].load(Ordering::Acquire) as f32;
      let trials_j = node.trial_counts[j].load(Ordering::Acquire) as f32;
      let e1 =
          1.0 + tree_cfg.prior_equiv * node.prior_values[j]
              + tree_cfg.mc_scale    * succs_j;
      let e2 =
          1.0 + 0.0f32.max(tree_cfg.prior_equiv * (1.0 - node.prior_values[j]))
              + 0.0f32.max(tree_cfg.mc_scale    * (trials_j - succs_j));
      self.net_succs[j] = e1 as f64;
      self.net_fails[j] = e2 as f64;
    }
  }
}

struct OmegaLikelihoodData {
  shared_data:  Rc<RefCell<OmegaNodeSharedData>>,
}

impl OmegaLikelihoodData {
  pub fn new(shared_data: Rc<RefCell<OmegaNodeSharedData>>) -> OmegaLikelihoodData {
    OmegaLikelihoodData{
      shared_data:  shared_data,
    }
  }
}

pub struct OmegaLikelihoodIntegral {
  integrand:    Integrand<OmegaLikelihoodData>,
  workspace:    IntegrationWorkspace,
}

impl OmegaLikelihoodIntegral {
  pub fn new(shared_data: Rc<RefCell<OmegaNodeSharedData>>) -> OmegaLikelihoodIntegral {
    OmegaLikelihoodIntegral{
      integrand:    Integrand{
        function:   omega_likelihood_extfun,
        data:       OmegaLikelihoodData::new(shared_data),
      },
      workspace:    IntegrationWorkspace::new(WORKSPACE_LEN),
    }
  }

  pub fn calculate_likelihood(&mut self) -> f32 {
    let &mut OmegaLikelihoodIntegral{
      ref mut integrand,
      ref mut workspace,
      .. } = self;
    let (res, _) = integrand.integrate_qags(
        0.0, 1.0,
        ABS_TOL, REL_TOL, MAX_INTERVALS,
        workspace,
    );
    res as f32
  }
}

struct OmegaDiagonalGFactorData {
  shared_data:  Rc<RefCell<OmegaNodeSharedData>>,
}

pub struct OmegaDiagonalGFactorIntegral {
  integrand:    Integrand<OmegaDiagonalGFactorData>,
  workspace:    IntegrationWorkspace,
}

impl OmegaDiagonalGFactorIntegral {
  pub fn new(shared_data: Rc<RefCell<OmegaNodeSharedData>>) -> OmegaDiagonalGFactorIntegral {
    OmegaDiagonalGFactorIntegral{
      integrand:    Integrand{
        function:   omega_diagonal_g_factor_extfun,
        data:       OmegaDiagonalGFactorData{
          shared_data:  shared_data,
        },
      },
      workspace:    IntegrationWorkspace::new(WORKSPACE_LEN),
    }
  }

  pub fn calculate_diagonal_g_factor(&mut self) -> f32 {
    let &mut OmegaDiagonalGFactorIntegral{
      ref mut integrand,
      ref mut workspace,
      .. } = self;
    let (res, _) = integrand.integrate_qags(
        0.0, 1.0,
        ABS_TOL, REL_TOL, MAX_INTERVALS,
        workspace,
    );
    res as f32
  }
}

struct OmegaCrossGFactorInnerData {
  shared_data:  Rc<RefCell<OmegaNodeSharedData>>,
  action_rank:  usize,
  bias:         f64,
}

struct OmegaCrossGFactorOuterData {
  shared_data:  Rc<RefCell<OmegaNodeSharedData>>,
  action_rank:  usize,
  inner_integrand:  Integrand<OmegaCrossGFactorInnerData>,
  workspace:        IntegrationWorkspace,
}

pub struct OmegaCrossGFactorIntegral {
  outer_integrand:  Integrand<OmegaCrossGFactorOuterData>,
  workspace:        IntegrationWorkspace,
}

impl OmegaCrossGFactorIntegral {
  pub fn new(shared_data: Rc<RefCell<OmegaNodeSharedData>>) -> OmegaCrossGFactorIntegral {
    OmegaCrossGFactorIntegral{
      outer_integrand:  Integrand{
        function:       omega_cross_g_factor_outer_extfun,
        data:           OmegaCrossGFactorOuterData{
          shared_data:  shared_data.clone(),
          action_rank:  0,
          inner_integrand:  Integrand{
            function:       omega_cross_g_factor_inner_extfun,
            data:           OmegaCrossGFactorInnerData{
              shared_data:  shared_data,
              action_rank:  0,
              bias:         0.0,
            },
          },
          workspace:        IntegrationWorkspace::new(WORKSPACE_LEN),
        },
      },
      workspace:        IntegrationWorkspace::new(WORKSPACE_LEN),
    }
  }

  pub fn calculate_cross_g_factor(&mut self, action_rank: usize) -> f32 {
    let &mut OmegaCrossGFactorIntegral{
      ref mut outer_integrand,
      ref mut workspace,
      .. } = self;
    outer_integrand.data.action_rank = action_rank;
    outer_integrand.data.inner_integrand.data.action_rank = action_rank;
    let (res, _) = outer_integrand.integrate_qags(
        0.0, 1.0,
        ABS_TOL, REL_TOL, MAX_INTERVALS,
        workspace,
    );
    res as f32
  }
}

pub struct OmegaTreeBatchWorker<W> where W: SearchPolicyWorker {
  tree_cfg:     Option<TreePolicyConfig>,
  memory:       Arc<RwLock<OmegaWorkerMemory>>,
  //inner_value:  f32,
  //batch_scale:  f32,
  shared_data:          Rc<RefCell<OmegaNodeSharedData>>,
  likelihood_int:       OmegaLikelihoodIntegral,
  diag_g_factor_int:    OmegaDiagonalGFactorIntegral,
  cross_g_factor_int:   OmegaCrossGFactorIntegral,
  //search_worker:        Rc<RefCell<ConvnetPolicyWorker>>,
  search_worker:        Rc<RefCell<W>>,
}

//impl OmegaTreeBatchWorker {
impl<W> OmegaTreeBatchWorker<W> where W: SearchPolicyWorker {
  //pub fn new(tree_cfg: TreePolicyConfig, /*objective: ObjectLevelObjective,*/ search_worker: Rc<RefCell<ConvnetPolicyWorker>>) -> OmegaTreeBatchWorker {
  pub fn new(memory: Arc<RwLock<OmegaWorkerMemory>>, search_worker: Rc<RefCell<W>>) -> OmegaTreeBatchWorker<W> {
    let shared_data = Rc::new(RefCell::new(OmegaNodeSharedData::new()));
    OmegaTreeBatchWorker{
      tree_cfg:     None, //tree_cfg,
      memory:       memory,
      //inner_value:  0.0,
      //batch_scale:  1.0,
      likelihood_int:       OmegaLikelihoodIntegral::new(shared_data.clone()),
      diag_g_factor_int:    OmegaDiagonalGFactorIntegral::new(shared_data.clone()),
      cross_g_factor_int:   OmegaCrossGFactorIntegral::new(shared_data.clone()),
      shared_data:          shared_data,
      search_worker:        search_worker,
      //traces:   vec![],
      //obj_label:    None,
    }
  }

  /*fn reset(&mut self, objective: ObjectLevelObjective, search_trace: &SearchTrace) {
    self.tree_cfg = None;
    // FIXME(20160301): inner value depends on the current objective.
    self.inner_value = search_trace.root_value.unwrap();
  }*/
}

//impl TreeBatchWorker for OmegaTreeBatchWorker {
impl<W> TreeBatchWorker for OmegaTreeBatchWorker<W> where W: SearchPolicyWorker {
  fn with_search_worker<F, V>(&mut self, mut f: F) -> V where F: FnOnce(&mut SearchPolicyWorker) -> V {
    let mut worker = self.search_worker.borrow_mut();
    f(&mut *worker)
  }

  fn with_prior_policy<F, V>(&mut self, mut f: F) -> V where F: FnOnce(&mut PriorPolicy) -> V {
    let mut worker = self.search_worker.borrow_mut();
    f(worker.prior_policy())
  }

  //fn start_instance(&mut self, tree_cfg: TreePolicyConfig) {
  fn start_instance(&mut self, search_trace: &SearchTrace) {
    self.tree_cfg = Some(search_trace.tree_cfg.unwrap());
    // FIXME(20160301): inner value depends on the current objective.
    //self.inner_value = search_trace.root_value.unwrap();
  }

  fn update_batch(&mut self, root_node: Arc<RwLock<ReconNode>>, trace_batch: &SearchTraceBatch) {
    for batch_idx in 0 .. trace_batch.batch_size {
      let mut node = root_node.clone();
      let tree_trace = &trace_batch.traj_traces[batch_idx].tree_trace;
      for decision in tree_trace.decisions.iter() {
        let next_node = {
          let node = node.read().unwrap();
          let turn = node.state.current_turn();

          self.shared_data.borrow_mut().reset(
              self.tree_cfg.unwrap(),
              decision.action_rank,
              decision.horizon,
              &*node,
          );

          let mut worker = self.search_worker.borrow_mut();
          let mut prior_policy = worker.diff_prior_policy();

          assert!(decision.horizon <= node.valid_actions.len());
          let inner_value = self.memory.read().unwrap().read_inner_value();
          let likelihood = self.likelihood_int.calculate_likelihood();
          for (j, &action_j) in node.valid_actions.iter().take(decision.horizon).enumerate() {
            let g_j = if decision.action == action_j {
              self.diag_g_factor_int.calculate_diagonal_g_factor()
            } else {
              self.cross_g_factor_int.calculate_cross_g_factor(j)
            };
            node.state.get_data().features.extract_relative_features(turn, prior_policy.expose_input_buffer(j));
            prior_policy.preload_action_label(j, action_j);
            // XXX(20160301): The inner value includes the baseline.
            prior_policy.preload_loss_weight(j, g_j * inner_value / likelihood);
          }

          prior_policy.load_inputs(decision.horizon);
          prior_policy.forward(decision.horizon);
          prior_policy.backward(decision.horizon);

          node.child_nodes[decision.action.idx()].clone()
        };
        node = next_node;
      }
    }
  }

  fn update_instance(&mut self, root_node: Arc<RwLock<ReconNode>>) {
  }
}

pub struct OmegaWorkerMemory {
  pub inner_values: Vec<f32>,

  t: usize,
}

impl OmegaWorkerMemory {
  pub fn new() -> OmegaWorkerMemory {
    OmegaWorkerMemory{
      inner_values: vec![0.0, 0.0],
      t: 0,
    }
  }

  pub fn reset(&mut self) {
    self.inner_values.clear();
  }

  pub fn set_state(&mut self, objective: ObjectLevelObjective, t: usize, bare_inner_value: f32) {
    // FIXME(20160301): currently specialized for TD0 objective.
    match t {
      0 => {
        self.inner_values[t] = bare_inner_value;
      }
      1 => {
        self.inner_values[t] = 1.0 - bare_inner_value;
      }
      _ => unreachable!(),
    }
    self.t = t;
  }

  pub fn read_inner_value(&self) -> f32 {
    self.inner_values[self.t]
  }
}

#[derive(Clone, Copy)]
pub enum ObjectLevelObjective {
  SupervisedKLLoss,
  TD0L2Loss,
}

//#[derive(Clone)]
pub enum ObjectLevelInput {
  SingleState{state: TxnState<TxnStateNodeData>},
  PairStates{state1: TxnState<TxnStateNodeData>, state2: TxnState<TxnStateNodeData>},
  SingleTrace{trace: SearchTrace},
  PairTraces{trace1: SearchTrace, trace2: SearchTrace},
}

#[derive(Clone, Copy)]
pub enum ObjectLevelLabel {
  Action{action: Action},
}

pub struct OmegaTrainingConfig {
  pub objective:    Option<ObjectLevelObjective>,
  pub step_size:    f32,
}

pub struct MetaLevelWorker<W> where W: SearchPolicyWorker {
  objective:    Option<ObjectLevelObjective>,
  step_size:    f32,
  recon_worker: Rc<RefCell<ParallelSearchReconWorker<OmegaTreeBatchWorker<W>, ()>>>,

  //traces:   Vec<TxnState<TxnStateNodeData>>,
  traces:       Vec<SearchTrace>,
  obj_label:    Option<ObjectLevelLabel>,
}

//impl MetaLevelWorker {
impl<W> MetaLevelWorker<W> where W: SearchPolicyWorker {
  pub fn new(recon_worker: Rc<RefCell<ParallelSearchReconWorker<OmegaTreeBatchWorker<W>, ()>>>) -> MetaLevelWorker<W> {
    MetaLevelWorker{
      objective:    None,
      step_size:    0.0,
      recon_worker: recon_worker,
      traces:       vec![],
      obj_label:    None,
    }
  }

  pub fn reset(&mut self, objective: ObjectLevelObjective, step_size: f32) {
    self.objective = Some(objective);
    self.step_size = step_size;
  }

  pub fn load_input(&mut self, input: ObjectLevelInput) {
    match (self.objective, input) {
      (Some(ObjectLevelObjective::SupervisedKLLoss), ObjectLevelInput::SingleTrace{trace}) => {
        self.traces.clear();
        self.traces.push(trace);
      }
      (Some(ObjectLevelObjective::TD0L2Loss), ObjectLevelInput::PairTraces{trace1, trace2}) => {
        self.traces.clear();
        self.traces.push(trace1);
        self.traces.push(trace2);
      }
      _ => unimplemented!(),
    }
  }

  pub fn load_label(&mut self, label: Option<ObjectLevelLabel>) {
    match (self.objective, label) {
      (Some(ObjectLevelObjective::SupervisedKLLoss), Some(ObjectLevelLabel::Action{action})) => {
        self.obj_label = label;
      }
      (Some(ObjectLevelObjective::TD0L2Loss), None) => {
        self.obj_label = None;
      }
      _ => unimplemented!(),
    }
  }

  pub fn train_sample(&mut self) {
    match self.objective {
      Some(ObjectLevelObjective::SupervisedKLLoss) => {
        // FIXME(20160301)
        unimplemented!();
      }
      Some(ObjectLevelObjective::TD0L2Loss) => {
        let &mut MetaLevelWorker{
          step_size,
          ref traces,
          ref mut recon_worker,
          .. } = self;
        let mut recon_worker = recon_worker.borrow_mut();

        let mut expected_values = [0.0, 0.0];

        // FIXME(20160301): this whole idea is likely misguided.
        // There is always a single working SearchTrace, which we do not want to
        // clone. So this data structure should process one tree/trace at a time.

        recon_worker.reconstruct_trace(&traces[0]);
        {
          let mut root_node = recon_worker.shared_tree.root_node.lock().unwrap();
          expected_values[0] = root_node.as_ref().unwrap().read().unwrap().get_value();
        }

        recon_worker.reconstruct_trace(&traces[1]);
        {
          let mut root_node = recon_worker.shared_tree.root_node.lock().unwrap();
          expected_values[1] = root_node.as_ref().unwrap().read().unwrap().get_value();
        }

        let step_scale = expected_values[0] + expected_values[1] - 1.0;

        recon_worker.with_tree_worker(|tree_worker| {
          tree_worker.with_search_worker(|search_worker| {
            let mut prior_policy = search_worker.diff_prior_policy();
            prior_policy.sync_gradients(GradSyncMode::Sum);
            prior_policy.descend_params(step_scale * step_size);
            prior_policy.reset_gradients();
          });
        });

        // FIXME(20160301)
        unimplemented!();
      }
      _ => {
        panic!("forgot to set TreePolicyConfig");
      }
    }
  }
}

pub struct MetaLevelSearch;

impl MetaLevelSearch {
  pub fn join<W>(&self,
      objective:    ObjectLevelObjective,
      step_size:    f32,
      baseline:     f32,
      search_cfg:   MonteCarloSearchConfig,
      tree_cfg:     TreePolicyConfig,
      server:       &ParallelMonteCarloSearchServer<W>,
      input:        ObjectLevelInput,
      //rng:          &mut Xorshiftplus128Rng,
      )
  where W: SearchPolicyWorker
  {
    let num_workers = server.num_workers();
    let (worker_batch_size, worker_num_batches) = search_cfg.worker_dims(num_workers);
    assert!(worker_batch_size >= 1);
    assert!(worker_num_batches >= 1);

    let cfg = SearchWorkerConfig{
      batch_size:       worker_batch_size,
      num_batches:      worker_num_batches,
      // FIXME(20160112): use correct komi value.
      komi:             7.5,
      prev_mean_score:  0.0,
    };

    match (objective, input) {
      (ObjectLevelObjective::TD0L2Loss, ObjectLevelInput::PairStates{state1, state2}) => {
        // FIXME(20160301): need to search and trace twice.
        for (t, state) in [state1, state2].into_iter().enumerate() {
          let shared_tree = SharedTree::new(tree_cfg);
          for tid in 0 .. num_workers {
            server.enqueue(tid, SearchWorkerCommand::ResetSearch{
              cfg:            cfg,
              shared_tree:    shared_tree.clone(),
              init_state:     state.clone(),
              record_search:  true,
            });
          }
          server.join();

          for tid in 0 .. num_workers {
            server.enqueue(tid, SearchWorkerCommand::TrainMetaLevelTD0Gradient{t: t});
          }
          server.join();
        }

        for tid in 0 .. num_workers {
          server.enqueue(tid, SearchWorkerCommand::TrainMetaLevelTD0Descent{step_size: step_size});
        }
        server.join();

        // FIXME(20160301)
        unimplemented!();
      }

      _ => unimplemented!(),
    }
  }
}
