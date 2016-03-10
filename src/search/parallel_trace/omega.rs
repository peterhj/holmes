//use array_util::{array_argmax};
use board::{Board, Action};
use data::{EpisodePreproc, LazyEpisodeLoader};
use search::parallel_policies::{
  SearchPolicyWorker,
  PriorPolicy, DiffPriorPolicy,
  GradAccumMode, GradSyncMode,
};
use search::parallel_policies::convnet::{
  ConvnetPolicyWorkerBuilder,
  ConvnetPolicyWorker,
  ConvnetPriorPolicy,
};
use search::parallel_trace::{
  TreeBatchWorker, RolloutTrajWorker,
  SearchTrace, SearchTraceBatch, ReconNode,
  ParallelSearchReconWorker,
};
use search::parallel_tree::{
  MonteCarloSearchConfig,
  TreePolicyConfig,
  SharedTree,
  ParallelMonteCarloSearchServer,
  SearchWorkerConfig,
  SearchWorkerBatchConfig,
  SearchWorkerCommand,
};
use txnstate::{TxnStateConfig, TxnState};
use txnstate::extras::{TxnStateNodeData};

use cuda::runtime::{CudaDevice};
use gsl::{Gsl, panic_error_handler};
use gsl::functions::{exp, exp_e10, beta, log_beta, norm_inc_beta_error, psi, beta_pdf, beta_cdf};
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
const MAX_INTERVALS:    usize = 4096;
const WORKSPACE_LEN:    usize = 4096;

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
  //let x = exp_e10(bias + t.ln().ln() - (1.0 - t).ln().ln() + (e1 - 1.0) * t.ln() + (e2 - 1.0) * (1.0 - t).ln()).value();
  let mut x = exp_e10(bias + (e1 - 1.0) * t.ln() + (e2 - 1.0) * (1.0 - t).ln()).value();
  //println!("DEBUG: cross g: inner fun value: {:6}", x);
  x *= (t / (1.0 - t)).ln();
  x
}

extern "C" fn omega_cross_g_factor_normalization_extfun(t: f64, unsafe_data: *mut c_void) -> f64 {
  // XXX(20160222): This computes the normalization of the "cross" G-factor:
  //
  //    $ t ^ (e1-1) * (1 - t) ^ (e2-1) $
  //
  // In other words it is the integrand of the (unnormalized) incomplete beta
  // function.
  let data: &OmegaCrossGFactorInnerData = unsafe { transmute(unsafe_data) };
  let d = &*data.shared_data.borrow();
  let j = data.action_rank;
  let bias = data.bias;
  let e1 = d.net_succs[j];
  let e2 = d.net_fails[j];
  let x = exp_e10(bias + (e1 - 1.0) * t.ln() + (e2 - 1.0) * (1.0 - t).ln()).value();
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
    ref mut norm_integrand,
    ref mut workspace} = data;
  let d = &*shared_data.borrow();
  let j = action_rank;
  let e1 = d.net_succs[j];
  let e2 = d.net_fails[j];
  let z = log_beta(e1, e2);
  inner_integrand.data.action_rank = j;
  inner_integrand.data.bias = -z;
  //let mut norm = norm_inc_beta(u, e1, e2);
  let (mut norm, _) = norm_inc_beta_error(u, e1, e2);
  let mut rare_case = false;
  if norm == 0.0 {
    norm_integrand.data.action_rank = j;
    let mut shift = 4.0;
    loop {
      norm_integrand.data.bias = -z + shift;
      // FIXME(20160306): integrate normalization.
      let (new_norm, _) = norm_integrand.integrate_qags(
          0.0, u,
          ABS_TOL, REL_TOL, MAX_INTERVALS,
          workspace,
      );
      if new_norm > 0.0 {
        //println!("DEBUG: cross g_j outer extfun: final norm: {:e} shift: {:.0}", new_norm, shift);
        inner_integrand.data.bias = -z + shift;
        norm = new_norm;
        break;
      } else {
        if shift >= 512.0 {
          /*println!("WARNING:  cross g_j outer extfun: shift reached limit");
          println!("          j: {} u: {:e} e1: {:e} e2: {:e}", j, u, e1, e2);
          panic!();*/
          rare_case = true;
          break;
        }
        shift *= 4.0;
      }
    }
  }
  let mut x = if !rare_case {
    let (mut x, _) = inner_integrand.integrate_qags(
        0.0, u,
        ABS_TOL, REL_TOL, MAX_INTERVALS,
        workspace,
    );
    x /= norm;
    x
  } else {
    (u / (1.0 - u)).ln()
  };
  x -= psi(e1) - psi(e2);
  x *= omega_likelihood_fun(u, d);
  x *= d.prior_equiv;
  /*if !x.is_finite() {
    panic!("cross g_j is nonfinite: {:?} {:?} {:?}",
        inner_integrand.data.bias, e1, e2);
  }*/
  x
}

#[derive(Debug)]
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
          1.0 + 0.0f32.max(tree_cfg.prior_equiv * node.prior_values[j])
              + 0.0f32.max(tree_cfg.mc_scale    * succs_j);
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

  pub fn calculate_likelihood(&mut self) -> f64 {
    let &mut OmegaLikelihoodIntegral{
      ref mut integrand,
      ref mut workspace,
      .. } = self;
    let (res, _) = integrand.integrate_qags(
        0.0, 1.0,
        ABS_TOL, REL_TOL, MAX_INTERVALS,
        workspace,
    );
    res
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

  pub fn calculate_diagonal_g_factor(&mut self) -> f64 {
    let &mut OmegaDiagonalGFactorIntegral{
      ref mut integrand,
      ref mut workspace,
      .. } = self;
    let (res, _) = integrand.integrate_qags(
        0.0, 1.0,
        ABS_TOL, REL_TOL, MAX_INTERVALS,
        workspace,
    );
    res
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
  norm_integrand:   Integrand<OmegaCrossGFactorInnerData>,
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
          action_rank:  0,
          inner_integrand:  Integrand{
            function:       omega_cross_g_factor_inner_extfun,
            data:           OmegaCrossGFactorInnerData{
              shared_data:  shared_data.clone(),
              action_rank:  0,
              bias:         0.0,
            },
          },
          norm_integrand:   Integrand{
            function:       omega_cross_g_factor_normalization_extfun,
            data:           OmegaCrossGFactorInnerData{
              shared_data:  shared_data.clone(),
              action_rank:  0,
              bias:         0.0,
            },
          },
          shared_data:  shared_data,
          workspace:    IntegrationWorkspace::new(WORKSPACE_LEN),
        },
      },
      workspace:        IntegrationWorkspace::new(WORKSPACE_LEN),
    }
  }

  pub fn calculate_cross_g_factor(&mut self, action_rank: usize) -> f64 {
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
    res
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
  //pub fn new(tree_cfg: TreePolicyConfig, /*objective: MetaLevelObjective,*/ search_worker: Rc<RefCell<ConvnetPolicyWorker>>) -> OmegaTreeBatchWorker {
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
    let inner_value = self.memory.read().unwrap().read_inner_value();
    for batch_idx in 0 .. trace_batch.batch_size {
      let mut node = root_node.clone();
      let tree_trace = &trace_batch.traj_traces[batch_idx].tree_trace;
      for (k, decision) in tree_trace.decisions.iter().enumerate() {
        /*println!("DEBUG: omega: batch_idx: {}/{} decision: {}/{} {:?}",
            batch_idx, trace_batch.batch_size,
            k, tree_trace.decisions.len(),
            decision,
        );*/
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
          let likelihood = self.likelihood_int.calculate_likelihood();
          /*println!("DEBUG: omega: batch_idx: {}/{} likelihood: {:.6}",
              batch_idx, trace_batch.batch_size,
              likelihood,
          );*/
          if !likelihood.is_finite() {
            /*panic!("WARNING: omega: likelihood is nonfinite: shared: {:?}",
                self.shared_data.borrow(),
            );*/
            println!("WARNING:  omega: likelihood is nonfinite:");
            println!("          batch_idx: {}/{} decision: {}/{} {:?}",
                batch_idx, trace_batch.batch_size,
                k, tree_trace.decisions.len(),
                decision,
            );
            println!("          shared: {:?}",
                self.shared_data.borrow(),
            );
            panic!();
          }
          for (j, &action_j) in node.valid_actions.iter().take(decision.horizon).enumerate() {
            let g_j = if decision.action == action_j {
              //println!("DEBUG: omega: calculating diagonal g: {:?} {} {:?}", decision, j, action_j);
              self.diag_g_factor_int.calculate_diagonal_g_factor()
            } else {
              //println!("DEBUG: omega: calculating cross g: {:?} {} {:?}", decision, j, action_j);
              self.cross_g_factor_int.calculate_cross_g_factor(j)
            };
            if !g_j.is_finite() {
              println!("WARNING:  omega: g_j is nonfinite:");
              println!("          batch_idx: {}/{} decision: {}/{} {:?}",
                  batch_idx, trace_batch.batch_size,
                  k, tree_trace.decisions.len(),
                  decision,
              );
              println!("          action: {} {:?} shared: {:?}",
                  j, action_j,
                  self.shared_data.borrow(),
              );
              panic!();
            }
            node.state.get_data().features.extract_relative_features(turn, prior_policy.expose_input_buffer(j));
            prior_policy.preload_action_label(j, action_j);
            // XXX(20160301): The inner value should include the baseline.
            prior_policy.preload_loss_weight(j, inner_value * (g_j / likelihood) as f32);
            /*println!("DEBUG: omega: batch_idx: {}/{} factor: {:.6} g_j: {:.6}",
                batch_idx, trace_batch.batch_size,
                inner_value * g_j / likelihood, g_j,
            );*/
          }

          prior_policy.load_inputs(decision.horizon);
          prior_policy.forward(decision.horizon);
          prior_policy.backward(decision.horizon);

          node.child_nodes[decision.action_rank].clone()
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
    //self.inner_values.clear();
    self.inner_values[0] = 0.0;
    self.inner_values[1] = 0.0;
  }

  pub fn set_state(&mut self, objective: MetaLevelObjective, t: usize, bare_inner_value: f32) {
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
pub enum MetaLevelObjective {
  SupervisedKLLoss,
  TD0L2Loss,
}

//#[derive(Clone)]
pub enum MetaLevelInput {
  SingleState{state: TxnState<TxnStateNodeData>},
  PairStates{state1: TxnState<TxnStateNodeData>, state2: TxnState<TxnStateNodeData>},
  SingleTrace{trace: SearchTrace},
  PairTraces{trace1: SearchTrace, trace2: SearchTrace},
}

#[derive(Clone, Copy)]
pub enum MetaLevelLabel {
  Action{action: Action},
}

pub struct OmegaTrainingConfig {
  pub objective:    Option<MetaLevelObjective>,
  pub step_size:    f32,
}

/*pub struct MetaLevelWorker<W> where W: SearchPolicyWorker {
  objective:    Option<MetaLevelObjective>,
  step_size:    f32,
  recon_worker: Rc<RefCell<ParallelSearchReconWorker<OmegaTreeBatchWorker<W>, ()>>>,

  //traces:   Vec<TxnState<TxnStateNodeData>>,
  traces:       Vec<SearchTrace>,
  obj_label:    Option<MetaLevelLabel>,
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

  pub fn reset(&mut self, objective: MetaLevelObjective, step_size: f32) {
    self.objective = Some(objective);
    self.step_size = step_size;
  }

  pub fn load_input(&mut self, input: MetaLevelInput) {
    match (self.objective, input) {
      (Some(MetaLevelObjective::SupervisedKLLoss), MetaLevelInput::SingleTrace{trace}) => {
        self.traces.clear();
        self.traces.push(trace);
      }
      (Some(MetaLevelObjective::TD0L2Loss), MetaLevelInput::PairTraces{trace1, trace2}) => {
        self.traces.clear();
        self.traces.push(trace1);
        self.traces.push(trace2);
      }
      _ => unimplemented!(),
    }
  }

  pub fn load_label(&mut self, label: Option<MetaLevelLabel>) {
    match (self.objective, label) {
      (Some(MetaLevelObjective::SupervisedKLLoss), Some(MetaLevelLabel::Action{action})) => {
        self.obj_label = label;
      }
      (Some(MetaLevelObjective::TD0L2Loss), None) => {
        self.obj_label = None;
      }
      _ => unimplemented!(),
    }
  }

  pub fn train_sample(&mut self) {
    match self.objective {
      Some(MetaLevelObjective::SupervisedKLLoss) => {
        // FIXME(20160301)
        unimplemented!();
      }
      Some(MetaLevelObjective::TD0L2Loss) => {
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
}*/

pub struct MetaLevelSearch;

impl MetaLevelSearch {
  pub fn join<W>(&self,
      iter:         usize,
      objective:    MetaLevelObjective,
      step_size:    f32,
      baseline:     f32,
      search_cfg:   MonteCarloSearchConfig,
      tree_cfg:     TreePolicyConfig,
      server:       &ParallelMonteCarloSearchServer<W>,
      input:        MetaLevelInput,
      //rng:          &mut Xorshiftplus128Rng,
      )
  where W: SearchPolicyWorker
  {
    let num_workers = server.num_workers();
    let (worker_batch_size, worker_num_batches) = search_cfg.worker_dims(num_workers);
    assert!(worker_batch_size >= 1);
    assert!(worker_num_batches >= 1);

    /*let worker_cfg = SearchWorkerConfig{
      batch_size:       worker_batch_size,
      num_batches:      worker_num_batches,
      // FIXME(20160112): use correct komi value.
      komi:             7.5,
      prev_mean_score:  0.0,
    };*/
    let worker_cfg = SearchWorkerConfig{
      batch_cfg:            SearchWorkerBatchConfig::Fixed{num_batches: worker_num_batches},
      tree_batch_size:      None,
      rollout_batch_size:   worker_batch_size,
    };

    match (objective, input) {
      (MetaLevelObjective::TD0L2Loss, MetaLevelInput::PairStates{state1, state2}) => {
        // XXX(20160301): Need to search and trace twice.
        for (t, state) in [state1, state2].into_iter().enumerate() {
          println!("DEBUG: MetaLevelSearch: frame {} (search)", t+1);
          let shared_tree = SharedTree::new(tree_cfg);
          for tid in 0 .. num_workers {
            server.enqueue(tid, SearchWorkerCommand::ResetSearch{
              cfg:            worker_cfg,
              shared_tree:    shared_tree.clone(),
              init_state:     state.clone(),
              record_search:  true,
            });
          }
          server.join();

          println!("DEBUG: MetaLevelSearch: frame {} (gradient)", t+1);
          for tid in 0 .. num_workers {
            server.enqueue(tid, SearchWorkerCommand::TrainMetaLevelTD0Gradient{t: t});
          }
          server.join();
        }

        println!("DEBUG: MetaLevelSearch: descent");
        for tid in 0 .. num_workers {
          server.enqueue(tid, SearchWorkerCommand::TrainMetaLevelTD0Descent{step_size: step_size});
        }
        server.join();
      }

      _ => unimplemented!(),
    }
  }
}

pub struct OmegaDriver {
  mc_cfg:   MonteCarloSearchConfig,
  tree_cfg: TreePolicyConfig,
  server:   ParallelMonteCarloSearchServer<ConvnetPolicyWorker>,
}

impl OmegaDriver {
  pub fn new(state_cfg: TxnStateConfig, search_cfg: MonteCarloSearchConfig, tree_cfg: TreePolicyConfig) -> OmegaDriver {
    //setup_ieee_env();
    Gsl::disable_error_handler();
    //Gsl::set_error_handler(panic_error_handler);
    let worker_tree_batch_capacity = 32;
    let worker_rollout_batch_capacity = 576;
    let num_workers = CudaDevice::count().unwrap();
    let server =  ParallelMonteCarloSearchServer::new(
        state_cfg,
        num_workers,
        worker_tree_batch_capacity,
        worker_rollout_batch_capacity,
        ConvnetPolicyWorkerBuilder::new(
            tree_cfg, num_workers,
            worker_tree_batch_capacity,
            worker_rollout_batch_capacity,
        ),
    );
    OmegaDriver{
      mc_cfg:   search_cfg,
      tree_cfg: tree_cfg,
      server:   server,
    }
  }

  pub fn train<P>(&mut self, loader: &mut LazyEpisodeLoader<P>) where P: EpisodePreproc {
    let mut iter = 0;
    loader.for_each_random_pair(|frame1, frame2| {
      println!("DEBUG: iter: {}", iter);
      let search = MetaLevelSearch;
      search.join(
          iter,
          MetaLevelObjective::TD0L2Loss,
          0.0001,
          0.0,
          self.mc_cfg,
          self.tree_cfg,
          &self.server,
          MetaLevelInput::PairStates{state1: frame1.0.clone(), state2: frame2.0.clone()},
      );
      iter += 1;
    });
  }
}
