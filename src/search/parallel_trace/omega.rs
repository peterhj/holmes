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
  ExploreBatchWorker, RolloutTrajWorker,
  SearchTraceBatch, ReconNode,
};
use search::parallel_tree::{
  TreePolicyConfig,
};

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
          1.0 + tree_cfg.prior_equiv * (1.0 - node.prior_values[j])
              + tree_cfg.mc_scale    * (trials_j - succs_j);
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

#[derive(Clone, Copy)]
pub enum MetaObjective {
  SupervisedKLLoss,
}

#[derive(Clone, Copy)]
pub enum MetaLabel {
  ActionLabel{action: Action},
}

pub struct OmegaBatchWorker {
  tree_cfg:             TreePolicyConfig,
  objective:            MetaObjective,
  shared_data:          Rc<RefCell<OmegaNodeSharedData>>,
  likelihood_int:       OmegaLikelihoodIntegral,
  diag_g_factor_int:    OmegaDiagonalGFactorIntegral,
  cross_g_factor_int:   OmegaCrossGFactorIntegral,
  prior_policy:         ConvnetPriorPolicy,
  policy_worker:        Rc<RefCell<ConvnetPolicyWorker>>,
}

impl OmegaBatchWorker {
  pub fn load_label(&mut self, label: MetaLabel) {
    match (self.objective, label) {
      (MetaObjective::SupervisedKLLoss, MetaLabel::ActionLabel{action}) => {
        // FIXME(20160225)
        unimplemented!();
      }
    }
  }

  /*pub fn finalize_instance(&mut self) {
    let mut worker = self.policy_worker.borrow_mut();
    worker.diff_prior_policy().synchronize_gradients(GradSyncMode::Sum);
    worker.diff_prior_policy().reset_gradients();

    // FIXME(20160222)
    unimplemented!();
  }*/
}

impl ExploreBatchWorker for OmegaBatchWorker {
  //fn with_prior_policy<V>(&mut self, mut f: &mut FnMut(&mut PriorPolicy) -> V) -> V {
  fn with_prior_policy<F, V>(&mut self, mut f: F) -> V where F: FnOnce(&mut PriorPolicy) -> V {
    let mut worker = self.policy_worker.borrow_mut();
    f(worker.prior_policy())
  }

  fn update_batch(&mut self, root_node: Arc<RwLock<ReconNode>>, trace_batch: &SearchTraceBatch) {
    for batch_idx in 0 .. trace_batch.batch_size {
      let mut node = root_node.clone();
      let explore_traj = &trace_batch.traj_traces[batch_idx].explore_traj;
      for decision in explore_traj.decisions.iter() {
        let next_node = {
          let node = node.read().unwrap();
          let turn = node.state.current_turn();

          self.shared_data.borrow_mut().reset(
              self.tree_cfg,
              decision.action_rank,
              decision.horizon,
              &*node,
          );

          let mut worker = self.policy_worker.borrow_mut();
          let mut prior_policy = worker.diff_prior_policy();

          assert!(decision.horizon <= node.valid_actions.len());
          let likelihood = self.likelihood_int.calculate_likelihood();
          for (j, &action_j) in node.valid_actions.iter().take(decision.horizon).enumerate() {
            let g_j = if decision.action == action_j {
              self.diag_g_factor_int.calculate_diagonal_g_factor()
            } else {
              self.cross_g_factor_int.calculate_cross_g_factor(j)
            };
            node.state.get_data().features.extract_relative_features(turn, prior_policy.expose_input_buffer(j));
            prior_policy.preload_action_label(j, action_j);
            prior_policy.preload_loss_weight(j, g_j / likelihood);
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
    let mut worker = self.policy_worker.borrow_mut();
    worker.diff_prior_policy().sync_gradients(GradSyncMode::Sum);
    worker.diff_prior_policy().reset_gradients();

    match self.objective {
      MetaObjective::SupervisedKLLoss => {
        // FIXME(20160225)
        unimplemented!();
      }
    }
  }
}
