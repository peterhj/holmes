use board::{Board};
use search::parallel_policies::{
  DiffPriorPolicy,
  GradAccumMode, GradSyncMode,
};
use search::parallel_policies::convnet::{
  ConvnetPriorPolicy,
};
use search::parallel_trace::{
  ExploreBatchWorker, RolloutTrajWorker,
  SearchTraceBatch, ReconNode,
};
use search::parallel_tree::{
  TreePolicyConfig,
};

use gsl::functions::{exp_e10, beta, norm_inc_beta, psi, beta_pdf, beta_cdf};
use gsl::integration::{IntegrationWorkspace, Integrand};

use libc::{c_void};
use std::cell::{RefCell};
use std::mem::{transmute};
use std::rc::{Rc};
use std::sync::{Arc, RwLock};
use std::sync::atomic::{Ordering};

const ABS_TOL:          f64 = 0.0;
const REL_TOL:          f64 = 1.0e-6;
const MAX_ITERS:        usize = 1024;
const WORKSPACE_LEN:    usize = 1024;

fn omega_likelihood_fun(u: f64, d: &OmegaNodeSharedData) -> f64 {
  // XXX(20160222): This computes the $ H(u;s_j,a_j) * p(u;s_j,a_j) $ term
  // in the integrand of the decision likelihood function.
  let diag_j = d.diag_rank;
  /*let mut x = beta_pdf(
      u,
      1.0 + d.prior_equiv * d.prior_values[diag_j]
          + d.mc_scale    * d.succs[diag_j],
      1.0 + d.prior_equiv * (1.0 - d.prior_values[diag_j])
          + d.mc_scale    * (d.trials[diag_j] - d.succs[diag_j]),
  );*/
  let diag_e1 = d.net_succs[diag_j];
  let diag_e2 = d.net_fails[diag_j];
  let mut x = beta_pdf(u, diag_e1, diag_e2);
  for j in 0 .. d.horizon {
    if j != d.diag_rank {
      /*x *= beta_cdf(
          u,
          1.0 + d.prior_equiv * d.prior_values[j]
              + d.mc_scale    * d.succs[j],
          1.0 + d.prior_equiv * (1.0 - d.prior_values[j])
              + d.mc_scale    * (d.trials[j] - d.succs[j]),
      );*/
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
  /*let d = &data.likelihood_data;
  let diag_j = d.diag_rank;
  let e1 =
      1.0 + d.prior_equiv * d.prior_values[diag_j]
          + d.mc_scale    * d.succs[diag_j];
  let e2 =
      1.0 + d.prior_equiv * (1.0 - d.prior_values[diag_j])
          + d.mc_scale    * (d.trials[diag_j] - d.succs[diag_j]);*/
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
  /*let d = &data.likelihood_data;
  let j = data.action_rank;
  let e1 =
      1.0 + d.prior_equiv * d.prior_values[j]
          + d.mc_scale    * d.succs[j];
  let e2 =
      1.0 + d.prior_equiv * (1.0 - d.prior_values[j])
          + d.mc_scale    * (d.trials[j] - d.succs[j]);*/
  let d = &*data.shared_data.borrow();
  let j = data.action_rank;
  let e1 = d.net_succs[j];
  let e2 = d.net_fails[j];
  let mut x = exp_e10((e1 - 1.0) * t.ln() + (e2 - 1.0) * (1.0 - t).ln()).value();
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
  /*let d = likelihood_data;
  let j = action_rank;
  let e1 =
      1.0 + d.prior_equiv * d.prior_values[j]
          + d.mc_scale    * d.succs[j];
  let e2 =
      1.0 + d.prior_equiv * (1.0 - d.prior_values[j])
          + d.mc_scale    * (d.trials[j] - d.succs[j]);*/
  let d = &*shared_data.borrow();
  let j = action_rank;
  let e1 = d.net_succs[j];
  let e2 = d.net_fails[j];
  let (mut x, _) = inner_integrand.integrate_qags(
      0.0, 1.0,
      ABS_TOL, REL_TOL, MAX_ITERS,
      workspace,
  );
  x /= norm_inc_beta(u, e1, e2) * beta(e1, e2);
  x -= psi(e1) - psi(e2);
  x *= omega_likelihood_fun(u, d);
  x *= d.prior_equiv;
  x
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

  pub fn calculate_likelihood(&mut self, tree_cfg: TreePolicyConfig, node: &ReconNode) -> f32 {
    // FIXME(20160222)
    unimplemented!();
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

  pub fn calculate_diagonal_g_factor(&mut self, tree_cfg: TreePolicyConfig, horizon: usize, node: &ReconNode) -> f32 {
    let &mut OmegaDiagonalGFactorIntegral{
      ref mut integrand,
      ref mut workspace,
      .. } = self;
    let (res, _) = integrand.integrate_qags(
        0.0, 1.0,
        ABS_TOL, REL_TOL, MAX_ITERS,
        workspace,
    );
    res as f32
  }
}

struct OmegaCrossGFactorInnerData {
  shared_data:  Rc<RefCell<OmegaNodeSharedData>>,
  action_rank:  usize,
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
            },
          },
          workspace:        IntegrationWorkspace::new(WORKSPACE_LEN),
        },
      },
      workspace:        IntegrationWorkspace::new(WORKSPACE_LEN),
    }
  }

  pub fn calculate_cross_g_factor(&mut self, tree_cfg: TreePolicyConfig, horizon: usize, node: &ReconNode) -> f32 {
    let &mut OmegaCrossGFactorIntegral{
      ref mut outer_integrand,
      ref mut workspace,
      .. } = self;
    let (res, _) = outer_integrand.integrate_qags(
        0.0, 1.0,
        ABS_TOL, REL_TOL, MAX_ITERS,
        workspace,
    );
    res as f32
  }
}

pub struct OmegaNodeSharedData {
  horizon:      usize,
  prior_equiv:  f64,
  diag_rank:    usize,
  net_succs:    Vec<f64>,
  net_fails:    Vec<f64>,
}

impl OmegaNodeSharedData {
  pub fn reset(&mut self, tree_cfg: TreePolicyConfig, action_rank: usize, horizon: usize, node: &ReconNode) {
    self.horizon = horizon;
    self.prior_equiv = tree_cfg.prior_equiv as f64;
    self.diag_rank = action_rank;
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

pub struct OmegaBatchWorker {
  tree_cfg:             TreePolicyConfig,
  shared_data:          Rc<RefCell<OmegaNodeSharedData>>,
  diag_g_factor_int:    OmegaDiagonalGFactorIntegral,
  cross_g_factor_int:   OmegaCrossGFactorIntegral,

  prior_policy: ConvnetPriorPolicy,
}

impl OmegaBatchWorker {
  pub fn finalize_instance(&mut self) {
    self.prior_policy().synchronize_gradients(GradSyncMode::Sum);
    self.prior_policy().reset_gradients();

    // FIXME(20160222)
    unimplemented!();
  }
}

impl ExploreBatchWorker for OmegaBatchWorker {
  fn prior_policy(&mut self) -> &mut DiffPriorPolicy {
    &mut self.prior_policy
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

          assert!(decision.horizon <= node.valid_actions.len());
          for (j, &action_j) in node.valid_actions.iter().take(decision.horizon).enumerate() {
            let g_j = if decision.action == action_j {
              self.diag_g_factor_int.calculate_diagonal_g_factor(self.tree_cfg, decision.horizon, &*node)
            } else {
              self.cross_g_factor_int.calculate_cross_g_factor(self.tree_cfg, decision.horizon, &*node)
            };

            node.state.get_data().features.extract_relative_features(turn, self.prior_policy().expose_input_buffer(j));
            self.prior_policy().preload_action_label(j, action_j);
            self.prior_policy().preload_loss_weight(j, g_j);
          }

          self.prior_policy().load_inputs(decision.horizon);
          self.prior_policy().forward(decision.horizon);
          self.prior_policy().backward(decision.horizon);
          self.prior_policy().accumulate_gradients(GradAccumMode::ScoreRatio);

          node.child_nodes[decision.action.idx()].clone()
        };
        node = next_node;
      }
    }
  }
}
