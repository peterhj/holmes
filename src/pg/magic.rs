use gsl::functions::{
  exp_e10, psi, beta_pdf, beta_cdf,
};
use gsl::integration::{
  IntegrationWorkspace, Integrand,
};

use libc::{c_void};
use std::mem::{transmute};

// FIXME(20151216)
const ACTION_SPACE_LEN: usize = 0;

const ABS_TOL:          f64 = 0.0;
const REL_TOL:          f64 = 1.0e-6;
const WORKSPACE_LEN:    usize = 1024;

struct ThetaPolicyMachine;

struct OmegaPolicyMachine {
  outer_dist_int:   Integrand<OuterIntegrandData>,
  outer_grad_int:   Integrand<OuterIntegrandData>,
  workspace:        IntegrationWorkspace,
}

impl OmegaPolicyMachine {
  pub extern "C" fn outer_integrand_f(u: f64, _data: *mut c_void) -> f64 {
    let mut data: &mut OuterIntegrandData = unsafe { transmute(_data) };

    data.dist_terms.clear();
    data.grad_terms.clear();

    for k in (0 .. ACTION_SPACE_LEN) {
      if k == data.kprime {
        data.dist_terms.push(1.0);
      } else {
        let dist_term = beta_cdf(u, 2.0, 2.0);
        data.dist_terms.push(dist_term);
      }
    }

    if data.calc_grad {
      for k in (0 .. ACTION_SPACE_LEN) {
        if k == data.kprime {
          data.grad_terms.push(0.0);
        } else {
          let inner_dist_upper = 1.0;
          data.inner_dist_int.data.reset(inner_dist_upper, 2.0, 2.0);
          let (inner_dist_result, _) = data.inner_dist_int.integrate_qags(
              0.0, inner_dist_upper,
              ABS_TOL, REL_TOL, 1000,
              &mut data.workspace,
          );

          let inner_grad_upper = 1.0;
          data.inner_grad_int.data.reset(inner_grad_upper, 2.0, 2.0);
          let (inner_grad_result, _) = data.inner_grad_int.integrate_qags(
              0.0, inner_grad_upper,
              ABS_TOL, REL_TOL, 1000,
              &mut data.workspace,
          );

          let grad_term = inner_grad_result / inner_dist_result + (psi(2.0) - psi(2.0));
          data.grad_terms.push(grad_term);
        }
      }
    }

    let mut dist_logprod = 0.0;
    for k in (0 .. ACTION_SPACE_LEN) {
      let dist_term = data.dist_terms[k];
      dist_logprod += dist_term.ln();
    }
    let dist_prod = dist_logprod.exp();

    let mut grad_sum = 0.0;
    if data.calc_grad {
      for k in (0 .. ACTION_SPACE_LEN) {
        let grad_term = data.grad_terms[k];
        grad_sum += grad_term;
      }
    } else {
      grad_sum = 1.0;
    }

    let measure = beta_pdf(u, 2.0, 2.0);

    let outer_value = dist_prod * grad_sum * measure;
    outer_value
  }

  pub extern "C" fn inner_integrand_f(t: f64, _data: *mut c_void) -> f64 {
    let mut data: &mut InnerIntegrandData = unsafe { transmute(_data) };

    let bias = data.bias;
    let e1 = data.e1;
    let e2 = data.e2;

    let mut inner_value = exp_e10(bias + e1 * (t.ln()) + e2 * ((1.0 - t).ln())).value();
    if data.calc_smooth {
      inner_value *= (t / (1.0 - t)).ln();
    }
    inner_value
  }

  pub fn new(kprime: usize) -> OmegaPolicyMachine {
    OmegaPolicyMachine{
      outer_dist_int:   Integrand{
        function:       OmegaPolicyMachine::outer_integrand_f,
        data:           OuterIntegrandData::new(kprime, false),
      },
      outer_grad_int:   Integrand{
        function:       OmegaPolicyMachine::outer_integrand_f,
        data:           OuterIntegrandData::new(kprime, true),
      },
      workspace:        IntegrationWorkspace::new(WORKSPACE_LEN),
    }
  }

  pub fn calculate(&mut self) {
    // FIXME(20151216)
    let epsilon = 1.0;
    let (outer_dist_result, _) = self.outer_dist_int.integrate_qags(
        0.0, 1.0,
        ABS_TOL, REL_TOL, 1000,
        &mut self.workspace,
    );
    let (outer_grad_result, _) = self.outer_grad_int.integrate_qags(
        0.0, 1.0,
        ABS_TOL, REL_TOL, 1000,
        &mut self.workspace,
    );
    let outer_scale = epsilon * outer_grad_result / outer_dist_result;

    // TODO(20151216)
    unimplemented!();
  }
}

struct OuterIntegrandData {
  kprime:       usize,
  calc_grad:    bool,
  term_params:  Vec<()>,

  dist_terms:       Vec<f64>,
  grad_terms:       Vec<f64>,
  inner_dist_int:   Integrand<InnerIntegrandData>,
  inner_grad_int:   Integrand<InnerIntegrandData>,
  workspace:        IntegrationWorkspace,
}

impl OuterIntegrandData {
  pub fn new(kprime: usize, calc_grad: bool) -> OuterIntegrandData {
    OuterIntegrandData{
      kprime:       kprime,
      calc_grad:    calc_grad,
      term_params:  Vec::with_capacity(ACTION_SPACE_LEN),
      dist_terms:       Vec::with_capacity(ACTION_SPACE_LEN),
      grad_terms:       Vec::with_capacity(ACTION_SPACE_LEN),
      inner_dist_int:   Integrand{
        function:       OmegaPolicyMachine::inner_integrand_f,
        data:           InnerIntegrandData::new(false),
      },
      inner_grad_int:   Integrand{
        function:       OmegaPolicyMachine::inner_integrand_f,
        data:           InnerIntegrandData::new(true),
      },
      workspace:        IntegrationWorkspace::new(WORKSPACE_LEN),
    }
  }
}

#[derive(Default)]
struct InnerIntegrandData {
  //upper_bound:  f64,
  calc_smooth:  bool,
  e1:           f64,
  e2:           f64,
  bias:         f64,
}

impl InnerIntegrandData {
  pub fn new(calc_smooth: bool) -> InnerIntegrandData {
    let mut data: InnerIntegrandData = Default::default();
    data.calc_smooth = calc_smooth;
    data
  }

  pub fn reset(&mut self, upper: f64, e1: f64, e2: f64) {
    assert!(upper >= 0.0);
    assert!(upper <= 1.0);
    assert!(e1 > 0.0);
    assert!(e2 > 0.0);
    //self.upper_bound = upper;
    self.e1 = e1;
    self.e2 = e2;

    let argmax = e1 / (e1 + e2);
    // FIXME(20151220): can add a small constant to bias for stability.
    self.bias = if upper >= argmax {
      -(e1 * (e1 / (e1 + e2)).ln() + e2 * (e2 / (e1 + e2)).ln())
    } else {
      -(e1 * upper.ln() + e2 * (1.0 - upper).ln())
    };
  }
}
