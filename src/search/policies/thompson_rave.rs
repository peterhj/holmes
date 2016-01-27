use array_util::{array_argmax};
use board::{Board, Stone, Point};
use hyper::{
  RAVE,
  RAVE_EQUIV,
  PRIOR_EQUIV,
};
use search::tree::{Node};
use search::policies::{TreePolicy};

use rand::{Rng};
use rand::distributions::{IndependentSample};
use rand::distributions::gamma::{Gamma};
use std::iter::{repeat};

pub struct ThompsonRaveTreePolicy {
  rave:         bool,
  rave_equiv:   f32,
  prior_equiv:  f32,

  tmp_values:   Vec<f32>,
}

impl ThompsonRaveTreePolicy {
  pub fn new() -> ThompsonRaveTreePolicy {
    ThompsonRaveTreePolicy{
      rave:         RAVE.with(|x| *x),
      rave_equiv:   RAVE_EQUIV.with(|x| *x),
      prior_equiv:  PRIOR_EQUIV.with(|x| *x),

      tmp_values:   repeat(0.0).take(Board::SIZE).collect(),
    }
  }
}

impl TreePolicy for ThompsonRaveTreePolicy {
  fn rave(&self) -> bool {
    self.rave
  }

  fn init_node(&mut self, node: &mut Node, rng: &mut Self::R) {
    // Do nothing.
  }

  fn execute_search(&mut self, node: &Node, rng: &mut Self::R) -> Option<(Point, usize)> {
    for j in 0 .. node.horizon {
      let n = node.num_trials[j];
      let s = node.num_succs[j];
      let pn = self.prior_equiv;
      let ps = node.prior_moves[j].1 * self.prior_equiv; // FIXME(20151124): rounding?
      let rn = node.num_trials_rave[j];
      let (es, ef) = if !self.rave || rn == 0.0 {
        ((s + ps), (n + pn) - (s + ps))
      } else {
        let rs = node.num_succs_rave[j];
        let beta_rave = rn / (rn + (n + pn) + (n + pn) * rn / self.rave_equiv);
        let ev =
            0.0f32.max(1.0 - beta_rave) * ((s + ps) / (n + pn))
            + beta_rave * (rs / rn)
        ;
        // FIXME(20151124): rounding?
        let es = ev * (n + pn);
        let ef = 0.0f32.max(1.0 - ev) * (n + pn);
        (es, ef)
      };
      // XXX: Sample two gamma distributions to get a beta distributed
      // random variable.
      let gamma_s = Gamma::new((es + 1.0) as f64, 1.0);
      let gamma_f = Gamma::new((ef + 1.0) as f64, 1.0);
      let xs = gamma_s.ind_sample(rng) as f32;
      let xf = gamma_f.ind_sample(rng) as f32;
      let u = xs / (xs + xf);
      self.tmp_values[j] = u;
    }
    array_argmax(&self.tmp_values[ .. node.horizon])
      .map(|j| (node.prior_moves[j].0, j))
  }

  fn backup_values(&mut self, node: &mut Node, rng: &mut Self::R) {
    // Do nothing.
  }
}
