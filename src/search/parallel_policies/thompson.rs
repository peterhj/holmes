use array_util::{array_argmax};
use board::{Board, Point};
//use random::{XorShift128PlusRng};
use search::parallel_policies::{TreePolicy};
use search::parallel_tree::{Node};

use rng::xorshift::{Xorshiftplus128Rng};

use rand::distributions::{IndependentSample};
use rand::distributions::gamma::{Gamma};
use std::iter::{repeat};
use std::sync::atomic::{Ordering};

pub struct ThompsonTreePolicy {
  prior:        bool,
  prior_equiv:  f32,
  rave:         bool,
  rave_equiv:   f32,
  tmp_values:   Vec<f32>,
}

impl ThompsonTreePolicy {
  pub fn new() -> ThompsonTreePolicy {
    ThompsonTreePolicy{
      prior:        true,
      prior_equiv:  100.0,
      rave:         true,
      rave_equiv:   3000.0,
      tmp_values:   repeat(0.0).take(Board::SIZE).collect(),
    }
  }
}

impl TreePolicy for ThompsonTreePolicy {
  fn execute_search(&mut self, node: &Node, rng: &mut Xorshiftplus128Rng) -> Option<(Point, usize)> {
    for j in 0 .. node.horizon {
      let n = node.values.num_trials[j].load(Ordering::Acquire) as f32;
      let s = node.values.num_succs[j].load(Ordering::Acquire) as f32;
      let (pn, ps) = if !self.prior {
        (0.0, 0.0)
      } else {
        (self.prior_equiv, self.prior_equiv * node.values.prior_values[j])
      };
      let rn = node.values.num_trials_rave[j].load(Ordering::Acquire) as f32;
      let (es, ef) = if !self.rave || rn == 0.0 {
        ((s + ps), (n + pn) - (s + ps))
      } else {
        let rs = node.values.num_succs_rave[j].load(Ordering::Acquire) as f32;
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
      .map(|j| (node.valid_moves[j], j))
  }
}
