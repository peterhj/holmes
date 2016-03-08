use array_util::{array_argmax};
use board::{Board, Point};
//use random::{XorShift128PlusRng};
use search::parallel_policies::{TreePolicy};
use search::parallel_tree::{TreePolicyConfig, HorizonConfig, NodeValues, Node};

use rng::xorshift::{Xorshiftplus128Rng};

use rand::distributions::{IndependentSample};
use rand::distributions::gamma::{Gamma};
use std::iter::{repeat};
use std::sync::atomic::{Ordering};

pub struct ThompsonTreePolicy {
  cfg:          TreePolicyConfig,
  /*mc_scale:     f32,
  prior:        bool,
  prior_equiv:  f32,
  rave:         bool,
  rave_equiv:   f32,*/
  tmp_values:   Vec<f32>,
}

impl ThompsonTreePolicy {
  pub fn new(tree_cfg: TreePolicyConfig) -> ThompsonTreePolicy {
    ThompsonTreePolicy{
      cfg:    tree_cfg,

      /*mc_scale:     1.0,
      //mc_scale:     0.5,

      /*prior:        true,
      prior_equiv:  100.0,*/
      prior:        true,
      prior_equiv:  16.0,
      //prior_equiv:  8.0,
      /*prior:        false,
      prior_equiv:  0.0,*/

      /*rave:         true,
      rave_equiv:   3000.0,*/
      rave:         false,
      rave_equiv:   0.0,*/

      tmp_values:   repeat(0.0).take(Board::SIZE).collect(),
    }
  }
}

impl TreePolicy for ThompsonTreePolicy {
  fn use_rave(&self) -> bool {
    self.cfg.rave
  }

  fn execute_search(&mut self, node: &Node, rng: &mut Xorshiftplus128Rng) -> (Option<(Point, usize)>, usize) {
    let horizon = node.values.horizon();
    for j in 0 .. horizon {
      let n = self.cfg.mc_scale * node.values.num_trials[j].load(Ordering::Acquire) as f32;
      let s = self.cfg.mc_scale * node.values.num_succs[j].load(Ordering::Acquire) as f32;
      /*let (pn, ps) = if !self.prior {
        (2.0, 1.0)
      } else {
        (self.prior_equiv, self.prior_equiv * node.values.prior_values[j])
      };*/
      let (pn, ps) = (self.cfg.prior_equiv, self.cfg.prior_equiv * node.values.prior_values[j]);
      let (es, ef) = if !self.cfg.rave {
        (0.0f32.max(s + ps), 0.0f32.max((n + pn) - (s + ps)))
      } else {
        // XXX(20160304)
        unimplemented!();
        /*let rn = node.values.num_trials_rave[j].load(Ordering::Acquire) as f32;
        if rn == 0.0 {
          ((s + ps), 0.0f32.max((n + pn) - (s + ps)))
        } else {
          let rs = node.values.num_succs_rave[j].load(Ordering::Acquire) as f32;
          let beta_rave = rn / (rn + (n + pn) + (n + pn) * rn / self.cfg.rave_equiv);
          let ev =
              0.0f32.max((1.0 - beta_rave) * ((s + ps) / (n + pn)))
              + beta_rave * (rs / rn)
          ;
          // FIXME(20151124): rounding?
          let es = ev * (n + pn);
          let ef = 0.0f32.max((1.0 - ev) * (n + pn));
          (es, ef)
        }*/
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
    let res = array_argmax(&self.tmp_values[ .. horizon])
      .map(|j| (node.valid_moves[j], j));
    (res, horizon)
  }
}
