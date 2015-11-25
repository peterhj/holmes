use array_util::{array_argmax};
use board::{Stone, Point};
use hyper::{
  UCB_C,
  RAVE,
  RAVE_EQUIV,
  PBIAS_EQUIV,
};
use search::{Node};
use search::policies::{TreePolicy};

pub struct UctRaveTreePolicy {
  ucb_c:        f32,
  rave:         bool,
  rave_equiv:   f32,
  pbias_equiv:  f32,
}

impl UctRaveTreePolicy {
  pub fn new() -> UctRaveTreePolicy {
    UctRaveTreePolicy{
      ucb_c:        UCB_C.with(|x| *x),
      rave:         RAVE.with(|x| *x),
      rave_equiv:   RAVE_EQUIV.with(|x| *x),
      pbias_equiv:  PBIAS_EQUIV.with(|x| *x),
    }
  }
}

impl TreePolicy for UctRaveTreePolicy {
  fn rave(&self) -> bool {
    self.rave
  }

  fn init_node(&mut self, node: &mut Node, rng: &mut Self::R) {
    self.backup_values(node, rng);
  }

  fn execute_search(&mut self, node: &Node, rng: &mut Self::R) -> Option<(Point, usize)> {
    array_argmax(&node.values[ .. node.horizon])
      .map(|j| (node.prior_moves[j].0, j))
  }

  fn backup_values(&mut self, node: &mut Node, rng: &mut Self::R) {
    let log_total_trials = node.total_trials.ln();
    let beta_pbias = (self.pbias_equiv / (node.total_trials + self.pbias_equiv)).sqrt();
    //let num_arms = node.values.len();
    //for j in (0 .. num_arms) {
    for j in (0 .. node.horizon) {
      // XXX: The UCB1-RAVE update rule.
      let n = node.num_trials[j];
      let s = node.num_succs[j];
      let rn = node.num_trials_rave[j];
      let prior_p = node.prior_moves[j].1;
      if !self.rave || rn == 0.0 {
        node.values[j] =
            s / n + self.ucb_c * (log_total_trials / n).sqrt()
            + beta_pbias * prior_p
        ;
      } else {
        let rs = node.num_succs_rave[j];
        let beta_rave = rn / (rn + n + n * rn / self.rave_equiv);
        node.values[j] =
            0.0f32.max(1.0 - beta_rave) * (s / n + self.ucb_c * (log_total_trials / n).sqrt())
            + beta_rave * (rs / rn)
            + beta_pbias * prior_p
        ;
      }
    }
  }
}
