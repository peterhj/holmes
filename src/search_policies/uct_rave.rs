use board::{Stone, Point};
use search_policies::{TreePolicy};
use search_tree::{Node};

use statistics_avx2::array::{array_argmax};

pub struct UctRaveTreePolicy {
  pub c: f32,
}

impl UctRaveTreePolicy {
  fn update_values(&self, node: &mut Node) {
    let equiv_rave = 3000.0;
    let equiv_prior = 100.0;
    let log_total_trials = node.total_trials.ln();
    let beta_prior = (equiv_prior / (node.total_trials + equiv_prior)).sqrt();
    let num_arms = node.values.len();
    for j in (0 .. num_arms) {
      // XXX: The UCB1-RAVE update rule.
      let n = node.num_trials[j];
      let s = node.num_succs[j];
      let rn = node.num_trials_rave[j];
      let prior_p = node.prior_moves[j].1;
      if rn == 0.0 {
        node.values[j] =
            s / n + self.c * (log_total_trials / n).sqrt()
            + beta_prior * prior_p
        ;
      } else {
        let rs = node.num_succs_rave[j];
        let beta_rave = rn / (rn + n + n * rn / equiv_rave);
        node.values[j] =
            (1.0 - beta_rave) * (s / n + self.c * (log_total_trials / n).sqrt())
            + beta_rave * (rs / rn)
            + beta_prior * prior_p
        ;
      }
    }
  }
}

impl TreePolicy for UctRaveTreePolicy {
  fn rave(&self) -> bool { true }

  fn init_node(&mut self, node: &mut Node) {
    self.update_values(node);
  }

  /*fn execute_greedy(&mut self, node: &Node) -> Option<Point> {
    array_argmax(&node.num_trials[ .. node.horizon]).map(|j| node.prior_moves[j].0)
  }*/

  fn execute_search(&mut self, node: &Node) -> Option<(Point, usize)> {
    /*let res = array_argmax(&node.values[ .. node.horizon]).map(|j| (node.prior_moves[j].0, j));
    if node.values[ .. node.horizon].len() > 0 {
      assert!(res.is_some());
    }*/
    let mut argmax: Option<(usize, f32)> = None;
    for j in (0 .. node.horizon) {
      let x = node.values[j];
      if argmax.is_none() || argmax.unwrap().1 < x {
        argmax = Some((j, x));
      }
    }
    if node.values[ .. node.horizon].len() > 0 {
      assert!(argmax.is_some());
    }
    argmax.map(|(j, _)| (node.prior_moves[j].0, j))
  }

  fn backup(&mut self, node: &mut Node) {
    self.update_values(node);
  }
}
