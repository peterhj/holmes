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
    let log_total_trials = node.total_trials.ln();
    let num_arms = node.values.len();
    for j in (0 .. num_arms) {
      // XXX: The UCB1-RAVE update rule.
      let n = node.num_trials[j];
      let s = node.num_succs[j];
      let rn = node.num_trials_rave[j];
      if rn == 0.0 {
        node.values[j] = s / n + self.c * (log_total_trials / n).sqrt();
      } else {
        let rs = node.num_succs_rave[j];
        let beta_rave = rn / (rn + n + n * rn / equiv_rave);
        node.values[j] =
            (1.0 - beta_rave) * (s / n + self.c * (log_total_trials / n).sqrt())
            + beta_rave * (rs / rn + self.c * (log_total_trials / rn).sqrt());
      }
      // TODO(20151111): can also compare the KL-UCB rule.
    }
  }
}

impl TreePolicy for UctRaveTreePolicy {
  fn init(&mut self, node: &mut Node) {
    self.update_values(node);
  }

  fn execute_greedy(&mut self, node: &Node) -> Option<Point> {
    array_argmax(&node.num_trials).map(|j| node.valid_moves[j])
  }

  fn execute_search(&mut self, node: &Node) -> Option<(Point, usize)> {
    array_argmax(&node.values).map(|j| (node.valid_moves[j], j))
  }

  fn backup(&mut self, node: &mut Node) {
    self.update_values(node);
  }
}
