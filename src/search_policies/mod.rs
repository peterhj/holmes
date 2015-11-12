use board::{Stone, Point};
use features::{TxnStateFeaturesData};
use search_tree::{Node};
use txnstate::{TxnState};

pub mod uct_rave;

pub trait PriorPolicy {
  fn init_prior(&mut self,
      node_state: &TxnState<TxnStateFeaturesData>,
      total_trials: &mut f32,
      num_trials: &mut [f32],
      num_succs: &mut [f32]);
}

pub trait TreePolicy {
  fn init(&mut self, node: &mut Node);
  fn execute_greedy(&mut self, node: &Node) -> Option<Point>;
  fn execute_search(&mut self, node: &Node) -> Option<(Point, usize)>;
  fn backup(&mut self, node: &mut Node);
}

pub trait RolloutPolicy {
}
