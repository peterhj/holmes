use board::{Stone, Point};
use features::{TxnStateFeaturesData};
use search_tree::{Node};
use txnstate::{TxnState, TxnStateNodeData};

pub mod convnet;
pub mod quasiuniform;
pub mod uct_rave;

pub trait PriorPolicy {
  //fn init_prior(&mut self, node: &mut Node, ctx: &Self::Ctx);
  fn fill_prior_probs(&mut self, state: &TxnState<TxnStateNodeData>, valid_moves: &[Point], prior_probs: &mut Vec<(Point, f32)>);
}

pub trait TreePolicy {
  fn rave(&self) -> bool;
  fn init_node(&mut self, node: &mut Node);
  //fn execute_greedy(&mut self, node: &Node) -> Option<Point>;
  fn execute_search(&mut self, node: &Node) -> Option<(Point, usize)>;
  fn backup(&mut self, node: &mut Node);
}

pub trait RolloutPolicy {
}
