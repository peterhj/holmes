use board::{Stone, Point};
use features::{TxnStateFeaturesData};
use search_tree::{Node};
use txnstate::{TxnState, TxnStateNodeData};

pub mod convnet;
pub mod quasiuniform;
pub mod uct_rave;

pub trait PriorPolicy {
  type Ctx;

  fn init_prior(&mut self,
      node_state: &TxnState<TxnStateNodeData>,
      total_trials: &mut f32,
      num_trials: &mut [f32],
      num_succs: &mut [f32],
      ctx: &Self::Ctx);
}

pub trait TreePolicy {
  fn rave(&self) -> bool;
  fn init_node(&mut self, node: &mut Node);
  fn execute_greedy(&mut self, node: &Node) -> Option<Point>;
  fn execute_search(&mut self, node: &Node) -> Option<(Point, usize)>;
  fn backup(&mut self, node: &mut Node);
}

pub trait RolloutPolicy {
}
