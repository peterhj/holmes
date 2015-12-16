use board::{Stone, Point};
use random::{XorShift128PlusRng};
use search::parallel_tree::{Walk, Trajectory, Node};
use txnstate::{TxnState};
use txnstate::extras::{TxnStateNodeData};

use rand::{Rng};

pub mod convnet;
pub mod thompson;

pub trait SearchPolicyWorkerBuilder: Send + Clone {
  type Worker: SearchPolicyWorker;

  fn build_worker(&self) -> Self::Worker;
}

pub trait SearchPolicyWorker {
  fn prior_policy(&mut self) -> &mut PriorPolicy;
  fn tree_policy(&mut self) -> &mut TreePolicy<R=XorShift128PlusRng>;
  fn rollout_policy(&mut self) -> &mut RolloutPolicy<R=XorShift128PlusRng>;
}

pub trait PriorPolicy {
  fn fill_prior_values(&mut self, state: &TxnState<TxnStateNodeData>, valid_moves: &[Point], prior_values: &mut Vec<f32>);
}

pub trait TreePolicy {
  type R: Rng = XorShift128PlusRng;

  fn init_node(&mut self, node: &mut Node, rng: &mut Self::R);
  fn execute_search(&mut self, node: &Node, rng: &mut Self::R) -> Option<(Point, usize)>;
  fn backup_values(&mut self, node: &Node, rng: &mut Self::R);
}

pub trait RolloutPolicy {
  type R: Rng = XorShift128PlusRng;

  fn batch_size(&self) -> usize;
  fn rollout_batch(&mut self, walks: &[Walk], trajs: &mut [Trajectory], rng: &mut Self::R);
}
