use board::{Stone, Point};
use random::{XorShift128PlusRng};
use search::parallel_tree::{TreeTraj, RolloutTraj, Node};
use txnstate::{TxnState};
use txnstate::extras::{TxnStateNodeData};

use rand::{Rng};

pub mod convnet;
pub mod thompson;

pub trait SearchPolicyWorkerBuilder: Send + Clone {
  type Worker: SearchPolicyWorker;

  fn build_worker(&self, tid: usize, worker_batch_size: usize) -> Self::Worker;
}

pub trait SearchPolicyWorker {
  fn prior_policy(&mut self) -> &mut PriorPolicy;
  fn tree_policy(&mut self) -> &mut TreePolicy<R=XorShift128PlusRng>;
  fn prior_and_tree_policies(&mut self) -> (&mut PriorPolicy, &mut TreePolicy<R=XorShift128PlusRng>);
  fn rollout_policy(&mut self) -> &mut RolloutPolicy<R=XorShift128PlusRng>;
}

pub trait PriorPolicy {
  fn fill_prior_values(&mut self, state: &TxnState<TxnStateNodeData>, valid_moves: &[Point], prior_values: &mut Vec<(Point, f32)>);
}

pub trait TreePolicy {
  type R: Rng = XorShift128PlusRng;

  fn execute_search(&mut self, node: &Node, rng: &mut Self::R) -> Option<(Point, usize)>;
}

pub trait RolloutPolicy {
  type R: Rng = XorShift128PlusRng;

  fn batch_size(&self) -> usize;
  fn rollout_batch(&mut self, tree_trajs: &[TreeTraj], rollout_trajs: &mut [RolloutTraj], rng: &mut Self::R);
}
