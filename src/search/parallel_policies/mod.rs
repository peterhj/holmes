use board::{Stone, Point};
//use random::{XorShift128PlusRng};
use search::parallel_tree::{TreeTraj, RolloutTraj, Trace, Node};
use txnstate::{TxnState};
use txnstate::extras::{TxnStateNodeData};

use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng};

pub mod convnet;
pub mod thompson;

pub trait SearchPolicyWorkerBuilder: Send + Clone {
  type Worker: SearchPolicyWorker;

  //fn build_worker(&self, tid: usize, worker_batch_size: usize) -> Self::Worker;
  fn into_worker(self, tid: usize, worker_batch_size: usize) -> Self::Worker;
}

pub trait SearchPolicyWorker {
  fn prior_policy(&mut self) -> &mut PriorPolicy;
  fn tree_policy(&mut self) -> &mut TreePolicy<R=Xorshiftplus128Rng>;
  fn prior_and_tree_policies(&mut self) -> (&mut PriorPolicy, &mut TreePolicy<R=Xorshiftplus128Rng>);
  fn rollout_policy(&mut self) -> &mut RolloutPolicy<R=Xorshiftplus128Rng>;
}

pub trait PriorPolicy {
  fn fill_prior_values(&mut self, state: &TxnState<TxnStateNodeData>, valid_moves: &[Point], prior_values: &mut Vec<(Point, f32)>);
}

pub trait TreePolicy {
  type R: Rng = Xorshiftplus128Rng;

  fn execute_search(&mut self, node: &Node, rng: &mut Self::R) -> Option<(Point, usize)>;
}

pub trait RolloutPolicyBuilder: Send + Clone {
  //type R: Rng = Xorshiftplus128Rng;
  type Policy: RolloutPolicy<R=Xorshiftplus128Rng>;

  fn build_rollout_policy(&self, tid: usize, batch_size: usize) -> Self::Policy;
}

pub enum RolloutLeafs<'a> {
  TreeTrajs(&'a [TreeTraj]),
  States(&'a [TxnState<TxnStateNodeData>]),
}

impl<'a> RolloutLeafs<'a> {
  pub fn with_leaf_state<F>(&self, idx: usize, mut f: F)
  where F: FnMut(&TxnState<TxnStateNodeData>) {
    match self {
      &RolloutLeafs::TreeTrajs(tree_trajs) => {
        f(&tree_trajs[idx].leaf_node.as_ref().unwrap().read().unwrap().state);
      }
      &RolloutLeafs::States(states) => {
        f(&states[idx]);
      }
    }
  }
}

pub enum RolloutMode {
  Simulation,
  BalanceTraining,
}

pub trait RolloutPolicy {
  type R: Rng = Xorshiftplus128Rng;

  fn batch_size(&self) -> usize;
  fn rollout_batch(&mut self, leafs: RolloutLeafs, rollout_trajs: &mut [RolloutTraj], mode: RolloutMode, rng: &mut Self::R);
  fn rollout_trace(&mut self, trace: &Trace, mode: RolloutMode);
  fn descend_params(&mut self, scale: f32);
}
