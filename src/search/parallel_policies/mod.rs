use board::{Stone, Action, Point};
//use random::{XorShift128PlusRng};
use search::parallel_tree::{TreeTraj, RolloutTraj, QuickTrace, Node};
use search::parallel_trace::{SearchTraceBatch};
use txnstate::{TxnState};
use txnstate::extras::{TxnStateNodeData};

use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng};
use std::path::{Path};

pub mod convnet;
//pub mod shaped_thompson;
pub mod thompson;
pub mod ucb;

pub trait SearchPolicyWorkerBuilder: Send + Clone {
  type Worker: SearchPolicyWorker;

  //fn build_worker(&self, tid: usize, worker_batch_size: usize) -> Self::Worker;
  fn into_worker(self, tid: usize, worker_batch_size: usize) -> Self::Worker;
}

pub trait SearchPolicyWorker {
  fn prior_policy(&mut self) -> &mut PriorPolicy;
  fn diff_prior_policy(&mut self) -> &mut DiffPriorPolicy;
  fn tree_policy(&mut self) -> &mut TreePolicy<R=Xorshiftplus128Rng>;
  fn exploration_policies(&mut self) -> (&mut PriorPolicy, &mut TreePolicy<R=Xorshiftplus128Rng>);
  fn rollout_policy(&mut self) -> &mut RolloutPolicy<R=Xorshiftplus128Rng>;
}

pub trait PriorPolicy {
  fn fill_prior_values(&mut self, state: &TxnState<TxnStateNodeData>, valid_moves: &[Point], prior_values: &mut Vec<(Point, f32)>);
}

pub enum GradAccumMode {
  Add{scale: f32},
}

pub enum GradSyncMode {
  Sum,
  //Average,
}

pub trait DiffPriorPolicy: PriorPolicy {
  fn expose_input_buffer(&mut self, batch_idx: usize) -> &mut [u8];
  fn preload_action_label(&mut self, batch_idx: usize, action: Action);
  fn preload_loss_weight(&mut self, batch_idx: usize, weight: f32);
  fn load_inputs(&mut self, batch_size: usize);
  fn forward(&mut self, batch_size: usize);
  fn backward(&mut self, batch_size: usize);
  fn read_values(&mut self, batch_size: usize);
  //fn accumulate_gradients(&mut self, accum_mode: GradAccumMode);
  fn sync_gradients(&mut self, sync_mode: GradSyncMode);
  fn reset_gradients(&mut self);
  fn descend_params(&mut self, step_size: f32);
}

pub trait TreePolicy {
  type R: Rng = Xorshiftplus128Rng;

  fn use_rave(&self) -> bool;
  fn execute_search(&mut self, node: &Node, rng: &mut Self::R) -> (Option<(Point, usize)>, usize);
}

pub trait RolloutPolicyBuilder: Send + Clone {
  //type R: Rng = Xorshiftplus128Rng;
  type Policy: RolloutPolicy<R=Xorshiftplus128Rng>;

  fn into_rollout_policy(self, tid: usize, batch_size: usize) -> Self::Policy;
}

pub enum RolloutLeafs<'a> {
  TreeTrajs(&'a [TreeTraj]),
  LeafStates(&'a [TxnState<TxnStateNodeData>]),
}

impl<'a> RolloutLeafs<'a> {
  pub fn with_leaf_state<F>(&self, idx: usize, mut f: F)
  where F: FnMut(&TxnState<TxnStateNodeData>) {
    match self {
      &RolloutLeafs::TreeTrajs(tree_trajs) => {
        f(&tree_trajs[idx].leaf_node.as_ref().unwrap().read().unwrap().state);
      }
      &RolloutLeafs::LeafStates(leaf_states) => {
        f(&leaf_states[idx]);
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
  fn max_rollout_len(&self) -> usize;
  fn rollout_batch(&mut self,
      batch_size:       usize,
      //green_stone:      Stone,
      leafs:            RolloutLeafs,
      rollout_trajs:    &mut [RolloutTraj],
      pass_only:        Option<Stone>,
      mut trace_batch:  Option<&mut SearchTraceBatch>,
      record_trace: bool, traces: &mut [QuickTrace], // FIXME(20160226): deprecated.
      rng:              &mut Self::R,
  );

  fn init_traces(&mut self);
  //fn rollout_trace(&mut self, trace: &Trace, baseline: f32) -> bool;
  fn rollout_trace(&mut self, trace: &QuickTrace, baseline: f32) -> bool;
  fn backup_traces(&mut self, learning_rate: f32, target_value: f32, eval_value: f32, num_traces: usize);

  fn save_params(&mut self, save_dir: &Path, t: usize);

  fn rollout_green_trace(&mut self, baseline: f32, green_stone: Stone, trace: &QuickTrace) -> bool;
  fn backup_green_traces(&mut self, step_size: f32, traces_count: usize, green_stone: Stone);
  fn load_green_params(&mut self, blob: &[u8]) { unimplemented!(); }
  fn load_red_params(&mut self, blob: &[u8]) { unimplemented!(); }
  fn save_green_params(&mut self) -> Vec<u8> { unimplemented!(); }
  fn save_red_params(&mut self) -> Vec<u8> { unimplemented!(); }
}
