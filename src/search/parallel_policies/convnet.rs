use board::{Point};
use convnet_new::{
  /*build_2layer16_19x19x16_arch,
  build_3layer32_19x19x16_arch,
  build_12layer384_19x19x30_arch,*/
};
use random::{XorShift128PlusRng};
use search::parallel_policies::{
  SearchPolicyWorkerBuilder, SearchPolicyWorker,
  PriorPolicy, TreePolicy, RolloutPolicy,
};
use search::parallel_policies::thompson::{ThompsonTreePolicy};
use search::parallel_tree::{TreeTraj, RolloutTraj};
use txnstate::{TxnState};
use txnstate::extras::{TxnStateNodeData};

use array_cuda::device::{DeviceContext};
//use rembrandt::arch_new::{PipelineArchWorker};

use std::rc::{Rc};

#[derive(Clone, Copy)]
pub struct ConvnetPolicyWorkerBuilder;

impl SearchPolicyWorkerBuilder for ConvnetPolicyWorkerBuilder {
  type Worker = ConvnetPolicyWorker;

  fn build_worker(&self, tid: usize, worker_batch_size: usize) -> ConvnetPolicyWorker {
    let context = Rc::new(DeviceContext::new(tid));
    ConvnetPolicyWorker{
      prior_policy:     ConvnetPriorPolicy{
        context:          context.clone(),
      },
      tree_policy:      ThompsonTreePolicy::new(),
      rollout_policy:   ConvnetRolloutPolicy{
        context:          context,
        batch_size:       worker_batch_size,
      }
    }
  }
}

pub struct ConvnetPolicyWorker {
  prior_policy:     ConvnetPriorPolicy,
  tree_policy:      ThompsonTreePolicy,
  rollout_policy:   ConvnetRolloutPolicy,
}

impl SearchPolicyWorker for ConvnetPolicyWorker {
  fn prior_policy(&mut self) -> &mut PriorPolicy {
    &mut self.prior_policy
  }

  fn tree_policy(&mut self) -> &mut TreePolicy<R=XorShift128PlusRng> {
    &mut self.tree_policy
  }

  fn prior_and_tree_policies(&mut self) -> (&mut PriorPolicy, &mut TreePolicy<R=XorShift128PlusRng>) {
    (&mut self.prior_policy, &mut self.tree_policy)
  }

  fn rollout_policy(&mut self) -> &mut RolloutPolicy<R=XorShift128PlusRng> {
    &mut self.rollout_policy
  }
}

pub struct ConvnetPriorPolicy {
  context:  Rc<DeviceContext>,
}

impl PriorPolicy for ConvnetPriorPolicy {
  fn fill_prior_values(&mut self, state: &TxnState<TxnStateNodeData>, valid_moves: &[Point], prior_values: &mut Vec<(Point, f32)>) {
    prior_values.clear();

    // TODO(20151222)
    unimplemented!();
  }
}

pub struct ConvnetRolloutPolicy {
  context:      Rc<DeviceContext>,
  batch_size:   usize,
}

impl RolloutPolicy for ConvnetRolloutPolicy {
  fn batch_size(&self) -> usize {
    self.batch_size
  }

  fn rollout_batch(&mut self, tree_trajs: &[TreeTraj], rollout_trajs: &mut [RolloutTraj], rng: &mut XorShift128PlusRng) {
    // TODO(20151224)
    unimplemented!();
  }
}
