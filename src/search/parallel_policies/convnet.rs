use random::{XorShift128PlusRng};
use search::parallel_policies::{
  SearchPolicyWorkerBuilder, SearchPolicyWorker,
  PriorPolicy, TreePolicy, RolloutPolicy,
};
use search::parallel_policies::thompson::{ThompsonTreePolicy};

#[derive(Clone, Copy)]
pub struct ConvnetPolicyWorkerBuilder;

impl SearchPolicyWorkerBuilder for ConvnetPolicyWorkerBuilder {
  type Worker = ConvnetPolicyWorker;

  fn build_worker(&self) -> ConvnetPolicyWorker {
    unimplemented!();
  }
}

pub struct ConvnetPolicyWorker {
  prior_policy:     ConvnetPriorPolicy,
  tree_policy:      ThompsonTreePolicy,
  rollout_policy:   ConvnetRolloutPolicy,
}

impl ConvnetPolicyWorker {
  pub fn new() -> ConvnetPolicyWorker {
    ConvnetPolicyWorker{
      prior_policy:     ConvnetPriorPolicy,
      tree_policy:      ThompsonTreePolicy,
      rollout_policy:   ConvnetRolloutPolicy,
    }
  }
}

impl SearchPolicyWorker for ConvnetPolicyWorker {
  fn prior_policy(&mut self) -> &mut PriorPolicy {
    unimplemented!();
  }

  fn tree_policy(&mut self) -> &mut TreePolicy<R=XorShift128PlusRng> {
    unimplemented!();
  }

  fn prior_and_tree_policies(&mut self) -> (&mut PriorPolicy, &mut TreePolicy<R=XorShift128PlusRng>) {
    unimplemented!();
  }

  fn rollout_policy(&mut self) -> &mut RolloutPolicy<R=XorShift128PlusRng> {
    unimplemented!();
  }
}

pub struct ConvnetPriorPolicy;

pub struct ConvnetRolloutPolicy;
