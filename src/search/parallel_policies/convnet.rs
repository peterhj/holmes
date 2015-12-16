use random::{XorShift128PlusRng};
use search::parallel_policies::{
  SearchPolicyWorkerBuilder, SearchPolicyWorker,
  PriorPolicy, TreePolicy, RolloutPolicy,
};

#[derive(Clone, Copy)]
pub struct ConvnetPolicyWorkerBuilder;

impl SearchPolicyWorkerBuilder for ConvnetPolicyWorkerBuilder {
  type Worker = ConvnetPolicyWorker;

  fn build_worker(&self) -> ConvnetPolicyWorker {
    unimplemented!();
  }
}

pub struct ConvnetPolicyWorker;

impl SearchPolicyWorker for ConvnetPolicyWorker {
  fn prior_policy(&mut self) -> &mut PriorPolicy {
    unimplemented!();
  }

  fn tree_policy(&mut self) -> &mut TreePolicy<R=XorShift128PlusRng> {
    unimplemented!();
  }

  fn rollout_policy(&mut self) -> &mut RolloutPolicy<R=XorShift128PlusRng> {
    unimplemented!();
  }
}
