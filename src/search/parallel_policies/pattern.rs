use search::parallel_policies::convnet::{};

pub struct ConvnetPatternPolicyWorker {
  prior_policy:     ConvnetPriorPolicy,
  tree_policy:      ThompsonTreePolicy,
  rollout_policy:   PatternRolloutPolicy,
}

pub struct PatternRolloutPolicy {
  batch_size:   usize,
}

impl RolloutPolicy for PatternRolloutPolicy {
  fn batch_size(&self) -> usize {
    self.batch_size
  }

  fn max_rollout_len(&self) -> usize {
    let max_iters = 361 + 361 / 2 + 1;
    max_iters
  }

  fn rollout_batch(&mut self,
      batch_size:       usize,
      leafs:            RolloutLeafs,
      rollout_trajs:    &mut [RolloutTraj],
      pass_only:        Option<Stone>,
      mut trace_batch:  Option<&mut SearchTraceBatch>,
      rng:              &mut Xorshiftplus128Rng)
  {
    // FIXME(20160413)
    unimplemented!();
  }
}
