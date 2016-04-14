use board::{Board, Action, Stone, Point};
use convnet_new::{
  build_13layer384multi3_19x19x32_arch_nodir,
};
use discrete::{DiscreteFilter};
use discrete::bfilter::{BFilter};
use random::{choose_without_replace};
use pattern::{
  PatternGammaDatabase,
  InvariantLibPattern3x3,
};
use search::parallel_policies::{
  SearchPolicyWorkerBuilder, SearchPolicyWorker,
  PriorPolicy, DiffPriorPolicy, TreePolicy,
  GradAccumMode, GradSyncMode,
  RolloutPolicyBuilder, RolloutMode, RolloutLeafs, RolloutPolicy,
};
use search::parallel_policies::convnet::{
  ConvnetPriorPolicy,
};
use search::parallel_policies::thompson::{
  ThompsonTreePolicy,
};
use search::parallel_tree::{TreePolicyConfig, TreeTraj, RolloutTraj, QuickTrace};
use search::parallel_trace::{SearchTraceBatch};
use txnstate::{TxnState, check_good_move_fast};
use txnstate::extras::{TxnStateNodeData, for_each_touched_empty};

use array_cuda::device::{DeviceContext, for_all_devices};
use float::ord::{F32SupNan};
use rembrandt::arch_new::{
  Worker, ArchWorker,
  PipelineArchConfig, PipelineArchSharedData, PipelineArchWorker,
};
use rembrandt::data_new::{SampleLabel};
use rembrandt::layer_new::{Phase};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, thread_rng};
use std::path::{Path, PathBuf};
use std::rc::{Rc};
use std::sync::{Arc};

#[derive(Clone)]
pub struct ConvnetPatternPolicyWorkerBuilder {
  tree_cfg:             TreePolicyConfig,
  prior_arch_cfg:       PipelineArchConfig,
  prior_save_path:      PathBuf,
  prior_shared:         Arc<PipelineArchSharedData>,
  prior_shared2:        Arc<()>,
}

impl ConvnetPatternPolicyWorkerBuilder {
  pub fn new(tree_cfg: TreePolicyConfig, num_workers: usize, worker_tree_batch_size: usize, worker_batch_size: usize) -> ConvnetPatternPolicyWorkerBuilder {
    let prior_arch_cfg = build_13layer384multi3_19x19x32_arch_nodir(worker_tree_batch_size);
    let prior_save_path = PathBuf::from("models/gogodb_w2015-preproc-alphav3m_19x19x32_13layer384multi3.saved");

    let prior_shared = for_all_devices(num_workers, |contexts| {
      Arc::new(PipelineArchSharedData::new(num_workers, &prior_arch_cfg, contexts))
    });

    ConvnetPatternPolicyWorkerBuilder{
      tree_cfg:             tree_cfg,
      prior_arch_cfg:       prior_arch_cfg,
      prior_save_path:      prior_save_path,
      prior_shared:         prior_shared,
      prior_shared2:        Arc::new(()),
    }
  }
}

impl SearchPolicyWorkerBuilder for ConvnetPatternPolicyWorkerBuilder {
  type Worker = ConvnetPatternPolicyWorker;

  fn into_worker(self, tid: usize, worker_tree_batch_size: usize, worker_batch_size: usize) -> ConvnetPatternPolicyWorker {
    assert_eq!(worker_tree_batch_size, worker_batch_size);
    let context = Rc::new(DeviceContext::new(tid));
    let ctx = (*context).as_ref();
    let mut prior_arch = PipelineArchWorker::new(
        worker_tree_batch_size,
        self.prior_arch_cfg,
        self.prior_save_path,
        tid,
        [thread_rng().next_u64(), thread_rng().next_u64()],
        &self.prior_shared,
        self.prior_shared2.clone(),
        &ctx,
    );
    prior_arch.load_layer_params(None, &ctx);
    let prior_policy = ConvnetPriorPolicy{
      context:  context.clone(),
      arch:     prior_arch,
    };
    let tree_policy = ThompsonTreePolicy::new(self.tree_cfg);
    let rollout_policy = PatternRolloutPolicy{
      batch_size:   worker_batch_size,
      pattern_db:   PatternGammaDatabase::open(&PathBuf::from("pat_gammas.db")),
    };
    ConvnetPatternPolicyWorker{
      prior_policy:     prior_policy,
      tree_policy:      tree_policy,
      rollout_policy:   rollout_policy,
    }
  }
}

pub struct ConvnetPatternPolicyWorker {
  prior_policy:     ConvnetPriorPolicy,
  tree_policy:      ThompsonTreePolicy,
  rollout_policy:   PatternRolloutPolicy,
}

impl SearchPolicyWorker for ConvnetPatternPolicyWorker {
  fn prior_policy(&mut self) -> &mut PriorPolicy {
    &mut self.prior_policy
  }

  fn diff_prior_policy(&mut self) -> &mut DiffPriorPolicy {
    &mut self.prior_policy
  }

  fn tree_policy(&mut self) -> &mut TreePolicy<R=Xorshiftplus128Rng> {
    &mut self.tree_policy
  }

  fn exploration_policies(&mut self) -> (&mut PriorPolicy, &mut TreePolicy<R=Xorshiftplus128Rng>) {
    (&mut self.prior_policy, &mut self.tree_policy)
  }

  fn rollout_policy(&mut self) -> &mut RolloutPolicy<R=Xorshiftplus128Rng> {
    &mut self.rollout_policy
  }
}

pub struct PatternRolloutPolicy {
  batch_size:   usize,
  pattern_db:   PatternGammaDatabase,
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
      _pass_only:       Option<Stone>,
      mut _trace_batch: Option<&mut SearchTraceBatch>,
      rng:              &mut Xorshiftplus128Rng)
  {
    assert!(batch_size <= rollout_trajs.len());
    assert!(_pass_only.is_none());
    assert!(_trace_batch.is_none());

    let mut valid_move_set = vec![vec![], vec![]];
    // XXX(20160124): Valid move iterator is an "upper bound" on the valid moves,
    // since it is easy to add elements but hard to remove them.
    let mut valid_move_iter: Vec<Vec<Vec<usize>>> = vec![vec![], vec![]];
    let mut bad_moves: Vec<Vec<Vec<Point>>> = vec![vec![], vec![]];
    let mut turn_pass = vec![vec![], vec![]];
    let mut num_both_passed = 0;
    let mut filters = vec![vec![], vec![]];
    for batch_idx in 0 .. batch_size {
      leafs.with_leaf_state(batch_idx, |leaf_state| {
        valid_move_set[0].push(leaf_state.get_data().legality.legal_points(Stone::Black));
        valid_move_set[1].push(leaf_state.get_data().legality.legal_points(Stone::White));
        valid_move_iter[0].push(valid_move_set[0][batch_idx].iter().collect());
        valid_move_iter[1].push(valid_move_set[1][batch_idx].iter().collect());
      });
      bad_moves[0].push(vec![]);
      bad_moves[1].push(vec![]);
      turn_pass[0].push(false);
      turn_pass[1].push(false);
      filters[0].push(BFilter::with_capacity(Board::SIZE));
      filters[1].push(BFilter::with_capacity(Board::SIZE));
    }

    for batch_idx in 0 .. batch_size {
      // FIXME(20160413): initialize discrete samplers.
      //filters[0][batch_idx].reset();
      //filters[1][batch_idx].reset();

      let max_iters = 361 + 361 / 2 + rng.gen_range(0, 2);
      for t in 0 .. max_iters {
        let turn = rollout_trajs[batch_idx].sim_state.current_turn();
        // FIXME(20160413)
        unimplemented!();
        /*let pat = InvariantLibPattern3x3::from_repr_bytes(
            rollout_trajs[batch_idx].sim_state.get_data().features,
        );*/
      }
    }

    // FIXME(20160413)
    unimplemented!();
  }
}
