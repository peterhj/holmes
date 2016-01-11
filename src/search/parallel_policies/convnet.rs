use board::{Board, Action, Stone, Point};
use convnet_new::{
  build_2layer16_19x19x16_arch,
  build_12layer128_19x19x16_arch,
  /*build_3layer32_19x19x16_arch,
  build_12layer384_19x19x30_arch,*/
};
use discrete::{DiscreteFilter};
use discrete::bfilter::{BFilter};
use search::parallel_policies::{
  SearchPolicyWorkerBuilder, SearchPolicyWorker,
  PriorPolicy, TreePolicy,
  RolloutPolicyBuilder, RolloutMode, RolloutLeafs, RolloutPolicy,
};
use search::parallel_policies::thompson::{ThompsonTreePolicy};
use search::parallel_tree::{TreeTraj, RolloutTraj, Trace};
use txnstate::{TxnState, check_good_move_fast};
use txnstate::extras::{TxnStateNodeData, for_each_touched_empty};

use array_cuda::device::{DeviceContext, for_all_devices};
use rembrandt::arch_new::{
  ArchWorker,
  PipelineArchConfig, PipelineArchSharedData, PipelineArchWorker,
};
use rembrandt::layer_new::{Phase};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng};
use std::path::{PathBuf};
use std::rc::{Rc};
use std::sync::{Arc};

#[derive(Clone)]
pub struct ConvnetPolicyWorkerBuilder {
  prior_arch_cfg:       PipelineArchConfig,
  prior_save_path:      PathBuf,
  prior_shared:         Arc<PipelineArchSharedData>,
  prior_shared2:        Arc<()>,
  rollout_arch_cfg:     PipelineArchConfig,
  rollout_save_path:    PathBuf,
  rollout_shared:       Arc<PipelineArchSharedData>,
  rollout_shared2:      Arc<()>,
}

impl ConvnetPolicyWorkerBuilder {
  pub fn new(num_workers: usize, worker_batch_size: usize) -> ConvnetPolicyWorkerBuilder {
    let (prior_arch_cfg, prior_save_path) = build_12layer128_19x19x16_arch(1);
    let (rollout_arch_cfg, rollout_save_path) = build_2layer16_19x19x16_arch(worker_batch_size);
    let prior_shared = for_all_devices(num_workers, |contexts| {
      Arc::new(PipelineArchSharedData::new(num_workers, &prior_arch_cfg, contexts))
    });
    let rollout_shared = for_all_devices(num_workers, |contexts| {
      Arc::new(PipelineArchSharedData::new(num_workers, &rollout_arch_cfg, contexts))
    });
    ConvnetPolicyWorkerBuilder{
      prior_arch_cfg:       prior_arch_cfg,
      prior_save_path:      prior_save_path,
      prior_shared:         prior_shared,
      prior_shared2:        Arc::new(()),
      rollout_arch_cfg:     rollout_arch_cfg,
      rollout_save_path:    rollout_save_path,
      rollout_shared:       rollout_shared,
      rollout_shared2:      Arc::new(()),
    }
  }
}

impl SearchPolicyWorkerBuilder for ConvnetPolicyWorkerBuilder {
  type Worker = ConvnetPolicyWorker;

  fn into_worker(self, tid: usize, worker_batch_size: usize) -> ConvnetPolicyWorker {
    let context = Rc::new(DeviceContext::new(tid));
    let ctx = (*context).as_ref();
    let mut prior_arch = PipelineArchWorker::new(
        1,
        self.prior_arch_cfg,
        self.prior_save_path,
        tid,
        &self.prior_shared,
        self.prior_shared2.clone(),
        &ctx,
    );
    prior_arch.load_layer_params(None, &ctx);
    let mut rollout_arch = PipelineArchWorker::new(
        worker_batch_size,
        self.rollout_arch_cfg,
        self.rollout_save_path,
        tid,
        &self.rollout_shared,
        self.rollout_shared2.clone(),
        &ctx,
    );
    rollout_arch.load_layer_params(None, &ctx);
    let prior_policy = ConvnetPriorPolicy{
      context:  context.clone(),
      arch:     prior_arch,
    };
    let tree_policy = ThompsonTreePolicy::new();
    let rollout_policy = ConvnetRolloutPolicy{
      context:      context.clone(),
      batch_size:   worker_batch_size,
      arch:         rollout_arch,
    };
    ConvnetPolicyWorker{
      prior_policy:     prior_policy,
      tree_policy:      tree_policy,
      rollout_policy:   rollout_policy,
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

  fn tree_policy(&mut self) -> &mut TreePolicy<R=Xorshiftplus128Rng> {
    &mut self.tree_policy
  }

  fn prior_and_tree_policies(&mut self) -> (&mut PriorPolicy, &mut TreePolicy<R=Xorshiftplus128Rng>) {
    (&mut self.prior_policy, &mut self.tree_policy)
  }

  fn rollout_policy(&mut self) -> &mut RolloutPolicy<R=Xorshiftplus128Rng> {
    &mut self.rollout_policy
  }
}

pub struct ConvnetPriorPolicy {
  context:  Rc<DeviceContext>,
  arch:     PipelineArchWorker<()>,
}

impl PriorPolicy for ConvnetPriorPolicy {
  fn fill_prior_values(&mut self, state: &TxnState<TxnStateNodeData>, valid_moves: &[Point], prior_values: &mut Vec<(Point, f32)>) {
    let ctx = (*self.context).as_ref();
    let turn = state.current_turn();
    state.get_data().features.extract_relative_features(turn, self.arch.input_layer().expose_host_frame_buf(0));
    self.arch.input_layer().load_frames(1, &ctx);
    self.arch.forward(1, Phase::Inference, &ctx);
    self.arch.loss_layer().store_probs(1, &ctx);
    let pred_probs = self.arch.loss_layer().get_probs(1);
    prior_values.clear();
    for &point in valid_moves.iter() {
      prior_values.push((point, pred_probs.as_slice()[point.idx()]));
    }
  }
}

#[derive(Clone, Copy)]
pub struct ConvnetRolloutPolicyBuilder;

impl RolloutPolicyBuilder for ConvnetRolloutPolicyBuilder {
  type Policy = ConvnetRolloutPolicy;

  fn build_rollout_policy(&self, tid: usize, batch_size: usize) -> ConvnetRolloutPolicy {
    ConvnetRolloutPolicy::new(tid, batch_size)
  }
}

pub struct ConvnetRolloutPolicy {
  context:      Rc<DeviceContext>,
  batch_size:   usize,
  //mode:         RolloutMode,
  arch:         PipelineArchWorker<()>,
}

impl ConvnetRolloutPolicy {
  pub fn new(tid: usize, batch_size: usize) -> ConvnetRolloutPolicy {
    let context = Rc::new(DeviceContext::new(tid));
    // FIXME(20160107)
    unimplemented!();
    /*ConvnetRolloutPolicy{
      context:    context,
      batch_size: batch_size,
      //mode:       mode,
      arch:       build_2layer16_19x19x16_arch(),
    }*/
  }
}

impl RolloutPolicy for ConvnetRolloutPolicy {
  fn batch_size(&self) -> usize {
    self.batch_size
  }

  fn rollout_batch(&mut self, leafs: RolloutLeafs, rollout_trajs: &mut [RolloutTraj], mode: RolloutMode, rng: &mut Xorshiftplus128Rng) {
    let ctx = (*self.context).as_ref();
    let batch_size = self.batch_size();
    assert!(batch_size <= rollout_trajs.len());

    let mut valid_moves = vec![vec![], vec![]];
    let mut bad_moves = vec![vec![], vec![]];
    let mut turn_pass = vec![vec![], vec![]];
    let mut num_both_passed = 0;
    let mut filters = vec![];
    for idx in 0 .. batch_size {
      /*let leaf_node = tree_trajs[idx].leaf_node.as_ref().unwrap().read().unwrap();
      valid_moves[0].push(leaf_node.state.get_data().legality.legal_points(Stone::Black));
      valid_moves[1].push(leaf_node.state.get_data().legality.legal_points(Stone::White));*/
      leafs.with_leaf_state(idx, |state| {
        valid_moves[0].push(state.get_data().legality.legal_points(Stone::Black));
        valid_moves[1].push(state.get_data().legality.legal_points(Stone::White));
      });
      bad_moves[0].push(vec![]);
      bad_moves[1].push(vec![]);
      turn_pass[0].push(false);
      turn_pass[1].push(false);
      filters.push(BFilter::with_capacity(Board::SIZE));
    }

    let max_iters = 361 + 361 / 2 + rng.gen_range(0, 2);
    for t in (0 .. max_iters) {
      if num_both_passed == batch_size {
        break;
      }

      for batch_idx in 0 .. batch_size {
        if !rollout_trajs[batch_idx].rollout {
          continue;
        }

        let turn = rollout_trajs[batch_idx].sim_state.current_turn();
        rollout_trajs[batch_idx].sim_state.get_data()
          .extract_relative_features(
              turn, self.arch.input_layer().expose_host_frame_buf(batch_idx));
        // FIXME(20151125): mask softmax output with valid moves.
        //self.arch.loss_layer().preload_mask_buf(batch_idx, valid_moves[turn.offset()][batch_idx].as_slice());
      }

      self.arch.input_layer().load_frames(batch_size, &ctx);
      // FIXME(20151125): mask softmax output with valid moves.
      //self.arch.loss_layer().load_masks(batch_size, &ctx);
      match mode {
        RolloutMode::Simulation => {
          self.arch.forward(batch_size, Phase::Inference, &ctx);
        }
        // FIXME(20160106): simulation balancing training.
        RolloutMode::BalanceTraining => {
          /*self.arch.forward(batch_size, Phase::Training, &ctx);
          //self.arch.backward(batch_size, , &ctx);*/
          unimplemented!();
        }
      }
      self.arch.loss_layer().store_probs(batch_size, &ctx);

      {
        let batch_probs = self.arch.loss_layer().get_probs(batch_size).as_slice();
        for batch_idx in 0 .. batch_size {
          if !rollout_trajs[batch_idx].rollout {
            continue;
          }
          filters[batch_idx].reset(&batch_probs[batch_idx * Board::SIZE .. (batch_idx + 1) * Board::SIZE]);
        }
      }

      for batch_idx in 0 .. batch_size {
        if !rollout_trajs[batch_idx].rollout {
          continue;
        }

        let sim_turn = rollout_trajs[batch_idx].sim_state.current_turn();
        let sim_turn_off = sim_turn.offset();
        if turn_pass[0][batch_idx] && turn_pass[1][batch_idx] {
          continue;
        }

        let mut made_move = false;
        let mut spin_count = 0;
        while !valid_moves[sim_turn_off][batch_idx].is_empty() {
          spin_count += 1;
          if let Some(p) = filters[batch_idx].sample(rng) {
            filters[batch_idx].zero(p);
            let sim_point = Point::from_idx(p);
            if !valid_moves[sim_turn_off][batch_idx].contains(&p) {
              continue;
            } else if !check_good_move_fast(
                &rollout_trajs[batch_idx].sim_state.position,
                &rollout_trajs[batch_idx].sim_state.chains,
                sim_turn, sim_point)
            {
              valid_moves[sim_turn_off][batch_idx].remove(&p);
              bad_moves[sim_turn_off][batch_idx].push(sim_point);
              continue;
            } else {
              valid_moves[sim_turn_off][batch_idx].remove(&p);
              if rollout_trajs[batch_idx].sim_state.try_place(sim_turn, sim_point).is_err() {
                rollout_trajs[batch_idx].sim_state.undo();
                continue;
              } else {
                rollout_trajs[batch_idx].sim_state.commit();
                for_each_touched_empty(
                    &rollout_trajs[batch_idx].sim_state.position,
                    &rollout_trajs[batch_idx].sim_state.chains,
                    |_, _, pt|
                {
                  valid_moves[sim_turn_off][batch_idx].insert(pt.idx());
                });
                made_move = true;
                break;
              }
            }
          } else {
            // XXX(20151207): Remaining moves were deemed to have probability zero.
            break;
          }
        }
        assert!(spin_count <= Board::SIZE);

        // XXX: Bad moves are not technically illegal.
        valid_moves[sim_turn_off][batch_idx].extend(bad_moves[sim_turn_off][batch_idx].iter().map(|&pt| pt.idx()));

        turn_pass[sim_turn_off][batch_idx] = false;
        if !made_move {
          rollout_trajs[batch_idx].sim_state.try_action(sim_turn, Action::Pass);
          rollout_trajs[batch_idx].sim_state.commit();
          for_each_touched_empty(&rollout_trajs[batch_idx].sim_state.position, &rollout_trajs[batch_idx].sim_state.chains, |position, chains, pt| {
            valid_moves[sim_turn_off][batch_idx].insert(pt.idx());
          });
          turn_pass[sim_turn_off][batch_idx] = true;
          if turn_pass[0][batch_idx] && turn_pass[1][batch_idx] {
            num_both_passed += 1;
            continue;
          }
        }
      }
    }
  }

  fn rollout_trace(&mut self, trace: &Trace, mode: RolloutMode) {
    let ctx = (*self.context).as_ref();

    let trace_len = trace.pairs.len();
    let value = trace.value.unwrap();
    let baseline = 0.5;

    for t in 0 .. trace_len {
      let state = &trace.pairs[t].0;
      let turn = state.current_turn();
      state.get_data()
        .extract_relative_features(
            turn, self.arch.input_layer().expose_host_frame_buf(t),
        );
    }
    match mode {
      RolloutMode::Simulation => {
        unimplemented!();
      }
      RolloutMode::BalanceTraining => {
        self.arch.input_layer().load_frames(trace_len, &ctx);
        self.arch.forward(trace_len, Phase::Training, &ctx);
        self.arch.backward(trace_len, (value - baseline) / (trace_len as f32), &ctx);
      }
    }

    // TODO(20160106)
    unimplemented!();
  }

  fn descend_params(&mut self, scale: f32) {
    let ctx = (*self.context).as_ref();
    self.arch.descend(scale, 0.0, &ctx);
    self.arch.reset_gradients(0.0, &ctx);
  }
}
