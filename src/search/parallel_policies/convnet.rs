use board::{Board, Action, Stone, Point};
use convnet_new::{
  build_2layer16_19x19x16_arch,
  build_2layer16_19x19x16_arch_nodir,
  build_12layer128_19x19x16_arch,
  /*build_3layer32_19x19x16_arch,
  build_12layer384_19x19x30_arch,*/
};
use discrete::{DiscreteFilter};
use discrete::bfilter::{BFilter};
use random::{choose_without_replace};
use search::parallel_policies::{
  SearchPolicyWorkerBuilder, SearchPolicyWorker,
  PriorPolicy, TreePolicy,
  RolloutPolicyBuilder, RolloutMode, RolloutLeafs, RolloutPolicy,
};
use search::parallel_policies::thompson::{ThompsonTreePolicy};
use search::parallel_tree::{TreeTraj, RolloutTraj, Trace, QuickTrace};
use txnstate::{TxnState, check_good_move_fast};
use txnstate::extras::{TxnStateNodeData, for_each_touched_empty};

use array_cuda::device::{DeviceContext, for_all_devices};
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
    // FIXME(20160121): temporarily using existing balance run.
    /*let rollout_arch_cfg = build_2layer16_19x19x16_arch_nodir(worker_batch_size);
    let rollout_save_path = PathBuf::from("experiments/models_balance/test");*/
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
        [thread_rng().next_u64(), thread_rng().next_u64()],
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
        [thread_rng().next_u64(), thread_rng().next_u64()],
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

  fn exploration_policies(&mut self) -> (&mut PriorPolicy, &mut TreePolicy<R=Xorshiftplus128Rng>) {
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

#[derive(Clone)]
pub struct ConvnetRolloutPolicyBuilder {
  num_workers:          usize,
  rollout_arch_cfg:     PipelineArchConfig,
  rollout_save_path:    PathBuf,
  rollout_shared:       Arc<PipelineArchSharedData>,
  rollout_shared2:      Arc<()>,
}

impl ConvnetRolloutPolicyBuilder {
  pub fn new(num_workers: usize, batch_size: usize) -> ConvnetRolloutPolicyBuilder {
    let (rollout_arch_cfg, rollout_save_path) = build_2layer16_19x19x16_arch(batch_size);
    let rollout_shared = for_all_devices(num_workers, |contexts| {
      Arc::new(PipelineArchSharedData::new(num_workers, &rollout_arch_cfg, contexts))
    });
    ConvnetRolloutPolicyBuilder{
      num_workers:          num_workers,
      rollout_arch_cfg:     rollout_arch_cfg,
      rollout_save_path:    rollout_save_path,
      rollout_shared:       rollout_shared,
      rollout_shared2:      Arc::new(()),
    }
  }
}

impl RolloutPolicyBuilder for ConvnetRolloutPolicyBuilder {
  type Policy = ConvnetRolloutPolicy;

  fn into_rollout_policy(self, tid: usize, batch_size: usize) -> ConvnetRolloutPolicy {
    assert!(tid < self.num_workers);
    let context = Rc::new(DeviceContext::new(tid));
    let ctx = (*context).as_ref();
    let mut rollout_arch = PipelineArchWorker::new(
        batch_size,
        self.rollout_arch_cfg,
        self.rollout_save_path,
        tid,
        [thread_rng().next_u64(), thread_rng().next_u64()],
        &self.rollout_shared,
        self.rollout_shared2.clone(),
        &ctx,
    );
    rollout_arch.load_layer_params(None, &ctx);
    ConvnetRolloutPolicy{
      context:      context.clone(),
      batch_size:   batch_size,
      arch:         rollout_arch,
    }
  }
}

pub struct ConvnetRolloutPolicy {
  context:      Rc<DeviceContext>,
  batch_size:   usize,
  arch:         PipelineArchWorker<()>,
}

/*impl ConvnetRolloutPolicy {
}*/

impl RolloutPolicy for ConvnetRolloutPolicy {
  fn batch_size(&self) -> usize {
    self.batch_size
  }

  fn max_rollout_len(&self) -> usize {
    let max_iters = 361 + 361 / 2 + 1;
    max_iters
  }

  fn rollout_batch(&mut self, batch_size: usize, leafs: RolloutLeafs, rollout_trajs: &mut [RolloutTraj], record_trace: bool, traces: &mut [QuickTrace], rng: &mut Xorshiftplus128Rng) {
    let ctx = (*self.context).as_ref();

    // FIXME(20160120): this allows us to be a little sloppy with how many
    // trajectories we allocate.
    //assert_eq!(batch_size, rollout_trajs.len());
    assert!(batch_size <= rollout_trajs.len());

    let mut valid_move_set = vec![vec![], vec![]];
    // XXX(20160124): Valid move iterator is an "upper bound" on the valid moves,
    // since it is easy to add elements but hard to remove them.
    let mut valid_move_iter: Vec<Vec<Vec<usize>>> = vec![vec![], vec![]];
    let mut bad_moves = vec![vec![], vec![]];
    let mut turn_pass = vec![vec![], vec![]];
    let mut num_both_passed = 0;
    let mut filters = vec![];
    for batch_idx in 0 .. batch_size {
      /*let leaf_node = tree_trajs[idx].leaf_node.as_ref().unwrap().read().unwrap();
      valid_move_set[0].push(leaf_node.state.get_data().legality.legal_points(Stone::Black));
      valid_move_set[1].push(leaf_node.state.get_data().legality.legal_points(Stone::White));*/
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
      filters.push(BFilter::with_capacity(Board::SIZE));
    }

    if record_trace {
      for batch_idx in 0 .. batch_size {
        if rollout_trajs[batch_idx].rollout {
          traces[batch_idx].reset();
          traces[batch_idx].init_state = Some(rollout_trajs[batch_idx].sim_state.clone());
        }
      }
    }

    let max_iters = 361 + 361 / 2 + rng.gen_range(0, 2);
    for t in 0 .. max_iters {
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
        //self.arch.loss_layer().preload_mask_buf(batch_idx, valid_move_set[turn.offset()][batch_idx].as_slice());
      }

      self.arch.input_layer().load_frames(batch_size, &ctx);
      // FIXME(20151125): mask softmax output with valid moves.
      //self.arch.loss_layer().load_masks(batch_size, &ctx);
      self.arch.forward(batch_size, Phase::Inference, &ctx);
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

        if record_trace {
          traces[batch_idx].actions.push((sim_turn, Action::Pass));
        }

        let mut made_move = false;
        let mut spin_count = 0;
        while !valid_move_set[sim_turn_off][batch_idx].is_empty() {
          if let Some(p) = filters[batch_idx].sample(rng) {
            filters[batch_idx].zero(p);
            let sim_point = Point::from_idx(p);
            if !valid_move_set[sim_turn_off][batch_idx].contains(&p) {
              spin_count += 1;
              continue;
            } else if !check_good_move_fast(
                &rollout_trajs[batch_idx].sim_state.position,
                &rollout_trajs[batch_idx].sim_state.chains,
                sim_turn, sim_point)
            {
              valid_move_set[sim_turn_off][batch_idx].remove(&p);
              bad_moves[sim_turn_off][batch_idx].push(sim_point);
              spin_count += 1;
              continue;
            } else {
              valid_move_set[sim_turn_off][batch_idx].remove(&p);
              if rollout_trajs[batch_idx].sim_state.try_place(sim_turn, sim_point).is_err() {
                rollout_trajs[batch_idx].sim_state.undo();
                spin_count += 1;
                continue;
              } else {
                if record_trace {
                  let trace_len = traces[batch_idx].actions.len();
                  traces[batch_idx].actions[trace_len - 1].1 = Action::Place{point: sim_point};
                }
                rollout_trajs[batch_idx].sim_state.commit();
                for_each_touched_empty(
                    &rollout_trajs[batch_idx].sim_state.position,
                    &rollout_trajs[batch_idx].sim_state.chains,
                    |_, _, pt|
                {
                  //valid_move_set[sim_turn_off][batch_idx].insert(pt.idx());
                  let pt = pt.idx();
                  if !valid_move_set[sim_turn_off][batch_idx].contains(&pt) {
                    valid_move_set[sim_turn_off][batch_idx].insert(pt);
                    valid_move_iter[sim_turn_off][batch_idx].push(pt);
                  }
                });
                made_move = true;
                spin_count += 1;
                break;
              }
            }
          } else {
            // XXX(20151207): Remaining moves were deemed to have probability zero,
            // so finish the rollout using uniform policy.
            /*break;*/
            while !valid_move_set[sim_turn_off][batch_idx].is_empty() {
              spin_count += 1;
              if let Some(p) = choose_without_replace(&mut valid_move_iter[sim_turn_off][batch_idx], rng) {
                let sim_point = Point::from_idx(p);
                if !valid_move_set[sim_turn_off][batch_idx].contains(&p) {
                  spin_count += 1;
                  continue;
                } else if !check_good_move_fast(
                    &rollout_trajs[batch_idx].sim_state.position,
                    &rollout_trajs[batch_idx].sim_state.chains,
                    sim_turn, sim_point)
                {
                  valid_move_set[sim_turn_off][batch_idx].remove(&p);
                  bad_moves[sim_turn_off][batch_idx].push(sim_point);
                  spin_count += 1;
                  continue;
                } else {
                  valid_move_set[sim_turn_off][batch_idx].remove(&p);
                  if rollout_trajs[batch_idx].sim_state.try_place(sim_turn, sim_point).is_err() {
                    rollout_trajs[batch_idx].sim_state.undo();
                    spin_count += 1;
                    continue;
                  } else {
                    if record_trace {
                      let trace_len = traces[batch_idx].actions.len();
                      traces[batch_idx].actions[trace_len - 1].1 = Action::Place{point: sim_point};
                    }
                    rollout_trajs[batch_idx].sim_state.commit();
                    for_each_touched_empty(
                        &rollout_trajs[batch_idx].sim_state.position,
                        &rollout_trajs[batch_idx].sim_state.chains,
                        |_, _, pt|
                    {
                      //valid_move_set[sim_turn_off][batch_idx].insert(pt.idx());
                      let pt = pt.idx();
                      if !valid_move_set[sim_turn_off][batch_idx].contains(&pt) {
                        valid_move_set[sim_turn_off][batch_idx].insert(pt);
                        valid_move_iter[sim_turn_off][batch_idx].push(pt);
                      }
                    });
                    made_move = true;
                    spin_count += 1;
                    break;
                  }
                }
              }
            }
          }
        }
        assert!(spin_count <= Board::SIZE);

        // XXX: Bad moves are not technically illegal.
        valid_move_set[sim_turn_off][batch_idx].extend(bad_moves[sim_turn_off][batch_idx].iter().map(|&pt| pt.idx()));

        turn_pass[sim_turn_off][batch_idx] = false;
        if !made_move {
          rollout_trajs[batch_idx].sim_state.try_action(sim_turn, Action::Pass);
          rollout_trajs[batch_idx].sim_state.commit();
          for_each_touched_empty(&rollout_trajs[batch_idx].sim_state.position, &rollout_trajs[batch_idx].sim_state.chains, |position, chains, pt| {
            valid_move_set[sim_turn_off][batch_idx].insert(pt.idx());
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

  fn init_traces(&mut self) {
    let ctx = (*self.context).as_ref();
    self.arch.reset_gradients(0.0, &ctx);
  }

  fn rollout_trace(&mut self, trace: &Trace, baseline: f32) -> bool {
    let ctx = (*self.context).as_ref();

    let trace_len = trace.pairs.len();
    if trace_len == 0 {
      return false;
    }
    let value = trace.value.unwrap();
    let baseline = baseline;

    for t in 0 .. trace_len {
      let turn = trace.pairs[t].0;
      let state = &trace.pairs[t].1;
      let action = trace.pairs[t].2;
      //let turn = state.current_turn();
      state
        .extract_relative_features(
            turn, self.arch.input_layer().expose_host_frame_buf(t),
        );
      let label = SampleLabel::Category{category: match action {
        Action::Place{point} => point.idx() as i32,
        _ => -1,
      }};
      self.arch.loss_layer().preload_label(t, &label, Phase::Inference);
    }

    self.arch.input_layer().load_frames(trace_len, &ctx);
    self.arch.loss_layer().load_labels(trace_len, &ctx);
    self.arch.forward(trace_len, Phase::Training, &ctx);
    self.arch.backward(trace_len, (value - baseline) / (trace_len as f32), &ctx);

    true
  }

  fn rollout_quicktrace(&mut self, trace: &QuickTrace, baseline: f32) -> bool {
    let ctx = (*self.context).as_ref();

    let mut trace_len = 0;
    let mut sim_state = match trace.init_state {
      Some(ref init_state) => init_state.clone(),
      None => return false,
    };
    for &(turn, action) in trace.actions.iter() {
      match action {
        Action::Place{point} => {
          let t = trace_len;
          sim_state.get_data()
            .extract_relative_features(
                turn, self.arch.input_layer().expose_host_frame_buf(t),
            );
          let label = SampleLabel::Category{category: point.idx() as i32};
          self.arch.loss_layer().preload_label(t, &label, Phase::Inference);
          trace_len += 1;
        }
        _ => {}
      }
      match sim_state.try_action(turn, action) {
        Ok(_) => sim_state.commit(),
        Err(_) => panic!("quicktrace actions should always be legal!"),
      }
    }
    if trace_len == 0 {
      return false;
    }

    let value = match trace.value {
      Some(value) => value,
      None => panic!("missing quicktrace value!"),
    };
    self.arch.input_layer().load_frames(trace_len, &ctx);
    self.arch.loss_layer().load_labels(trace_len, &ctx);
    self.arch.forward(trace_len, Phase::Training, &ctx);
    self.arch.backward(trace_len, (value - baseline) / (trace_len as f32), &ctx);

    true
  }

  fn backup_traces(&mut self, learning_rate: f32, target_value: f32, eval_value: f32, num_traces: usize) {
    let ctx = (*self.context).as_ref();
    self.arch.dev_allreduce_sum_gradients(&ctx);
    // FIXME(20160122): forgot to scale by total number of traces, not the
    // number of traces processed by this worker...
    // as an approximation, just multiply local number of traces by number of
    // workers.
    //self.arch.descend(learning_rate * (target_value - eval_value) / (num_traces as f32), 0.0, &ctx);
    let num_workers = self.arch.num_workers();
    self.arch.descend(learning_rate * (target_value - eval_value) / ((num_workers * num_traces) as f32), 0.0, &ctx);
  }

  fn save_params(&mut self, save_dir: &Path, t: usize) {
    if self.arch.tid() == 0 {
      let ctx = (*self.context).as_ref();
      self.arch.save_layer_params_dir(save_dir, t, &ctx);
    }
  }
}
