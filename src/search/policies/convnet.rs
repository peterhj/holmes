use board::{Board, Stone, Point, Action};
use convnet::arch::{
  build_pgbalance_action_2layer_19x19x16_arch,
  build_action_2layer_19x19x16_arch,
  build_action_12layer_19x19x16_arch,
  /*build_action_3layer_arch,
  build_action_6layer_arch,
  build_action_narrow_3layer_19x19x16_arch,
  build_action_3layer_19x19x16_arch,
  build_action_6layer_19x19x16_arch,
  build_action3_narrow_3layer_19x19x16_arch,
  build_action3_12layer_19x19x16_arch,*/
};
use discrete::{DiscreteFilter};
use discrete::bfilter::{BFilter};
use search::tree::{Walk, Trajectory};
use search::policies::{PriorPolicy, RolloutPolicy};
use txnstate::{
  TxnState,
  check_good_move_fast,
};
use txnstate::extras::{
  TxnStateNodeData,
  for_each_touched_empty,
};

use async_cuda::context::{DeviceContext};
use cuda::runtime::{CudaDevice};
use rembrandt::layer::{Layer};
use rembrandt::net::{NetArch, LinearNetArch};
use rembrandt::opt::{OptPhase};

use rand::{Rng};

pub struct ConvnetPriorPolicy {
  ctx:  DeviceContext,
  arch: LinearNetArch,
}

impl ConvnetPriorPolicy {
  pub fn new() -> ConvnetPriorPolicy {
    let ctx = DeviceContext::new(0);

    let arch = build_action_12layer_19x19x16_arch(1, &ctx);
    //let arch = build_action3_12layer_19x19x16_arch(1, &ctx);

    //let arch = build_action_3layer_arch(1, &ctx);
    //let arch = build_action_6layer_arch(1, &ctx);
    //let arch = build_action_3layer_19x19x16_arch(1, &ctx);
    //let arch = build_action_6layer_19x19x16_arch(1, &ctx);

    ConvnetPriorPolicy{
      ctx:  ctx,
      arch: arch,
    }
  }
}

impl PriorPolicy for ConvnetPriorPolicy {
  //type Ctx = DeviceContext;

  /*fn init_prior(&mut self, node: &mut Node, ctx: &DeviceContext) {
    // TODO(20151113)
    /*let turn = node_state.current_turn();
    node_state.get_data().features.extract_relative_features(
        turn, self.arch.data_layer().expose_host_frame_buf(0));
    self.arch.data_layer().load_frames(1, ctx);
    self.arch.evaluate(OptPhase::Inference, ctx);*/
  }*/

  fn fill_prior_probs(&mut self, state: &TxnState<TxnStateNodeData>, valid_moves: &[Point], prior_moves: &mut Vec<(Point, f32)>) {
    // TODO(20151113)
    let turn = state.current_turn();
    state.get_data().features.extract_relative_features(turn, self.arch.data_layer().expose_host_frame_buf(0));
    self.ctx.set_device();
    self.arch.data_layer().load_frames(1, &self.ctx);
    self.arch.evaluate(OptPhase::Inference, &self.ctx);
    self.arch.loss_layer().store_probs(1, &self.ctx);
    let pred_probs = self.arch.loss_layer().predict_probs(1, &self.ctx);
    for &point in valid_moves.iter() {
      prior_moves.push((point, pred_probs.as_slice()[point.idx()]));
    }
  }
}

pub struct BatchConvnetRolloutPolicy {
  batch_size: usize,
  ctx:  DeviceContext,
  arch: LinearNetArch,
}

impl BatchConvnetRolloutPolicy {
  pub fn new(batch_size: usize) -> BatchConvnetRolloutPolicy {
    let ctx = DeviceContext::new(0);

    // XXX(20160123): testing out simulation balanced params.
    let arch = build_action_2layer_19x19x16_arch(batch_size, &ctx);
    //let arch = build_pgbalance_action_2layer_19x19x16_arch(batch_size, &ctx);

    //let arch = build_action_narrow_3layer_19x19x16_arch(batch_size, &ctx);
    //let arch = build_action3_narrow_3layer_19x19x16_arch(batch_size, &ctx);

    BatchConvnetRolloutPolicy{
      batch_size: batch_size,
      ctx:  ctx,
      arch: arch,
    }
  }
}

impl RolloutPolicy for BatchConvnetRolloutPolicy {
  fn rollout(&self, walk: &Walk, traj: &mut Trajectory, rng: &mut Self::R) {
    unimplemented!();
  }

  fn do_batch(&self) -> bool {
    true
  }

  fn batch_size(&self) -> usize {
    self.batch_size
  }

  fn rollout_batch(&mut self, walks: &[Walk], trajs: &mut [Trajectory], rng: &mut Self::R) {
    let batch_size = self.batch_size();
    assert_eq!(batch_size, trajs.len());

    let mut valid_moves = vec![vec![], vec![]];
    let mut bad_moves = vec![vec![], vec![]];
    let mut turn_pass = vec![vec![], vec![]];
    let mut num_both_passed = 0;
    let mut filters = vec![];
    for idx in 0 .. batch_size {
      let leaf_node = walks[idx].leaf_node.as_ref().unwrap().borrow();
      valid_moves[0].push(leaf_node.state.get_data().legality.legal_points(Stone::Black));
      valid_moves[1].push(leaf_node.state.get_data().legality.legal_points(Stone::White));
      bad_moves[0].push(vec![]);
      bad_moves[1].push(vec![]);
      turn_pass[0].push(false);
      turn_pass[1].push(false);
      filters.push(BFilter::with_capacity(Board::SIZE));
    }

    let max_iters = 361 + 361 / 2 + rng.gen_range(0, 2);
    for t in 0 .. max_iters {
      //println!("DEBUG: BatchConvnetRolloutPolicy: iter {} / {}", t, max_iters);
      if num_both_passed == batch_size {
        //println!("DEBUG: BatchConvnetRolloutPolicy: all passed");
        break;
      }

      for idx in 0 .. batch_size {
        if !trajs[idx].rollout {
          continue;
        }

        let turn = trajs[idx].sim_state.current_turn();
        trajs[idx].sim_state.get_data().features
          .extract_relative_features(
              turn, self.arch.data_layer().expose_host_frame_buf(idx));
        // FIXME(20151125): mask softmax output with valid moves.
        //self.arch.loss_layer().preload_mask_buf(idx, valid_moves[turn.offset()][idx].as_slice());
      }

      self.arch.data_layer().load_frames(batch_size, &self.ctx);
      // FIXME(20151125): mask softmax output with valid moves.
      //self.arch.loss_layer().load_masks(batch_size, &self.ctx);
      self.arch.evaluate(OptPhase::Inference, &self.ctx);
      self.arch.loss_layer().store_probs(batch_size, &self.ctx);

      {
        let batch_probs = self.arch.loss_layer().predict_probs(batch_size, &self.ctx).as_slice();
        for idx in 0 .. batch_size {
          if !trajs[idx].rollout {
            continue;
          }
          filters[idx].reset(&batch_probs[idx * Board::SIZE .. (idx + 1) * Board::SIZE]);
        }
      }

      for idx in 0 .. batch_size {
        if !trajs[idx].rollout {
          continue;
        }

        let sim_turn = trajs[idx].sim_state.current_turn();
        let sim_turn_off = sim_turn.offset();
        if turn_pass[0][idx] && turn_pass[1][idx] {
          continue;
        }

        let mut made_move = false;
        let mut spin_count = 0;
        while !valid_moves[sim_turn_off][idx].is_empty() {
          spin_count += 1;
          if let Some(p) = filters[idx].sample(rng) {
            filters[idx].zero(p);
            let sim_point = Point::from_idx(p);
            if !valid_moves[sim_turn_off][idx].contains(p) {
              continue;
            } else if !check_good_move_fast(&trajs[idx].sim_state.position, &trajs[idx].sim_state.chains, sim_turn, sim_point) {
              valid_moves[sim_turn_off][idx].remove(p);
              bad_moves[sim_turn_off][idx].push(sim_point);
              continue;
            } else {
              valid_moves[sim_turn_off][idx].remove(p);
              if trajs[idx].sim_state.try_place(sim_turn, sim_point).is_err() {
                trajs[idx].sim_state.undo();
                continue;
              } else {
                trajs[idx].sim_state.commit();
                for_each_touched_empty(&trajs[idx].sim_state.position, &trajs[idx].sim_state.chains, |position, chains, pt| {
                  valid_moves[sim_turn_off][idx].insert(pt.idx());
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
        /*println!("DEBUG: BatchConvnetRolloutPolicy: iter {}: idx {}: spin count: {}",
            t, idx, spin_count);*/
        assert!(spin_count <= Board::SIZE);

        // XXX: Bad moves are not technically illegal.
        valid_moves[sim_turn_off][idx].extend(bad_moves[sim_turn_off][idx].iter().map(|&pt| pt.idx()));

        turn_pass[sim_turn_off][idx] = false;
        if !made_move {
          trajs[idx].sim_state.try_action(sim_turn, Action::Pass);
          trajs[idx].sim_state.commit();
          for_each_touched_empty(&trajs[idx].sim_state.position, &trajs[idx].sim_state.chains, |position, chains, pt| {
            valid_moves[sim_turn_off][idx].insert(pt.idx());
          });
          turn_pass[sim_turn_off][idx] = true;
          if turn_pass[0][idx] && turn_pass[1][idx] {
            num_both_passed += 1;
            continue;
          }
        }
      }
    }
  }
}

pub struct ParallelBatchConvnetRolloutPolicy {
  total_batch_size: usize,
  part_batch_size:  usize,
  num_parts:        usize,
  ctxs:   Vec<DeviceContext>,
  archs:  Vec<LinearNetArch>,
}

impl ParallelBatchConvnetRolloutPolicy {
  pub fn new(batch_size: usize) -> ParallelBatchConvnetRolloutPolicy {
    let device_count = CudaDevice::count().unwrap();
    assert_eq!(0, batch_size % device_count);
    let part_batch_size = batch_size / device_count;
    let mut ctxs = vec![];
    let mut archs = vec![];
    for device_idx in 0 .. device_count {
      let ctx = DeviceContext::new(device_idx);

      // XXX(20160123): testing out simulation balanced params.
      let arch = build_action_2layer_19x19x16_arch(part_batch_size, &ctx);
      //let arch = build_pgbalance_action_2layer_19x19x16_arch(part_batch_size, &ctx);

      ctxs.push(ctx);
      archs.push(arch);
    }
    ParallelBatchConvnetRolloutPolicy{
      total_batch_size: batch_size,
      part_batch_size:  part_batch_size,
      num_parts:        device_count,
      ctxs:   ctxs,
      archs:  archs,
    }
  }
}

impl RolloutPolicy for ParallelBatchConvnetRolloutPolicy {
  fn rollout(&self, walk: &Walk, traj: &mut Trajectory, rng: &mut Self::R) {
    unimplemented!();
  }

  fn do_batch(&self) -> bool {
    true
  }

  fn batch_size(&self) -> usize {
    self.total_batch_size
  }

  fn rollout_batch(&mut self, walks: &[Walk], trajs: &mut [Trajectory], rng: &mut Self::R) {
    let batch_size = self.total_batch_size;
    let part_batch_size = self.part_batch_size;
    assert_eq!(batch_size, trajs.len());

    let mut valid_moves = vec![vec![], vec![]];
    let mut bad_moves = vec![vec![], vec![]];
    let mut turn_pass = vec![vec![], vec![]];
    let mut num_both_passed = 0;
    let mut filters = vec![];
    for idx in 0 .. batch_size {
      let leaf_node = walks[idx].leaf_node.as_ref().unwrap().borrow();
      valid_moves[0].push(leaf_node.state.get_data().legality.legal_points(Stone::Black));
      valid_moves[1].push(leaf_node.state.get_data().legality.legal_points(Stone::White));
      bad_moves[0].push(vec![]);
      bad_moves[1].push(vec![]);
      turn_pass[0].push(false);
      turn_pass[1].push(false);
      filters.push(BFilter::with_capacity(Board::SIZE));
    }

    let max_iters = 361 + 361 / 2 + rng.gen_range(0, 2);
    for t in 0 .. max_iters {
      //println!("DEBUG: ParallelBatchConvnetRolloutPolicy: iter {} / {}", t, max_iters);
      if num_both_passed == batch_size {
        //println!("DEBUG: ParallelBatchConvnetRolloutPolicy: all passed");
        break;
      }

      for part in 0 .. self.num_parts {
        for idx in part * part_batch_size .. (part + 1) * part_batch_size {
          if !trajs[idx].rollout {
            continue;
          }

          let part_idx = idx - part * part_batch_size;
          let turn = trajs[idx].sim_state.current_turn();
          trajs[idx].sim_state.get_data().features
            .extract_relative_features(
                turn, self.archs[part].data_layer().expose_host_frame_buf(part_idx));
          // FIXME(20151125): mask softmax output with valid moves.
          //self.arch.loss_layer().preload_mask_buf(idx, valid_moves[turn.offset()][idx].as_slice());
        }
      }

      // FIXME(20151211): parallel rollouts.
      for part in 0 .. self.num_parts {
        self.ctxs[part].set_device();
        self.archs[part].data_layer().load_frames(part_batch_size, &self.ctxs[part]);
      }
      for part in 0 .. self.num_parts {
        self.ctxs[part].set_device();
        self.archs[part].evaluate(OptPhase::Inference, &self.ctxs[part]);
      }
      for part in 0 .. self.num_parts {
        self.ctxs[part].set_device();
        self.archs[part].loss_layer().store_probs(part_batch_size, &self.ctxs[part]);
      }

      {
        for part in 0 .. self.num_parts {
          self.ctxs[part].set_device();
          let batch_probs = self.archs[part].loss_layer().predict_probs(part_batch_size, &self.ctxs[part]).as_slice();
          for idx in part * part_batch_size .. (part + 1) * part_batch_size {
            if !trajs[idx].rollout {
              continue;
            }
            let part_idx = idx - part * part_batch_size;
            filters[idx].reset(&batch_probs[part_idx * Board::SIZE .. (part_idx + 1) * Board::SIZE]);
          }
        }
      }

      for idx in 0 .. batch_size {
        if !trajs[idx].rollout {
          continue;
        }

        let sim_turn = trajs[idx].sim_state.current_turn();
        let sim_turn_off = sim_turn.offset();
        if turn_pass[0][idx] && turn_pass[1][idx] {
          continue;
        }

        let mut made_move = false;
        let mut spin_count = 0;
        while !valid_moves[sim_turn_off][idx].is_empty() {
          spin_count += 1;
          if let Some(p) = filters[idx].sample(rng) {
            filters[idx].zero(p);
            let sim_point = Point::from_idx(p);
            if !valid_moves[sim_turn_off][idx].contains(p) {
              continue;
            } else if !check_good_move_fast(&trajs[idx].sim_state.position, &trajs[idx].sim_state.chains, sim_turn, sim_point) {
              valid_moves[sim_turn_off][idx].remove(p);
              bad_moves[sim_turn_off][idx].push(sim_point);
              continue;
            } else {
              valid_moves[sim_turn_off][idx].remove(p);
              if trajs[idx].sim_state.try_place(sim_turn, sim_point).is_err() {
                trajs[idx].sim_state.undo();
                continue;
              } else {
                trajs[idx].sim_state.commit();
                for_each_touched_empty(&trajs[idx].sim_state.position, &trajs[idx].sim_state.chains, |position, chains, pt| {
                  valid_moves[sim_turn_off][idx].insert(pt.idx());
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
        /*println!("DEBUG: ParallelBatchConvnetRolloutPolicy: iter {}: idx {}: spin count: {}",
            t, idx, spin_count);*/
        assert!(spin_count <= Board::SIZE);

        // XXX: Bad moves are not technically illegal.
        valid_moves[sim_turn_off][idx].extend(bad_moves[sim_turn_off][idx].iter().map(|&pt| pt.idx()));

        turn_pass[sim_turn_off][idx] = false;
        if !made_move {
          trajs[idx].sim_state.try_action(sim_turn, Action::Pass);
          trajs[idx].sim_state.commit();
          for_each_touched_empty(&trajs[idx].sim_state.position, &trajs[idx].sim_state.chains, |position, chains, pt| {
            valid_moves[sim_turn_off][idx].insert(pt.idx());
          });
          turn_pass[sim_turn_off][idx] = true;
          if turn_pass[0][idx] && turn_pass[1][idx] {
            num_both_passed += 1;
            continue;
          }
        }
      }
    }
  }
}
