use board::{Board, Stone, Point, Action};
use convnet::{
  build_action_3layer_arch,
  build_action_6layer_arch,
  build_action_2layer_19x19x16_arch,
  build_action_6layer_19x19x16_arch,
};
use discrete::{DiscreteFilter};
use discrete::bfilter::{BFilter};
use search::{Trajectory};
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
    //let arch = build_action_3layer_arch(1, &ctx);
    //let arch = build_action_6layer_arch(1, &ctx);
    let arch = build_action_6layer_19x19x16_arch(1, &ctx);
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
    self.arch.evaluate(OptPhase::Evaluation, ctx);*/
  }*/

  fn fill_prior_probs(&mut self, state: &TxnState<TxnStateNodeData>, valid_moves: &[Point], prior_moves: &mut Vec<(Point, f32)>) {
    // TODO(20151113)
    let turn = state.current_turn();
    state.get_data().features.extract_relative_features(turn, self.arch.data_layer().expose_host_frame_buf(0));
    self.arch.data_layer().load_frames(1, &self.ctx);
    self.arch.evaluate(OptPhase::Evaluation, &self.ctx);
    self.arch.loss_layer().store_probs(1, &self.ctx);
    let pred_probs = self.arch.loss_layer().predict_probs(1, &self.ctx);
    for &point in valid_moves.iter() {
      prior_moves.push((point, pred_probs.as_slice()[point.idx()]));
    }
  }
}

pub struct BatchConvnetRolloutPolicy {
  ctx:  DeviceContext,
  arch: LinearNetArch,
}

impl BatchConvnetRolloutPolicy {
  pub fn new() -> BatchConvnetRolloutPolicy {
    let ctx = DeviceContext::new(0);
    let arch = build_action_2layer_19x19x16_arch(64, &ctx);
    BatchConvnetRolloutPolicy{
      ctx:  ctx,
      arch: arch,
    }
  }
}

impl RolloutPolicy for BatchConvnetRolloutPolicy {
  fn rollout(&self, traj: &mut Trajectory, rng: &mut Self::R) {
    unimplemented!();
  }

  fn batch_size(&self) -> usize {
    64
  }

  fn rollout_batch(&mut self, trajs: &mut [Trajectory], rng: &mut Self::R) {
    let batch_size = self.batch_size();
    assert_eq!(batch_size, trajs.len());

    let mut valid_moves = vec![vec![], vec![]];
    let mut bad_moves = vec![vec![], vec![]];
    let mut turn_pass = vec![vec![], vec![]];
    let mut filters = vec![];
    for idx in (0 .. batch_size) {
      let leaf_node = trajs[idx].leaf_node.as_ref().unwrap().borrow();
      valid_moves[0].push(leaf_node.state.get_data().legality.legal_points(Stone::Black));
      valid_moves[1].push(leaf_node.state.get_data().legality.legal_points(Stone::White));
      bad_moves[0].push(vec![]);
      bad_moves[1].push(vec![]);
      turn_pass[0].push(false);
      turn_pass[1].push(false);
      filters.push(BFilter::with_capacity(Board::SIZE));
    }

    let max_iters = 361 + 361 / 2 + rng.gen_range(0, 2);
    for t in (0 .. max_iters) {
      for idx in (0 .. batch_size) {
        let turn = trajs[idx].sim_state.current_turn();
        trajs[idx].sim_state.get_data()
          .extract_relative_features(
              turn, self.arch.data_layer().expose_host_frame_buf(idx));
        // FIXME(20151125): mask softmax output with valid moves.
        //self.arch.loss_layer().preload_mask_buf(idx, valid_moves[turn.offset()][idx].as_slice());
      }

      self.arch.data_layer().load_frames(batch_size, &self.ctx);
      // FIXME(20151125): mask softmax output with valid moves.
      //self.arch.loss_layer().load_masks(batch_size, &self.ctx);
      self.arch.evaluate(OptPhase::Evaluation, &self.ctx);
      self.arch.loss_layer().store_probs(batch_size, &self.ctx);

      {
        let batch_probs = self.arch.loss_layer().predict_probs(batch_size, &self.ctx).as_slice();
        for idx in (0 .. batch_size) {
          filters[idx].reset(&batch_probs[idx * Board::SIZE .. (idx + 1) * Board::SIZE]);
        }
      }

      for idx in (0 .. batch_size) {
        let sim_turn = trajs[idx].sim_state.current_turn();
        let sim_turn_off = sim_turn.offset();
        if turn_pass[0][idx] && turn_pass[1][idx] {
          continue;
        }

        let mut made_move = false;
        while !valid_moves[sim_turn_off][idx].is_empty() {
          if let Some(p) = filters[idx].sample(rng) {
            let sim_point = Point::from_idx(p);
            filters[idx].zero(p);
            if !valid_moves[sim_turn_off][idx].contains(&p) {
              continue;
            } else if !check_good_move_fast(&trajs[idx].sim_state.position, &trajs[idx].sim_state.chains, sim_turn, sim_point) {
              valid_moves[sim_turn_off][idx].remove(&p);
              bad_moves[sim_turn_off][idx].push(sim_point);
              continue;
            } else {
              valid_moves[sim_turn_off][idx].remove(&p);
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
            unreachable!();
          }
        }

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
        }
      }
    }
  }
}
