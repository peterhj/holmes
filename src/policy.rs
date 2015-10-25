use fastboard::{PosExt, Pos, Action, Stone, FastBoard, FastBoardAux, FastBoardWork};
use fasttree::{ExecutionBehavior, Trajectory, NodeId, FastSearchNode, FastSearchTree};
use mctree::{McTrajectory, McNode, McSearchTree};
use random::{XorShift128PlusRng, choose_without_replace, sample_discrete_cdf};

use async_cuda::context::{DeviceContext};
use rembrandt::layer::{
  Layer,
  ParamsInitialization, ActivationFunction,
  DataLayerConfig, Conv2dLayerConfig, SoftmaxLossLayerConfig,
  DataLayer, Conv2dLayer, SoftmaxLossLayer,
};
use rembrandt::net::{NetArch, LinearNetArch};
use statistics_avx2::array::{array_argmax};
use statistics_avx2::random::{StreamRng, XorShift128PlusStreamRng};

use bit_set::{BitSet};
use rand::{Rng, SeedableRng, thread_rng};
use std::path::{PathBuf};

pub trait SearchPolicy {
  fn backup(&self, tree: &mut FastSearchTree, traj: &Trajectory) -> usize;
  fn execute_search(&self, node: &FastSearchNode) -> Action;
  fn execute_best(&self, node: &FastSearchNode) -> Action;

  fn backup_new(&self, traj: &mut McTrajectory) { unimplemented!(); }
  fn execute_best_new(&self, node: &McNode) -> Action { unimplemented!(); }
}

#[derive(Clone, Copy)]
pub struct UctSearchPolicy {
  pub c:  f32,
}

impl UctSearchPolicy {
  fn backup_node(&self, node: &mut FastSearchNode, update_mov: Pos, outcome: f32) -> bool {
    if !node.rev_moves.contains_key(&update_mov) {
      //println!("WARNING: uct: node does not have update mov: {:?}",
      //    update_mov);
      return false;
    }
    let update_j = node.rev_moves[&update_mov];
    let reward = if outcome >= 0.0 {
      if let Stone::White = node.state.current_turn() {
        1.0
      } else {
        0.0
      }
    } else {
      if let Stone::Black = node.state.current_turn() {
        1.0
      } else {
        0.0
      }
    };
    node.credit_one_arm(update_j, reward);
    // TODO(20151022): could do a SIMD version of this.
    for j in (0 .. node.value.len()) {
      node.value[j] = node.succ_ratio[j] + self.c * (node.total_trials.ln() / node.num_trials[j]).sqrt();
    }
    true
  }

  fn backup_node_new(&self, node: &mut McNode, update_mov: Pos, update_j: usize, outcome: f32) {
    if node.valid_moves[update_j] != update_mov {
      println!("WARNING: uct policy: node valid mov index {} is not {:?}!",
          update_j, update_mov);
      return;
    }
    let reward = if outcome >= 0.0 {
      if let Stone::White = node.state.current_turn() {
        1.0
      } else {
        0.0
      }
    } else {
      if let Stone::Black = node.state.current_turn() {
        1.0
      } else {
        0.0
      }
    };
    node.credit_one_arm(update_j, reward);
    for j in (0 .. node.value.len()) {
      node.value[j] = (node.succ_trials[j] / node.num_trials[j]) + self.c * (node.total_visits.ln() / node.num_trials[j]).sqrt();
    }
  }
}

impl SearchPolicy for UctSearchPolicy {
  fn backup(&self, tree: &mut FastSearchTree, traj: &Trajectory) -> usize {
    // XXX(20151022): Outcome and reward conventions:
    // Outcomes are absolutely valued:
    //    W win: >= 0.0 outcome.
    //    B win: < 0.0 outcome.
    // Rewards are relatively valued:
    //    curr turn win:  +1 reward
    //    oppo turn win:   0 reward
    // FIXME(20151023): use tree and traj to backup multiple nodes.
    let (leaf_id, leaf_turn, _) = traj.search_result.unwrap();
    if traj.rollout_moves.len() == 0 {
      //println!("WARNING: during backup, leaf node had no rollout moves!");
      return 0;
    }
    let mut success = 1;
    if !self.backup_node(tree.nodes.get_mut(&leaf_id).unwrap(), traj.rollout_moves[0].1, traj.outcome) {
      println!("WARNING: uct: leaf backup failed: id: {:?} mov: {:?}",
          leaf_id, traj.rollout_moves[0].1);
      success = 0;
    }
    for &(id, _, mov) in traj.search_pairs.iter().rev() {
      if !self.backup_node(tree.nodes.get_mut(&id).unwrap(), mov, traj.outcome) {
        println!("WARNING: uct: inner backup failed: id: {:?} mov: {:?} search movs: {:?}",
            id, mov, traj.search_pairs);
      }
    }
    success
  }

  fn execute_search(&self, node: &FastSearchNode) -> Action {
    if node.moves.len() > 0 {
      let j = array_argmax(&node.value);
      Action::Place{pos: node.moves[j]}
    } else {
      Action::Pass
    }
  }

  fn execute_best(&self, node: &FastSearchNode) -> Action {
    if node.moves.len() > 0 {
      let j = array_argmax(&node.num_trials);
      Action::Place{pos: node.moves[j]}
    } else {
      Action::Pass
    }
  }

  fn backup_new(&self, traj: &mut McTrajectory) {
    let outcome = traj.score.expect("FATAL: uct policy: backup: missing outcome!");
    /*if traj.rollout_moves.is_empty() {
      println!("WARNING: uct policy: backup: no rollouts in this trajectory!");
      return;
    }*/
    if !traj.rollout_moves.is_empty() {
      if let Some(ref mut leaf_node) = traj.leaf_node {
        let update_mov = traj.rollout_moves[0];
        let mut update_j = None;
        for (j, &mov) in leaf_node.borrow().valid_moves.iter().enumerate() {
          if mov == update_mov {
            update_j = Some(j);
            break;
          }
        }
        if update_j.is_none() {
          println!("WARNING: uct policy: backup: leaf node is missing first rollout move!");
          return;
        }
        self.backup_node_new(&mut leaf_node.borrow_mut(), update_mov, update_j.unwrap(), outcome);
      } else {
        println!("WARNING: uct policy: backup: no leaf node in this trajectory!");
        return;
      }
    }
    for &mut (ref mut node, mov, j) in traj.search_pairs.iter_mut().rev() {
      if node.borrow().valid_moves[j] != mov {
        println!("WARNING: uct policy: backup: inner node has mismatched move!");
        return;
      }
      self.backup_node_new(&mut node.borrow_mut(), mov, j, outcome);
    }
  }

  fn execute_best_new(&self, node: &McNode) -> Action {
    println!("DEBUG: uct policy: executing best...");
    if !node.valid_moves.is_empty() {
      let j = array_argmax(&node.num_trials);
      println!("DEBUG: uct policy:   argmax: ({}, {}) trials: {:?}",
          j, node.num_trials[j], node.num_trials);
      Action::Place{pos: node.valid_moves[j]}
    } else {
      Action::Pass
    }
  }
}

#[derive(Clone, Copy)]
pub struct RaveSearchPolicy;

#[derive(Clone, Copy)]
pub struct ThompsonSearchPolicy;

impl SearchPolicy for ThompsonSearchPolicy {
  fn backup(&self, tree: &mut FastSearchTree, traj: &Trajectory) -> usize {
    // TODO
    unimplemented!();
  }

  fn execute_search(&self, node: &FastSearchNode) -> Action {
    // TODO
    unimplemented!();
    /*if node.moves.len() > 0 {
      let j = array_argmax(&node.value);
      Action::Place{pos: node.moves[j]}
    } else {
      Action::Pass
    }*/
  }

  fn execute_best(&self, node: &FastSearchNode) -> Action {
    if node.moves.len() > 0 {
      let j = array_argmax(&node.num_trials);
      Action::Place{pos: node.moves[j]}
    } else {
      Action::Pass
    }
  }
}

pub trait RolloutPolicy {
  /*fn execute(&mut self, init_state: &FastBoard, init_aux_state: &FastBoardAux) -> Stone {
    unimplemented!();
  }*/

  /*fn reset(&mut self, init_state: &FastBoard, init_aux_state: &FastBoardAux) {
    unimplemented!();
  }*/

  fn batch_size(&self) -> usize;
  fn execution_behavior(&self) -> ExecutionBehavior;

  fn preload_batch_state(&mut self, batch_idx: usize, state: &FastBoard) {
    unimplemented!();
  }

  fn execute_batch(&mut self, batch_size: usize/*, mut f: &mut FnMut(usize, &mut RolloutPolicy)*/) {
    unimplemented!();
  }

  fn read_policy(&mut self, batch_size: usize) -> &[f32] {
    unimplemented!();
  }

  fn read_policy_cdfs(&mut self, batch_size: usize) -> &[f32] {
    unimplemented!();
  }
}

/*#[derive(Clone)]
pub struct QuasiUniformRolloutPolicy {
  rng:        XorShift128PlusRng,
  state:      FastBoard,
  work:       FastBoardWork,
  moves:      Vec<Pos>,
  max_plies:  usize,
}

impl QuasiUniformRolloutPolicy {
  pub fn new() -> QuasiUniformRolloutPolicy {
    QuasiUniformRolloutPolicy{
      // FIXME(20151008)
      rng:        XorShift128PlusRng::from_seed([thread_rng().next_u64(), thread_rng().next_u64()]),
      state:      FastBoard::new(),
      work:       FastBoardWork::new(),
      moves:      Vec::with_capacity(FastBoard::BOARD_SIZE),
      max_plies:  FastBoard::BOARD_SIZE,
    }
  }
}

impl RolloutPolicy for QuasiUniformRolloutPolicy {
  fn batch_size(&self) -> usize { 1 }
  fn execution_behavior(&self) -> ExecutionBehavior { ExecutionBehavior::Uniform }

  fn execute(&mut self, init_state: &FastBoard, init_aux_state: &FastBoardAux) -> Stone {
    let leaf_turn = init_state.current_turn();
    self.state.clone_from(init_state);
    // TODO(20151008): initialize moves list.
    self.moves.clear();
    self.moves.extend(init_aux_state.get_legal_positions(leaf_turn).iter()
      .map(|p| p as Pos));
    if self.moves.len() > 0 {
      'rollout: for _ in (0 .. self.max_plies) {
        let ply_turn = self.state.current_turn();
        loop {
          let pos = match choose_without_replace(&mut self.moves, &mut self.rng) {
            Some(pos) => pos,
            None => break 'rollout,
          };
          if self.state.is_legal_move_fast(ply_turn, pos) {
            let action = Action::Place{pos: pos};
            self.state.play(ply_turn, action, &mut self.work, &mut None, false);
            self.moves.extend(self.state.last_captures());
            break;
          } else if self.moves.len() == 0 {
            break 'rollout;
          }
        }
      }
    }
    // TODO(20151008): set correct komi.
    // XXX: Settle ties in favor of W player.
    if self.state.score_fast(6.5) >= 0.0 {
      Stone::White
    } else {
      Stone::Black
    }
  }
}*/

pub struct UniformRolloutPolicy;

impl UniformRolloutPolicy {
  pub fn new() -> UniformRolloutPolicy {
    UniformRolloutPolicy
  }
}

impl RolloutPolicy for UniformRolloutPolicy {
  fn batch_size(&self) -> usize { 1 }
  fn execution_behavior(&self) -> ExecutionBehavior { ExecutionBehavior::Uniform }

  fn preload_batch_state(&mut self, batch_idx: usize, state: &FastBoard) {
    // Do nothing.
  }

  fn execute_batch(&mut self, batch_size: usize) {
    // Do nothing.
  }
}

pub struct ConvNetBatchRolloutPolicy {
  //rng:    XorShift128PlusStreamRng,
  ctx:    DeviceContext,
  arch:   LinearNetArch,
}

impl ConvNetBatchRolloutPolicy {
  pub fn new() -> ConvNetBatchRolloutPolicy {
    let ctx = DeviceContext::new(0);
    let batch_size = 256;
    let num_hidden = 16;
    let data_layer_cfg = DataLayerConfig{
      raw_width: 19, raw_height: 19,
      crop_width: 19, crop_height: 19,
      channels: 4,
    };
    let conv1_layer_cfg = Conv2dLayerConfig{
      in_width: 19, in_height: 19, in_channels: 4,
      conv_size: 9, conv_stride: 1, conv_pad: 4,
      out_channels: num_hidden,
      act_fun: ActivationFunction::Rect,
      init_weights: ParamsInitialization::Normal{mean: 0.0, std: 0.01},
    };
    let conv2_layer_cfg = Conv2dLayerConfig{
      in_width: 19, in_height: 19, in_channels: num_hidden,
      conv_size: 3, conv_stride: 1, conv_pad: 1,
      out_channels: 1,
      act_fun: ActivationFunction::Identity,
      init_weights: ParamsInitialization::Normal{mean: 0.0, std: 0.01},
    };
    let loss_layer_cfg = SoftmaxLossLayerConfig{
      num_categories: 361,
      // FIXME(20151023): masking.
      do_mask: false,
      //do_mask: true,
    };
    let data_layer = DataLayer::new(0, data_layer_cfg, batch_size);
    let conv1_layer = Conv2dLayer::new(0, conv1_layer_cfg, batch_size, Some(&data_layer), &ctx);
    let conv2_layer = Conv2dLayer::new(0, conv2_layer_cfg, batch_size, Some(&conv1_layer), &ctx);
    let softmax_layer = SoftmaxLossLayer::new(0, loss_layer_cfg, batch_size, Some(&conv2_layer));
    let mut arch = LinearNetArch::new(
        PathBuf::from("experiments/models/convnet_19x19x4_conv_9x9x16R_conv_3x3x1"),
        batch_size,
        data_layer,
        softmax_layer,
        vec![
          Box::new(conv1_layer),
          Box::new(conv2_layer),
        ],
    );
    arch.load_layer_params(None, &ctx);

    ConvNetBatchRolloutPolicy {
      ctx:  ctx,
      arch: arch,
    }
  }
}

impl RolloutPolicy for ConvNetBatchRolloutPolicy {
  fn batch_size(&self) -> usize { self.arch.batch_size() }
  fn execution_behavior(&self) -> ExecutionBehavior { ExecutionBehavior::DiscreteDist }

  fn preload_batch_state(&mut self, batch_idx: usize, state: &FastBoard) {
    let &mut ConvNetBatchRolloutPolicy{ref ctx, ref mut arch, ..} = self;
    let turn = state.current_turn();
    // XXX: FastBoard features are absolutely arranged; the features we feed to
    // the net must be relatively arranged. Do this by performing a special
    // permutation of the feature planes.
    arch.data_layer().preload_frame_permute(batch_idx, state.extract_features(), turn.offset(), ctx);
    //arch.loss_layer().preload_mask(batch_idx, state.extract_mask(turn), ctx);
  }

  fn execute_batch(&mut self, batch_size: usize) {
    let &mut ConvNetBatchRolloutPolicy{
      ref ctx, ref mut arch, ..} = self;
    arch.data_layer().load_frames(batch_size, ctx);
    //arch.loss_layer().load_masks(batch_size, ctx);
    arch.evaluate(ctx);
    arch.loss_layer().store_labels(batch_size, &self.ctx);
    arch.loss_layer().store_cdfs(batch_size, &self.ctx);
    arch.loss_layer().store_probs(batch_size, &self.ctx);
    let labels = arch.loss_layer().predict_labels(batch_size, &self.ctx);
    //println!("DEBUG: convnet policy: labels: {:?}", labels);
  }

  fn read_policy(&mut self, batch_idx: usize) -> &[f32] {
    let batch_size = self.arch.batch_size();
    let batch_probs = &self.arch.loss_layer().predict_probs(batch_size, &self.ctx)
      .as_slice()[batch_idx * FastBoard::BOARD_SIZE .. (batch_idx + 1) * FastBoard::BOARD_SIZE];
    batch_probs
  }

  fn read_policy_cdfs(&mut self, batch_idx: usize) -> &[f32] {
    let batch_size = self.arch.batch_size();
    let batch_cdfs = &self.arch.loss_layer().predict_cdfs(batch_size, &self.ctx)
      .as_slice()[batch_idx * FastBoard::BOARD_SIZE .. (batch_idx + 1) * FastBoard::BOARD_SIZE];
    batch_cdfs
  }
}
