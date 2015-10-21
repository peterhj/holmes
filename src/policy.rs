use fastboard::{PosExt, Pos, Action, Stone, FastBoard, FastBoardAux, FastBoardWork};
use fasttree::{FastSearchNode};
use random::{XorShift128PlusRng, choose_without_replace, sample_discrete_cdf};

//use array::{Buffer};
use async_cuda::context::{DeviceContext};
use rembrandt::layer::{Layer};
use rembrandt::net::{NetArch, LinearNetArch};
use statistics_avx2::array::{array_argmax};
use statistics_avx2::random::{StreamRng, XorShift128PlusStreamRng};

use bit_set::{BitSet};
use rand::{Rng, SeedableRng, thread_rng};

pub trait SearchPolicy {
  fn reset(&mut self);
  fn backup(&self, n: f32, num_trials: &[f32], succ_ratio: &[f32], value: &mut [f32]);
  fn execute_search(&self, node: &FastSearchNode) -> Action;
  fn execute_best(&self, node: &FastSearchNode) -> Action;
}

#[derive(Clone, Copy)]
pub struct UctSearchPolicy {
  pub c:  f32,
}

impl SearchPolicy for UctSearchPolicy {
  fn reset(&mut self) {
    // Do nothing.
  }

  fn backup(&self, n: f32, num_trials: &[f32], succ_ratio: &[f32], value: &mut [f32]) {
    for j in (0 .. value.len()) {
      value[j] = succ_ratio[j] + self.c * (n.ln() / num_trials[j]).sqrt();
    }
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
}

#[derive(Clone, Copy)]
pub struct RaveSearchPolicy;

#[derive(Clone, Copy)]
pub struct ThompsonSearchPolicyConfig {
  pub max_trials: i32,
}

#[derive(Clone, Copy)]
pub struct ThompsonSearchPolicy {
  config: ThompsonSearchPolicyConfig,
}

pub trait RolloutPolicy {
  fn execute(&mut self, init_state: &FastBoard, init_aux_state: &FastBoardAux) -> Stone {
    unimplemented!();
  }

  /*fn reset(&mut self, init_state: &FastBoard, init_aux_state: &FastBoardAux) {
    unimplemented!();
  }*/

  fn batch_size(&self) -> usize;

  fn preload_batch_state(&mut self, batch_idx: usize, state: &FastBoard) {
    unimplemented!();
  }

  fn execute_batch_step(&mut self, batch_size: usize) -> &[f32] {
    unimplemented!();
  }
}

#[derive(Clone)]
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
}

pub struct ConvNetBatchRolloutPolicy {
  rng:    XorShift128PlusStreamRng,
  moves:  Vec<Pos>,
  //mask:   BitSet,

  //host_input_buf: Buffer<u8>,
  ctx:    DeviceContext,
  arch:   LinearNetArch,
}

impl RolloutPolicy for ConvNetBatchRolloutPolicy {
  fn batch_size(&self) -> usize { self.arch.batch_size() }

  fn preload_batch_state(&mut self, batch_idx: usize, state: &FastBoard) {
    let &mut ConvNetBatchRolloutPolicy{ref ctx, ref mut arch, ..} = self;
    arch.data_layer().preload_frame(batch_idx, state.extract_features(), ctx);
    arch.loss_layer().premask_batch(batch_idx, state.extract_mask(), ctx);
  }

  fn execute_batch_step(&mut self, batch_size: usize) -> &[f32] {
    {
      let &mut ConvNetBatchRolloutPolicy{
        ref mut rng, ref mut moves,
        ref ctx, ref mut arch, ..} = self;
      arch.data_layer().load_frames(batch_size, ctx);
      arch.loss_layer().mask_batch(batch_size, ctx);
      arch.evaluate(ctx);
      let batch_cdfs = arch.loss_layer().predict_cdfs(batch_size, ctx).as_slice();
      batch_cdfs
      /*for batch_idx in (0 .. batch_size) {
        let cdf = &batch_cdfs[batch_idx * FastBoard::BOARD_SIZE .. (batch_idx + 1) * FastBoard::BOARD_SIZE];
        let j = sample_discrete_cdf(cdf, rng);
        moves[batch_idx] = j as Pos;
      }*/
    }
    //let &mut ConvNetBatchRolloutPolicy{ref moves, ..} = self;
    //moves
  }
}
