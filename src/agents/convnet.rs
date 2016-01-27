use agents::{Agent};
use board::{Board, RuleSet, PlayerRank, Stone, Point, Action};
use convnet::arch::{
  build_action_3layer_arch,
  build_action_6layer_arch,
  build_action_3layer_19x19x16_arch,
  build_action_6layer_19x19x16_arch,
  build_action_12layer_19x19x16_arch,
};
use txnstate::{TxnState};
use txnstate::features::{
  TxnStateFeaturesData,
  TxnStateLibFeaturesData,
};

use async_cuda::context::{DeviceContext};
use rembrandt::layer::{Layer};
use rembrandt::net::{NetArch, LinearNetArch};
use rembrandt::opt::{OptPhase};

use std::path::{PathBuf};

pub struct ConvnetAgent {
  komi:     f32,
  player:   Option<Stone>,

  history:  Vec<(TxnState<TxnStateLibFeaturesData>, Action)>,
  state:    TxnState<TxnStateLibFeaturesData>,

  ctx:      DeviceContext,
  arch:     LinearNetArch,
}

impl ConvnetAgent {
  pub fn new() -> ConvnetAgent {
    let ctx = DeviceContext::new(0);
    //let arch = build_action_3layer_arch(1, &ctx);
    //let arch = build_action_6layer_arch(1, &ctx);
    //let arch = build_action_3layer_19x19x16_arch(1, &ctx);
    //let arch = build_action_6layer_19x19x16_arch(1, &ctx);
    let arch = build_action_12layer_19x19x16_arch(1, &ctx);
    ConvnetAgent{
      komi:     0.0,
      player:   None,
      history:  vec![],
      state:    TxnState::new(
          [PlayerRank::Dan(9), PlayerRank::Dan(9)],
          RuleSet::KgsJapanese.rules(),
          TxnStateLibFeaturesData::new(),
      ),
      ctx:      ctx,
      arch:     arch,
    }
  }
}

impl Agent for ConvnetAgent {
  fn reset(&mut self) {
    self.komi = 7.5;
    self.player = None;

    self.history.clear();
    self.state.reset();
  }

  fn board_dim(&mut self, board_dim: usize) {
    assert_eq!(Board::DIM, board_dim);
  }

  fn komi(&mut self, komi: f32) {
    self.komi = komi;
  }

  fn player(&mut self, stone: Stone) {
    self.player = Some(Stone::Black);
  }

  fn apply_action(&mut self, turn: Stone, action: Action) {
    self.history.push((self.state.clone(), action));
    match self.state.try_action(turn, action) {
      Ok(_)   => { self.state.commit(); }
      Err(_)  => { panic!("agent tried to apply an illegal action!"); }
    }
  }

  fn undo(&mut self) {
    // FIXME(20151108): track ply; limit to how far we can undo.
    if let Some((prev_state, _)) = self.history.pop() {
      self.state.clone_from(&prev_state);
    } else {
      self.state.reset();
    }
  }

  fn act(&mut self, turn: Stone) -> Action {
    if self.history.len() > 0 {
      match self.history[self.history.len() - 1].1 {
        Action::Resign | Action::Pass => {
          return Action::Pass;
        }
        _ => {}
      }
    }
    let &mut ConvnetAgent{
      ref mut state, ref ctx, ref mut arch, .. } = self;
    state.get_data().extract_relative_features(turn, arch.data_layer().expose_host_frame_buf(0));
    arch.data_layer().load_frames(1, ctx);
    arch.evaluate(OptPhase::Inference, ctx);
    arch.loss_layer().store_ranked_labels(1, ctx);
    let ranked_labels = arch.loss_layer().predict_ranked_labels(1);
    let mut action = Action::Pass;
    for k in 0 .. Board::SIZE {
      let place_point = Point(ranked_labels[k] as i16);
      let res = state.try_place(turn, place_point);
      state.undo();
      match res {
        Ok(_) => {
          action = Action::Place{point: place_point};
          break;
        }
        Err(_) => {}
      }
    }
    action
  }
}
