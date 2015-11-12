use agents::{Agent};
use board::{Board, RuleSet, Stone, Point, Action};
use convnet::{
  build_action_3layer_arch,
  build_action_6layer_arch,
};
use features::{TxnStateFeaturesData};
use txnstate::{TxnState};

use async_cuda::context::{DeviceContext};
use rembrandt::layer::{Layer};
use rembrandt::net::{NetArch, LinearNetArch};
use rembrandt::opt::{OptPhase};

use std::path::{PathBuf};

pub struct SearchAgent {
  komi:     f32,

  history:  Vec<(TxnState<TxnStateFeaturesData>, Action)>,
  state:    TxnState<TxnStateFeaturesData>,

  ctx:          DeviceContext,
  prior_arch:   LinearNetArch,
}

impl SearchAgent {
  pub fn new() -> SearchAgent {
    let ctx = DeviceContext::new(0);
    let prior_arch = build_action_6layer_arch(1, &ctx);
    //let prior_arch = build_action_3layer_arch(1, &ctx);
    SearchAgent{
      komi:     0.0,
      history:  vec![],
      state:    TxnState::new(RuleSet::KgsJapanese.rules(), TxnStateFeaturesData::new()),
      ctx:          ctx,
      prior_arch:   prior_arch,
    }
  }
}

impl Agent for SearchAgent {
  fn reset(&mut self) {
    self.komi = 6.5;

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
    // TODO(20151111)
  }

  fn apply_action(&mut self, turn: Stone, action: Action) {
  }

  fn undo(&mut self) {
  }

  fn act(&mut self, turn: Stone) -> Action {
    // TODO(20151111)
    unimplemented!();
  }
}
