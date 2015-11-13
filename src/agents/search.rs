use agents::{Agent};
use board::{Board, RuleSet, Stone, Point, Action};
use convnet::{
  build_action_3layer_arch,
  build_action_6layer_arch,
};
use search_policies::quasiuniform::{QuasiUniformRolloutPolicy};
use search_policies::uct_rave::{UctRaveTreePolicy};
use search_tree::{Tree, Trajectory, SequentialSearch};
use txnstate::{TxnState, TxnStateNodeData};

use async_cuda::context::{DeviceContext};
use rembrandt::layer::{Layer};
use rembrandt::net::{NetArch, LinearNetArch};
use rembrandt::opt::{OptPhase};

use std::path::{PathBuf};

pub struct SearchAgent {
  komi:     f32,

  history:  Vec<(TxnState<TxnStateNodeData>, Action)>,
  state:    TxnState<TxnStateNodeData>,

  ctx:      DeviceContext,
}

impl SearchAgent {
  pub fn new() -> SearchAgent {
    let ctx = DeviceContext::new(0);
    SearchAgent{
      komi:     0.0,
      history:  vec![],
      state:    TxnState::new(RuleSet::KgsJapanese.rules(), TxnStateNodeData::new()),
      ctx:      ctx,
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
    let search = SequentialSearch{
      num_rollouts: 1000,
    };
    let mut tree = Tree::new(self.state.clone());
    let mut traj = Trajectory::new();
    let mut tree_policy = UctRaveTreePolicy{c: 0.3};
    let mut roll_policy = QuasiUniformRolloutPolicy;
    let (_, action) = search.join(&mut tree, &mut traj, &mut tree_policy, &mut roll_policy);
    action
  }
}
