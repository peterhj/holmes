use agents::{Agent};
use board::{Board, RuleSet, Stone, Point, Action};
use search::{Tree, Trajectory, SequentialSearch};
use search::policies::convnet::{ConvnetPriorPolicy};
use search::policies::quasiuniform::{QuasiUniformRolloutPolicy};
use search::policies::uct_rave::{UctRaveTreePolicy};
use txnstate::{TxnState};
use txnstate::extras::{TxnStateNodeData};

use async_cuda::context::{DeviceContext};

use std::path::{PathBuf};

pub struct SearchAgent {
  komi:     f32,
  player:   Option<Stone>,

  history:  Vec<(TxnState<TxnStateNodeData>, Action)>,
  state:    TxnState<TxnStateNodeData>,

  //ctx:      DeviceContext,
  prior_policy: ConvnetPriorPolicy,
  tree_policy:  UctRaveTreePolicy,
  roll_policy:  QuasiUniformRolloutPolicy,
}

impl SearchAgent {
  pub fn new() -> SearchAgent {
    //let ctx = DeviceContext::new(0);
    let mut prior_policy = ConvnetPriorPolicy::new();
    let mut tree_policy = UctRaveTreePolicy{c: 0.5};
    let mut roll_policy = QuasiUniformRolloutPolicy;
    SearchAgent{
      komi:     0.0,
      player:   None,
      history:  vec![],
      state:    TxnState::new(RuleSet::KgsJapanese.rules(), TxnStateNodeData::new()),
      //ctx:      ctx,
      prior_policy: prior_policy,
      tree_policy:  tree_policy,
      roll_policy:  roll_policy,
    }
  }
}

impl Agent for SearchAgent {
  fn reset(&mut self) {
    self.komi = 6.5;
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
    if self.history.len() > 0 {
      match self.history[self.history.len() - 1].1 {
        Action::Resign | Action::Pass => {
          return Action::Pass;
        }
        _ => {}
      }
    }
    if self.player.is_none() {
      self.player = Some(turn);
      self.state.set_turn(turn);
    }
    assert_eq!(turn, self.state.current_turn());
    let mut search = SequentialSearch{
      num_rollouts: 10000,
      stats: Default::default(),
    };
    let mut tree = Tree::new(self.state.clone(), &mut self.prior_policy, &mut self.tree_policy);
    let mut traj = Trajectory::new();
    let (_, action) = search.join(&mut tree, &mut traj, &mut self.prior_policy, &mut self.tree_policy, &mut self.roll_policy);
    println!("DEBUG: search stats: {:?}", search.stats);
    action
  }
}
