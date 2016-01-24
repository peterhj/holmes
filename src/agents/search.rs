use agents::{Agent};
use board::{Board, RuleSet, PlayerRank, Stone, Point, Action};
use search::{SearchResult};
use search::tree::{Tree, Trajectory, Search};
use search::policies::convnet::{
  ConvnetPriorPolicy,
  BatchConvnetRolloutPolicy,
  ParallelBatchConvnetRolloutPolicy,
};
use search::policies::quasiuniform::{QuasiUniformRolloutPolicy};
use search::policies::thompson_rave::{ThompsonRaveTreePolicy};
use search::policies::uct_rave::{UctRaveTreePolicy};
use txnstate::{TxnState};
use txnstate::extras::{TxnStateNodeData};

use async_cuda::context::{DeviceContext};

use std::path::{PathBuf};

pub struct SearchAgent {
  komi:     f32,
  player:   Option<Stone>,

  history:  Vec<(TxnState<TxnStateNodeData>, Action, Option<SearchResult>)>,
  state:    TxnState<TxnStateNodeData>,
  result:   Option<SearchResult>,

  //ctx:      DeviceContext,
  prior_policy: ConvnetPriorPolicy,
  //tree_policy:  UctRaveTreePolicy,
  tree_policy:  ThompsonRaveTreePolicy,
  //roll_policy:  QuasiUniformRolloutPolicy,
  roll_policy:  BatchConvnetRolloutPolicy,
  //roll_policy:  ParallelBatchConvnetRolloutPolicy,
}

impl SearchAgent {
  pub fn new() -> SearchAgent {
    //let ctx = DeviceContext::new(0);
    let mut prior_policy = ConvnetPriorPolicy::new();
    //let mut tree_policy = UctRaveTreePolicy::new();
    let mut tree_policy = ThompsonRaveTreePolicy::new();
    let batch_size = 256;
    //let mut roll_policy = QuasiUniformRolloutPolicy;
    let mut roll_policy = BatchConvnetRolloutPolicy::new(batch_size);
    //let mut roll_policy = ParallelBatchConvnetRolloutPolicy::new(batch_size);
    SearchAgent{
      komi:     0.0,
      player:   None,
      history:  vec![],
      state:    TxnState::new(
          [PlayerRank::Dan(9), PlayerRank::Dan(9)],
          RuleSet::KgsJapanese.rules(),
          TxnStateNodeData::new(),
      ),
      result:   None,
      //ctx:      ctx,
      prior_policy: prior_policy,
      tree_policy:  tree_policy,
      roll_policy:  roll_policy,
    }
  }
}

impl Agent for SearchAgent {
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
    // TODO(20151111)
  }

  fn apply_action(&mut self, turn: Stone, action: Action) {
    self.history.push((self.state.clone(), action, self.result));
    match self.state.try_action(turn, action) {
      Ok(_)   => { self.state.commit(); }
      Err(_)  => { panic!("agent tried to apply an illegal action!"); }
    }
  }

  fn undo(&mut self) {
    // FIXME(20151108): track ply; limit to how far we can undo.
    if let Some((prev_state, _, prev_result)) = self.history.pop() {
      self.state.clone_from(&prev_state);
      self.result = prev_result;
    } else {
      self.state.reset();
      self.result = None;
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
      self.state.unsafe_set_current_turn(turn);
    }
    assert_eq!(turn, self.state.current_turn());
    let mut search = Search::new(5120);
    let mut tree = Tree::new(self.state.clone(), &mut self.prior_policy, &mut self.tree_policy);
    let result = search.join(
        &mut tree,
        &mut self.prior_policy,
        &mut self.tree_policy,
        &mut self.roll_policy,
        self.komi,
        self.result.as_ref());
    println!("DEBUG: search stats:  {:?}", search.stats);
    println!("DEBUG: search result: {:?}", result);
    self.result = Some(result);
    result.action
  }
}
