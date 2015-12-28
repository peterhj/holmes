use agents::{Agent};
use board::{Board, RuleSet, PlayerRank, Stone, Point, Action};
use search::{SearchResult};
use search::parallel_policies::convnet::{
  ConvnetPolicyWorkerBuilder, ConvnetPolicyWorker,
};
use search::parallel_tree::{
  ParallelMonteCarloSearchServer, ParallelMonteCarloSearch,
};
use txnstate::{TxnState};
use txnstate::extras::{TxnStateNodeData};

use async_cuda::context::{DeviceContext};
use rng::xorshift::{Xorshiftplus128Rng};

use std::path::{PathBuf};

pub struct ParallelMonteCarloSearchAgent {
  komi:     f32,
  player:   Option<Stone>,

  history:  Vec<(TxnState<TxnStateNodeData>, Action, Option<SearchResult>)>,
  state:    TxnState<TxnStateNodeData>,
  result:   Option<SearchResult>,

  server:   ParallelMonteCarloSearchServer<ConvnetPolicyWorker>,
}

impl ParallelMonteCarloSearchAgent {
  pub fn new(num_workers: usize) -> ParallelMonteCarloSearchAgent {
    let batch_size = 256;
    ParallelMonteCarloSearchAgent{
      komi:     0.0,
      player:   None,
      history:  vec![],
      state:    TxnState::new(
          [PlayerRank::Dan(9), PlayerRank::Dan(9)],
          RuleSet::KgsJapanese.rules(),
          TxnStateNodeData::new(),
      ),
      result:   None,
      server:   ParallelMonteCarloSearchServer::new(num_workers, batch_size / num_workers, ConvnetPolicyWorkerBuilder),
    }
  }
}

impl Agent for ParallelMonteCarloSearchAgent {
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
      self.state.set_turn(turn);
    }
    assert_eq!(turn, self.state.current_turn());

    /*let mut search = Search::new(5120);
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
    result.action*/

    let mut search = ParallelMonteCarloSearch::new();
    //search.join();

    // TODO(20151225)
    unimplemented!();
  }
}
