use agents::{Agent};
use board::{Board, RuleSet, PlayerRank, Stone, Point, Action};
use search::{SearchResult};
use search::parallel_policies::convnet::{
  ConvnetPolicyWorkerBuilder, ConvnetPolicyWorker,
};
use search::parallel_tree::{
  MonteCarloSearchResult,
  ParallelMonteCarloSearchServer, ParallelMonteCarloSearch,
};
use txnstate::{TxnState};
use txnstate::extras::{TxnStateNodeData};

use async_cuda::context::{DeviceContext};
use cuda::runtime::{CudaDevice};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, thread_rng};
use std::cmp::{min};
use std::path::{PathBuf};

pub struct ParallelMonteCarloSearchAgent {
  komi:     f32,
  player:   Option<Stone>,

  history:  Vec<(TxnState<TxnStateNodeData>, Action, Option<MonteCarloSearchResult>)>,
  state:    TxnState<TxnStateNodeData>,
  //result:   Option<SearchResult>,
  result:   Option<MonteCarloSearchResult>,

  server:   ParallelMonteCarloSearchServer<ConvnetPolicyWorker>,
}

impl ParallelMonteCarloSearchAgent {
  pub fn new(num_workers: Option<usize>) -> ParallelMonteCarloSearchAgent {
    let batch_size = 256;
    let num_devices = CudaDevice::count().unwrap();
    let num_workers = num_workers.unwrap_or(num_devices);
    let num_workers = min(num_workers, num_devices);
    let worker_batch_size = batch_size / num_workers;
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
      server:   ParallelMonteCarloSearchServer::new(
                    num_workers, worker_batch_size,
                    ConvnetPolicyWorkerBuilder::new(num_workers, worker_batch_size),
                ),
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

    let num_rollouts = 5120;
    let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
    let mut search = ParallelMonteCarloSearch::new();
    let search_res = search.join(num_rollouts, &mut self.server, &self.state, &mut rng);
    println!("DEBUG: search result: {:?}", search_res);
    self.result = Some(search_res);
    search_res.action
  }
}
