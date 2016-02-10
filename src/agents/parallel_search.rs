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
use txnstate::{TxnStateConfig, TxnState};
use txnstate::extras::{TxnStateNodeData};

use async_cuda::context::{DeviceContext};
use cuda::runtime::{CudaDevice};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, thread_rng};
use std::cmp::{min};
use std::path::{PathBuf};

pub struct MonteCarloConfig {
  pub num_rollouts: usize,
  pub batch_size:   usize,
}

pub struct ParallelMonteCarloSearchAgent {
  config:   MonteCarloConfig,
  komi:     f32,
  player:   Option<Stone>,

  history:  Vec<(TxnState<TxnStateNodeData>, Action, Option<MonteCarloSearchResult>)>,
  ply:      usize,
  state:    TxnState<TxnStateNodeData>,
  result:   Option<MonteCarloSearchResult>,

  rng:      Xorshiftplus128Rng,
  server:   ParallelMonteCarloSearchServer<ConvnetPolicyWorker>,
}

impl ParallelMonteCarloSearchAgent {
  pub fn new(config: MonteCarloConfig, num_workers: Option<usize>) -> ParallelMonteCarloSearchAgent {
    let batch_capacity = 256;
    let num_devices = CudaDevice::count().unwrap();
    let num_workers = num_workers.unwrap_or(num_devices);
    let num_workers = min(num_workers, num_devices);
    let worker_batch_capacity = batch_capacity / num_workers;
    ParallelMonteCarloSearchAgent{
      config:   config,
      komi:     0.0,
      player:   None,
      history:  vec![],
      ply:      0,
      state:    TxnState::new(
          TxnStateConfig{
            rules:  RuleSet::KgsJapanese.rules(),
            ranks:  [PlayerRank::Dan(9), PlayerRank::Dan(9)],
          },
          TxnStateNodeData::new(),
      ),
      result:   None,
      rng:      Xorshiftplus128Rng::new(&mut thread_rng()),
      server:   ParallelMonteCarloSearchServer::new(
          num_workers, worker_batch_capacity,
          ConvnetPolicyWorkerBuilder::new(num_workers, worker_batch_capacity),
      ),
    }
  }
}

impl Agent for ParallelMonteCarloSearchAgent {
  fn reset(&mut self) {
    self.komi = 7.5;
    self.player = None;

    self.history.clear();
    self.ply = 0;
    self.state.reset();
    self.result = None;
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
    // FIXME(20160114): the search result may be stale here since we do not yet
    // support pondering.
    self.history.push((self.state.clone(), action, self.result));
    self.ply += 1;
    assert_eq!(self.history.len(), self.ply);
    match self.state.try_action(turn, action) {
      Ok(_)   => { self.state.commit(); }
      Err(_)  => { panic!("agent tried to apply an illegal action!"); }
    }
  }

  fn undo(&mut self) {
    // FIXME(20151108): track ply; limit to how far we can undo.
    if let Some((prev_state, _, prev_result)) = self.history.pop() {
      self.ply -= 1;
      self.state.clone_from(&prev_state);
      self.result = prev_result;
    } else {
      self.ply = 0;
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

    // FIXME(20160114): read remaining time and apply a time management policy.
    //let num_rollouts = 5120;
    //let num_rollouts = 10240;

    let num_rollouts = self.config.num_rollouts;
    let batch_size = self.config.batch_size;

    let mut search = ParallelMonteCarloSearch::new();
    let (search_res, search_stats) = search.join(
        num_rollouts,
        batch_size,
        &mut self.server,
        self.player.unwrap(),
        &self.state,
        self.result.as_ref(),
        &mut self.rng);
    println!("DEBUG: search result: {:?}", search_res);
    println!("DEBUG: search stats:  {:?}", search_stats);
    self.result = Some(search_res);
    search_res.action
  }
}
