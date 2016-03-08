use agents::{Agent};
use board::{Board, RuleSet, PlayerRank, Stone, Point, Action};
use search::{SearchResult};
use search::parallel_policies::convnet::{
  ConvnetPolicyWorkerBuilder, ConvnetPolicyWorker,
};
use search::parallel_tree::{
  MonteCarloSearchConfig,
  TreePolicyConfig,
  SharedTree,
  SearchWorkerConfig,
  SearchWorkerBatchConfig,
  MonteCarloSearchResult,
  ParallelMonteCarloSearchServer,
  ParallelMonteCarloSearch,
};
use txnstate::{TxnStateConfig, TxnState};
use txnstate::extras::{TxnStateNodeData};

use async_cuda::context::{DeviceContext};
use cuda::runtime::{CudaDevice};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, thread_rng};
use std::cmp::{min};
use std::path::{PathBuf};

/*pub struct MonteCarloConfig {
  pub num_rollouts: usize,
  pub batch_size:   usize,
}*/

pub struct ParallelMonteCarloSearchAgent {
  config:       MonteCarloSearchConfig,
  tree_cfg:     TreePolicyConfig,
  komi:         f32,
  player:       Option<Stone>,

  history:      Vec<(TxnState<TxnStateNodeData>, Action, Option<MonteCarloSearchResult>)>,
  ply:          usize,
  state:        TxnState<TxnStateNodeData>,
  result:       Option<MonteCarloSearchResult>,
  tree:         Option<SharedTree>,

  rng:          Xorshiftplus128Rng,
  server:       ParallelMonteCarloSearchServer<ConvnetPolicyWorker>,
}

impl ParallelMonteCarloSearchAgent {
  pub fn new(config: MonteCarloSearchConfig, tree_cfg: TreePolicyConfig, num_workers: Option<usize>) -> ParallelMonteCarloSearchAgent {
    println!("DEBUG: parallel search agent: search config: {:?}", config);
    println!("DEBUG: parallel search agent: tree policy config: {:?}", tree_cfg);
    let state_cfg = TxnStateConfig::default();
    let num_devices = CudaDevice::count().unwrap();
    let num_workers = num_workers.unwrap_or(num_devices);
    let num_workers = min(num_workers, num_devices);
    //let batch_capacity = 256;
    //let worker_batch_capacity = batch_capacity / num_workers;
    //let worker_batch_capacity = 576;
    let worker_batch_capacity = 256;
    ParallelMonteCarloSearchAgent{
      config:   config,
      tree_cfg: tree_cfg,
      komi:     0.0,
      player:   None,
      history:  vec![],
      ply:      0,
      state:    TxnState::new(
          state_cfg,
          TxnStateNodeData::new(),
      ),
      result:   None,
      tree:     None,
      rng:      Xorshiftplus128Rng::new(&mut thread_rng()),
      server:   ParallelMonteCarloSearchServer::new(
          state_cfg,
          num_workers, 1, worker_batch_capacity,
          ConvnetPolicyWorkerBuilder::new(tree_cfg, num_workers, 1, worker_batch_capacity),
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
    self.tree = None;
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
    self.history.push((self.state.clone(), action, self.result.clone()));
    self.ply += 1;
    assert_eq!(self.history.len(), self.ply);
    match self.state.try_action(turn, action) {
      Ok(_)   => { self.state.commit(); }
      Err(_)  => { panic!("agent tried to apply an illegal action!"); }
    }

    // XXX(20160210): Step forward the tree, if possible.
    let mut advance_success = false;
    if self.tree.is_none() {
      println!("DEBUG: ParallelSearchAgent: no tree to advance");
    }
    if let Some(ref tree) = self.tree {
      if tree.try_advance(turn, action) {
        advance_success = true;
      }
    }
    if !advance_success {
      self.tree = None;
    }
  }

  fn undo(&mut self) {
    // FIXME(20151108): track ply; limit to how far we can undo.
    if let Some((prev_state, _, prev_result)) = self.history.pop() {
      self.ply -= 1;
      self.state.clone_from(&prev_state);
      self.tree = None;
      self.result = prev_result;
    } else {
      self.ply = 0;
      self.state.reset();
      self.tree = None;
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

    let num_rollouts = self.config.num_rollouts;
    let batch_size = self.config.batch_size;

    let shared_tree = if self.tree.is_none() {
      let shared_tree = SharedTree::new(self.tree_cfg);
      self.tree = Some(shared_tree.clone());
      shared_tree
    } else {
      self.tree.as_ref().unwrap().clone()
    };
    let worker_cfg = SearchWorkerConfig{
      batch_cfg:    SearchWorkerBatchConfig::Fixed{num_batches: num_rollouts / batch_size},
      tree_batch_size:      None,
      rollout_batch_size:   batch_size,
    };
    let mut search = ParallelMonteCarloSearch::new();
    let (search_res, search_stats) = search.join(
        worker_cfg,
        &mut self.server,
        self.player.unwrap(),
        &self.state,
        shared_tree,
        //self.result.as_ref(),
        &mut self.rng);
    let action = search_res.action;
    println!("DEBUG: search result: {:?}", search_res);
    println!("DEBUG: search stats:  {:?}", search_stats);
    self.result = Some(search_res);
    action
  }
}
