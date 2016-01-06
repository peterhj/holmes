use array_util::{array_argmax};
use board::{Board, RuleSet, PlayerRank, Stone, Point, Action};
use hyper::{load_hyperparam};
//use random::{XorShift128PlusRng};
use search::{SearchResult, SearchStats};
use search::parallel_policies::{
  SearchPolicyWorkerBuilder, SearchPolicyWorker,
  PriorPolicy, TreePolicy, RolloutPolicy,
};
use txnstate::{TxnState, check_good_move_fast};
use txnstate::extras::{TxnStateNodeData};
use txnstate::features::{
  TxnStateLibFeaturesData,
  TxnStateExtLibFeatsData,
};

use float::ord::{F32InfNan};
use rng::xorshift::{Xorshiftplus128Rng};
use threadpool::{ThreadPool};

use bit_set::{BitSet};
use rand::{Rng, SeedableRng, thread_rng};
use std::cmp::{max};
use std::iter::{repeat};
use std::marker::{PhantomData};
use std::sync::{Arc, Barrier, RwLock};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Sender, Receiver, channel};
use vec_map::{VecMap};

pub struct HyperparamConfig {
  pub prior:        bool,
  pub prior_equiv:  f32,
  pub pwide:        bool,
  pub pwide_mu:     f32,
  pub rave:         bool,
  pub rave_equiv:   f32,
}

impl HyperparamConfig {
  pub fn new() -> HyperparamConfig {
    HyperparamConfig{
      prior:        load_hyperparam("prior"),
      prior_equiv:  load_hyperparam("prior_equiv"),
      pwide:        load_hyperparam("pwide"),
      pwide_mu:     load_hyperparam("pwide_mu"),
      rave:         load_hyperparam("rave"),
      rave_equiv:   load_hyperparam("rave_equiv"),
    }
  }
}

#[derive(Clone)]
pub struct TreeTraj {
  pub backup_triples: Vec<(Arc<RwLock<Node>>, Point, usize)>,
  pub leaf_node:      Option<Arc<RwLock<Node>>>,
}

impl TreeTraj {
  pub fn new() -> TreeTraj {
    TreeTraj{
      backup_triples: vec![],
      leaf_node:      None,
    }
  }

  pub fn reset(&mut self) {
    self.backup_triples.clear();
    self.leaf_node = None;
  }
}

#[derive(Clone)]
pub struct RolloutTraj {
  pub rollout:      bool,
  pub sim_state:    TxnState<TxnStateLibFeaturesData>,
  pub sim_pairs:    Vec<(Stone, Point)>,

  pub raw_score:    Option<f32>,
  pub adj_score:    Option<[f32; 2]>,

  // FIXME(20151222): should probably move this to a `CommonTraj` data structure
  // but not critical.
  rave_mask:        Vec<BitSet>,
}

impl RolloutTraj {
  pub fn new() -> RolloutTraj {
    RolloutTraj{
      rollout:      false,
      sim_state:    TxnState::new(
          [PlayerRank::Dan(9), PlayerRank::Dan(9)],
          RuleSet::KgsJapanese.rules(),
          TxnStateLibFeaturesData::new(),
      ),
      sim_pairs:    vec![],
      raw_score:    None,
      adj_score:    None,
      rave_mask:    vec![
        BitSet::with_capacity(Board::SIZE),
        BitSet::with_capacity(Board::SIZE),
      ],
    }
  }

  pub fn reset_terminal(&mut self) {
    self.rollout = false;
    self.sim_state.reset();
    self.sim_pairs.clear();
    self.raw_score = None;
    self.adj_score = None;
    self.rave_mask[0].clear();
    self.rave_mask[1].clear();
  }

  pub fn reset_rollout(&mut self, tree_traj: &TreeTraj) {
    // TODO(20151222)
    unimplemented!();
  }
}

pub struct NodeValues {
  pub prior_values:     Vec<f32>,
  pub total_trials:     AtomicUsize,
  pub num_trials:       Vec<AtomicUsize>,
  pub num_succs:        Vec<AtomicUsize>,
  pub num_trials_rave:  Vec<AtomicUsize>,
  pub num_succs_rave:   Vec<AtomicUsize>,
}

impl NodeValues {
  pub fn new(num_arms: usize) -> NodeValues {
    let mut num_trials = Vec::with_capacity(num_arms);
    for _ in 0 .. num_arms {
      num_trials.push(AtomicUsize::new(0));
    }
    let mut num_succs = Vec::with_capacity(num_arms);
    for _ in 0 .. num_arms {
      num_succs.push(AtomicUsize::new(0));
    }
    let mut num_trials_rave = Vec::with_capacity(num_arms);
    for _ in 0 .. num_arms {
      num_trials_rave.push(AtomicUsize::new(0));
    }
    let mut num_succs_rave = Vec::with_capacity(num_arms);
    for _ in 0 .. num_arms {
      num_succs_rave.push(AtomicUsize::new(0));
    }
    NodeValues{
      prior_values:     repeat(0.5).take(num_arms).collect(),
      total_trials:     AtomicUsize::new(0),
      num_trials:       num_trials,
      num_succs:        num_succs,
      num_trials_rave:  num_trials_rave,
      num_succs_rave:   num_succs_rave,
    }
  }

  pub fn num_trials_float(&self) -> Vec<f32> {
    let mut ns = vec![];
    for j in 0 .. self.num_trials.len() {
      ns.push(self.num_trials[j].load(Ordering::Acquire) as f32);
    }
    ns
  }
}

pub struct Node {
  pub state:        TxnState<TxnStateNodeData>,

  pub horizon:      usize,
  pub valid_moves:  Vec<Point>,
  pub child_nodes:  Vec<Option<Arc<RwLock<Node>>>>,
  pub action_idxs:  VecMap<usize>,
  pub values:       NodeValues,
}

impl Node {
  pub fn new(state: TxnState<TxnStateNodeData>, prior_policy: &mut PriorPolicy) -> Node {
    let mut valid_moves = vec![];
    state.get_data().legality.fill_legal_points(state.current_turn(), &mut valid_moves);
    let num_arms = valid_moves.len();
    // FIXME(20151224): progressive widening.
    let init_horizon = num_arms;
    let mut values = NodeValues::new(num_arms);
    let mut action_priors = vec![];
    let mut action_idxs = VecMap::with_capacity(Board::SIZE);

    // XXX(20151224): Sort moves by descending value for progressive widening.
    prior_policy.fill_prior_values(&state, &valid_moves, &mut action_priors);
    action_priors.sort_by(|left, right| {
      F32InfNan(right.1).cmp(&F32InfNan(left.1))
    });
    for j in (0 .. num_arms) {
      valid_moves[j] = action_priors[j].0;
      values.prior_values[j] = action_priors[j].1;
    }

    let child_nodes: Vec<_> = repeat(None).take(num_arms).collect();
    for j in (0 .. num_arms) {
      action_idxs.insert(valid_moves[j].idx(), j);
    }
    Node{
      state:        state,
      horizon:      init_horizon,
      valid_moves:  valid_moves,
      child_nodes:  child_nodes,
      action_idxs:  action_idxs,
      values:       values,
    }
  }

  pub fn is_terminal(&self) -> bool {
    self.valid_moves.is_empty()
  }

  pub fn update_visits(&self) {
    self.values.total_trials.fetch_add(1, Ordering::AcqRel);
  }

  pub fn update_arm(&self, j: usize, score: f32) {
    let turn = self.state.current_turn();
    self.values.num_trials[j].fetch_add(1, Ordering::AcqRel);
    if (Stone::White == turn && score >= 0.0) ||
        (Stone::Black == turn && score < 0.0)
    {
      self.values.num_succs[j].fetch_add(1, Ordering::AcqRel);
    }
  }

  pub fn rave_update_arm(&self, j: usize, score: f32) {
    let turn = self.state.current_turn();
    self.values.num_trials_rave[j].fetch_add(1, Ordering::AcqRel);
    if (Stone::White == turn && score >= 0.0) ||
        (Stone::Black == turn && score < 0.0)
    {
      self.values.num_succs_rave[j].fetch_add(1, Ordering::AcqRel);
    }
  }
}

pub enum TreeResult {
  NonTerminal,
  Terminal,
}

pub struct Tree {
  //pwide_cfg:    ProgWideConfig,
  //count:        Arc<AtomicUsize>,
  //tree_idx:     usize,
  root_node:    Arc<RwLock<Node>>,
}

impl Drop for Tree {
  fn drop(&mut self) {
    //self.count.fetch_sub(1, Ordering::SeqCst);
  }
}

impl Clone for Tree {
  fn clone(&self) -> Tree {
    //let tree_idx = self.count.fetch_add(1, Ordering::SeqCst);
    Tree{
      //count:        self.count.clone(),
      //tree_idx:     tree_idx,
      root_node:    self.root_node.clone(),
    }
  }
}

impl Tree {
  pub fn new(init_state: TxnState<TxnStateNodeData>, prior_policy: &mut PriorPolicy) -> Tree {
    Tree{
      //count:        Arc::new(AtomicUsize::new(1)),
      //tree_idx:     0,
      root_node:    Arc::new(RwLock::new(Node::new(init_state, prior_policy))),
    }
  }

  pub fn walk(&self,
      tree_traj: &mut TreeTraj,
      prior_policy: &mut PriorPolicy,
      tree_policy: &mut TreePolicy<R=Xorshiftplus128Rng>,
      stats: &mut SearchStats,
      rng: &mut Xorshiftplus128Rng)
      -> TreeResult
  {
    tree_traj.reset();

    let mut ply = 0;
    let mut cursor_node = self.root_node.clone();
    loop {
      // At the cursor node, decide to walk or rollout depending on the total
      // number of trials.
      ply += 1;
      let cursor_trials = cursor_node.read().unwrap().values.total_trials.load(Ordering::Acquire);
      if cursor_trials >= 1 {
        // Try to walk through the current node using the exploration policy.
        stats.edge_count += 1;
        let res = tree_policy.execute_search(&*cursor_node.read().unwrap(), rng);
        match res {
          Some((place_point, j)) => {
            tree_traj.backup_triples.push((cursor_node.clone(), place_point, j));
            let has_child = cursor_node.read().unwrap().child_nodes[j].is_some();
            if has_child {
              // Existing inner node, simply update the cursor.
              let child_node = cursor_node.read().unwrap().child_nodes[j].as_ref().unwrap().clone();
              cursor_node = child_node;
              stats.inner_edge_count += 1;
            } else {
              // Create a new leaf node and stop the walk.
              let mut leaf_state = cursor_node.read().unwrap().state.clone();
              let turn = leaf_state.current_turn();
              match leaf_state.try_place(turn, place_point) {
                Ok(_) => {
                  leaf_state.commit();
                }
                Err(e) => {
                  // XXX: this means the legal moves features gave an incorrect result.
                  panic!("walk failed due to illegal move: {:?}", e);
                }
              }
              let mut leaf_node = Arc::new(RwLock::new(Node::new(leaf_state, prior_policy)));
              cursor_node.write().unwrap().child_nodes[j] = Some(leaf_node.clone());
              cursor_node = leaf_node;
              stats.new_leaf_count += 1;
              break;
            }
          }
          None => {
            // Terminal node, stop the walk.
            stats.term_edge_count += 1;
            break;
          }
        }
      } else {
        // Not enough trials, stop the walk and do a rollout.
        stats.old_leaf_count += 1;
        break;
      }
    }

    // TODO(20151111): if the cursor node is terminal, score it now and backup;
    // otherwise do a rollout and then backup.
    stats.max_ply = max(stats.max_ply, ply);
    tree_traj.leaf_node = Some(cursor_node.clone());
    let terminal = cursor_node.read().unwrap().is_terminal();
    if terminal {
      TreeResult::Terminal
    } else {
      TreeResult::NonTerminal
    }
  }

  pub fn backup(&self, tree_traj: &TreeTraj, rollout_traj: &mut RolloutTraj, rng: &mut Xorshiftplus128Rng) {
    rollout_traj.rave_mask[0].clear();
    rollout_traj.rave_mask[1].clear();
    let raw_score = rollout_traj.raw_score.unwrap();
    let adj_score = rollout_traj.adj_score.unwrap();

    if rollout_traj.sim_pairs.len() >= 1 {
      assert!(rollout_traj.rollout);
      let leaf_node = tree_traj.leaf_node.as_ref().unwrap().read().unwrap();
      let leaf_turn = leaf_node.state.current_turn();

      // XXX(20151120): Currently not tracking pass pairs, so need to check that
      // the first rollout pair belongs to the leaf node turn.
      let (update_turn, update_point) = rollout_traj.sim_pairs[0];
      if leaf_turn == update_turn {
        if let Some(&update_j) = leaf_node.action_idxs.get(&update_point.idx()) {
          assert_eq!(update_point, leaf_node.valid_moves[update_j]);
          leaf_node.update_arm(update_j, adj_score[update_turn.offset()]);
        } else {
          panic!("WARNING: leaf_node action_idxs does not contain update arm: {:?}", update_point);
        }
      }

      for &(sim_turn, sim_point) in rollout_traj.sim_pairs.iter() {
        rollout_traj.rave_mask[sim_turn.offset()].insert(sim_point.idx());
      }
      for sim_p in rollout_traj.rave_mask[update_turn.offset()].iter() {
        let sim_point = Point::from_idx(sim_p);
        if let Some(&sim_j) = leaf_node.action_idxs.get(&sim_point.idx()) {
          assert_eq!(sim_point, leaf_node.valid_moves[sim_j]);
          leaf_node.rave_update_arm(sim_j, adj_score[update_turn.offset()]);
        }
      }
    }

    {
      let leaf_node = tree_traj.leaf_node.as_ref().unwrap().read().unwrap();
      leaf_node.update_visits();
    }

    for &(ref node, update_point, update_j) in tree_traj.backup_triples.iter().rev() {
      let node = node.read().unwrap();

      assert_eq!(update_point, node.valid_moves[update_j]);
      let update_turn = node.state.current_turn();
      node.update_arm(update_j, adj_score[update_turn.offset()]);

      rollout_traj.rave_mask[update_turn.offset()].insert(update_point.idx());
      for sim_p in rollout_traj.rave_mask[update_turn.offset()].iter() {
        let sim_point = Point::from_idx(sim_p);
        if let Some(&sim_j) = node.action_idxs.get(&sim_point.idx()) {
          assert_eq!(sim_point, node.valid_moves[sim_j]);
          node.rave_update_arm(sim_j, adj_score[update_turn.offset()]);
        }
      }

      node.update_visits();
    }

    // FIXME(20151222): backup accumulators.
    //self.backup_count += 1;
    //self.mean_raw_score += (raw_score - self.mean_raw_score) / self.backup_count as f32;
  }
}

pub struct ParallelMonteCarloEvalServer/*<W> where W: SearchPolicyWorker*/ {
  num_workers:          usize,
  worker_batch_size:    usize,
  barrier:  Arc<Barrier>,
  pool:     ThreadPool,
  in_txs:   Vec<Sender<()>>,
  out_rx:   Receiver<()>,
  //_marker:  PhantomData<W>,
}

impl ParallelMonteCarloEvalServer {
  pub fn new<B>(num_workers: usize, worker_batch_size: usize, worker_builder: B) -> ParallelMonteCarloEvalServer {
    let barrier = Arc::new(Barrier::new(num_workers + 1));
    let pool = ThreadPool::new(num_workers);
    let mut in_txs = vec![];
    let (out_tx, out_rx) = channel();

    for tid in 0 .. num_workers {
      let builder = worker_builder.clone();
      let barrier = barrier.clone();

      let (in_tx, in_rx) = channel();
      let out_tx = out_tx.clone();
      in_txs.push(in_tx);

      pool.execute(move || {
        let mut rng = Xorshiftplus128Rng::from_seed([123, 456]);
        let mut worker = builder.build_worker(tid, worker_batch_size);
        let barrier = barrier;
        let in_rx = in_rx;
        let out_tx = out_tx;

        loop {
          // FIXME(20151222): for real time search, num batches is just an
          // estimate; should check for termination within inner batch loop.
          let cmd: SearchWorkerCommand = in_rx.recv().unwrap();
          let (cfg, tree) = match cmd {
            SearchWorkerCommand::Search{cfg, init_state} => {
              (cfg, Tree::new(init_state, worker.prior_policy()))
            }
            SearchWorkerCommand::Quit => break,
          };
          let batch_size = cfg.batch_size;
          //let num_batches = cfg.num_batches.unwrap();

          // TODO(20160106): initialize tree trajectories with leaf state.
          let mut tree_trajs: Vec<_> = repeat(TreeTraj::new()).take(batch_size).collect();
          let mut rollout_trajs: Vec<_> = repeat(RolloutTraj::new()).take(batch_size).collect();
          // FIXME(20151222): should share stats between workers.
          let mut stats: SearchStats = Default::default();

          //for batch in 0 .. num_batches {
          {
            /*let (prior_policy, tree_policy) = worker.prior_and_tree_policies();
            for batch_idx in 0 .. batch_size {
              let tree_traj = &mut tree_trajs[batch_idx];
              tree.walk(tree_traj, prior_policy, tree_policy, &mut stats, &mut rng);
            }*/
          }

          worker.rollout_policy().rollout_batch(&tree_trajs, &mut rollout_trajs, &mut rng);

          /*for batch_idx in 0 .. batch_size {
            let tree_traj = &tree_trajs[batch_idx];
            let rollout_traj = &mut rollout_trajs[batch_idx];
            tree.backup(tree_traj, rollout_traj, &mut rng);
          }*/

          barrier.wait();
          //}

        }

        out_tx.send(()).unwrap();
      });
    }

    ParallelMonteCarloSearchServer{
      num_workers:          num_workers,
      worker_batch_size:    worker_batch_size,
      barrier:  barrier,
      pool:     pool,
      in_txs:   in_txs,
      out_rx:   out_rx,
      _marker:  PhantomData,
    }
  }
}

#[derive(Clone)]
pub enum SearchWorkerCommand {
  Search{cfg: SearchWorkerConfig, init_state: TxnState<TxnStateNodeData>},
  Quit,
}

#[derive(Clone, Copy)]
pub struct SearchWorkerConfig {
  pub batch_size:   usize,
}

pub struct ParallelMonteCarloSearchServer<W> where W: SearchPolicyWorker {
  num_workers:          usize,
  worker_batch_size:    usize,
  barrier:  Arc<Barrier>,
  pool:     ThreadPool,
  in_txs:   Vec<Sender<SearchWorkerCommand>>,
  out_rx:   Receiver<()>,
  _marker:  PhantomData<W>,
}

impl<W> ParallelMonteCarloSearchServer<W> where W: SearchPolicyWorker {
  pub fn new<B>(num_workers: usize, worker_batch_size: usize, worker_builder: B) -> ParallelMonteCarloSearchServer<W>
  where B: 'static + SearchPolicyWorkerBuilder<Worker=W> {
    let barrier = Arc::new(Barrier::new(num_workers + 1));
    let pool = ThreadPool::new(num_workers);
    let mut in_txs = vec![];
    let (out_tx, out_rx) = channel();

    for tid in 0 .. num_workers {
      let builder = worker_builder.clone();
      let barrier = barrier.clone();

      let (in_tx, in_rx) = channel();
      let out_tx = out_tx.clone();
      in_txs.push(in_tx);

      pool.execute(move || {
        let mut rng = Xorshiftplus128Rng::from_seed([123, 456]);
        let mut worker = builder.build_worker(tid, worker_batch_size);
        let barrier = barrier;
        let in_rx = in_rx;
        let out_tx = out_tx;

        loop {
          // FIXME(20151222): for real time search, num batches is just an
          // estimate; should check for termination within inner batch loop.
          let cmd: SearchWorkerCommand = in_rx.recv().unwrap();
          let (cfg, tree) = match cmd {
            SearchWorkerCommand::Search{cfg, init_state} => {
              (cfg, Tree::new(init_state, worker.prior_policy()))
            }
            SearchWorkerCommand::Quit => break,
          };
          let batch_size = cfg.batch_size;
          //let num_batches = cfg.num_batches.unwrap();

          let mut tree_trajs: Vec<_> = repeat(TreeTraj::new()).take(batch_size).collect();
          let mut rollout_trajs: Vec<_> = repeat(RolloutTraj::new()).take(batch_size).collect();
          // FIXME(20151222): should share stats between workers.
          let mut stats: SearchStats = Default::default();

          //for batch in 0 .. num_batches {
          {
            let (prior_policy, tree_policy) = worker.prior_and_tree_policies();
            for batch_idx in 0 .. batch_size {
              let tree_traj = &mut tree_trajs[batch_idx];
              tree.walk(tree_traj, prior_policy, tree_policy, &mut stats, &mut rng);
            }
          }

          {
            worker.rollout_policy().rollout_batch(&tree_trajs, &mut rollout_trajs, &mut rng);
          }

          for batch_idx in 0 .. batch_size {
            let tree_traj = &tree_trajs[batch_idx];
            let rollout_traj = &mut rollout_trajs[batch_idx];
            tree.backup(tree_traj, rollout_traj, &mut rng);
          }

          barrier.wait();
          //}

        }

        out_tx.send(()).unwrap();
      });
    }

    ParallelMonteCarloSearchServer{
      num_workers:          num_workers,
      worker_batch_size:    worker_batch_size,
      barrier:  barrier,
      pool:     pool,
      in_txs:   in_txs,
      out_rx:   out_rx,
      _marker:  PhantomData,
    }
  }

  pub fn num_workers(&self) -> usize {
    self.num_workers
  }

  pub fn worker_batch_size(&self) -> usize {
    self.worker_batch_size
  }

  pub fn enqueue(&self, tid: usize, cmd: SearchWorkerCommand) {
    self.in_txs[tid].send(cmd).unwrap();
  }

  pub fn sync(&self) {
    self.barrier.wait();
  }

  pub fn join(&self) {
    for _ in self.out_rx.iter().take(self.num_workers) {
    }
  }
}

#[derive(Default)]
pub struct ParallelMonteCarloSearchStats {
  pub argmax_j:         AtomicUsize,
  pub argmax_ntrials:   AtomicUsize,
}

pub struct ParallelMonteCarloSearch {
  pub stats:        Arc<ParallelMonteCarloSearchStats>,
}

impl ParallelMonteCarloSearch {
  pub fn new() -> ParallelMonteCarloSearch {
    ParallelMonteCarloSearch{
      stats:        Arc::new(Default::default()),
    }
  }

  pub fn join<W>(&self,
      //total_batch_size:   usize,
      total_num_rollouts: usize,
      server:       &ParallelMonteCarloSearchServer<W>,
      init_state:   &TxnState<TxnStateNodeData>,
      //tree:     Tree,
      rng:          &mut Xorshiftplus128Rng)
      -> SearchResult
      where W: SearchPolicyWorker
  {
    let num_workers = server.num_workers();
    let num_rollouts = (total_num_rollouts + num_workers - 1) / num_workers * num_workers;
    let num_worker_rollouts = num_rollouts / num_workers;
    //let worker_batch_size = (total_batch_size + num_workers - 1) / num_workers;
    let worker_batch_size = server.worker_batch_size();
    let num_batches = (num_worker_rollouts + worker_batch_size - 1) / worker_batch_size;
    assert!(worker_batch_size >= 1);
    assert!(num_batches >= 1);

    // TODO(20160106): reset stats.

    let cfg = SearchWorkerConfig{
      batch_size:   worker_batch_size,
    };
    for batch in 0 .. num_batches {
      for tid in 0 .. num_workers {
        // FIXME(20160106): currently, Tree requires a PriorPolicy for
        // initialization; instead, reset it inside the search worker.
        server.enqueue(tid, SearchWorkerCommand::Search{cfg: cfg, init_state: init_state.clone()});
      }
      server.sync();
    }
    for tid in 0 .. num_workers {
      server.enqueue(tid, SearchWorkerCommand::Quit);
    }
    server.join();

    // TODO(20151225): update stats.
    unimplemented!();
    /*let root_node = tree.root_node.read().unwrap();
    let root_trials = root_node.values.num_trials_float();
    let action = if let Some(argmax_j) = array_argmax(&root_trials) {
      //self.stats.argmax_rank = argmax_j as i32;
      //self.stats.argmax_trials = root_trials[argmax_j] as i32;
      let argmax_point = root_node.valid_moves[argmax_j];
      Action::Place{point: argmax_point}
    } else {
      //self.stats.argmax_rank = -1;
      Action::Pass
    };
    SearchResult{
      turn:   root_node.state.current_turn(),
      action: action,
      expected_score: 0.0, //tree.mean_raw_score,
    }*/
  }
}
