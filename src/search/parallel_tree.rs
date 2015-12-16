use board::{Board, RuleSet, PlayerRank, Stone, Point, Action};
use random::{XorShift128PlusRng};
use search::{SearchResult, SearchStats};
use search::parallel_policies::{
  SearchPolicyWorkerBuilder, SearchPolicyWorker,
  PriorPolicy, TreePolicy, RolloutPolicy,
};
use txnstate::{TxnState, check_good_move_fast};
use txnstate::extras::{TxnStateNodeData};
use txnstate::features::{TxnStateLibFeaturesData};

use threadpool::{ThreadPool};

use bit_set::{BitSet};
use rand::{Rng, SeedableRng, thread_rng};
use std::iter::{repeat};
use std::marker::{PhantomData};
use std::sync::{Arc, Barrier, RwLock};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Sender, Receiver, channel};
use vec_map::{VecMap};

#[derive(Clone)]
pub struct Walk {
  pub backup_triples: Vec<(Arc<RwLock<Node>>, Point, usize)>,
  pub leaf_node:      Option<Arc<RwLock<Node>>>,
}

impl Walk {
  pub fn new() -> Walk {
    Walk{
      backup_triples: vec![],
      leaf_node:      None,
    }
  }
}

#[derive(Clone)]
pub struct Trajectory {
  pub sim_state:    TxnState<TxnStateLibFeaturesData>,
  pub sim_pairs:    Vec<(Stone, Point)>,

  pub raw_score:    Option<f32>,
  pub adj_score:    Option<[f32; 2]>,

  rave_mask:        Vec<BitSet>,
}

impl Trajectory {
  pub fn new() -> Trajectory {
    Trajectory{
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
}

pub struct NodeValues {
  pub total_trials:     AtomicUsize,
  pub prior_values:     Vec<AtomicUsize>,
  pub num_trials:       Vec<AtomicUsize>,
  pub num_succs:        Vec<AtomicUsize>,
  pub num_trials_rave:  Vec<AtomicUsize>,
  pub num_succs_rave:   Vec<AtomicUsize>,
}

pub struct Node {
  pub state:    TxnState<TxnStateNodeData>,

  pub horizon:      usize,
  pub child_nodes:  Vec<Option<Arc<RwLock<Node>>>>,
  pub valid_moves:  Vec<Point>,
  pub arm_indexes:  VecMap<usize>,
  pub values:       NodeValues,
}

impl Node {
  pub fn new(state: TxnState<TxnStateNodeData>) -> Node {
    // TODO(20151214)
    unimplemented!();
    /*Node{
      state:    state,
    }*/
  }

  pub fn is_terminal(&self) -> bool {
    self.valid_moves.is_empty()
  }

  pub fn update_visits(&self) {
    self.values.total_trials.fetch_add(1, Ordering::SeqCst);
  }

  pub fn update_arm(&self, j: usize, score: f32) {
    let turn = self.state.current_turn();
    self.values.num_trials[j].fetch_add(1, Ordering::SeqCst);
    if (Stone::White == turn && score >= 0.0) ||
        (Stone::Black == turn && score < 0.0)
    {
      self.values.num_succs[j].fetch_add(1, Ordering::SeqCst);
    }
  }

  pub fn rave_update_arm(&self, j: usize, score: f32) {
    let turn = self.state.current_turn();
    self.values.num_trials_rave[j].fetch_add(1, Ordering::SeqCst);
    if (Stone::White == turn && score >= 0.0) ||
        (Stone::Black == turn && score < 0.0)
    {
      self.values.num_succs_rave[j].fetch_add(1, Ordering::SeqCst);
    }
  }
}

pub struct Tree {
  //pwide_cfg:    ProgWideConfig,
  count:        Arc<AtomicUsize>,
  tree_idx:     usize,
  root_node:    Arc<RwLock<Node>>,
}

impl Drop for Tree {
  fn drop(&mut self) {
    self.count.fetch_sub(1, Ordering::SeqCst);
  }
}

impl Clone for Tree {
  fn clone(&self) -> Tree {
    let tree_idx = self.count.fetch_add(1, Ordering::SeqCst);
    Tree{
      count:        self.count.clone(),
      tree_idx:     tree_idx,
      root_node:    self.root_node.clone(),
    }
  }
}

impl Tree {
  pub fn new(init_state: TxnState<TxnStateNodeData>) -> Tree {
    Tree{
      count:        Arc::new(AtomicUsize::new(1)),
      tree_idx:     0,
      root_node:    Arc::new(RwLock::new(Node::new(init_state))),
    }
  }

  pub fn walk(&self, walk: &mut Walk) {
  }

  pub fn backup(&self, walk: &Walk, traj: &Trajectory) {
  }
}

#[derive(Clone, Copy)]
pub struct SearchThreadConfig {
  pub num_batches:  usize,
  pub batch_size:   usize,
}

pub struct ParallelSearchServer<W> where W: SearchPolicyWorker {
  num_threads:  usize,
  pool:     ThreadPool,
  in_txs:   Vec<Sender<(SearchThreadConfig, Tree)>>,
  out_rx:   Receiver<()>,
  _marker:  PhantomData<W>,
}

impl<W> ParallelSearchServer<W> where W: SearchPolicyWorker {
  pub fn new<B>(num_threads: usize, worker_builder: B) -> ParallelSearchServer<W>
  where B: 'static + SearchPolicyWorkerBuilder<Worker=W> {
    let pool = ThreadPool::new(num_threads);
    let barrier = Arc::new(Barrier::new(num_threads));
    let mut in_txs = vec![];
    let (out_tx, out_rx) = channel();

    for _ in (0 .. num_threads) {
      let builder = worker_builder.clone();
      let barrier = barrier.clone();
      let (in_tx, in_rx) = channel();
      let out_tx = out_tx.clone();
      in_txs.push(in_tx);

      pool.execute(move || {
        let worker = builder.build_worker();
        let barrier = barrier;
        let in_rx = in_rx;
        let out_tx = out_tx;

        loop {
          let (cfg, tree): (SearchThreadConfig, Tree) = in_rx.recv().unwrap();
          let num_batches = cfg.num_batches;
          let batch_size = cfg.batch_size;

          let mut walks: Vec<_> = repeat(Walk::new()).take(batch_size).collect();
          let mut trajs: Vec<_> = repeat(Trajectory::new()).take(batch_size).collect();

          for batch in (0 .. num_batches) {
            for batch_idx in (0 .. batch_size) {
              let walk = &mut walks[batch_idx];
              tree.walk(walk);
            }

            // TODO(20151214)
            //worker.rollout_policy().rollout_batch(&walks, &mut trajs, ...);

            for batch_idx in (0 .. batch_size) {
              let walk = &walks[batch_idx];
              let traj = &mut trajs[batch_idx];
              tree.backup(walk, traj);
            }

            barrier.wait();
          }

          out_tx.send(()).unwrap();
        }
      });
    }

    ParallelSearchServer{
      num_threads:  num_threads,
      pool:     pool,
      in_txs:   in_txs,
      out_rx:   out_rx,
      _marker:  PhantomData,
    }
  }

  pub fn num_threads(&self) -> usize {
    self.num_threads
  }

  pub fn enqueue(&self, tid: usize, cfg: SearchThreadConfig, tree: Tree) {
    self.in_txs[tid].send((cfg, tree)).unwrap();
  }

  pub fn join(&self) {
    for _ in self.out_rx.iter().take(self.num_threads) {
    }
  }
}

pub struct ParallelSearch {
  pub num_rollouts: usize,
  pub stats:        SearchStats,
}

impl ParallelSearch {
  pub fn new(num_rollouts: usize, num_threads: usize) -> ParallelSearch {
    ParallelSearch{
      num_rollouts: num_rollouts,
      stats:        Default::default(),
    }
  }

  pub fn join<W>(&mut self,
      server:   &ParallelSearchServer<W>,
      tree:     Tree,
      rng:      &mut XorShift128PlusRng)
      -> SearchResult
      where W: SearchPolicyWorker
  {
    //assert_eq!(tree.tree_idx, 0);
    let num_threads = server.num_threads();
    let num_rollouts = (self.num_rollouts + num_threads - 1) / num_threads * num_threads;
    let num_thread_rollouts = num_rollouts / num_threads;
    let total_batch_size = 256;
    let thread_batch_size = (total_batch_size + num_threads - 1) / num_threads;
    let num_batches = (num_thread_rollouts + thread_batch_size - 1) / thread_batch_size;
    assert!(thread_batch_size >= 1);
    assert!(num_batches >= 1);

    let cfg = SearchThreadConfig{
      num_batches:  num_batches,
      batch_size:   thread_batch_size,
    };
    for tid in (0 .. num_threads) {
      server.enqueue(tid, cfg, tree.clone());
    }
    server.join();

    // TODO(20151214)
    unimplemented!();
  }
}
