use array_util::{array_argmax};
use board::{Board, RuleSet, PlayerRank, Stone, Point, Action};
use hyper::{load_hyperparam};
use search::{SearchStats};
use search::parallel_policies::{
  SearchPolicyWorkerBuilder, SearchPolicyWorker,
  PriorPolicy, TreePolicy,
  RolloutPolicyBuilder, RolloutMode, RolloutLeafs, RolloutPolicy,
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
use std::cell::{RefCell, Ref, RefMut};
use std::cmp::{max, min};
use std::iter::{repeat};
use std::marker::{PhantomData};
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::rc::{Rc};
use std::sync::{Arc, Barrier, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::sync::atomic::{AtomicUsize, Ordering, fence};
use std::sync::mpsc::{Sender, Receiver, channel};
use time::{get_time};
use vec_map::{VecMap};

/*fn slow_atomic_increment(x: &AtomicUsize) -> usize {
  //x.fetch_add(1, Ordering::AcqRel);
  loop {
    let prev_x = x.load(Ordering::Acquire);
    let next_x = prev_x + 1;
    let curr_x = x.compare_and_swap(prev_x, next_x, Ordering::AcqRel);
    if prev_x == curr_x {
      return next_x;
    }
  }
}*/

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

  pub fn reset_terminal(&mut self, leaf_state: &TxnState<TxnStateNodeData>) {
    self.rollout = false;
    //self.sim_state.reset();
    self.sim_state.replace_clone_from(leaf_state, leaf_state.get_data().features.clone());
    self.sim_pairs.clear();
    self.raw_score = None;
    self.adj_score = None;
    self.rave_mask[0].clear();
    self.rave_mask[1].clear();
  }

  pub fn reset_rollout(&mut self, leaf_state: &TxnState<TxnStateNodeData>) {
    self.rollout = true;
    self.sim_state.replace_clone_from(leaf_state, leaf_state.get_data().features.clone());
    self.sim_pairs.clear();
    self.raw_score = None;
    self.adj_score = None;
    self.rave_mask[0].clear();
    self.rave_mask[1].clear();
  }

  pub fn score(&mut self, komi: f32, expected_score: f32) {
    /*if !self.rollout {
      // FIXME(20151114)
      self.raw_score = Some(0.0);
      self.adj_score = Some([0.0, 0.0]);
    } else {
    }*/
    // XXX(20151125): A version of dynamic komi. When the current player is
    // ahead, the expected score in their favor is deducted into the effective
    // komi:
    // - B ahead, expected score < 0.0, komi should be increased
    // - W ahead, expected score > 0.0, komi should be decreased
    // This implements the heuristic, "when ahead, stay ahead."
    // FIXME(20151125): one complication is how dynamic komi interacts with
    // prior values.
    self.raw_score = Some(self.sim_state.current_score_rollout(komi));
    let b_adj_score = self.sim_state.current_score_rollout(komi - 0.0f32.min(expected_score));
    let w_adj_score = self.sim_state.current_score_rollout(komi - 0.0f32.max(expected_score));
    self.adj_score = Some([b_adj_score, w_adj_score]);
  }
}

#[derive(Clone)]
pub struct Trace {
  //pub pairs:    Vec<(TxnState<TxnStateExtLibFeatsData>, Action)>,
  //pub pairs:    Vec<(TxnState<TxnStateLibFeaturesData>, Action)>,
  pub pairs:    Vec<(Stone, TxnStateLibFeaturesData, Action)>,
  pub value:    Option<f32>,
}

impl Trace {
  pub fn new() -> Trace {
    Trace{
      pairs:    Vec::with_capacity(544),
      value:    None,
    }
  }
}

#[derive(Clone)]
pub struct QuickTrace {
  pub init_state:   Option<TxnState<TxnStateLibFeaturesData>>,
  pub actions:      Vec<(Stone, Action)>,
  pub value:        Option<f32>,
}

impl QuickTrace {
  pub fn new() -> QuickTrace {
    QuickTrace{
      init_state:   None,
      actions:      Vec::with_capacity(544),
      value:        None,
    }
  }

  pub fn reset(&mut self) {
    self.init_state = None;
    self.actions.clear();
    self.value = None;
  }
}

pub trait NodeBox: Sized {
  type V: NodeValues;

  fn map<U>(&self, f: &mut FnMut(&Node<Self, Self::V>) -> U) -> U;
  fn map_mut<U>(&self, f: &mut FnMut(&mut Node<Self, Self::V>) -> U) -> U;
}

pub struct RcNodeBox {
  inner:      Rc<RefCell<Node<RcNodeBox, RefCell<FloatNodeValues>>>>,
}

impl NodeBox for RcNodeBox {
  type V = RefCell<FloatNodeValues>;

  fn map<U>(&self, f: &mut FnMut(&Node<Self, Self::V>) -> U) -> U {
    let node = self.inner.borrow();
    f(&*node)
  }

  fn map_mut<U>(&self, f: &mut FnMut(&mut Node<Self, Self::V>) -> U) -> U {
    let mut node = self.inner.borrow_mut();
    f(&mut *node)
  }
}

pub struct ArcNodeBox {
  inner:    Arc<RwLock<Node<ArcNodeBox, AtomicNodeValues>>>,
}

impl NodeBox for ArcNodeBox {
  type V = AtomicNodeValues;

  fn map<U>(&self, f: &mut FnMut(&Node<Self, Self::V>) -> U) -> U {
    let node = self.inner.read().unwrap();
    f(&*node)
  }

  fn map_mut<U>(&self, f: &mut FnMut(&mut Node<Self, Self::V>) -> U) -> U {
    let mut node = self.inner.write().unwrap();
    f(&mut *node)
  }
}

pub trait NodeValues {
  fn new(num_arms: usize) -> Self where Self: Sized;
  fn horizon(&self) -> usize;
  fn num_trials_float(&self) -> Vec<f32>;
}

pub struct FloatNodeValues {
  pub prior_values:     Vec<f32>,
  pub horizon:          usize,
  pub total_trials:     f32,
  pub num_trials:       Vec<f32>,
  pub num_succs:        Vec<f32>,
  pub num_raw_succs:    Vec<f32>,
  pub num_trials_rave:  Vec<f32>,
  pub num_succs_rave:   Vec<f32>,
}

impl NodeValues for RefCell<FloatNodeValues> {
  fn new(num_arms: usize) -> Self where Self: Sized {
    RefCell::new(FloatNodeValues{
      prior_values:     repeat(0.5).take(num_arms).collect(),
      horizon:          min(1, num_arms),
      total_trials:     0.0,
      num_trials:       repeat(0.0).take(num_arms).collect(),
      num_succs:        repeat(0.0).take(num_arms).collect(),
      num_raw_succs:    repeat(0.0).take(num_arms).collect(),
      num_trials_rave:  repeat(0.0).take(num_arms).collect(),
      num_succs_rave:   repeat(0.0).take(num_arms).collect(),
    })
  }

  fn horizon(&self) -> usize {
    let values = self.borrow();
    values.horizon
  }

  fn num_trials_float(&self) -> Vec<f32> {
    let values = self.borrow();
    values.num_trials.clone()
  }
}

pub struct AtomicNodeValues {
  pub prior_values:     Vec<f32>,
  pub horizon:          AtomicUsize,
  pub total_trials:     AtomicUsize,
  pub num_trials:       Vec<AtomicUsize>,
  pub num_succs:        Vec<AtomicUsize>,
  pub num_raw_succs:    Vec<AtomicUsize>,
  pub num_trials_rave:  Vec<AtomicUsize>,
  pub num_succs_rave:   Vec<AtomicUsize>,
}

impl NodeValues for AtomicNodeValues {
  fn new(num_arms: usize) -> AtomicNodeValues {
    let mut num_trials = Vec::with_capacity(num_arms);
    for _ in 0 .. num_arms {
      num_trials.push(AtomicUsize::new(0));
    }
    let mut num_succs = Vec::with_capacity(num_arms);
    for _ in 0 .. num_arms {
      num_succs.push(AtomicUsize::new(0));
    }
    let mut num_raw_succs = Vec::with_capacity(num_arms);
    for _ in 0 .. num_arms {
      num_raw_succs.push(AtomicUsize::new(0));
    }
    let mut num_trials_rave = Vec::with_capacity(num_arms);
    for _ in 0 .. num_arms {
      num_trials_rave.push(AtomicUsize::new(0));
    }
    let mut num_succs_rave = Vec::with_capacity(num_arms);
    for _ in 0 .. num_arms {
      num_succs_rave.push(AtomicUsize::new(0));
    }
    AtomicNodeValues{
      prior_values:     repeat(0.5).take(num_arms).collect(),
      // FIXME(20160125): switch on progressive widening.
      horizon:          AtomicUsize::new(min(1, num_arms)),
      total_trials:     AtomicUsize::new(0),
      num_trials:       num_trials,
      num_succs:        num_succs,
      num_raw_succs:    num_raw_succs,
      num_trials_rave:  num_trials_rave,
      num_succs_rave:   num_succs_rave,
    }
  }

  fn horizon(&self) -> usize {
    self.horizon.load(Ordering::Acquire)
  }

  fn num_trials_float(&self) -> Vec<f32> {
    let mut ns = vec![];
    for j in 0 .. self.num_trials.len() {
      ns.push(self.num_trials[j].load(Ordering::Acquire) as f32);
    }
    ns
  }
}

pub struct Node<N=ArcNodeBox, V=AtomicNodeValues> where N: NodeBox, V: NodeValues {
  pub state:        TxnState<TxnStateNodeData>,
  //pub horizon:      AtomicUsize,
  pub valid_moves:  Vec<Point>,
  pub action_idxs:  VecMap<usize>,
  pub child_nodes:  Vec<Option<Arc<RwLock<Node>>>>,
  //pub child_nodes:  Vec<Option<ArcNodeBox>>,
  pub values:       AtomicNodeValues,
  _marker:  PhantomData<(N, V)>,
}

impl<N, V> Node<N, V> where N: NodeBox, V: NodeValues {
  pub fn new(state: TxnState<TxnStateNodeData>, prior_policy: &mut PriorPolicy) -> Node<N, V> {
    let mut valid_moves = vec![];
    state.get_data().legality.fill_legal_points(state.current_turn(), &mut valid_moves);
    let num_arms = valid_moves.len();

    // FIXME(20160109): depends on progressive widening.
    //let init_horizon = min(1, num_arms);

    // XXX(20151224): Sort moves by descending value for progressive widening.
    let mut values = AtomicNodeValues::new(num_arms);
    let mut action_priors = Vec::with_capacity(num_arms);
    prior_policy.fill_prior_values(&state, &valid_moves, &mut action_priors);
    action_priors.sort_by(|left, right| {
      F32InfNan(right.1).cmp(&F32InfNan(left.1))
    });
    //println!("DEBUG: node top actions: {:?}", &action_priors[ .. min(10, action_priors.len())]);
    for j in 0 .. num_arms {
      valid_moves[j] = action_priors[j].0;
      values.prior_values[j] = action_priors[j].1;
    }

    let child_nodes: Vec<_> = repeat(None).take(num_arms).collect();
    let mut action_idxs = VecMap::with_capacity(Board::SIZE);
    for j in 0 .. num_arms {
      action_idxs.insert(valid_moves[j].idx(), j);
    }
    Node{
      state:        state,
      //horizon:      AtomicUsize::new(init_horizon),
      valid_moves:  valid_moves,
      child_nodes:  child_nodes,
      action_idxs:  action_idxs,
      values:       values,
      _marker:  PhantomData,
    }
  }

  pub fn is_terminal(&self) -> bool {
    self.valid_moves.is_empty()
  }

  pub fn update_visits(&self) {
    self.values.total_trials.fetch_add(1, Ordering::AcqRel);
    let num_arms = self.valid_moves.len();
    // FIXME(20160111): read progressive widening hyperparameter.
    let pwide_mu = 1.8f32;
    loop {
      // XXX(20160111): Need to read from `total_trials` again because other
      // threads may also be updating this node's horizon.
      let total_trials = self.values.total_trials.load(Ordering::Acquire);
      let next_horizon = min(num_arms, (((1 + total_trials) as f32).ln() / pwide_mu.ln()).ceil() as usize);
      let prev_horizon = self.values.horizon.load(Ordering::Acquire);
      let curr_horizon = self.values.horizon.compare_and_swap(prev_horizon, next_horizon, Ordering::AcqRel);
      if prev_horizon == curr_horizon {
        return;
      }
    }
  }

  pub fn update_arm(&self, j: usize, score: f32, raw_score: f32) {
    let turn = self.state.current_turn();
    self.values.num_trials[j].fetch_add(1, Ordering::AcqRel);
    if (Stone::White == turn && score >= 0.0) ||
        (Stone::Black == turn && score < 0.0)
    {
      self.values.num_succs[j].fetch_add(1, Ordering::AcqRel);
    }
    if (Stone::White == turn && raw_score >= 0.0) ||
        (Stone::Black == turn && raw_score < 0.0)
    {
      self.values.num_raw_succs[j].fetch_add(1, Ordering::AcqRel);
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

pub struct UniqueTree<N> where N: NodeBox {
  root_node:        Option<N>,
  mean_raw_score:   f32,
}

struct InnerTree {
  root_node:        Option<Arc<RwLock<Node<ArcNodeBox, AtomicNodeValues>>>>,
  mean_raw_score:   f32,
}

#[derive(Clone)]
pub struct SharedTree {
  inner:    Arc<Mutex<InnerTree>>,
}

impl SharedTree {
  pub fn new() -> SharedTree {
    SharedTree{
      inner:    Arc::new(Mutex::new(InnerTree{
        root_node:      None,
        mean_raw_score: 0.0,
      })),
    }
  }

  pub fn try_reset(&self, init_state: TxnState<TxnStateNodeData>, prior_policy: &mut PriorPolicy) {
    let mut inner = self.inner.lock().unwrap();
    if inner.root_node.is_none() {
      inner.root_node = Some(Arc::new(RwLock::new(Node::new(init_state, prior_policy))));
      inner.mean_raw_score = 0.0;
    }
  }
}

pub trait TreeOpsTrait {
  type N: NodeBox;

  fn traverse(
      root_node:    Self::N,
      tree_traj:    &mut TreeTraj,
      prior_policy: &mut PriorPolicy,
      tree_policy:  &mut TreePolicy<R=Xorshiftplus128Rng>,
      rng:          &mut Xorshiftplus128Rng)
      -> TreeResult;
  fn backup(
      tree_traj: &TreeTraj,
      rollout_traj: &mut RolloutTraj,
      rng: &mut Xorshiftplus128Rng);
}

pub struct TreeOps;

impl TreeOps {
  pub fn traverse(
      root_node:    Arc<RwLock<Node>>,
      tree_traj:    &mut TreeTraj,
      prior_policy: &mut PriorPolicy,
      tree_policy:  &mut TreePolicy<R=Xorshiftplus128Rng>,
      stats:        &mut SearchStats,
      rng:          &mut Xorshiftplus128Rng)
      -> TreeResult
  {
    tree_traj.reset();

    let mut ply = 0;
    let mut cursor_node: Arc<RwLock<Node>> = root_node;
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
              let leaf_node = {
                // XXX(20160111): Try to insert a new leaf node, but check for a
                // race if another thread has done so first.
                let mut cursor_node = cursor_node.write().unwrap();
                if cursor_node.child_nodes[j].is_none() {
                  // Create a new leaf node and stop the walk.
                  let mut leaf_state = cursor_node.state.clone();
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
                  cursor_node.child_nodes[j] = Some(leaf_node.clone());
                  //cursor_node = leaf_node;
                  stats.new_leaf_count += 1;
                  leaf_node
                } else {
                  cursor_node.child_nodes[j].as_ref().unwrap().clone()
                }
              };
              cursor_node = leaf_node;
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

  pub fn backup(
      tree_traj: &TreeTraj,
      rollout_traj: &mut RolloutTraj,
      rng: &mut Xorshiftplus128Rng)
  {
    rollout_traj.rave_mask[0].clear();
    rollout_traj.rave_mask[1].clear();
    let raw_score = match rollout_traj.raw_score {
      Some(raw_score) => raw_score,
      None => panic!("missing raw score for backup!"),
    };
    let adj_score = match rollout_traj.adj_score {
      Some(adj_score) => adj_score,
      None => panic!("missing adj score for backup!"),
    };

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
          leaf_node.update_arm(update_j, adj_score[update_turn.offset()], raw_score);
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
      node.update_arm(update_j, adj_score[update_turn.offset()], raw_score);

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
  }
}

#[derive(Clone)]
pub enum EvalWorkerCommand {
  Eval{cfg: EvalWorkerConfig, init_state: TxnState<TxnStateNodeData>},
  EvalGradientsAndBackup,
  Quit,
}

#[derive(Clone, Copy)]
pub struct EvalWorkerConfig {
  pub batch_size:   usize,
  pub num_batches:  usize,
  pub komi:                 f32,
  pub prev_expected_score:  f32,
}

pub struct ParallelMonteCarloEvalServer<W> where W: RolloutPolicy<R=Xorshiftplus128Rng> {
  num_workers:          usize,
  worker_batch_size:    usize,
  pool:         ThreadPool,
  in_txs:       Vec<Sender<EvalWorkerCommand>>,
  out_barrier:  Arc<Barrier>,
  _marker:      PhantomData<W>,
}

impl<W> Drop for ParallelMonteCarloEvalServer<W> where W: RolloutPolicy<R=Xorshiftplus128Rng> {
  fn drop(&mut self) {
    // TODO(20160114)
  }
}

impl<W> ParallelMonteCarloEvalServer<W> where W: RolloutPolicy<R=Xorshiftplus128Rng> {
  pub fn new<B>(num_workers: usize, worker_batch_size: usize, rollout_policy_builder: B) -> ParallelMonteCarloEvalServer<W>
  where B: 'static + RolloutPolicyBuilder<Policy=W> {
    let barrier = Arc::new(Barrier::new(num_workers));
    let pool = ThreadPool::new(num_workers);
    let mut in_txs = vec![];
    let out_barrier = Arc::new(Barrier::new(num_workers + 1));

    for tid in 0 .. num_workers {
      let rollout_policy_builder = rollout_policy_builder.clone();
      let barrier = barrier.clone();
      let out_barrier = out_barrier.clone();

      let (in_tx, in_rx) = channel();
      in_txs.push(in_tx);

      pool.execute(move || {
        let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
        let mut rollout_policy = rollout_policy_builder.into_rollout_policy(tid, worker_batch_size);
        let barrier = barrier;
        let in_rx = in_rx;
        let out_barrier = out_barrier;

        let mut leaf_states: Vec<_> = repeat(TxnState::new(
            [PlayerRank::Dan(9), PlayerRank::Dan(9)],
            RuleSet::KgsJapanese.rules(),
            TxnStateNodeData::new(),
        )).take(worker_batch_size).collect();
        let mut rollout_trajs: Vec<_> = repeat(RolloutTraj::new()).take(worker_batch_size).collect();
        let mut traces: Vec<_> = repeat(QuickTrace::new()).take(worker_batch_size).collect();

        loop {
          let cmd: EvalWorkerCommand = in_rx.recv().unwrap();
          let (cfg, init_state) = match cmd {
            EvalWorkerCommand::Eval{cfg, init_state} => {
              (cfg, init_state)
            }
            EvalWorkerCommand::EvalGradientsAndBackup => {
              // FIXME(20160117)
              unimplemented!();
            }
            EvalWorkerCommand::Quit => break,
          };

          let batch_size = cfg.batch_size;
          let num_batches = cfg.num_batches;
          assert!(batch_size <= worker_batch_size);

          let mut worker_mean_score: f32 = 0.0;
          let mut worker_backup_count: f32 = 0.0;

          for batch in 0 .. num_batches {
            {
              for batch_idx in 0 .. batch_size {
                let rollout_traj = &mut rollout_trajs[batch_idx];
                let leaf_state = &leaf_states[batch_idx];
                rollout_traj.reset_rollout(leaf_state);
              }
            }
            barrier.wait();
            fence(Ordering::AcqRel);

            rollout_policy.rollout_batch(
                batch_size,
                RolloutLeafs::LeafStates(&leaf_states),
                &mut rollout_trajs,
                true, &mut traces,
                &mut rng);
            barrier.wait();
            fence(Ordering::AcqRel);

            for batch_idx in 0 .. batch_size {
              let rollout_traj = &mut rollout_trajs[batch_idx];
              // FIXME(20160109): use correct values of komi and previous score.
              rollout_traj.score(cfg.komi, cfg.prev_expected_score);
              let raw_score = rollout_traj.raw_score.unwrap();
              worker_backup_count += 1.0;
              worker_mean_score += (raw_score - worker_mean_score) / worker_backup_count;
            }
            barrier.wait();
            fence(Ordering::AcqRel);
          }

          /*{
            let mut mean_raw_score = tree.mean_raw_score.lock().unwrap();
            *mean_raw_score += worker_mean_score / (num_workers as f32);
          }*/

          out_barrier.wait();
        }

        out_barrier.wait();
      });
    }

    ParallelMonteCarloEvalServer{
      num_workers:          num_workers,
      worker_batch_size:    worker_batch_size,
      pool:         pool,
      in_txs:       in_txs,
      out_barrier:  out_barrier,
      _marker:      PhantomData,
    }
  }

  pub fn num_workers(&self) -> usize {
    self.num_workers
  }

  pub fn worker_batch_size(&self) -> usize {
    self.worker_batch_size
  }

  pub fn enqueue(&self, tid: usize, cmd: EvalWorkerCommand) {
    self.in_txs[tid].send(cmd).unwrap();
  }

  pub fn join(&self) {
    self.out_barrier.wait();
  }
}

#[derive(Clone)]
pub enum SearchWorkerCommand {
  ResetSerialSearch{
    cfg:        SearchWorkerConfig,
    init_state: TxnState<TxnStateNodeData>,
  },
  ResetSearch{
    cfg:            SearchWorkerConfig,
    shared_tree:    SharedTree,
    init_state:     TxnState<TxnStateNodeData>,
  },
  ResetEval{
    cfg:            SearchWorkerConfig,
    shared_data:    EvalSharedData,
    record_trace:   bool,
    init_state:     TxnState<TxnStateNodeData>,
  },
  EvalGradientsAndBackup{
    learning_rate:  f32,
    baseline:       f32,
    target_value:   f32,
    eval_value:     f32,
  },
  SaveRolloutParams{
    save_dir:   PathBuf,
    num_iters:  usize,
  },
  Quit,
}

#[derive(Clone, Copy)]
pub struct SearchWorkerConfig {
  pub batch_size:   usize,
  pub num_batches:  usize,

  pub komi:                 f32,
  pub prev_expected_score:  f32,
}

pub struct ParallelMonteCarloSearchServer<W> where W: SearchPolicyWorker {
  num_workers:          usize,
  worker_batch_size:    usize,
  pool:         ThreadPool,
  in_txs:       Vec<Sender<SearchWorkerCommand>>,
  out_barrier:  Arc<Barrier>,
  _marker:      PhantomData<W>,
}

impl<W> Drop for ParallelMonteCarloSearchServer<W> where W: SearchPolicyWorker {
  fn drop(&mut self) {
    for tid in 0 .. self.num_workers {
      self.enqueue(tid, SearchWorkerCommand::Quit);
    }
    self.join();
  }
}

impl<W> ParallelMonteCarloSearchServer<W> where W: SearchPolicyWorker {
  pub fn new<B>(num_workers: usize, worker_batch_size: usize, worker_builder: B) -> ParallelMonteCarloSearchServer<W>
  where B: 'static + SearchPolicyWorkerBuilder<Worker=W> {
    let barrier = Arc::new(Barrier::new(num_workers));
    let pool = ThreadPool::new(num_workers);
    let mut in_txs = vec![];
    let out_barrier = Arc::new(Barrier::new(num_workers + 1));

    for tid in 0 .. num_workers {
      let worker_builder = worker_builder.clone();
      let barrier = barrier.clone();
      let out_barrier = out_barrier.clone();

      let (in_tx, in_rx) = channel();
      in_txs.push(in_tx);

      pool.execute(move || {
        let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
        let mut worker = worker_builder.into_worker(tid, worker_batch_size);
        let barrier = barrier;
        let in_rx = in_rx;
        let out_barrier = out_barrier;

        let mut tree_trajs: Vec<_> = repeat(TreeTraj::new()).take(worker_batch_size).collect();
        let mut leaf_states: Vec<_> = repeat(TxnState::new(
            [PlayerRank::Dan(9), PlayerRank::Dan(9)],
            RuleSet::KgsJapanese.rules(),
            TxnStateNodeData::new(),
        )).take(worker_batch_size).collect();
        let mut rollout_trajs: Vec<_> = repeat(RolloutTraj::new()).take(worker_batch_size).collect();
        let mut traces: Vec<_> = repeat(QuickTrace::new()).take(worker_batch_size * 40).collect();

        loop {
          // FIXME(20151222): for real time search, num batches is just an
          // estimate; should check for termination within inner batch loop.
          let cmd: SearchWorkerCommand = in_rx.recv().unwrap();
          match cmd {
            SearchWorkerCommand::ResetSerialSearch{cfg, init_state} => {
              // FIXME(20160124)
              unimplemented!();
            }

            SearchWorkerCommand::ResetSearch{cfg, shared_tree, init_state} => {
              // XXX(20160107): If the tree has no root node, this sets it;
              // otherwise use the existing root node.
              let tree = shared_tree;
              tree.try_reset(init_state, worker.prior_policy());

              let root_node = {
                let inner = tree.inner.lock().unwrap();
                inner.root_node.as_ref().unwrap().clone()
              };

              let batch_size = cfg.batch_size;
              let num_batches = cfg.num_batches;
              assert!(batch_size <= worker_batch_size);

              let mut worker_mean_score: f32 = 0.0;
              let mut worker_backup_count: f32 = 0.0;

              // FIXME(20151222): should share stats between workers.
              let mut stats: SearchStats = Default::default();

              /*for trace in traces.iter_mut() {
                trace.reset();
              }*/
              if traces.len() < num_batches * batch_size {
                for _ in traces.len() .. num_batches * batch_size {
                  traces.push(QuickTrace::new());
                }
              }

              for batch in 0 .. num_batches {
                /*println!("DEBUG: worker {} batch {}/{}",
                    tid, batch, num_batches);*/

                {
                  let (prior_policy, tree_policy) = worker.exploration_policies();
                  for batch_idx in 0 .. batch_size {
                    let tree_traj = &mut tree_trajs[batch_idx];
                    let rollout_traj = &mut rollout_trajs[batch_idx];
                    match TreeOps::traverse(root_node.clone(), tree_traj, prior_policy, tree_policy, &mut stats, &mut rng) {
                      TreeResult::Terminal => {
                        let leaf_state = &tree_traj.leaf_node.as_ref().unwrap().read().unwrap().state;
                        rollout_traj.reset_terminal(leaf_state);
                      }
                      TreeResult::NonTerminal => {
                        let leaf_state = &tree_traj.leaf_node.as_ref().unwrap().read().unwrap().state;
                        rollout_traj.reset_rollout(leaf_state);
                      }
                    }
                  }
                }
                barrier.wait();
                fence(Ordering::AcqRel);

                worker.rollout_policy().rollout_batch(
                    batch_size,
                    RolloutLeafs::TreeTrajs(&tree_trajs),
                    &mut rollout_trajs,
                    false, &mut traces[batch * batch_size .. (batch + 1) * batch_size],
                    &mut rng);
                barrier.wait();
                fence(Ordering::AcqRel);

                for batch_idx in 0 .. batch_size {
                  let tree_traj = &tree_trajs[batch_idx];
                  let rollout_traj = &mut rollout_trajs[batch_idx];
                  rollout_traj.score(cfg.komi, cfg.prev_expected_score);
                  TreeOps::backup(tree_traj, rollout_traj, &mut rng);
                  let raw_score = rollout_traj.raw_score.unwrap();
                  worker_backup_count += 1.0;
                  worker_mean_score += (raw_score - worker_mean_score) / worker_backup_count;
                }
                barrier.wait();
                fence(Ordering::AcqRel);
              }

              {
                let mut inner = tree.inner.lock().unwrap();
                inner.mean_raw_score += worker_mean_score / (num_workers as f32);
              }

              out_barrier.wait();
            }

            SearchWorkerCommand::ResetEval{cfg, shared_data, record_trace, init_state} => {
              let batch_size = cfg.batch_size;
              let num_batches = cfg.num_batches;
              assert!(batch_size <= worker_batch_size);

              let mut worker_mean_value: f32 = 0.0;
              let mut worker_backup_count: f32 = 0.0;

              let init_turn = init_state.current_turn();

              /*for trace in traces.iter_mut() {
                trace.reset();
              }*/
              if traces.len() < num_batches * batch_size {
                for _ in traces.len() .. num_batches * batch_size {
                  traces.push(QuickTrace::new());
                }
              }

              for batch in 0 .. num_batches {
                for batch_idx in 0 .. batch_size {
                  let rollout_traj = &mut rollout_trajs[batch_idx];
                  let leaf_state = &leaf_states[batch_idx];
                  rollout_traj.reset_rollout(leaf_state);
                }
                barrier.wait();
                fence(Ordering::AcqRel);

                worker.rollout_policy().rollout_batch(
                    batch_size,
                    RolloutLeafs::LeafStates(&leaf_states),
                    &mut rollout_trajs,
                    record_trace, &mut traces[batch * batch_size .. (batch + 1) * batch_size],
                    &mut rng);
                barrier.wait();
                fence(Ordering::AcqRel);

                for batch_idx in 0 .. batch_size {
                  let rollout_traj = &mut rollout_trajs[batch_idx];
                  rollout_traj.score(cfg.komi, cfg.prev_expected_score);
                  let raw_score = rollout_traj.raw_score.unwrap();
                  let value = match init_turn {
                    Stone::Black => {
                      if raw_score < 0.0 {
                        1.0
                      } else {
                        0.0
                      }
                    }
                    Stone::White => {
                      if raw_score >= 0.0 {
                        1.0
                      } else {
                        0.0
                      }
                    }
                    _ => unreachable!(),
                  };
                  if record_trace {
                    traces[batch * batch_size + batch_idx].value = Some(value);
                  }
                  worker_backup_count += 1.0;
                  worker_mean_value += (value - worker_mean_value) / worker_backup_count;
                }
                barrier.wait();
                fence(Ordering::AcqRel);
              }

              {
                let mut expected_value = shared_data.expected_value.lock().unwrap();
                *expected_value += worker_mean_value / (num_workers as f32);
              }

              out_barrier.wait();
            }

            SearchWorkerCommand::EvalGradientsAndBackup{learning_rate, baseline, target_value, eval_value} => {
              let mut num_traces = 0;
              worker.rollout_policy().init_traces();
              for trace in traces.iter() {
                if worker.rollout_policy().rollout_quicktrace(trace, baseline) {
                  num_traces += 1;
                }
              }
              if num_traces > 0 {
                worker.rollout_policy().backup_traces(learning_rate, target_value, eval_value, num_traces);
              }

              out_barrier.wait();
            }

            SearchWorkerCommand::SaveRolloutParams{save_dir, num_iters} => {
              worker.rollout_policy().save_params(&save_dir, num_iters);

              out_barrier.wait();
            }

            SearchWorkerCommand::Quit => {
              break;
            }
          };

        }

        out_barrier.wait();
      });
    }

    ParallelMonteCarloSearchServer{
      num_workers:          num_workers,
      worker_batch_size:    worker_batch_size,
      pool:         pool,
      in_txs:       in_txs,
      out_barrier:  out_barrier,
      _marker:      PhantomData,
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

  pub fn join(&self) {
    self.out_barrier.wait();
  }
}

#[derive(Clone, Copy, Debug)]
pub struct MonteCarloSearchResult {
  pub turn:             Stone,
  pub action:           Action,
  pub expected_score:   f32,
  pub expected_adj_val: f32,
  pub expected_value:   f32,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct MonteCarloSearchStats {
  pub argmax_rank:      Option<usize>,
  pub argmax_ntrials:   usize,
  pub elapsed_ms:       usize,
}

/*#[derive(Default)]
pub struct ParallelMonteCarloSearchStats {
  pub argmax_rank:      AtomicUsize,
  pub argmax_ntrials:   AtomicUsize,
}*/

pub struct SerialMonteCarloSearch;

impl SerialMonteCarloSearch {
  pub fn new() -> SerialMonteCarloSearch {
    SerialMonteCarloSearch
  }

  pub fn join<W>(&self,
      total_num_rollouts: usize,
      total_batch_size:   usize,
      server:       &ParallelMonteCarloSearchServer<W>,
      init_state:   &TxnState<TxnStateNodeData>,
      prev_result:  Option<&MonteCarloSearchResult>,
      rng:          &mut Xorshiftplus128Rng)
      -> (MonteCarloSearchResult, MonteCarloSearchStats)
      where W: SearchPolicyWorker
  {
    unimplemented!();
  }
}

pub struct ParallelMonteCarloSearch;

impl ParallelMonteCarloSearch {
  pub fn new() -> ParallelMonteCarloSearch {
    ParallelMonteCarloSearch
  }

  pub fn join<W>(&self,
      total_num_rollouts: usize,
      total_batch_size:   usize,
      server:       &ParallelMonteCarloSearchServer<W>,
      init_state:   &TxnState<TxnStateNodeData>,
      prev_result:  Option<&MonteCarloSearchResult>,
      rng:          &mut Xorshiftplus128Rng)
      -> (MonteCarloSearchResult, MonteCarloSearchStats)
      where W: SearchPolicyWorker
  {
    let num_workers = server.num_workers();
    let num_rollouts = (total_num_rollouts + num_workers - 1) / num_workers * num_workers;
    let worker_num_rollouts = num_rollouts / num_workers;
    //let worker_batch_size = server.worker_batch_size();
    let worker_batch_size = (total_batch_size + num_workers - 1) / num_workers;
    let worker_num_batches = (worker_num_rollouts + worker_batch_size - 1) / worker_batch_size;
    assert!(worker_batch_size >= 1);
    assert!(worker_num_batches >= 1);
    /*println!("DEBUG: server config: num workers:          {}", num_workers);
    println!("DEBUG: server config: worker num rollouts:  {}", worker_num_rollouts);
    println!("DEBUG: server config: worker batch size:    {}", worker_batch_size);
    println!("DEBUG: server config: worker num batches:   {}", worker_num_batches);*/
    /*println!("DEBUG: search config: {} {} {}",
        num_workers, worker_batch_size, worker_num_batches);*/

    // TODO(20160106): reset stats.

    let tree = SharedTree::new();
    let cfg = SearchWorkerConfig{
      batch_size:   worker_batch_size,
      num_batches:  worker_num_batches,
      komi:                 7.5, // FIXME(20160112): use correct komi value.
      prev_expected_score:  prev_result.map_or(0.0, |r| r.expected_score),
    };

    let start_time = get_time();
    for tid in 0 .. num_workers {
      server.enqueue(tid, SearchWorkerCommand::ResetSearch{
        cfg:            cfg,
        shared_tree:    tree.clone(),
        init_state:     init_state.clone(),
      });
    }
    server.join();
    let end_time = get_time();
    let elapsed_ms = (end_time - start_time).num_milliseconds();

    let mut stats: MonteCarloSearchStats = Default::default();
    stats.elapsed_ms = elapsed_ms as usize;

    /*let root_node_opt = tree.root_node.read().unwrap();
    let root_node = root_node_opt.as_ref().unwrap().read().unwrap();*/
    let (root_node, mean_raw_score) = {
      let inner_tree = tree.inner.lock().unwrap();
      (inner_tree.root_node.as_ref().unwrap().clone(), inner_tree.mean_raw_score)
    };
    let root_node = root_node.read().unwrap();
    let root_trials = root_node.values.num_trials_float();
    let (action, adj_value, raw_value) = if let Some(argmax_j) = array_argmax(&root_trials) {
      stats.argmax_rank = Some(argmax_j);
      stats.argmax_ntrials = root_trials[argmax_j] as usize;
      let argmax_point = root_node.valid_moves[argmax_j];
      let j_adj_succs = root_node.values.num_succs[argmax_j].load(Ordering::Acquire);
      let j_raw_succs = root_node.values.num_raw_succs[argmax_j].load(Ordering::Acquire);
      let j_trials = root_trials[argmax_j];
      let adj_value = j_adj_succs as f32 / j_trials;
      let raw_value = j_raw_succs as f32 / j_trials;
      (Action::Place{point: argmax_point}, adj_value, raw_value)
    } else {
      stats.argmax_rank = None;
      (Action::Pass, 0.5, 0.5)
    };
    (MonteCarloSearchResult{
      turn:   root_node.state.current_turn(),
      action: action,
      /*expected_score: {
        let mean_score = tree.mean_raw_score.lock().unwrap();
        *mean_score
      },*/
      expected_score:   mean_raw_score,
      expected_adj_val: adj_value,
      expected_value:   raw_value,
    }, stats)
  }
}

#[derive(Clone, Copy, Debug)]
pub struct MonteCarloEvalResult {
  pub expected_value:   f32,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct MonteCarloEvalStats {
  pub elapsed_ms:   usize,
}

#[derive(Clone)]
pub struct EvalSharedData {
  expected_value:   Arc<Mutex<f32>>,
}

impl EvalSharedData {
  pub fn new() -> EvalSharedData {
    EvalSharedData{
      expected_value:   Arc::new(Mutex::new(0.0)),
    }
  }
}

pub struct ParallelMonteCarloEval;

impl ParallelMonteCarloEval {
  pub fn new() -> ParallelMonteCarloEval {
    ParallelMonteCarloEval
  }

  pub fn join<W>(&self,
      total_num_rollouts: usize,
      total_batch_size:   usize,
      server:       &ParallelMonteCarloSearchServer<W>,
      init_state:   &TxnState<TxnStateNodeData>,
      //eval_mode:    RolloutMode,
      rng:          &mut Xorshiftplus128Rng)
      -> (MonteCarloEvalResult, MonteCarloEvalStats)
      where W: SearchPolicyWorker
  {
    let num_workers = server.num_workers();
    let num_rollouts = (total_num_rollouts + num_workers - 1) / num_workers * num_workers;
    let worker_num_rollouts = num_rollouts / num_workers;
    //let worker_batch_size = server.worker_batch_size();
    let worker_batch_size = (total_batch_size + num_workers - 1) / num_workers;
    let worker_num_batches = (worker_num_rollouts + worker_batch_size - 1) / worker_batch_size;
    assert!(worker_batch_size >= 1);
    assert!(worker_num_batches >= 1);
    /*println!("DEBUG: server config: num workers:          {}", num_workers);
    println!("DEBUG: server config: worker num rollouts:  {}", worker_num_rollouts);
    println!("DEBUG: server config: worker batch size:    {}", worker_batch_size);
    println!("DEBUG: server config: worker num batches:   {}", worker_num_batches);*/
    /*println!("DEBUG: eval config: {} {} {}",
        num_workers, worker_batch_size, worker_num_batches);*/

    let cfg = SearchWorkerConfig{
      batch_size:   worker_batch_size,
      num_batches:  worker_num_batches,
      komi:                 7.5, // FIXME(20160112): use correct komi value.
      prev_expected_score:  0.0, // FIXME(20160116): use previous score?
    };
    let shared_data = EvalSharedData::new();

    let start_time = get_time();
    for tid in 0 .. num_workers {
      server.enqueue(tid, SearchWorkerCommand::ResetEval{
        cfg:            cfg,
        shared_data:    shared_data.clone(),
        record_trace:   false,
        init_state:     init_state.clone(),
      });
    }
    server.join();
    let end_time = get_time();
    let elapsed_ms = (end_time - start_time).num_milliseconds();

    (MonteCarloEvalResult{
      expected_value:   {
        let expected_value = shared_data.expected_value.lock().unwrap();
        *expected_value
      },
    }, MonteCarloEvalStats{
      elapsed_ms:   elapsed_ms as usize,
    })
  }
}

pub struct ParallelMonteCarloBackup;

impl ParallelMonteCarloBackup {
  pub fn new() -> ParallelMonteCarloBackup {
    ParallelMonteCarloBackup
  }

  pub fn join<W>(&self,
      total_num_rollouts: usize,
      total_batch_size:   usize,
      learning_rate:      f32,
      baseline:           f32,
      target_value:       f32,
      eval_value:         f32,
      server:       &ParallelMonteCarloSearchServer<W>,
      init_state:   &TxnState<TxnStateNodeData>,
      //eval_mode:    RolloutMode,
      rng:          &mut Xorshiftplus128Rng)
      -> MonteCarloEvalStats
      where W: SearchPolicyWorker
  {
    let num_workers = server.num_workers();
    let num_rollouts = (total_num_rollouts + num_workers - 1) / num_workers * num_workers;
    let worker_num_rollouts = num_rollouts / num_workers;
    let worker_batch_size = (total_batch_size + num_workers - 1) / num_workers;
    let worker_num_batches = (worker_num_rollouts + worker_batch_size - 1) / worker_batch_size;
    assert!(worker_batch_size >= 1);
    assert!(worker_num_batches >= 1);
    /*println!("DEBUG: server config: num workers:          {}", num_workers);
    println!("DEBUG: server config: worker num rollouts:  {}", worker_num_rollouts);
    println!("DEBUG: server config: worker batch size:    {}", worker_batch_size);
    println!("DEBUG: server config: worker num batches:   {}", worker_num_batches);*/
    /*println!("DEBUG: backup config: {} {} {}",
        num_workers, worker_batch_size, worker_num_batches);*/

    let cfg = SearchWorkerConfig{
      batch_size:   worker_batch_size,
      num_batches:  worker_num_batches,
      komi:                 7.5, // FIXME(20160112): use correct komi value.
      prev_expected_score:  0.0, // FIXME(20160116): use previous score?
    };
    let shared_data = EvalSharedData::new();

    let start_time = get_time();
    // XXX(20160119): Need to run statistically independent evaluations before
    // backing up their traces.
    for tid in 0 .. num_workers {
      server.enqueue(tid, SearchWorkerCommand::ResetEval{
        cfg:            cfg,
        shared_data:    shared_data.clone(),
        record_trace:   true,
        init_state:     init_state.clone(),
      });
    }
    server.join();
    for tid in 0 .. num_workers {
      server.enqueue(tid, SearchWorkerCommand::EvalGradientsAndBackup{
        learning_rate:  learning_rate,
        baseline:       baseline,
        target_value:   target_value,
        eval_value:     eval_value,
      });
    }
    server.join();
    let end_time = get_time();
    let elapsed_ms = (end_time - start_time).num_milliseconds();

    MonteCarloEvalStats{
      elapsed_ms:   elapsed_ms as usize,
    }
  }
}

pub struct ParallelMonteCarloSave;

impl ParallelMonteCarloSave {
  pub fn new() -> ParallelMonteCarloSave {
    ParallelMonteCarloSave
  }

  pub fn join<W>(&self,
      save_dir:     &Path,
      num_iters:    usize,
      server:       &ParallelMonteCarloSearchServer<W>)
      where W: SearchPolicyWorker
  {
    let num_workers = server.num_workers();
    for tid in 0 .. num_workers {
      server.enqueue(tid, SearchWorkerCommand::SaveRolloutParams{
        save_dir:   PathBuf::from(save_dir),
        num_iters:  num_iters,
      });
    }
    server.join();
  }
}
