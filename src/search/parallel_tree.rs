use array_util::{array_argmax};
use board::{Board, RuleSet, PlayerRank, Stone, Point, Action};
use hyper::{load_hyperparam};
use search::{SearchStats, translate_score_to_reward};
use search::parallel_policies::{
  SearchPolicyWorkerBuilder, SearchPolicyWorker,
  PriorPolicy, TreePolicy,
  RolloutPolicyBuilder, RolloutMode, RolloutLeafs, RolloutPolicy,
  GradSyncMode,
};
use search::parallel_trace::{
  SearchTrace, SearchTrajTrace,
  TreeTrajTrace, TreeDecisionTrace, TreeExpansionTrace,
  RolloutTrajTrace,
  ParallelSearchReconWorkerBuilder,
};
use search::parallel_trace::omega::{
  OmegaTreeBatchWorker,
  OmegaWorkerMemory,
  MetaLevelObjective,
  //MetaLevelWorker,
};
use txnstate::{
  TxnStateConfig, TxnState,
  //BensonScratch,
  check_good_move_fast, is_eyelike,
};
use txnstate::extras::{
  TxnStateNodeData,
  TxnStateRolloutData,
};
use txnstate::features::{
  TxnStateLibFeaturesData,
  TxnStateExtLibFeatsData,
};

use bincode::rustc_serialize as bincode;
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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering, fence};
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

#[derive(Clone, Copy, Debug)]
pub struct TreePolicyConfig {
  pub horizon_cfg:  HorizonConfig,
  pub visit_thresh: usize,
  pub mc_scale:     f32,
  pub prior_equiv:  f32,
  pub rave:         bool,
  pub rave_equiv:   f32,
}

#[derive(Clone, Copy, Debug)]
pub enum HorizonConfig {
  All,
  Fixed{max_horizon: usize},
  Pwide{mu: f32},
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
  pub init_state:   TxnState<TxnStateNodeData>,
  //pub sim_state:    TxnState<TxnStateLibFeaturesData>,
  pub sim_state:    TxnState<TxnStateRolloutData>,
  pub sim_pairs:    Vec<(Stone, Point)>,

  pub score:    Option<f32>,
  //pub raw_score:    Option<f32>,
  //pub adj_score:    Option<[f32; 2]>,

  // FIXME(20151222): should probably move this to a `CommonTraj` data structure
  // but not critical.
  rave_mask:        Vec<BitSet>,

  // XXX(20160217): For counting live stones at terminal states.
  //term_count:       Vec<Vec<usize>>,

  // XXX(20160219): For counting Monte Carlo live stones.
  mc_live_mask:     Vec<BitSet>,

  // XXX(20160226): For Tromp-Taylor scoring.
  tt_scratch:       Vec<u8>,
}

impl RolloutTraj {
  pub fn new() -> RolloutTraj {
    RolloutTraj{
      rollout:      false,
      init_state:   TxnState::new(
          TxnStateConfig::default(),
          TxnStateNodeData::new(),
      ),
      sim_state:    TxnState::new(
          /*TxnStateConfig{
            rules:  RuleSet::KgsJapanese.rules(),
            ranks:  [PlayerRank::Dan(9), PlayerRank::Dan(9)],
          },*/
          TxnStateConfig::default(),
          //TxnStateLibFeaturesData::new(),
          TxnStateRolloutData::new(),
      ),
      sim_pairs:    vec![],
      score:        None,
      /*raw_score:    None,
      adj_score:    None,*/
      rave_mask:    vec![
        BitSet::with_capacity(Board::SIZE),
        BitSet::with_capacity(Board::SIZE),
      ],
      mc_live_mask: vec![
        BitSet::with_capacity(Board::SIZE),
        BitSet::with_capacity(Board::SIZE),
      ],
      tt_scratch:   repeat(0).take(Board::SIZE).collect(),
    }
  }

  /*pub fn reset_terminal(&mut self, leaf_state: &TxnState<TxnStateNodeData>) {
    self.rollout = false;
    self.init_state = leaf_state.clone();
    //self.sim_state.reset();
    //self.sim_state.replace_clone_from(leaf_state, TxnStateRolloutData::with_features(leaf_state.get_data().features.clone()));
    self.sim_state.replace_clone_from(leaf_state, TxnStateRolloutData::from_node_data(leaf_state.get_data()));
    self.sim_pairs.clear();
    self.raw_score = None;
    self.adj_score = None;
    self.rave_mask[0].clear();
    self.rave_mask[1].clear();
    /*for p in 0 .. Board::SIZE {
      self.term_count[0][p] = 0;
      self.term_count[1][p] = 0;
    }*/
    self.mc_live_mask[0].clear();
    self.mc_live_mask[1].clear();
  }*/

  pub fn reset_rollout(&mut self, leaf_state: &TxnState<TxnStateNodeData>) {
    self.rollout = true;
    self.init_state = leaf_state.clone();
    //self.sim_state.replace_clone_from(leaf_state, TxnStateRolloutData::with_features(leaf_state.get_data().features.clone()));
    self.sim_state.replace_clone_from(leaf_state, TxnStateRolloutData::from_node_data(leaf_state.get_data()));
    self.sim_pairs.clear();
    self.score = None;
    /*self.raw_score = None;
    self.adj_score = None;*/
    self.rave_mask[0].clear();
    self.rave_mask[1].clear();
    self.mc_live_mask[0].clear();
    self.mc_live_mask[1].clear();
  }

  //pub fn score(&mut self, komi: f32, expected_score: f32) {
  pub fn update_score(&mut self, komi: f32) {
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
    /*self.raw_score = Some(self.sim_state.current_score_rollout(komi));
    let b_adj_score = self.sim_state.current_score_rollout(komi - 0.0f32.min(expected_score));
    let w_adj_score = self.sim_state.current_score_rollout(komi - 0.0f32.max(expected_score));
    self.adj_score = Some([b_adj_score, w_adj_score]);*/

    //let score = self.sim_state.current_score_rollout(komi);
    let score = self.sim_state.current_score_tromp_taylor_undead(komi, &mut self.tt_scratch);
    self.score = Some(score);
    /*self.raw_score = Some(score);
    self.adj_score = Some([score, score]);*/
  }

  pub fn update_mc_live_counts(&mut self, mc_live_counts: &mut [Vec<usize>]) {
    for p in 0 .. Board::SIZE {
      let pt = Point::from_idx(p);
      let stone = self.sim_state.current_stone(pt);
      match stone {
        Stone::Black => {
          mc_live_counts[0][p] += 1;
        }
        Stone::White => {
          mc_live_counts[1][p] += 1;
        }
        Stone::Empty => {
          if is_eyelike(&self.sim_state.position, &self.sim_state.chains, Stone::Black, pt) {
            mc_live_counts[0][p] += 1;
          } else if is_eyelike(&self.sim_state.position, &self.sim_state.chains, Stone::White, pt) {
            mc_live_counts[1][p] += 1;
          }
        }
      }
    }
  }

  pub fn update_mc_live_mask(&mut self) {
    for p in 0 .. Board::SIZE {
      let pt = Point::from_idx(p);
      let stone = self.sim_state.current_stone(pt);
      match stone {
        Stone::Black => {
          self.mc_live_mask[0].insert(p);
        }
        Stone::White => {
          self.mc_live_mask[1].insert(p);
        }
        Stone::Empty => {
          if is_eyelike(&self.sim_state.position, &self.sim_state.chains, Stone::Black, pt) {
            self.mc_live_mask[0].insert(p);
          } else if is_eyelike(&self.sim_state.position, &self.sim_state.chains, Stone::White, pt) {
            self.mc_live_mask[1].insert(p);
          }
        }
      }
    }
  }

  pub fn backup_mc_live_counts(&self, turn: Stone, live_counts: &[AtomicUsize]) {
    let turn_off = turn.offset();
    for p in 0 .. Board::SIZE {
      if self.mc_live_mask[turn_off].contains(p) {
        live_counts[p].fetch_add(1, Ordering::AcqRel);
      }
    }
  }
}

/*#[derive(Clone)]
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
}*/

#[derive(Clone)]
pub struct QuickTrace {
  //pub init_state:   Option<TxnState<TxnStateRolloutData>>,
  pub init_state:   Option<TxnState<TxnStateNodeData>>,
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

/*pub trait NodeBox: Sized {
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
}*/

pub trait NodeValues {
  fn new(num_arms: usize, horizon_cfg: HorizonConfig) -> Self where Self: Sized;
  fn horizon(&self) -> usize;
  fn succ_ratios_float(&self) -> Vec<f32>;
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
  fn new(num_arms: usize, horizon_cfg: HorizonConfig) -> Self where Self: Sized {
    RefCell::new(FloatNodeValues{
      prior_values:     repeat(0.5).take(num_arms).collect(),
      // FIXME(20160208): horizon.
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

  fn succ_ratios_float(&self) -> Vec<f32> {
    // FIXME(20160209)
    unimplemented!();
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
  pub total_score:      AtomicUsize,
  pub num_trials:       Vec<AtomicUsize>,
  pub num_succs:        Vec<AtomicUsize>,
  //pub num_raw_succs:    Vec<AtomicUsize>,
  /*pub num_trials_rave:  Vec<AtomicUsize>,
  pub num_succs_rave:   Vec<AtomicUsize>,*/

  pub scores:           Vec<AtomicUsize>,

  /*pub total_live_stats: Vec<Vec<AtomicUsize>>,
  //pub live_counts:      Vec<AtomicUsize>,
  //pub live_counts_var:  Vec<AtomicUsize>,
  pub total_live_succs: AtomicUsize,
  pub live_succs:       Vec<AtomicUsize>,*/
}

impl AtomicNodeValues {
  pub fn mean_score(&self, min_trials_count: f32) -> f32 {
    let total_trials = self.total_trials.load(Ordering::Acquire) as f32;
    if total_trials >= min_trials_count {
      let total_score = 0.5 * self.total_score.load(Ordering::Acquire) as isize as f32;
      let mean_score = total_score / total_trials;
      mean_score
    } else {
      0.0
    }
  }

  pub fn arm_score(&self, j: usize) -> f32 {
    let trials = self.num_trials[j].load(Ordering::Acquire) as f32;
    let score = 0.5 * self.scores[j].load(Ordering::Acquire) as isize as f32;
    let j_score = score / trials;
    j_score
  }
}

impl NodeValues for AtomicNodeValues {
  fn new(num_arms: usize, horizon_cfg: HorizonConfig) -> AtomicNodeValues {
    let mut num_trials = Vec::with_capacity(num_arms);
    for _ in 0 .. num_arms {
      num_trials.push(AtomicUsize::new(0));
    }
    let mut num_succs = Vec::with_capacity(num_arms);
    for _ in 0 .. num_arms {
      num_succs.push(AtomicUsize::new(0));
    }
    /*let mut num_raw_succs = Vec::with_capacity(num_arms);
    for _ in 0 .. num_arms {
      num_raw_succs.push(AtomicUsize::new(0));
    }*/
    let mut num_trials_rave = Vec::with_capacity(num_arms);
    for _ in 0 .. num_arms {
      num_trials_rave.push(AtomicUsize::new(0));
    }
    let mut num_succs_rave = Vec::with_capacity(num_arms);
    for _ in 0 .. num_arms {
      num_succs_rave.push(AtomicUsize::new(0));
    }
    let mut scores = Vec::with_capacity(num_arms);
    for _ in 0 .. num_arms {
      scores.push(AtomicUsize::new(0));
    }
    /*let mut total_live_stats = vec![];
    for _ in 0 .. 2 {
      let mut sub_total_live_stats = Vec::with_capacity(num_arms);
      for _ in 0 .. num_arms {
        sub_total_live_stats.push(AtomicUsize::new(0));
      }
      total_live_stats.push(sub_total_live_stats);
    }
    /*let mut live_counts = Vec::with_capacity(num_arms);
    for _ in 0 .. num_arms {
      live_counts.push(AtomicUsize::new(0));
    }*/
    /*let mut live_counts_var = Vec::with_capacity(num_arms);
    for _ in 0 .. num_arms {
      live_counts_var.push(AtomicUsize::new(0));
    }*/
    let mut live_succs = Vec::with_capacity(num_arms);
    for _ in 0 .. num_arms {
      live_succs.push(AtomicUsize::new(0));
    }*/
    AtomicNodeValues{
      prior_values:     repeat(0.5).take(num_arms).collect(),
      horizon:          match horizon_cfg {
        HorizonConfig::All => {
          AtomicUsize::new(num_arms)
        }
        HorizonConfig::Fixed{max_horizon} => {
          AtomicUsize::new(min(max_horizon, num_arms))
        }
        HorizonConfig::Pwide{..} => {
          AtomicUsize::new(min(1, num_arms))
        }
      },
      total_trials:     AtomicUsize::new(0),
      total_score:      AtomicUsize::new(0),
      num_trials:       num_trials,
      num_succs:        num_succs,
      //num_raw_succs:    num_raw_succs,
      /*num_trials_rave:  num_trials_rave,
      num_succs_rave:   num_succs_rave,*/
      scores:           scores,
      /*total_live_stats: total_live_stats,
      //live_counts:      live_counts,
      //live_counts_var:  live_counts_var,
      total_live_succs: AtomicUsize::new(0),
      live_succs:       live_succs,*/
    }
  }

  fn horizon(&self) -> usize {
    self.horizon.load(Ordering::Acquire)
  }

  fn succ_ratios_float(&self) -> Vec<f32> {
    let mut xs = vec![];
    //for j in 0 .. self.num_trials.len() {
    for j in 0 .. self.horizon() {
      // XXX(20160210): Using the "raw" count of successes.
      //let s_j = self.num_raw_succs[j].load(Ordering::Acquire) as f32;
      let s_j = self.num_succs[j].load(Ordering::Acquire) as f32;
      let n_j = self.num_trials[j].load(Ordering::Acquire) as f32;
      if n_j < 32.0 {
        xs.push(0.0);
      } else {
        xs.push(s_j / n_j);
      }
    }
    xs
  }

  fn num_trials_float(&self) -> Vec<f32> {
    let mut ns = vec![];
    //for j in 0 .. self.num_trials.len() {
    for j in 0 .. self.horizon() {
      ns.push(self.num_trials[j].load(Ordering::Acquire) as f32);
    }
    ns
  }
}

//pub struct Node<N=ArcNodeBox, V=AtomicNodeValues> where N: NodeBox, V: NodeValues {
pub struct Node {
  pub state:        TxnState<TxnStateNodeData>,
  pub valid_moves:  Vec<Point>,
  pub action_idxs:  VecMap<usize>,
  pub child_nodes:  Vec<Option<Arc<RwLock<Node>>>>,
  //pub child_nodes:  Vec<Option<ArcNodeBox>>,
  pub values:       AtomicNodeValues,
  //_marker:  PhantomData<(N, V)>,
}

//impl<N, V> Node<N, V> where N: NodeBox, V: NodeValues {
impl Node {
  pub fn new(state: TxnState<TxnStateNodeData>, prior_policy: &mut PriorPolicy, horizon_cfg: HorizonConfig) -> Node/*<N, V>*/ {
    let mut valid_moves = vec![];
    state.get_data().legality.fill_legal_points(state.current_turn(), &mut valid_moves);
    let num_arms = valid_moves.len();

    // FIXME(20160109): depends on progressive widening.
    //let init_horizon = min(1, num_arms);
    //let horizon_cfg = HorizonConfig::Fixed{max_horizon: 10};
    //let horizon_cfg = HorizonConfig::Pwide{mu: 1.8};

    // XXX(20151224): Sort moves by descending value for progressive widening.
    let mut values = AtomicNodeValues::new(num_arms, horizon_cfg);
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
      //_marker:  PhantomData,
    }
  }

  pub fn is_terminal(&self) -> bool {
    self.valid_moves.is_empty()
  }

  pub fn get_argmax_value(&self) -> Option<(Action, f32)> {
    let num_trials = self.values.num_trials_float();
    if let Some(argmax_j) = array_argmax(&num_trials) {
      let argmax_point = self.valid_moves[argmax_j];
      let s_j = self.values.num_succs[argmax_j].load(Ordering::Acquire);
      let n_j = num_trials[argmax_j];
      let value = s_j as f32 / n_j;
      Some((Action::Place{point: argmax_point}, value))
    } else {
      None
    }
  }

  pub fn update_visits(&self, horizon_cfg: HorizonConfig, score: f32) {
    self.values.total_trials.fetch_add(1, Ordering::AcqRel);

    let uint_score = (2.0 * score).round() as isize as usize;
    self.values.total_score.fetch_add(uint_score, Ordering::AcqRel);

    match horizon_cfg {
      HorizonConfig::All => {}
      HorizonConfig::Fixed{..} => {}
      HorizonConfig::Pwide{mu} => {
        let num_arms = self.valid_moves.len();
        let pwide_mu = mu;
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
    }
  }

  //pub fn update_arm(&self, j: usize, adj_score: f32, raw_score: f32) {
  pub fn update_arm(&self, j: usize, score: f32) {
    self.values.num_trials[j].fetch_add(1, Ordering::AcqRel);

    //let uint_score = (2.0 * raw_score).round() as isize as usize;
    let uint_score = (2.0 * score).round() as isize as usize;
    self.values.scores[j].fetch_add(uint_score, Ordering::AcqRel);

    let turn = self.state.current_turn();
    if (Stone::White == turn && score >= 0.0) ||
        (Stone::Black == turn && score < 0.0)
    {
      self.values.num_succs[j].fetch_add(1, Ordering::AcqRel);
    }
    /*if (Stone::White == turn && raw_score >= 0.0) ||
        (Stone::Black == turn && raw_score < 0.0)
    {
      self.values.num_raw_succs[j].fetch_add(1, Ordering::AcqRel);
    }*/
  }

  pub fn update_total_live_stats(&self, rollout_traj: &RolloutTraj) -> isize {
    unimplemented!();
    /*let turn = self.state.current_turn();
    let turn_off = turn.offset();
    let mut counts: [isize; 2] = [0, 0];
    for offset in 0 .. 2 {
      let rel_offset = match turn_off {
        0 => offset,
        1 => 1 - offset,
        _ => unreachable!(),
      };
      for p in 0 .. Board::SIZE {
        if rollout_traj.mc_live_mask[offset].contains(p) {
          self.values.total_live_stats[rel_offset][p].fetch_add(1, Ordering::AcqRel);
          counts[rel_offset] += 1;
        }
      }
    }
    counts[0] - counts[1]*/
  }

  pub fn update_arm_live_succs(&self, j: usize, live_diff: isize, komi: f32) {
    unimplemented!();
    /*// FIXME(20160301): this first version of shaping is pretty broken.
    // Need to define the potential function at each backup.
    let rel_komi = match self.state.current_turn() {
      Stone::Black => -komi,
      Stone::White => komi,
      _ => unreachable!(),
    };
    if live_diff as f32 + rel_komi >= 0.0 {
      self.values.total_live_succs.fetch_add(1, Ordering::AcqRel);
      self.values.live_succs[j].fetch_add(1, Ordering::AcqRel);
    }*/
  }

  /*pub fn update_arm_live_counts(&self, j: usize, alive_count: usize) {
    let prev_sum = self.values.live_counts[j].fetch_add(alive_count, Ordering::AcqRel);
    // FIXME(20160301): update variance estimate (requires fixed point atomic).
    /*let curr_sum = prev_sum + alive_count;
    loop {
      let prev_var = self.values.live_counts_var[j].load(Ordering::Acquire);
      let var_update = (...);
      let prev_var_2 = self.values.live_counts_var[j].fetch_add(var_update, Ordering::AcqRel);
      if prev_var == prev_var_2 {
        break;
      }
    }*/
  }*/

  pub fn rave_update_arm(&self, j: usize, score: f32) {
    unimplemented!();
    /*let turn = self.state.current_turn();
    self.values.num_trials_rave[j].fetch_add(1, Ordering::AcqRel);
    if (Stone::White == turn && score >= 0.0) ||
        (Stone::Black == turn && score < 0.0)
    {
      self.values.num_succs_rave[j].fetch_add(1, Ordering::AcqRel);
    }*/
  }
}

pub fn search_principal_variation(init_node: Arc<RwLock<Node>>, depth: usize) -> Vec<(Stone, Action, usize, usize, f32, f32)> {
  let mut pv = vec![];
  let mut node = init_node;
  for t in 0 .. depth {
    let mut next_node = None;
    {
      let node = node.read().unwrap();
      let turn = node.state.current_turn();
      //let score = node.values.mean_score();
      let total_trials = node.values.total_trials.load(Ordering::Acquire);
      let num_trials = node.values.num_trials_float();
      let (action, arm_trials, arm_score, raw_value) = if let Some(argmax_j) = array_argmax(&num_trials) {
        next_node = node.child_nodes[argmax_j].clone();
        let argmax_point = node.valid_moves[argmax_j];
        let j_score = node.values.arm_score(argmax_j);
        let j_trials = num_trials[argmax_j];
        //let j_raw_succs = node.values.num_raw_succs[argmax_j].load(Ordering::Acquire);
        let j_succs = node.values.num_succs[argmax_j].load(Ordering::Acquire);
        let value = j_succs as f32 / j_trials;
        //let raw_value = j_raw_succs as f32 / j_trials;
        //if raw_value < 0.1 {
        if value < 0.1 {
          (Action::Resign, j_trials as usize, j_score, value)
        } else {
          (Action::Place{point: argmax_point}, j_trials as usize, j_score, value)
        }
      } else {
        (Action::Pass, 0, 0.0, 0.5)
      };
      pv.push((turn, action, total_trials, arm_trials, arm_score, raw_value));
      if action == Action::Resign || action == Action::Pass {
        break;
      }
    };
    if let Some(n) = next_node {
      node = n;
    } else {
      break;
    }
  }
  pv
}

/*pub fn search_principal_multi_variation(init_node: Arc<RwLock<Node>>, depth: usize) -> Vec<(Stone, Vec<(Action, usize, usize, f32, f32)>)> {
  let mut pv = vec![];
  let mut node = init_node;
  for t in 0 .. depth {
    let mut next_node = None;
    {
      let node = node.read().unwrap();
      let turn = node.state.current_turn();
      pv.push((turn, vec![]));
      //let score = node.values.mean_score();
      let total_trials = node.values.total_trials.load(Ordering::Acquire);
      let num_trials = node.values.num_trials_float();
      let (action, arm_trials, arm_score, raw_value) = if let Some(argmax_j) = array_argmax(&num_trials) {
        next_node = node.child_nodes[argmax_j].clone();
        let argmax_point = node.valid_moves[argmax_j];
        let j_score = node.values.arm_score(argmax_j);
        let j_trials = num_trials[argmax_j];
        //let j_raw_succs = node.values.num_raw_succs[argmax_j].load(Ordering::Acquire);
        let j_adj_succs = node.values.num_succs[argmax_j].load(Ordering::Acquire);
        //let adj_value = j_adj_succs as f32 / j_trials;
        let raw_value = j_raw_succs as f32 / j_trials;
        if raw_value < 0.1 {
          (Action::Resign, j_trials as usize, j_score, raw_value)
        } else {
          (Action::Place{point: argmax_point}, j_trials as usize, j_score, raw_value)
        }
      } else {
        (Action::Pass, 0, 0.0, 0.5)
      };
      let pv_idx = pv.len()-1;
      pv[pv_idx].1.push((action, total_trials, arm_trials, arm_score, raw_value));
      if action == Action::Resign || action == Action::Pass {
        break;
      }
    };
    if let Some(n) = next_node {
      node = n;
    } else {
      break;
    }
  }
  pv
}*/

pub enum TreeResult {
  NonTerminal,
  Terminal,
}

/*pub struct UniqueTree<N> where N: NodeBox {
  root_node:        Option<N>,
  mean_raw_score:   f32,
}*/

struct InnerTree {
  //root_node:        Option<Arc<RwLock<Node<ArcNodeBox, AtomicNodeValues>>>>,
  root_node:        Option<Arc<RwLock<Node>>>,

  mean_raw_score:   f32,
  mc_live_counts:   Vec<Arc<Vec<AtomicUsize>>>,

  explore_elapsed_ms:   AtomicUsize,
  rollout_elapsed_ms:   AtomicUsize,
}

#[derive(Clone)]
pub struct SharedTree {
  inner:        Arc<Mutex<InnerTree>>,
  //horizon_cfg:  HorizonConfig,
  tree_cfg:     TreePolicyConfig,
}

impl SharedTree {
  pub fn new(tree_cfg: TreePolicyConfig) -> SharedTree {
    let mut b_mc_live_counts = Vec::with_capacity(Board::SIZE);
    let mut w_mc_live_counts = Vec::with_capacity(Board::SIZE);
    for p in 0 .. Board::SIZE {
      b_mc_live_counts.push(AtomicUsize::new(0));
      w_mc_live_counts.push(AtomicUsize::new(0));
    }
    SharedTree{
      inner:    Arc::new(Mutex::new(InnerTree{
        root_node:      None,

        mean_raw_score: 0.0,
        mc_live_counts: vec![
          //Arc::new(repeat(AtomicUsize::new(0)).take(Board::SIZE).collect()),
          //Arc::new(repeat(AtomicUsize::new(0)).take(Board::SIZE).collect()),
          Arc::new(b_mc_live_counts),
          Arc::new(w_mc_live_counts),
        ],

        explore_elapsed_ms: AtomicUsize::new(0),
        rollout_elapsed_ms: AtomicUsize::new(0),
      })),

      // XXX(20160208): HACK: This is where the horizon policy is specified.
      // It should really belong to the hyperparam file, but whatever.
      //horizon_cfg: HorizonConfig::All,
      //horizon_cfg: HorizonConfig::Pwide{mu: 1.8},
      //horizon_cfg: HorizonConfig::Fixed{max_horizon: 3},
      //horizon_cfg: HorizonConfig::Fixed{max_horizon: 20},
      tree_cfg: tree_cfg,
    }
  }

  pub fn try_reset(&self, init_state: TxnState<TxnStateNodeData>, prior_policy: &mut PriorPolicy) {
    let mut inner = self.inner.lock().unwrap();
    if inner.root_node.is_none() {
      inner.root_node = Some(Arc::new(RwLock::new(Node::new(init_state, prior_policy, self.tree_cfg.horizon_cfg))));
      inner.mean_raw_score = 0.0;
      // FIXME(20160223): should reset other stats here too.
    }
  }

  pub fn try_advance(&self, turn: Stone, action: Action) -> bool {
    match action {
      Action::Resign |
      Action::Pass => {
        //println!("DEBUG: try_advance: do not advance on null move");
        false
      }
      Action::Place{point} => {
        let mut inner = self.inner.lock().unwrap();
        let maybe_next_node = {
          let p = point.idx();
          let maybe_root_node = inner.root_node.as_ref().map(|n| n.read().unwrap());
          if maybe_root_node.is_none() {
            println!("WARNING: try_advance: root node is None");
            return false;
          }
          let root_node = maybe_root_node.unwrap();
          let root_turn = root_node.state.current_turn();
          if root_turn != turn {
            println!("WARNING: try_advance: root node turn is {:?} but attempted turn is {:?}",
                root_turn, turn);
            return false;
          }
          let maybe_j = root_node.action_idxs.get(p).map(|j| *j);
          if maybe_j.is_none() {
            println!("WARNING: try_advance: j-th action idx does not exist");
            return false;
          }
          let j = maybe_j.unwrap();
          root_node.child_nodes[j].clone()
        };
        if maybe_next_node.is_none() {
          //println!("DEBUG: try_advance: j-th child node does not exist yet");
          return false;
        }
        inner.root_node = maybe_next_node;
        true
      }
    }
  }
}

/*pub trait TreeOpsTrait {
  type N: NodeBox;

  fn traverse(
      root_node:    Self::N,
      tree_traj:    &mut TreeTraj,
      prior_policy: &mut PriorPolicy,
      tree_policy:  &mut TreePolicy<R=Xorshiftplus128Rng>,
      rng:          &mut Xorshiftplus128Rng)
      -> TreeResult;
  fn backup(
      horizon_cfg: HorizonConfig,
      tree_traj: &TreeTraj,
      rollout_traj: &mut RolloutTraj,
      rng: &mut Xorshiftplus128Rng);
}*/

pub struct TreeOps;

impl TreeOps {
  pub fn traverse(
      //horizon_cfg:      HorizonConfig,
      tree_cfg:         TreePolicyConfig,
      root_node:        Arc<RwLock<Node>>,
      tree_traj:        &mut TreeTraj,
      //mut traj_trace:   Option<&mut SearchTrajTrace>,
      mut tree_trace:   Option<&mut TreeTrajTrace>,
      prior_policy:     &mut PriorPolicy,
      tree_policy:      &mut TreePolicy<R=Xorshiftplus128Rng>,
      //stats:            &mut SearchStats,
      rng:              &mut Xorshiftplus128Rng)
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
      if cursor_trials < tree_cfg.visit_thresh {
        // Not enough trials, stop the walk and do a rollout.
        //stats.old_leaf_count += 1;
        break;
      //if cursor_trials >= 1 {
      } else {
        // Try to walk through the current node using the exploration policy.
        //stats.edge_count += 1;
        let (res, horizon) = tree_policy.execute_search(&*cursor_node.read().unwrap(), rng);
        match res {
          Some((place_point, j)) => {
            tree_traj.backup_triples.push((cursor_node.clone(), place_point, j));
            if let Some(ref mut tree_trace) = tree_trace {
              tree_trace.decisions.push(TreeDecisionTrace::new(Action::Place{point: place_point}, j, horizon));
            }
            let has_child = cursor_node.read().unwrap().child_nodes[j].is_some();
            if has_child {
              // Existing inner node, simply update the cursor.
              let child_node = cursor_node.read().unwrap().child_nodes[j].as_ref().unwrap().clone();
              cursor_node = child_node;
              //stats.inner_edge_count += 1;
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
                  let inner_leaf_node = Node::new(leaf_state, prior_policy, tree_cfg.horizon_cfg);
                  if let Some(ref mut tree_trace) = tree_trace {
                    tree_trace.expansion = Some(TreeExpansionTrace::new(&inner_leaf_node));
                  }
                  let mut leaf_node = Arc::new(RwLock::new(inner_leaf_node));
                  cursor_node.child_nodes[j] = Some(leaf_node.clone());
                  //stats.new_leaf_count += 1;
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
            //stats.term_edge_count += 1;
            break;
          }
        }
      }
    }

    //stats.max_ply = max(stats.max_ply, ply);
    tree_traj.leaf_node = Some(cursor_node.clone());
    let terminal = cursor_node.read().unwrap().is_terminal();
    if terminal {
      TreeResult::Terminal
    } else {
      /*if let Some(traj_trace) = traj_trace {
        traj_trace.tree_traj.rollout = true;
      }*/
      TreeResult::NonTerminal
    }
  }

  pub fn backup(
      use_rave:         bool,
      komi:             f32,
      horizon_cfg:      HorizonConfig,
      tree_traj:        &TreeTraj,
      rollout_traj:     &mut RolloutTraj,
      //mut traj_trace:   Option<&mut SearchTrajTrace>,
      mut rollout_trace:    Option<&mut RolloutTrajTrace>,
      rng:              &mut Xorshiftplus128Rng)
  {
    /*let raw_score = match rollout_traj.raw_score {
      Some(raw_score) => raw_score,
      None => panic!("missing raw score for backup!"),
    };
    let adj_score = match rollout_traj.adj_score {
      Some(adj_score) => adj_score,
      None => panic!("missing adj score for backup!"),
    };*/
    rollout_traj.update_score(komi);
    //rollout_traj.update_mc_live_mask();

    let score = match rollout_traj.score {
      Some(score) => score,
      None => panic!("missing raw score for backup!"),
    };

    if let Some(ref mut rollout_trace) = rollout_trace {
      rollout_trace.score = Some(score);
    }

    if use_rave {
      rollout_traj.rave_mask[0].clear();
      rollout_traj.rave_mask[1].clear();
    }

    if rollout_traj.sim_pairs.len() >= 1 {
      assert!(rollout_traj.rollout);
      let leaf_node = tree_traj.leaf_node.as_ref().unwrap().read().unwrap();
      let leaf_turn = leaf_node.state.current_turn();

      // XXX(20151120): Currently not tracking pass pairs, so need to check that
      // the first rollout pair belongs to the leaf node turn.
      let (update_turn, update_point) = rollout_traj.sim_pairs[0];
      if leaf_turn == update_turn {
        if let Some(&update_j) = leaf_node.action_idxs.get(update_point.idx()) {
          assert_eq!(update_point, leaf_node.valid_moves[update_j]);
          //leaf_node.update_arm(update_j, adj_score[update_turn.offset()], raw_score);
          leaf_node.update_arm(update_j, score);

          /*let arm_alive_diff = leaf_node.update_total_live_stats(rollout_traj);
          leaf_node.update_arm_live_succs(update_j, arm_alive_diff, komi);*/
          /*if !tree_traj.backup_triples.is_empty() {
            let tree_traj_len = tree_traj.backup_triples.len();
            let (pred_node, pred_update_j) = (
              tree_traj.backup_triples[tree_traj_len - 1].0.read().unwrap(),
              tree_traj.backup_triples[tree_traj_len - 1].2,
            );
            pred_node.update_arm_live_counts(pred_update_j, arm_alive_count)
          }*/

          /*println!("DEBUG: search: rollout: ({:?}, {:?}, {:?})",
              update_turn, update_point, update_j,
          );*/
        } else {
          panic!("WARNING: leaf_node action_idxs does not contain update arm: {:?}", update_point);
        }
      }

      if use_rave {
        for &(sim_turn, sim_point) in rollout_traj.sim_pairs.iter() {
          rollout_traj.rave_mask[sim_turn.offset()].insert(sim_point.idx());
        }
        for sim_p in rollout_traj.rave_mask[update_turn.offset()].iter() {
          let sim_point = Point::from_idx(sim_p);
          if let Some(&sim_j) = leaf_node.action_idxs.get(sim_point.idx()) {
            assert_eq!(sim_point, leaf_node.valid_moves[sim_j]);
            //leaf_node.rave_update_arm(sim_j, adj_score[update_turn.offset()]);
            leaf_node.rave_update_arm(sim_j, score);
          }
        }
      }
    }

    {
      let leaf_node = tree_traj.leaf_node.as_ref().unwrap().read().unwrap();
      leaf_node.update_visits(horizon_cfg, score);
    }

    for (i, &(ref node, update_point, update_j)) in tree_traj.backup_triples.iter().enumerate().rev() {
      let node = node.read().unwrap();

      assert_eq!(update_point, node.valid_moves[update_j]);
      let update_turn = node.state.current_turn();
      //node.update_arm(update_j, adj_score[update_turn.offset()], raw_score);
      node.update_arm(update_j, score);

      /*let arm_alive_diff = node.update_total_live_stats(rollout_traj);
      node.update_arm_live_succs(update_j, arm_alive_diff, komi);*/
      /*if i >= 1 {
        let (pred_node, pred_update_j) = (
          tree_traj.backup_triples[i - 1].0.read().unwrap(),
          tree_traj.backup_triples[i - 1].2,
        );
        //pred_node.update_arm_live_counts(pred_update_j, arm_alive_count);
      }*/

      if use_rave {
        rollout_traj.rave_mask[update_turn.offset()].insert(update_point.idx());
        for sim_p in rollout_traj.rave_mask[update_turn.offset()].iter() {
          let sim_point = Point::from_idx(sim_p);
          if let Some(&sim_j) = node.action_idxs.get(sim_point.idx()) {
            assert_eq!(sim_point, node.valid_moves[sim_j]);
            //node.rave_update_arm(sim_j, adj_score[update_turn.offset()]);
            node.rave_update_arm(sim_j, score);
          }
        }
      }

      node.update_visits(horizon_cfg, score);
    }
  }
}

/*#[derive(Clone)]
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
  pub prev_mean_score:  f32,
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
            TxnStateConfig{
              rules:  RuleSet::KgsJapanese.rules(),
              ranks:  [PlayerRank::Dan(9), PlayerRank::Dan(9)],
            },
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
              rollout_traj.score(cfg.komi, cfg.prev_mean_score);
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

  /*pub fn worker_batch_size(&self) -> usize {
    self.worker_batch_size
  }*/

  pub fn enqueue(&self, tid: usize, cmd: EvalWorkerCommand) {
    self.in_txs[tid].send(cmd).unwrap();
  }

  pub fn join(&self) {
    self.out_barrier.wait();
  }
}*/

/*#[derive(Clone, Default)]
pub struct RolloutStats {
  pub rewards:  Arc<AtomicUsize>,
  pub count:    Arc<AtomicUsize>,
}*/

#[derive(Clone)]
pub enum SearchWorkerCommand {
  Noop,
  Quit,
  ResetSerialSearch{
    cfg:        SearchWorkerConfig,
    init_state: TxnState<TxnStateNodeData>,
  },
  ResetSearch{
    cfg:            SearchWorkerConfig,
    shared_tree:    SharedTree,
    init_state:     TxnState<TxnStateNodeData>,
    record_search:  bool,
  },
  Rollout{
    worker_cfg:     SearchWorkerConfig,
    leaf_states:    Vec<TxnState<TxnStateNodeData>>,
    record_search:  bool,
  },
  TrainMetaLevelTD0Gradient{
    t: usize,
  },
  TrainMetaLevelTD0Descent{
    step_size:  f32,
  },
  /*ResetEval{
    cfg:            SearchWorkerConfig,
    shared_data:    EvalSharedData,
    record_trace:   bool,
    green_stone:    Stone,
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
  ResetRollout{
    cfg:            SearchWorkerConfig,
    //shared_data:    EvalSharedData,
    green_stone:    Stone,
    init_state:     TxnState<TxnStateNodeData>,
    stats:          RolloutStats,
    // FIXME(20160222)
    //record_search:  bool,
  },
  TraceDescend{
    step_size:      f32,
    baseline:       f32,
    batch_size:     usize,
    green_stone:    Stone,
  },
  SaveFriendlyRolloutParamsToMem,
  LoadOpponentRolloutParamsFromMem{
    params_blob:    Arc<Vec<u8>>,
  },*/
}

pub enum SearchWorkerOutput {
  SearchTrace{
    tid:            usize,
    search_trace:   SearchTrace,
  },
  OpponentRolloutParams{
    params_blob:    Vec<u8>,
  },
}

#[derive(Clone, Copy, Debug)]
pub struct SearchWorkerConfig {
  pub batch_cfg:    SearchWorkerBatchConfig,
  pub tree_batch_size:      Option<usize>,
  pub rollout_batch_size:   usize,
  //pub batch_size:   usize,
  //pub num_batches:      usize,
  //pub komi:         f32,
  //pub prev_mean_score:  f32,
  //pub _tree_batch_size:     usize,
  //pub _rollout_batch_size:  usize,
}

#[derive(Clone, Copy, Debug)]
pub enum SearchWorkerBatchConfig {
  Fixed{num_batches: usize},
  TimeLimit{budget_ms: usize, tol_ms: usize},
}

pub struct ParallelMonteCarloSearchServer<W> where W: SearchPolicyWorker {
  num_workers:              usize,
  worker_batch_capacity:    usize,

  // XXX(20160308): Search workers perform the tree and rollout policies and
  // backup the tree after rollouts.
  search_pool:  ThreadPool,
  in_txs:       Vec<Sender<SearchWorkerCommand>>,
  out_rx:       Receiver<SearchWorkerOutput>,
  out_barrier:  Arc<Barrier>,

  // FIXME(20160308): Prior workers are for async prior evaluations.
  //prior_pool:   ThreadPool,

  // FIXME(20160308): Master worker (only one) is for distributed rollouts.
  //master_pool:  ThreadPool,

  // FIXME(20160308): Backup workers are for distributed rollouts.
  //backup_pool:  ThreadPool,

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
  pub fn new<B>(
      state_cfg: TxnStateConfig,
      num_workers: usize,
      worker_tree_batch_capacity: usize,
      worker_batch_capacity: usize,
      worker_builder: B) -> ParallelMonteCarloSearchServer<W>
  where B: 'static + SearchPolicyWorkerBuilder<Worker=W>
  {
    let search_pool = ThreadPool::new(num_workers);
    let barrier = Arc::new(Barrier::new(num_workers));
    let mut in_txs = vec![];
    let (out_tx, out_rx) = channel();
    let out_barrier = Arc::new(Barrier::new(num_workers + 1));

    // FIXME(20160308): for async priors and distributed rollouts.
    /*let prior_pool = ThreadPool::new(num_workers);
    let prior_barrier = Arc::new(Barrier::new(num_workers));
    let prior_ext_barrier = Arc::new(Barrier::new(num_workers + 1));
    let mut search_to_prior_txs = vec![];
    let mut search_to_prior_rxs = vec![];
    let mut prior_to_search_txs = vec![];
    let mut prior_to_search_rxs = vec![];

    let master_pool = ThreadPool::new(1);

    let backup_pool = ThreadPool::new(num_workers);*/

    /*let shared_rollout_rewards = Arc::new(AtomicUsize::new(0));
    let shared_rollout_count = Arc::new(AtomicUsize::new(0));
    let shared_traces_count = Arc::new(AtomicUsize::new(0));*/

    let recon_worker_builder = ParallelSearchReconWorkerBuilder::new(num_workers);
    let omega_memory = Arc::new(RwLock::new(OmegaWorkerMemory::new()));
    let shared_signal = Arc::new(AtomicBool::new(false));

    for tid in 0 .. num_workers {
      let worker_builder = worker_builder.clone();
      let recon_worker_builder = recon_worker_builder.clone();
      let omega_memory = omega_memory.clone();
      let shared_signal = shared_signal.clone();

      let barrier = barrier.clone();
      let out_barrier = out_barrier.clone();

      let (in_tx, in_rx) = channel();
      in_txs.push(in_tx);
      let out_tx = out_tx.clone();

      /*// FIXME(20160225): these `shared_*` variables are deprecated.
      let shared_rollout_rewards = shared_rollout_rewards.clone();
      let shared_rollout_count = shared_rollout_count.clone();
      let shared_traces_count = shared_traces_count.clone();*/

      search_pool.execute(move || {
        let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
        let worker = Rc::new(RefCell::new(worker_builder.into_worker(tid, worker_tree_batch_capacity, worker_batch_capacity)));
        let omega_worker = OmegaTreeBatchWorker::new(omega_memory.clone(), worker.clone());
        let recon_worker = Rc::new(RefCell::new(recon_worker_builder.into_worker(tid, omega_worker, ())));
        //let meta_worker = MetaLevelWorker::new(recon_worker.clone());
        let omega_memory = omega_memory;
        let shared_signal = shared_signal;

        let barrier = barrier;
        let in_rx = in_rx;
        let out_barrier = out_barrier;

        let use_rave = {
          let mut worker = worker.borrow_mut();
          let (_, tree_policy) = worker.exploration_policies();
          tree_policy.use_rave()
        };

        let mut tree_trajs: Vec<_> = repeat(TreeTraj::new()).take(worker_batch_capacity).collect();
        let mut leaf_states: Vec<_> = repeat(TxnState::new(state_cfg, TxnStateNodeData::new())).take(worker_batch_capacity).collect();
        let mut rollout_trajs: Vec<_> = repeat(RolloutTraj::new()).take(worker_batch_capacity).collect();

        // FIXME(20160225): QuickTrace is deprecated.
        //let mut traces: Vec<_> = repeat(QuickTrace::new()).take(worker_batch_capacity * 40).collect();

        /*let tree_cfg = TreePolicyConfig{
          horizon_cfg:  HorizonConfig::Fixed{max_horizon: 20},
          visit_thresh: 1,
          mc_scale:     1.0,
          prior_equiv:  16.0,
          rave:         false,
          rave_equiv:   0.0,
        };*/
        let mut search_trace = SearchTrace::new();

        let mut mc_live_counts: Vec<Vec<usize>> = vec![
          repeat(0).take(Board::SIZE).collect(),
          repeat(0).take(Board::SIZE).collect(),
        ];

        loop {
          // FIXME(20151222): for real time search, num batches is just an
          // estimate; should check for termination within inner batch loop.
          let cmd: SearchWorkerCommand = in_rx.recv().unwrap();
          match cmd {
            SearchWorkerCommand::Noop => {
              // Do nothing (other than synchronize).
              out_barrier.wait();
            }

            SearchWorkerCommand::Quit => {
              break;
            }

            SearchWorkerCommand::ResetSerialSearch{cfg, init_state} => {
              // FIXME(20160124)
              unimplemented!();
            }

            SearchWorkerCommand::ResetSearch{cfg, shared_tree, /*green_stone,*/ init_state, record_search} => {
              let timer_start = get_time();

              // XXX(20160107): If the tree has no root node, this sets it;
              // otherwise use the existing root node.
              let tree = shared_tree;
              tree.try_reset(init_state, worker.borrow_mut().prior_policy());

              // FIXME(20160308): handle different tree and rollout batch sizes.
              // If tree batch size is greater, offload to remote workers.
              //let batch_size = cfg.batch_size;
              //let num_batches = cfg.num_batches;
              let batch_size = cfg.rollout_batch_size / num_workers;
              /*println!("DEBUG: batch size: {} batch capacity: {}",
                  batch_size, worker_batch_capacity);
              assert!(batch_size <= worker_batch_capacity);*/
              if batch_size > worker_batch_capacity {
                panic!("WARNING: batch_size ({}) exceeds batch capacity ({})",
                    batch_size, worker_batch_capacity);
              }
              let tree_cfg = tree.tree_cfg;

              let root_node = {
                let inner = tree.inner.lock().unwrap();

                if tid == 0 {
                  let shared_mc_live_counts = inner.mc_live_counts.clone();
                  for p in 0 .. Board::SIZE {
                    shared_mc_live_counts[0][p].store(0, Ordering::Release);
                    shared_mc_live_counts[1][p].store(0, Ordering::Release);
                  }
                }

                inner.root_node.as_ref().unwrap().clone()
              };
              let komi = root_node.read().unwrap().state.config.komi;

              let mut explore_elapsed_ms = 0;
              let mut rollout_elapsed_ms = 0;
              /*let mut worker_mean_score: f32 = 0.0;
              let mut worker_backup_count: f32 = 0.0;*/
              for p in 0 .. Board::SIZE {
                mc_live_counts[0][p] = 0;
                mc_live_counts[1][p] = 0;
              }

              if record_search {
                let root_node = root_node.read().unwrap();
                search_trace.start_instance(tree_cfg, &*root_node);
              }

              // FIXME(20151222): should share stats between workers.
              //let mut stats: SearchStats = Default::default();

              // FIXME(20160308): old QuickTrace is deprecated.
              /*for trace in traces.iter_mut() {
                trace.reset();
              }*/
              /*if traces.len() < num_batches * batch_size {
                for _ in traces.len() .. num_batches * batch_size {
                  traces.push(QuickTrace::new());
                }
              }*/

              if tid == 0 {
                shared_signal.store(false, Ordering::Release);
              }
              barrier.wait();
              fence(Ordering::AcqRel);

              //for batch in 0 .. num_batches {
              let mut batch = 0;
              loop {
                /*println!("DEBUG: worker {} batch {}/{}",
                    tid, batch, num_batches);*/

                let start_time = get_time();

                if record_search {
                  search_trace.start_batch(batch_size);
                }

                {
                  let mut worker = worker.borrow_mut();
                  let (prior_policy, tree_policy) = worker.exploration_policies();
                  for batch_idx in 0 .. batch_size {
                    let tree_traj = &mut tree_trajs[batch_idx];
                    let rollout_traj = &mut rollout_trajs[batch_idx];
                    let tree_trace = if record_search {
                      Some(&mut search_trace.batches[batch].traj_traces[batch_idx].tree_trace)
                    } else {
                      None
                    };
                    match TreeOps::traverse(tree_cfg, root_node.clone(), tree_traj, tree_trace, prior_policy, tree_policy, /*&mut stats,*/ &mut rng) {
                      /*TreeResult::Terminal => {
                        let leaf_state = &tree_traj.leaf_node.as_ref().unwrap().read().unwrap().state;
                        rollout_traj.reset_terminal(leaf_state);
                      }
                      TreeResult::NonTerminal => {
                        let leaf_state = &tree_traj.leaf_node.as_ref().unwrap().read().unwrap().state;
                        rollout_traj.reset_rollout(leaf_state);
                      }*/
                      // XXX(20160223): Rolling out whether the leaf state is
                      // terminal or not. This is because the tree traversal
                      // does not account for passing at all, whereas rollouts
                      // do and should be less biased.
                      TreeResult::NonTerminal | TreeResult::Terminal => {
                        let leaf_state = &tree_traj.leaf_node.as_ref().unwrap().read().unwrap().state;
                        rollout_traj.reset_rollout(leaf_state);
                      }
                    }

                    /*let triples: Vec<_> = tree_traj.backup_triples.iter().map(|x| (x.1, x.2)).collect();
                    println!("DEBUG: search: batch: {}/{} batch_idx: {}/{} tree traj: {:?}",
                        batch, num_batches,
                        batch_idx, batch_size,
                        triples,
                    );*/
                  }
                }
                barrier.wait();
                fence(Ordering::AcqRel);

                let mid_time = get_time();

                {
                  let traj_trace_batch = if record_search {
                    Some(&mut search_trace.batches[batch])
                  } else {
                    None
                  };
                  worker.borrow_mut().rollout_policy().rollout_batch(
                      batch_size,
                      //green_stone,
                      RolloutLeafs::TreeTrajs(&tree_trajs),
                      &mut rollout_trajs,
                      None,
                      traj_trace_batch,
                      //false, &mut traces[batch * batch_size .. (batch + 1) * batch_size],
                      &mut rng);
                }
                barrier.wait();
                fence(Ordering::AcqRel);

                for batch_idx in 0 .. batch_size {
                  let tree_traj = &tree_trajs[batch_idx];
                  let rollout_traj = &mut rollout_trajs[batch_idx];
                  let rollout_trace = if record_search {
                    Some(&mut search_trace.batches[batch].traj_traces[batch_idx].rollout_trace)
                  } else {
                    None
                  };
                  rollout_traj.update_mc_live_counts(&mut mc_live_counts);
                  TreeOps::backup(use_rave, komi, tree_cfg.horizon_cfg, tree_traj, rollout_traj, rollout_trace, &mut rng);

                  /*//let raw_score = rollout_traj.raw_score.unwrap();
                  let score = rollout_traj.score.unwrap();
                  worker_backup_count += 1.0;
                  worker_mean_score += (score - worker_mean_score) / worker_backup_count;*/
                }
                barrier.wait();
                fence(Ordering::AcqRel);

                let end_time = get_time();

                explore_elapsed_ms += (mid_time - start_time).num_milliseconds() as usize;
                rollout_elapsed_ms += (end_time - mid_time).num_milliseconds() as usize;

                /*{
                  let root = root_node.read().unwrap();
                  println!("DEBUG: search: batch: {}/{} visits: {} stats: {}/{}",
                      batch, num_batches,
                      root.values.total_trials.load(Ordering::Acquire),
                      root.values.num_succs[0].load(Ordering::Acquire),
                      root.values.num_trials[0].load(Ordering::Acquire),
                  );
                }*/

                batch += 1;
                match cfg.batch_cfg {
                  SearchWorkerBatchConfig::Fixed{num_batches} => {
                    if batch >= num_batches {
                      break;
                    }
                  }
                  SearchWorkerBatchConfig::TimeLimit{budget_ms, tol_ms} => {
                    if tid == 0 {
                      let timer_lap = get_time();
                      let elapsed_ms = (timer_lap - timer_start).num_milliseconds() as usize;
                      if elapsed_ms + tol_ms > budget_ms {
                        shared_signal.store(true, Ordering::Release);
                      }
                    }
                    barrier.wait();
                    fence(Ordering::AcqRel);
                    let signal = shared_signal.load(Ordering::Acquire);
                    if signal {
                      println!("DEBUG: server: reached time limit, num batches: {}", batch);
                      break;
                    }
                  }
                }
              }

              if record_search {
                let root_node = root_node.read().unwrap();
                search_trace.update_instance(&*root_node);
              }

              let shared_mc_live_counts = {
                let mut inner = tree.inner.lock().unwrap();

                //inner.mean_raw_score += worker_mean_score / (num_workers as f32);
                inner.explore_elapsed_ms.fetch_add(explore_elapsed_ms, Ordering::AcqRel);
                inner.rollout_elapsed_ms.fetch_add(rollout_elapsed_ms, Ordering::AcqRel);

                let shared_mc_live_counts = inner.mc_live_counts.clone();
                shared_mc_live_counts
              };
              for p in 0 .. Board::SIZE {
                shared_mc_live_counts[0][p].fetch_add(mc_live_counts[0][p], Ordering::AcqRel);
                shared_mc_live_counts[1][p].fetch_add(mc_live_counts[1][p], Ordering::AcqRel);
              }

              out_barrier.wait();
              fence(Ordering::AcqRel);
            }

            SearchWorkerCommand::Rollout{worker_cfg, leaf_states, record_search} => {
              // FIXME(20160308)
              unimplemented!();
            }

            SearchWorkerCommand::TrainMetaLevelTD0Gradient{t} => {
              if tid == 0 {
                let mut omega_memory = omega_memory.write().unwrap();
                if t == 0 {
                  omega_memory.reset();
                }
                omega_memory.set_state(MetaLevelObjective::TD0L2Loss, t, search_trace.root_value.unwrap());
                println!("DEBUG: server: root value[{}]: {:.6}", t, omega_memory.inner_values[t]);
              }

              let mut recon_worker = recon_worker.borrow_mut();
              recon_worker.reconstruct_trace(&search_trace);
              if tid == 0 {
                let mut root_node = recon_worker.shared_tree.root_node.lock().unwrap();
                let root_value = root_node.as_ref().unwrap().read().unwrap().get_value();
                //assert_eq!(root_value, omega_memory.read().unwrap().inner_values[t]);
                let correct_root_value = match t {
                  0 => root_value,
                  1 => 1.0 - root_value,
                  _ => unreachable!(),
                };
                if correct_root_value != omega_memory.read().unwrap().inner_values[t] {
                  println!("WARNING: server: root values mismatch: {} {:.6} {:.6}",
                      t, correct_root_value, omega_memory.read().unwrap().inner_values[t]);
                }
              }

              out_barrier.wait();
              fence(Ordering::AcqRel);
            }

            SearchWorkerCommand::TrainMetaLevelTD0Descent{step_size} => {
              if tid == 0 {
                let omega_memory = omega_memory.read().unwrap();
                println!("DEBUG: TD0 values: {:.6} {:.6}", omega_memory.inner_values[0], omega_memory.inner_values[1]);
                let step_scale = omega_memory.inner_values[0] - omega_memory.inner_values[1];
                let mut worker = worker.borrow_mut();
                let mut prior_policy = worker.diff_prior_policy();
                prior_policy.sync_gradients(GradSyncMode::Sum);
                prior_policy.descend_params(step_scale * step_size);
                prior_policy.reset_gradients();
              }

              out_barrier.wait();
              fence(Ordering::AcqRel);
            }

            /*SearchWorkerCommand::ResetEval{cfg, shared_data, record_trace, green_stone, init_state} => {
              let batch_size = cfg.batch_size;
              let num_batches = cfg.num_batches;
              assert!(batch_size <= worker_batch_capacity);

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
                  leaf_states[batch_idx] = init_state.clone();
                  let rollout_traj = &mut rollout_trajs[batch_idx];
                  rollout_traj.reset_rollout(&init_state);
                }
                barrier.wait();
                fence(Ordering::AcqRel);

                worker.rollout_policy().rollout_batch(
                    batch_size,
                    // FIXME(20160222)
                    //green_stone,
                    RolloutLeafs::LeafStates(&leaf_states),
                    &mut rollout_trajs,
                    None,
                    None,
                    record_trace, &mut traces[batch * batch_size .. (batch + 1) * batch_size],
                    &mut rng);
                barrier.wait();
                fence(Ordering::AcqRel);

                for batch_idx in 0 .. batch_size {
                  let rollout_traj = &mut rollout_trajs[batch_idx];
                  //rollout_traj.score(cfg.komi, cfg.prev_mean_score);
                  rollout_traj.update_score(cfg.komi);
                  //let raw_score = rollout_traj.raw_score.unwrap();
                  let raw_score = rollout_traj.score.unwrap();
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
            }*/

            /*SearchWorkerCommand::EvalGradientsAndBackup{learning_rate, baseline, target_value, eval_value} => {
              let mut num_traces = 0;
              worker.rollout_policy().init_traces();
              for trace in traces.iter() {
                if worker.rollout_policy().rollout_trace(trace, baseline) {
                  num_traces += 1;
                }
              }
              if num_traces > 0 {
                worker.rollout_policy().backup_traces(learning_rate, target_value, eval_value, num_traces);
              }

              out_barrier.wait();
            }*/

            /*SearchWorkerCommand::SaveRolloutParams{save_dir, num_iters} => {
              worker.rollout_policy().save_params(&save_dir, num_iters);

              out_barrier.wait();
            }*/

            /*SearchWorkerCommand::ResetRollout{cfg, green_stone, init_state, stats} => {
              let batch_size = cfg.batch_size;
              let num_batches = cfg.num_batches;
              assert!(batch_size <= worker_batch_capacity);
              assert_eq!(num_batches, 1);

              // FIXME(20151222): should share stats between workers.
              //let mut stats: SearchStats = Default::default();

              if traces.len() < batch_size {
                for _ in traces.len() .. batch_size {
                  traces.push(QuickTrace::new());
                }
              }
              for batch_idx in 0 .. batch_size {
                leaf_states[batch_idx] = init_state.clone();
                let rollout_traj = &mut rollout_trajs[batch_idx];
                rollout_traj.reset_rollout(&init_state);
              }
              if tid == 0 {
                shared_rollout_rewards.store(0, Ordering::Release);
                shared_rollout_count.store(0, Ordering::Release);
              }
              barrier.wait();
              fence(Ordering::AcqRel);

              worker.rollout_policy().rollout_batch(
                  batch_size,
                  // FIXME(20160222)
                  //green_stone,
                  RolloutLeafs::LeafStates(&leaf_states),
                  &mut rollout_trajs,
                  None,
                  None,
                  true, &mut traces[0 .. batch_size],
                  &mut rng);
              barrier.wait();
              fence(Ordering::AcqRel);

              for batch_idx in 0 .. batch_size {
                let rollout_traj = &mut rollout_trajs[batch_idx];
                //rollout_traj.score(cfg.komi, cfg.prev_mean_score);
                rollout_traj.update_score(cfg.komi);
                //let raw_score = rollout_traj.raw_score.unwrap();
                let raw_score = rollout_traj.score.unwrap();
                // XXX(20160208): Compute terminal value.
                let reward = translate_score_to_reward(green_stone, raw_score);
                traces[batch_idx].value = Some(reward as f32);
                shared_rollout_rewards.fetch_add(reward, Ordering::AcqRel);
                shared_rollout_count.fetch_add(1, Ordering::AcqRel);
              }
              barrier.wait();
              fence(Ordering::AcqRel);

              if tid == 0 {
                stats.rewards.store(shared_rollout_rewards.load(Ordering::Acquire), Ordering::Release);
                stats.count.store(shared_rollout_count.load(Ordering::Acquire), Ordering::Release);
              }

              out_barrier.wait();
              fence(Ordering::AcqRel);
            }*/

            /*SearchWorkerCommand::TraceDescend{step_size, baseline, batch_size, green_stone} => {
              if tid == 0 {
                shared_traces_count.store(0, Ordering::Release);
              }
              barrier.wait();
              fence(Ordering::AcqRel);

              let mut worker_traces_count = 0;
              worker.rollout_policy().init_traces();
              for trace in &traces[ .. batch_size] {
                if worker.rollout_policy().rollout_green_trace(baseline, green_stone, trace) {
                  worker_traces_count += 1;
                }
              }
              shared_traces_count.fetch_add(worker_traces_count, Ordering::AcqRel);
              barrier.wait();
              fence(Ordering::AcqRel);

              let traces_count = shared_traces_count.load(Ordering::Acquire);
              if worker_traces_count > 0 {
                worker.rollout_policy().backup_green_traces(step_size, traces_count, green_stone);
              }

              out_barrier.wait();
              fence(Ordering::AcqRel);
            }*/

            /*SearchWorkerCommand::SaveFriendlyRolloutParamsToMem => {
              if tid == 0 {
                let params_blob = worker.rollout_policy().save_green_params();
                out_tx.send(SearchWorkerOutput::OpponentRolloutParams{
                  params_blob:  params_blob,
                }).unwrap();
              }
              out_barrier.wait();
            }*/

            /*SearchWorkerCommand::LoadOpponentRolloutParamsFromMem{params_blob} => {
              worker.rollout_policy().load_red_params(&params_blob);
              out_barrier.wait();
            }*/

            /*_ => {
              unimplemented!();
            }*/
          };

        }

        out_barrier.wait();
      });
    }

    ParallelMonteCarloSearchServer{
      num_workers:              num_workers,
      worker_batch_capacity:    worker_batch_capacity,
      search_pool:  search_pool,
      in_txs:       in_txs,
      out_rx:       out_rx,
      out_barrier:  out_barrier,
      _marker:      PhantomData,
    }
  }

  pub fn num_workers(&self) -> usize {
    self.num_workers
  }

  /*pub fn worker_batch_size(&self) -> usize {
    self.worker_batch_capacity
  }*/

  pub fn enqueue(&self, tid: usize, cmd: SearchWorkerCommand) {
    self.in_txs[tid].send(cmd).unwrap();
  }

  pub fn join(&self) {
    self.out_barrier.wait();
  }

  pub fn recv_output(&self) -> SearchWorkerOutput {
    self.out_rx.recv().unwrap()
  }

  pub fn wait_ready(&self) {
    let num_workers = self.num_workers();
    for tid in 0 .. num_workers {
      self.enqueue(tid, SearchWorkerCommand::Noop);
    }
    self.join();
  }
}

#[derive(Clone, Debug)]
pub struct MonteCarloSearchResult {
  pub turn:             Stone,
  pub action:           Action,
  pub expected_score:   f32,
  //pub expected_adj_val: f32,
  pub expected_value:   f32,
  pub b_mc_alive:       usize,
  pub w_mc_alive:       usize,
  /*pub b_alive_ch:       usize,
  pub b_alive_ter:      usize,
  pub w_alive_ch:       usize,
  pub w_alive_ter:      usize,*/
  pub top_prior_values: Vec<(Action, f32)>,
  pub pv:               Vec<(Stone, Action, usize, usize, f32, f32)>,
  pub dead_stones:      Vec<Vec<Point>>,
  pub live_stones:      Vec<Vec<Point>>,
  pub territory:        Vec<Vec<Point>>,
  pub outcome:          Option<Stone>,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct MonteCarloSearchStats {
  pub argmax_rank:      Option<usize>,
  pub argmax_ntrials:   usize,
  pub elapsed_ms:       usize,
  pub avg_explore_elapsed_ms:   usize,
  pub avg_rollout_elapsed_ms:   usize,
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
      //total_num_rollouts: usize,
      //total_batch_size: usize,
      //batch_cfg:    SearchWorkerBatchConfig,
      worker_cfg:   SearchWorkerConfig,
      server:       &ParallelMonteCarloSearchServer<W>,
      green_stone:  Stone,
      init_state:   &TxnState<TxnStateNodeData>,
      shared_tree:  SharedTree,
      //prev_result:  Option<&MonteCarloSearchResult>,
      rng:          &mut Xorshiftplus128Rng)
      -> (MonteCarloSearchResult, MonteCarloSearchStats)
      where W: SearchPolicyWorker
  {
    let num_workers = server.num_workers();
    /*let num_rollouts = (total_num_rollouts + num_workers - 1) / num_workers * num_workers;
    let worker_num_rollouts = num_rollouts / num_workers;
    //let worker_batch_size = server.worker_batch_size();
    let worker_batch_size = (total_batch_size + num_workers - 1) / num_workers;
    let worker_num_batches = (worker_num_rollouts + worker_batch_size - 1) / worker_batch_size;
    assert!(worker_batch_size >= 1);
    assert!(worker_num_batches >= 1);*/

    /*println!("DEBUG: server config: num workers:          {}", num_workers);
    println!("DEBUG: server config: worker num rollouts:  {}", worker_num_rollouts);
    println!("DEBUG: server config: worker batch size:    {}", worker_batch_size);
    println!("DEBUG: server config: worker num batches:   {}", worker_num_batches);*/
    /*println!("DEBUG: search config: {} {} {}",
        num_workers, worker_batch_size, worker_num_batches);*/

    // TODO(20160106): reset stats.

    /*let prev_mean_score = {
      let inner_tree = shared_tree.inner.lock().unwrap();
      if let Some(ref root_node) = inner_tree.root_node {
        let root_node = root_node.read().unwrap();
        root_node.values.mean_score(32.0)
      } else {
        0.0
      }
    };*/
    /*let cfg = SearchWorkerConfig{
      batch_cfg:    batch_cfg,
      batch_size:   worker_batch_size,
      //num_batches:      worker_num_batches,
      // FIXME(20160112): use correct komi value.
      //komi:             7.5,
      // FIXME(20160219): use correct root node score if it has sufficient
      // sample size, otherwise...?
      //prev_mean_score:  prev_result.map_or(0.0, |r| r.expected_score),
      //prev_mean_score:  prev_mean_score,
      //prev_mean_score:  0.0,
      //_tree_batch_size:     worker_batch_size,
      //_rollout_batch_size:  worker_batch_size,
    };*/
    //println!("DEBUG: ParallelMonteCarloSearch: worker config: {:?}", cfg);

    let start_time = get_time();
    for tid in 0 .. num_workers {
      server.enqueue(tid, SearchWorkerCommand::ResetSearch{
        cfg:            worker_cfg,
        shared_tree:    shared_tree.clone(),
        //green_stone:    Stone::White, // FIXME FIXME FIXME(20160208): use correct color.
        init_state:     init_state.clone(),
        record_search:  false,
      });
    }
    server.join();
    let end_time = get_time();
    let elapsed_ms = (end_time - start_time).num_milliseconds();

    let mut stats: MonteCarloSearchStats = Default::default();
    stats.elapsed_ms = elapsed_ms as usize;

    /*let root_node_opt = tree.root_node.read().unwrap();
    let root_node = root_node_opt.as_ref().unwrap().read().unwrap();*/
    let (root_node, /*mean_raw_score,*/ shared_mc_live_counts) = {
      let inner_tree = shared_tree.inner.lock().unwrap();

      stats.avg_explore_elapsed_ms = inner_tree.explore_elapsed_ms.load(Ordering::Acquire) / num_workers;
      stats.avg_rollout_elapsed_ms = inner_tree.rollout_elapsed_ms.load(Ordering::Acquire) / num_workers;

      // FIXME(20160219): should reset these above.
      inner_tree.explore_elapsed_ms.store(0, Ordering::Release);
      inner_tree.rollout_elapsed_ms.store(0, Ordering::Release);

      ( inner_tree.root_node.as_ref().unwrap().clone(),
        //inner_tree.mean_raw_score,
        inner_tree.mc_live_counts.clone(),
      )
    };

    // FIXME(20160308): count monte carlo dead/alive;
    // requires number of trajectories.
    // FIXME(20160315): track live/dead stones (required for Category A).
    let mut b_mc_alive = 0;
    let mut w_mc_alive = 0;
    let mut dead_stones = vec![vec![], vec![]];
    let mut live_stones = vec![vec![], vec![]];
    let mut territory = vec![vec![], vec![]];
    let mut outcome = None;
    //let live_thresh = (0.9 * (worker_batch_size * worker_num_batches * num_workers) as f32).ceil() as usize;
    {
      let root_node = root_node.read().unwrap();
      let rollout_count = root_node.values.total_trials.load(Ordering::Acquire);
      let live_thresh = (0.9 * rollout_count as f32).ceil() as usize;
      for p in 0 .. Board::SIZE {
        let pt = Point::from_idx(p);
        if shared_mc_live_counts[0][p].load(Ordering::Acquire) >= live_thresh {
          b_mc_alive += 1;
          match root_node.state.current_stone(pt) {
            Stone::Black => live_stones[0].push(pt),
            Stone::Empty => territory[0].push(pt),
            _ => {}
          }
        } else {
          match root_node.state.current_stone(pt) {
            Stone::Black => dead_stones[0].push(pt),
            _ => {}
          }
        }
        if shared_mc_live_counts[1][p].load(Ordering::Acquire) >= live_thresh {
          w_mc_alive += 1;
          match root_node.state.current_stone(pt) {
            Stone::White => live_stones[1].push(pt),
            Stone::Empty => territory[1].push(pt),
            _ => {}
          }
        } else {
          match root_node.state.current_stone(pt) {
            Stone::White => dead_stones[1].push(pt),
            _ => {}
          }
        }
      }
      if b_mc_alive > w_mc_alive {
        outcome = Some(Stone::Black);
      } else if b_mc_alive < w_mc_alive {
        outcome = Some(Stone::White);
      }
    }

    let root_pv = search_principal_variation(root_node.clone(), 3);

    let root_node = root_node.read().unwrap();
    let root_turn = root_node.state.current_turn();

    let mut top_prior_values = Vec::with_capacity(5);
    for k in 0 .. min(5, root_node.valid_moves.len()) {
      let pt = Action::Place{point: root_node.valid_moves[k]};
      let prior = root_node.values.prior_values[k];
      top_prior_values.push((pt, prior));
    }

    /*let mut scratch = BensonScratch::new();
    let (b_alive_ch, b_alive_ter) = root_node.state.count_unconditionally_alive(Stone::Black, &mut scratch);
    let (w_alive_ch, w_alive_ter) = root_node.state.count_unconditionally_alive(Stone::White, &mut scratch);*/

    // XXX(20160209): Maximize the winning rate rather than the number of plays.
    // The argument is that it should be highly unlikely for an arm with high
    // value to be played only a small number of times.
    // FIXME(20160209): the above, except in the case of progressive widening
    // horizon, where there _may_ be such arms; a hacky remedy may be to stop
    // widening at some fixed fraction of the total rollout number.
    //let root_ratios = root_node.values.succ_ratios_float();
    //let (action, adj_value, raw_value) = if let Some(argmax_j) = array_argmax(&root_ratios) {
    let root_trials = root_node.values.num_trials_float();
    let (action, expected_score, value) = if let Some(argmax_j) = array_argmax(&root_trials) {
      stats.argmax_rank = Some(argmax_j);
      stats.argmax_ntrials = root_trials[argmax_j] as usize;

      let argmax_point = root_node.valid_moves[argmax_j];
      let j_score = root_node.values.arm_score(argmax_j);
      let j_adj_succs = root_node.values.num_succs[argmax_j].load(Ordering::Acquire);
      //let j_raw_succs = root_node.values.num_raw_succs[argmax_j].load(Ordering::Acquire);
      let j_trials = root_trials[argmax_j];
      let value = j_adj_succs as f32 / j_trials;
      //let raw_value = j_raw_succs as f32 / j_trials;

      /*if root_turn == Stone::Black && -(b_mc_alive as f32) + cfg.komi < 0.0 {
        (Action::Pass, j_score, value, raw_value)
      } else if root_turn == Stone::White && w_mc_alive as f32 + cfg.komi >= 0.0 {
        (Action::Pass, j_score, value, raw_value)
      } else */

      //if raw_value < 0.1 {
      if value < 0.1 {
        (Action::Resign, j_score, value)
      } else {
        (Action::Place{point: argmax_point}, j_score, value)
      }
    } else {
      stats.argmax_rank = None;
      (Action::Pass, 0.0, 0.5)
    };

    (MonteCarloSearchResult{
      turn:   root_node.state.current_turn(),
      action: action,
      //expected_score:   mean_raw_score,
      expected_score:   expected_score,
      //expected_adj_val: value,
      expected_value:   value,
      b_mc_alive:       b_mc_alive,
      w_mc_alive:       w_mc_alive,
      /*b_alive_ch:       b_alive_ch,
      b_alive_ter:      b_alive_ter,
      w_alive_ch:       w_alive_ch,
      w_alive_ter:      w_alive_ter,*/
      top_prior_values: top_prior_values,
      pv:               root_pv,
      dead_stones:      dead_stones,
      live_stones:      live_stones,
      territory:        territory,
      outcome:          outcome,
    }, stats)
  }
}

#[derive(Clone, Copy, Debug)]
pub struct MonteCarloSearchConfig {
  pub num_rollouts: usize,
  pub batch_size:   usize,
}

impl MonteCarloSearchConfig {
  pub fn worker_dims(&self, num_workers: usize) -> (usize, usize) {
    let num_rollouts = (self.num_rollouts + num_workers - 1) / num_workers * num_workers;
    let worker_num_rollouts = num_rollouts / num_workers;
    let worker_batch_size = (self.batch_size + num_workers - 1) / num_workers;
    let worker_num_batches = (worker_num_rollouts + worker_batch_size - 1) / worker_batch_size;
    (worker_batch_size, worker_num_batches)
  }
}

/*pub struct ParallelMonteCarloSearchAndTrace {
  pub search_traces:    VecMap<SearchTrace>,
}

impl ParallelMonteCarloSearchAndTrace {
  pub fn new() -> ParallelMonteCarloSearchAndTrace {
    ParallelMonteCarloSearchAndTrace{
      search_traces:    VecMap::new(),
    }
  }

  pub fn join<W>(&mut self,
      search_cfg:   MonteCarloSearchConfig,
      server:       &ParallelMonteCarloSearchServer<W>,
      shared_tree:  SharedTree,
      init_state:   &TxnState<TxnStateNodeData>,
      rng:          &mut Xorshiftplus128Rng)
      -> (MonteCarloSearchResult, MonteCarloSearchStats)
      where W: SearchPolicyWorker
  {
    let num_workers = server.num_workers();
    /*let num_rollouts = (search_cfg.num_rollouts + num_workers - 1) / num_workers * num_workers;
    let worker_num_rollouts = num_rollouts / num_workers;
    let worker_batch_size = (total_batch_size + num_workers - 1) / num_workers;
    let worker_num_batches = (worker_num_rollouts + worker_batch_size - 1) / worker_batch_size;*/
    let (worker_batch_size, worker_num_batches) = search_cfg.worker_dims(num_workers);
    assert!(worker_batch_size >= 1);
    assert!(worker_num_batches >= 1);

    let cfg = SearchWorkerConfig{
      batch_size:       worker_batch_size,
      num_batches:      worker_num_batches,
      // FIXME(20160112): use correct komi value.
      //komi:             7.5,
      //prev_mean_score:  0.0,
    };

    let start_time = get_time();
    for tid in 0 .. num_workers {
      server.enqueue(tid, SearchWorkerCommand::ResetSearch{
        cfg:            cfg,
        shared_tree:    shared_tree.clone(),
        init_state:     init_state.clone(),
        record_search:  true,
      });
    }
    server.join();
    let end_time = get_time();
    let elapsed_ms = (end_time - start_time).num_milliseconds();

    for _ in 0 .. num_workers {
      match server.out_rx.recv().unwrap() {
        SearchWorkerOutput::SearchTrace{tid, search_trace} => {
          self.search_traces.insert(tid, search_trace);
        }
        _ => { unreachable!(); }
      }
    }

    let mut stats: MonteCarloSearchStats = Default::default();
    stats.elapsed_ms = elapsed_ms as usize;

    // FIXME(20160225)
    unimplemented!();
  }
}*/

/*#[derive(Clone, Copy, Debug)]
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
      green_stone:  Stone,
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
      komi:             7.5, // FIXME(20160112): use correct komi value.
      prev_mean_score:  0.0, // FIXME(20160116): use previous score?
    };
    let shared_data = EvalSharedData::new();

    let start_time = get_time();
    for tid in 0 .. num_workers {
      server.enqueue(tid, SearchWorkerCommand::ResetEval{
        cfg:            cfg,
        shared_data:    shared_data.clone(),
        record_trace:   false,
        green_stone:    green_stone,
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
      green_stone:  Stone,
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
      komi:             7.5, // FIXME(20160112): use correct komi value.
      prev_mean_score:  0.0, // FIXME(20160116): use previous score?
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
        green_stone:    green_stone,
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

pub struct ParallelMonteCarloRollout;

impl ParallelMonteCarloRollout {
  pub fn join<W>(&self,
      server:       &ParallelMonteCarloSearchServer<W>,
      batch_size:   usize,
      green_stone:  Stone,
      init_state:   TxnState<TxnStateNodeData>,
      rng:          &mut Xorshiftplus128Rng)
      -> (usize, usize)
      where W: SearchPolicyWorker
  {
    let num_workers = server.num_workers();
    let worker_batch_size = (batch_size + num_workers - 1) / num_workers;

    let stats: RolloutStats = Default::default();
    let cfg = SearchWorkerConfig{
      batch_size:   worker_batch_size,
      num_batches:  1,
      komi:             7.5, // FIXME(20160112): use correct komi value.
      prev_mean_score:  0.0, // FIXME(20160116): use previous score?
    };

    for tid in 0 .. num_workers {
      server.enqueue(tid, SearchWorkerCommand::ResetRollout{
        cfg:            cfg,
        green_stone:    green_stone,
        init_state:     init_state.clone(),
        stats:          stats.clone(),
      });
    }
    server.join();
    (stats.rewards.load(Ordering::Acquire), stats.count.load(Ordering::Acquire))
  }
}

pub struct ParallelMonteCarloRolloutBackup;

impl ParallelMonteCarloRolloutBackup {
  pub fn join<W>(&self,
      server:       &ParallelMonteCarloSearchServer<W>,
      step_size:    f32,
      baseline:     f32,
      batch_size:   usize,
      green_stone:  Stone,
      rng:          &mut Xorshiftplus128Rng)
      where W: SearchPolicyWorker
  {
    let num_workers = server.num_workers();
    let worker_batch_size = (batch_size + num_workers - 1) / num_workers;

    for tid in 0 .. num_workers {
      server.enqueue(tid, SearchWorkerCommand::TraceDescend{
        step_size:      step_size,
        baseline:       baseline,
        batch_size:     worker_batch_size,
        green_stone:    green_stone,
      });
    }
    server.join();
  }
}

pub struct ParallelMonteCarloSaveFriendlyParams;

impl ParallelMonteCarloSaveFriendlyParams {
  pub fn join<W>(&self,
      server:       &ParallelMonteCarloSearchServer<W>)
      where W: SearchPolicyWorker
  {
    let num_workers = server.num_workers();
    for tid in 0 .. num_workers {
      server.enqueue(tid, SearchWorkerCommand::SaveFriendlyRolloutParamsToMem);
    }
    server.join();
  }
}

pub struct ParallelMonteCarloLoadOpponentParams;

impl ParallelMonteCarloLoadOpponentParams {
  pub fn join<W>(&self,
      server:       &ParallelMonteCarloSearchServer<W>,
      params_blob:  Arc<Vec<u8>>)
      where W: SearchPolicyWorker
  {
    let num_workers = server.num_workers();
    for tid in 0 .. num_workers {
      server.enqueue(tid, SearchWorkerCommand::LoadOpponentRolloutParamsFromMem{
        params_blob:    params_blob.clone(),
      });
    }
    server.join();
  }
}*/
