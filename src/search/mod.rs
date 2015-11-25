use array_util::{array_argmax};
use board::{RuleSet, Board, Stone, Point, Action};
//use features::{TxnStateFeaturesData};
//use policy::{PriorPolicy, SearchPolicy, RolloutPolicy};
use random::{XorShift128PlusRng, choose_without_replace};
use search::policies::{PriorPolicy, TreePolicy, RolloutPolicy};
use txnstate::{TxnState, check_good_move_fast};
use txnstate::extras::{TxnStateNodeData};
use txnstate::features::{TxnStateLibFeaturesData};

use bit_set::{BitSet};
use rand::{Rng, SeedableRng, thread_rng};
use std::cell::{RefCell};
use std::cmp::{max, min};
use std::collections::{HashMap};
use std::iter::{repeat};
use std::io::{Write, stderr};
use std::rc::{Rc, Weak};
use time::{get_time};
use vec_map::{VecMap};

pub mod policies;

#[derive(Default, Debug)]
pub struct SearchStats {
  pub elapsed_ms:       i64,
  pub edge_count:       i32,
  pub inner_edge_count: i32,
  pub term_edge_count:  i32,
  pub old_leaf_count:   i32,
  pub new_leaf_count:   i32,
  pub term_count:       i32,
  pub nonterm_count:    i32,
}

#[derive(Clone)]
pub struct Trajectory {
  pub backup_triples: Vec<(Rc<RefCell<Node>>, Point, usize)>,
  pub leaf_node:      Option<Rc<RefCell<Node>>>,

  pub rollout:    bool,
  pub sim_state:  TxnState<TxnStateLibFeaturesData>,
  //pub sim_state:  TxnState<TxnStateNodeData>,
  pub sim_pairs:  Vec<(Stone, Point)>,

  pub score:      Option<f32>,

  // Temporary variables during rollouts/backups.
  rave_mask:  Vec<BitSet>,
  dead_mask:  Vec<BitSet>,
}

impl Trajectory {
  pub fn new() -> Trajectory {
    Trajectory{
      backup_triples: vec![],
      leaf_node:      None,
      rollout:    false,
      sim_state:  TxnState::new(RuleSet::KgsJapanese.rules(), TxnStateLibFeaturesData::new()),
      //sim_state:  TxnState::new(RuleSet::KgsJapanese.rules(), TxnStateNodeData::new()),
      sim_pairs:  vec![],
      score:      None,
      rave_mask:  vec![
        BitSet::with_capacity(Board::SIZE),
        BitSet::with_capacity(Board::SIZE),
      ],
      dead_mask:  vec![
        BitSet::with_capacity(Board::SIZE),
        BitSet::with_capacity(Board::SIZE),
      ],
    }
  }

  pub fn reset(&mut self) {
    self.backup_triples.clear();
    self.leaf_node = None;
    self.rollout = false;
    self.sim_state.reset();
    self.sim_pairs.clear();
    self.score = None;
    self.rave_mask[0].clear();
    self.rave_mask[1].clear();
  }

  pub fn reset_terminal(&mut self) {
    self.rollout = false;
    self.sim_pairs.clear();
  }

  pub fn reset_rollout(&mut self) {
    self.rollout = true;
    {
      let &mut Trajectory{ref leaf_node, ref mut sim_state, .. } = self;
      //sim_state.shrink_clone_from(&self.leaf_node.as_ref().unwrap().borrow().state);
      let state = &self.leaf_node.as_ref().unwrap().borrow().state;
      sim_state.replace_clone_from(state, state.get_data().features.clone());
    }
    self.sim_pairs.clear();
    self.score = None;
    self.rave_mask[0].clear();
    self.rave_mask[1].clear();
  }

  pub fn score(&mut self) {
    if !self.rollout {
      // FIXME(20151114)
      //unimplemented!();
      self.score = Some(0.0);
    } else {
      // FIXME(20151125): dynamic komi.
      self.score = Some(self.sim_state.current_score_rollout(6.5));
    }
  }
}

pub struct Node {
  pub parent_node:  Option<Weak<RefCell<Node>>>,
  pub state:        TxnState<TxnStateNodeData>,

  pub total_trials:     f32,
  pub horizon:          usize,
  pub child_nodes:      Vec<Option<Rc<RefCell<Node>>>>,
  //pub valid_moves:      Vec<Point>,
  pub prior_moves:      Vec<(Point, f32)>,
  pub arm_indexes:      VecMap<usize>,
  pub num_trials:       Vec<f32>,
  pub num_succs:        Vec<f32>,
  pub num_trials_rave:  Vec<f32>,
  pub num_succs_rave:   Vec<f32>,
  pub values:           Vec<f32>,
}

impl Node {
  pub fn new(parent_node: Option<&Rc<RefCell<Node>>>, state: TxnState<TxnStateNodeData>, prior_policy: &mut PriorPolicy) -> Node {
    let mut valid_moves = vec![];
    state.get_data().legality.fill_legal_points(state.current_turn(), &mut valid_moves);
    let num_arms = valid_moves.len();
    let mut prior_moves = Vec::with_capacity(num_arms);
    prior_policy.fill_prior_probs(&state, &valid_moves, &mut prior_moves);
    prior_moves.sort_by(|left, right| ((right.1 * 1.0e6) as i32).cmp(&((left.1 * 1.0e6) as i32)));
    let mut arm_indexes = VecMap::with_capacity(Board::SIZE);
    for j in (0 .. num_arms) {
      arm_indexes.insert(prior_moves[j].0.idx(), j);
    }
    let child_nodes: Vec<_> = repeat(None).take(num_arms).collect();
    let zeros: Vec<_> = repeat(0.0f32).take(num_arms).collect();
    // XXX: Choose a subset of arms to play.
    let horizon = min(1, num_arms);
    //println!("DEBUG: node horizon: {} top priors: {:?}", horizon, &prior_moves[.. min(10, num_arms)]);
    Node{
      parent_node:  parent_node.map(|node| Rc::downgrade(node)),
      state:        state,
      total_trials:     0.0,
      horizon:          horizon,
      child_nodes:      child_nodes,
      //valid_moves:      valid_moves,
      prior_moves:      prior_moves,
      arm_indexes:      arm_indexes,
      num_trials:       zeros.clone(),
      num_succs:        zeros.clone(),
      num_trials_rave:  zeros.clone(),
      num_succs_rave:   zeros.clone(),
      values:           zeros,
    }
  }

  pub fn is_terminal(&self) -> bool {
    self.child_nodes.is_empty()
  }

  pub fn update_visits(&mut self) {
    // FIXME(20151124): read progressive widening mu from hyperparam file.
    let mu = 2.0f32;
    self.total_trials += 1.0;
    // XXX: Progressive widening.
    self.horizon = min((1.0 + (1.0 + self.total_trials).ln() / mu.ln()).ceil() as usize, self.prior_moves.len());
  }

  pub fn update_arm(&mut self, j: usize, score: f32) {
    let turn = self.state.current_turn();
    self.num_trials[j] += 1.0;
    if (Stone::White == turn && score >= 0.0) ||
        (Stone::Black == turn && score < 0.0) {
      self.num_succs[j] += 1.0;
    }
  }

  pub fn rave_update_arm(&mut self, j: usize, score: f32) {
    let turn = self.state.current_turn();
    self.num_trials_rave[j] += 1.0;
    if (Stone::White == turn && score >= 0.0) ||
        (Stone::Black == turn && score < 0.0) {
      self.num_succs_rave[j] += 1.0;
    }
  }
}

pub enum WalkResult {
  /*Terminal(Rc<RefCell<Node>>),
  NonTerminal(Rc<RefCell<Node>>),*/
  Terminal,
  NonTerminal,
}

pub struct Tree {
  root_node:  Rc<RefCell<Node>>,

  // Accumulators across rollouts/backups.
  dead_count: VecMap<i32>,
}

impl Tree {
  pub fn new(init_state: TxnState<TxnStateNodeData>, prior_policy: &mut PriorPolicy, tree_policy: &mut TreePolicy<R=XorShift128PlusRng>/*, rng: &mut XorShift128PlusRng*/) -> Tree {
    let mut root_node = Rc::new(RefCell::new(Node::new(None, init_state, prior_policy)));
    // FIXME(20151124): calling this in SearchProblem.join() because RNG is
    // initialized there.
    //tree_policy.init_node(&mut *root_node.borrow_mut(), rng);
    Tree{
      root_node:  root_node,
      dead_count: VecMap::with_capacity(Board::SIZE),
    }
  }

  pub fn current_turn(&self) -> Stone {
    self.root_node.borrow().state.current_turn()
  }

  pub fn walk(&mut self,
      prior_policy: &mut PriorPolicy,
      tree_policy: &mut TreePolicy<R=XorShift128PlusRng>,
      traj: &mut Trajectory, stats: &mut SearchStats,
      rng: &mut XorShift128PlusRng)
      -> WalkResult
  {
    traj.reset();

    let mut cursor_node = self.root_node.clone();
    loop {
      // At the cursor node, decide to walk or rollout depending on the total
      // number of trials.
      let cursor_trials = cursor_node.borrow().total_trials;
      if cursor_trials >= 1.0 {
        // Try to walk through the current node using the exploration policy.
        stats.edge_count += 1;
        let res = tree_policy.execute_search(&*cursor_node.borrow(), rng);
        match res {
          Some((place_point, j)) => {
            traj.backup_triples.push((cursor_node.clone(), place_point, j));
            let has_child = cursor_node.borrow().child_nodes[j].is_some();
            if has_child {
              // Existing inner node, simply update the cursor.
              stats.inner_edge_count += 1;
              let child_node = cursor_node.borrow().child_nodes[j].as_ref().unwrap().clone();
              cursor_node = child_node;
            } else {
              // Create a new leaf node and stop the walk.
              stats.new_leaf_count += 1;
              let mut leaf_state = cursor_node.borrow().state.clone();
              let turn = leaf_state.current_turn();
              match leaf_state.try_place(turn, place_point) {
                Ok(_) => { leaf_state.commit(); }
                Err(e) => { panic!("walk failed due to illegal move: {:?}", e); } // XXX: this means the legal moves features gave an incorrect result.
              }
              let mut leaf_node = Rc::new(RefCell::new(Node::new(Some(&cursor_node), leaf_state, prior_policy)));
              tree_policy.init_node(&mut *leaf_node.borrow_mut(), rng);
              cursor_node.borrow_mut().child_nodes[j] = Some(leaf_node.clone());
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
    traj.leaf_node = Some(cursor_node.clone());
    let terminal = cursor_node.borrow().is_terminal();
    if terminal {
      //WalkResult::Terminal(cursor_node.clone())
      WalkResult::Terminal
    } else {
      //WalkResult::NonTerminal(cursor_node.clone())
      WalkResult::NonTerminal
    }
  }

  pub fn backup(&mut self, tree_policy: &mut TreePolicy<R=XorShift128PlusRng>, traj: &mut Trajectory, rng: &mut XorShift128PlusRng) {
    //writeln!(&mut stderr(), "DEBUG: backup idx: {}", i);
    traj.rave_mask[0].clear();
    traj.rave_mask[1].clear();
    let score = traj.score.unwrap();

    if traj.sim_pairs.len() >= 1 {
      assert!(traj.rollout);
      let mut leaf_node = traj.leaf_node.as_ref().unwrap().borrow_mut();
      let leaf_turn = leaf_node.state.current_turn();

      // XXX(20151120): Currently not tracking pass pairs, so need to check that
      // the first rollout pair belongs to the leaf node turn.
      let (update_turn, update_point) = traj.sim_pairs[0];
      if leaf_turn == update_turn {
        if let Some(&update_j) = leaf_node.arm_indexes.get(&update_point.idx()) {
          assert_eq!(update_point, leaf_node.prior_moves[update_j].0);
          leaf_node.update_arm(update_j, score);
        } else {
          writeln!(&mut stderr(), "WARNING: leaf_node arm_indexes does not contain update arm: {:?}", update_point);
          panic!();
        }
      }

      if tree_policy.rave() {
        // TODO(20151113)
        for &(sim_turn, sim_point) in traj.sim_pairs.iter() {
          traj.rave_mask[sim_turn.offset()].insert(sim_point.idx());
        }
        for sim_p in traj.rave_mask[update_turn.offset()].iter() {
          let sim_point = Point::from_idx(sim_p);
          if let Some(&sim_j) = leaf_node.arm_indexes.get(&sim_point.idx()) {
            assert_eq!(sim_point, leaf_node.prior_moves[sim_j].0);
            leaf_node.rave_update_arm(sim_j, score);
          }
        }
      }
    }

    {
      let mut leaf_node = traj.leaf_node.as_ref().unwrap().borrow_mut();
      leaf_node.update_visits();
      tree_policy.backup_values(&mut *leaf_node, rng);
    }

    for &(ref node, update_point, update_j) in traj.backup_triples.iter().rev() {
      let mut node = node.borrow_mut();

      assert_eq!(update_point, node.prior_moves[update_j].0);
      node.update_arm(update_j, score);

      if tree_policy.rave() {
        let update_turn = node.state.current_turn();
        traj.rave_mask[update_turn.offset()].insert(update_point.idx());
        for sim_p in traj.rave_mask[update_turn.offset()].iter() {
          let sim_point = Point::from_idx(sim_p);
          if node.arm_indexes.capacity() < sim_point.idx()+1 {
            panic!("WARNING: node.arm_indexes will overflow (324)!");
          }
          if let Some(&sim_j) = node.arm_indexes.get(&sim_point.idx()) {
            assert_eq!(sim_point, node.prior_moves[sim_j].0);
            node.rave_update_arm(sim_j, score);
          }
        }
      }

      node.update_visits();
      tree_policy.backup_values(&mut *node, rng);
    }
  }
}

pub struct SequentialSearch {
  pub num_rollouts: usize,
  pub stats:        SearchStats,
}

impl SequentialSearch {
  pub fn join(&mut self,
      tree: &mut Tree,
      prior_policy: &mut PriorPolicy,
      tree_policy: &mut TreePolicy<R=XorShift128PlusRng>,
      roll_policy: &mut RolloutPolicy<R=XorShift128PlusRng>)
      -> (Stone, Action)
  {
    let mut slow_rng = thread_rng();
    let mut rng = XorShift128PlusRng::from_seed([slow_rng.next_u64(), slow_rng.next_u64()]);
    let root_turn = tree.current_turn();

    let start_time = get_time();

    // FIXME(20151124): calling this here because RNG is initialized here.
    tree_policy.init_node(&mut *tree.root_node.borrow_mut(), &mut rng);

    let batch_size = roll_policy.batch_size();
    assert!(batch_size >= 1);

    match batch_size {
      1 => {
        let mut traj = Trajectory::new();
        for i in (0 .. self.num_rollouts) {
          match tree.walk(prior_policy, tree_policy, &mut traj, &mut self.stats, &mut rng) {
            WalkResult::Terminal => {
              self.stats.term_count += 1;
              traj.reset_terminal();
              traj.score();
              tree.backup(tree_policy, &mut traj, &mut rng);
            }
            WalkResult::NonTerminal => {
              self.stats.nonterm_count += 1;
              traj.reset_rollout();
              roll_policy.rollout(&mut traj, &mut rng);
              traj.score();
              tree.backup(tree_policy, &mut traj, &mut rng);
            }
          }
        }
      }

      batch_size => {
        let mut trajs: Vec<_> = repeat(Trajectory::new()).take(batch_size).collect();

        let num_batches = (self.num_rollouts + batch_size - 1) / batch_size;
        for batch in (0 .. num_batches) {
          // TODO(20151125): walks can happen in parallel within a batch.
          for batch_idx in (0 .. batch_size) {
            let traj = &mut trajs[batch_idx];
            match tree.walk(prior_policy, tree_policy, traj, &mut self.stats, &mut rng) {
              WalkResult::Terminal => {
                self.stats.term_count += 1;
                traj.reset_terminal();
              }
              WalkResult::NonTerminal => {
                self.stats.nonterm_count += 1;
                traj.reset_rollout();
              }
            }
          }

          roll_policy.rollout_batch(&mut trajs, &mut rng);

          for batch_idx in (0 .. batch_size) {
            let traj = &mut trajs[batch_idx];
            traj.score();
            tree.backup(tree_policy, traj, &mut rng);
          }
        }
      }
    }

    let end_time = get_time();
    let elapsed_ms = (end_time - start_time).num_milliseconds();
    self.stats.elapsed_ms = elapsed_ms;

    if let Some(argmax_j) = array_argmax(&tree.root_node.borrow().num_trials) {
      let root_node = tree.root_node.borrow();
      if root_node.num_succs[argmax_j] / root_node.num_trials[argmax_j] <= 0.001 {
        (root_turn, Action::Resign)
      } else {
        let argmax_point = root_node.prior_moves[argmax_j].0;
        (root_turn, Action::Place{point: argmax_point})
      }
    } else {
      (root_turn, Action::Pass)
    }
  }
}
