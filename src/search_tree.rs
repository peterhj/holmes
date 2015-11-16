use board::{RuleSet, Board, Stone, Point, Action};
//use features::{TxnStateFeaturesData};
//use policy::{PriorPolicy, SearchPolicy, RolloutPolicy};
use random::{XorShift128PlusRng, choose_without_replace};
use search_policies::{PriorPolicy, TreePolicy, RolloutPolicy};
use txnstate::{TxnState, TxnStateNodeData};

use statistics_avx2::array::{array_argmax};
use statistics_avx2::random::{StreamRng};

use bit_set::{BitSet};
use rand::{Rng, SeedableRng, thread_rng};
use std::cell::{RefCell};
use std::cmp::{max, min};
use std::collections::{HashMap};
use std::iter::{repeat};
use std::rc::{Rc, Weak};
use vec_map::{VecMap};

#[derive(Default, Debug)]
pub struct SearchStats {
  pub edge_count:       i32,
  pub inner_edge_count: i32,
  pub term_edge_count:  i32,
  pub old_leaf_count:   i32,
  pub new_leaf_count:   i32,
  pub term_count:       i32,
  pub nonterm_count:    i32,
}

pub struct Trajectory {
  pub backup_triples: Vec<(Rc<RefCell<Node>>, Point, usize)>,
  pub leaf_node:      Option<Rc<RefCell<Node>>>,

  pub rollout:    bool,
  pub sim_state:  TxnState,
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
      sim_state:  TxnState::new(RuleSet::KgsJapanese.rules(), ()),
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
  }

  pub fn init_rollout(&mut self, state: &TxnState<TxnStateNodeData>) {
    self.rollout = true;
    self.sim_state.shrink_clone_from(state);
  }

  pub fn score(&mut self) {
    if !self.rollout {
      // FIXME(20151114)
      //unimplemented!();
      self.score = Some(0.0);
    } else {
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
    let mu = 2.0f32;
    self.total_trials += 1.0;
    // XXX: Progressive widening.
    self.horizon = min((1.0 + (1.0 + self.total_trials).ln() / mu.ln()).ceil() as usize, self.child_nodes.len());
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
  Terminal(Rc<RefCell<Node>>),
  Leaf(Rc<RefCell<Node>>),
}

pub struct Tree {
  root_node:  Rc<RefCell<Node>>,

  // Accumulators across rollouts/backups.
  dead_count: VecMap<i32>,
}

impl Tree {
  pub fn new(init_state: TxnState<TxnStateNodeData>, prior_policy: &mut PriorPolicy, tree_policy: &mut TreePolicy) -> Tree {
    let mut root_node = Rc::new(RefCell::new(Node::new(None, init_state, prior_policy)));
    tree_policy.init_node(&mut *root_node.borrow_mut());
    Tree{
      root_node:  root_node,
      dead_count: VecMap::with_capacity(Board::SIZE),
    }
  }

  pub fn current_turn(&self) -> Stone {
    self.root_node.borrow().state.current_turn()
  }

  pub fn walk(&mut self, prior_policy: &mut PriorPolicy, tree_policy: &mut TreePolicy, traj: &mut Trajectory, stats: &mut SearchStats) -> WalkResult {
    traj.reset();

    let mut cursor_node = self.root_node.clone();
    loop {
      // At the cursor node, decide to walk or rollout depending on the total
      // number of trials.
      let cursor_trials = cursor_node.borrow().total_trials;
      if cursor_trials >= 1.0 {
        // Try to walk through the current node using the exploration policy.
        stats.edge_count += 1;
        let res = tree_policy.execute_search(&*cursor_node.borrow());
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
              tree_policy.init_node(&mut *leaf_node.borrow_mut());
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
      WalkResult::Terminal(cursor_node.clone())
    } else {
      WalkResult::Leaf(cursor_node.clone())
    }
  }

  pub fn backup(&mut self, tree_policy: &mut TreePolicy, traj: &mut Trajectory) {
    traj.rave_mask[0].clear();
    traj.rave_mask[1].clear();
    let score = match traj.score {
      Some(score) => score,
      None => panic!("Trajectory failed to score!"),
    };

    if traj.sim_pairs.len() >= 1 {
      let mut leaf_node = traj.leaf_node.as_ref().unwrap().borrow_mut();

      let (update_turn, update_point) = traj.sim_pairs[0];
      /*let mut update_j = None;
      for j in (0 .. leaf_node.prior_moves.len()) {
        if leaf_node.prior_moves[j].0 == update_point {
          update_j = Some(j);
          break;
        }
      }
      assert!(update_j.is_some());
      let update_j = update_j.unwrap();*/
      let update_j = leaf_node.arm_indexes[update_point.idx()];
      assert_eq!(update_point, leaf_node.prior_moves[update_j].0);
      leaf_node.update_visits();
      leaf_node.update_arm(update_j, score);

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

    for &(ref node, update_point, update_j) in traj.backup_triples.iter().rev() {
      let mut node = node.borrow_mut();

      assert_eq!(update_point, node.prior_moves[update_j].0);
      node.update_visits();
      node.update_arm(update_j, score);

      if tree_policy.rave() {
        let update_turn = node.state.current_turn();
        traj.rave_mask[update_turn.offset()].insert(update_point.idx());
        for sim_p in traj.rave_mask[update_turn.offset()].iter() {
          let sim_point = Point::from_idx(sim_p);
          if let Some(&sim_j) = node.arm_indexes.get(&sim_point.idx()) {
            assert_eq!(sim_point, node.prior_moves[sim_j].0);
            node.rave_update_arm(sim_j, score);
          }
        }
      }
    }
  }
}

pub struct SequentialSearch {
  pub num_rollouts: usize,
  pub stats:        SearchStats,
}

impl SequentialSearch {
  pub fn join(&mut self, tree: &mut Tree, traj: &mut Trajectory, prior_policy: &mut PriorPolicy, tree_policy: &mut TreePolicy, roll_policy: &mut RolloutPolicy) -> (Stone, Action) {
    let mut slow_rng = thread_rng();
    let mut rng = XorShift128PlusRng::from_seed([slow_rng.next_u64(), slow_rng.next_u64()]);
    let root_turn = tree.current_turn();

    for i in (0 .. self.num_rollouts) {
      match tree.walk(prior_policy, tree_policy, traj, &mut self.stats) {
        WalkResult::Terminal(node) => {
          //println!("DEBUG: terminal node");
          self.stats.term_count += 1;
          //node.borrow_mut().update_visits();
          traj.score();
          tree.backup(tree_policy, traj);
        }
        WalkResult::Leaf(node) => {
          //println!("DEBUG: leaf node");
          self.stats.nonterm_count += 1;

          //node.borrow_mut().update_visits();
          traj.init_rollout(&node.borrow().state);

          // TODO(20151112): do rollout.
          let mut valid_moves = [vec![], vec![]];
          node.borrow().state.get_data().legality.fill_legal_points(Stone::Black, &mut valid_moves[0]);
          node.borrow().state.get_data().legality.fill_legal_points(Stone::White, &mut valid_moves[1]);
          let mut sim_turn = traj.sim_state.current_turn();
          let mut sim_pass = [false, false];
          for _ in (0 .. 400) {
            sim_pass[sim_turn.offset()] = false;
            let mut made_move = false;
            while !valid_moves[sim_turn.offset()].is_empty() {
              if let Some(point) = choose_without_replace(&mut valid_moves[sim_turn.offset()], &mut rng) {
                if traj.sim_state.try_place(sim_turn, point).is_ok() {
                  traj.sim_state.commit();
                  traj.sim_pairs.push((sim_turn, point));
                  made_move = true;
                  break;
                } else {
                  traj.sim_state.undo();
                }
              }
            }
            if !made_move {
              sim_pass[sim_turn.offset()] = true;
            }
            if sim_pass[0] && sim_pass[1] {
              break;
            }
            sim_turn = sim_turn.opponent();
          }

          traj.score();
          tree.backup(tree_policy, traj);
        }
      }
    }

    if let Some(argmax_j) = array_argmax(&tree.root_node.borrow().num_trials) {
      let argmax_point = tree.root_node.borrow().prior_moves[argmax_j].0;
      (root_turn, Action::Place{point: argmax_point})
    } else {
      (root_turn, Action::Pass)
    }
  }
}
