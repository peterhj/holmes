use board::{RuleSet, Stone, Point, Action};
//use features::{TxnStateFeaturesData};
//use policy::{PriorPolicy, SearchPolicy, RolloutPolicy};
use random::{XorShift128PlusRng, arg_choose, choose_without_replace, sample_discrete_cdf};
use search_policies::{TreePolicy, RolloutPolicy};
use txnstate::{TxnState, TxnStateNodeData};

use statistics_avx2::array::{array_prefix_sum};
use statistics_avx2::random::{StreamRng};

use bit_set::{BitSet};
use rand::{Rng, SeedableRng, thread_rng};
use std::cell::{RefCell};
use std::cmp::{max, min};
use std::collections::{HashMap};
use std::iter::{repeat};
use std::rc::{Rc, Weak};

pub struct Trajectory {
  pub backup_triples: Vec<(Rc<RefCell<Node>>, Point, usize)>,
  pub leaf_node:      Option<Rc<RefCell<Node>>>,

  pub sim_state:  TxnState,
  pub sim_pairs:  Vec<(Stone, Point)>,
  pub score:      Option<f32>,
}

impl Trajectory {
  pub fn new() -> Trajectory {
    Trajectory{
      backup_triples: vec![],
      leaf_node:      None,
      sim_state:      TxnState::new(RuleSet::KgsJapanese.rules(), ()),
      sim_pairs:      vec![],
      score:          None,
    }
  }

  pub fn reset(&mut self) {
    self.backup_triples.clear();
    self.leaf_node = None;

    self.sim_state.reset();
    self.sim_pairs.clear();
    self.score = None;
  }

  pub fn init_rollout(&mut self, state: &TxnState<TxnStateNodeData>) {
    self.sim_state.shrink_clone_from(state);
  }
}

pub struct Node {
  pub parent_node:  Option<Weak<RefCell<Node>>>,
  pub state:        TxnState<TxnStateNodeData>,

  pub total_trials:     f32,
  pub child_nodes:      Vec<Option<Rc<RefCell<Node>>>>,
  pub valid_moves:      Vec<Point>,
  pub num_trials:       Vec<f32>,
  pub num_succs:        Vec<f32>,
  pub num_trials_rave:  Vec<f32>,
  pub num_succs_rave:   Vec<f32>,
  pub values:           Vec<f32>,
}

impl Node {
  pub fn new(parent_node: Option<&Rc<RefCell<Node>>>, state: TxnState<TxnStateNodeData>) -> Node {
    Node{
      parent_node:  parent_node.map(|node| Rc::downgrade(node)),
      state:        state,

      total_trials: 0.0,
      child_nodes:  vec![],
      valid_moves:  vec![],
      num_trials:   vec![],
      num_succs:    vec![],
      num_trials_rave:  vec![],
      num_succs_rave:   vec![],
      values:       vec![],
    }
  }

  pub fn is_terminal(&self) -> bool {
    // TODO(20151111)
    unimplemented!();
  }

  pub fn visit(&mut self) {
    self.total_trials += 1.0;
  }

  pub fn update_arm(&mut self, j: usize, score: f32) {
    self.num_trials[j] += 1.0;
    if Stone::White == self.state.current_turn() && score >= 0.0 {
      self.num_succs[j] += 1.0;
    } else if Stone::Black == self.state.current_turn() && score < 0.0 {
      self.num_succs[j] += 1.0;
    }
  }

  pub fn rave_update_arm(&mut self, j: usize, score: f32) {
    self.num_trials_rave[j] += 1.0;
    if Stone::White == self.state.current_turn() && score >= 0.0 {
      self.num_succs_rave[j] += 1.0;
    } else if Stone::Black == self.state.current_turn() && score < 0.0 {
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
}

impl Tree {
  pub fn new(init_state: TxnState<TxnStateNodeData>) -> Tree {
    let mut root_node = Rc::new(RefCell::new(Node::new(None, init_state)));
    Tree{
      root_node:  root_node,
    }
  }

  pub fn current_turn(&self) -> Stone {
    self.root_node.borrow().state.current_turn()
  }

  pub fn walk(&mut self, tree_policy: &mut TreePolicy, traj: &mut Trajectory) -> WalkResult {
    traj.reset();

    let mut cursor_node = self.root_node.clone();
    loop {
      // At the cursor node, decide to walk or rollout depending on the total
      // number of trials.
      let cursor_trials = cursor_node.borrow().total_trials;
      if cursor_trials >= 1.0 {
        // Try to walk through the current node using the exploration policy.
        let res = tree_policy.execute_search(&*cursor_node.borrow());
        match res {
          Some((place_point, j)) => {
            traj.backup_triples.push((cursor_node.clone(), place_point, j));
            let has_child = cursor_node.borrow().child_nodes[j].is_some();
            if has_child {
              // Existing inner node, simply update the cursor.
              let child_node = cursor_node.borrow().child_nodes[j].as_ref().unwrap().clone();
              cursor_node = child_node;
            } else {
              // Create a new leaf node and stop the walk.
              let mut leaf_state = cursor_node.borrow().state.clone();
              let turn = leaf_state.current_turn();
              match leaf_state.try_place(turn, place_point) {
                Ok(_) => { leaf_state.commit(); }
                Err(e) => { panic!("walk failed due to illegal move: {:?}", e); }
              }
              let mut leaf_node = Rc::new(RefCell::new(Node::new(Some(&cursor_node), leaf_state)));
              tree_policy.init_node(&mut *leaf_node.borrow_mut());
              cursor_node.borrow_mut().child_nodes[j] = Some(leaf_node.clone());
              cursor_node = leaf_node;
              break;
            }
          }
          None => {
            // Terminal node, stop the walk.
            break;
          }
        }
      } else {
        // Not enough trials, stop the walk and do a rollout.
        break;
      }
    }

    // TODO(20151111): if the cursor node is terminal, score it now and backup;
    // otherwise do a rollout and then backup.
    let terminal = cursor_node.borrow().is_terminal();
    if terminal {
      WalkResult::Terminal(cursor_node.clone())
    } else {
      WalkResult::Leaf(cursor_node.clone())
    }
  }

  pub fn backup(&mut self, tree_policy: &mut TreePolicy, traj: &Trajectory) {
    if tree_policy.rave() {
    }

    for &(ref node, place_point, arm) in traj.backup_triples.iter().rev() {
    }

    // TODO(20151112)
    unimplemented!();
  }
}

pub struct SequentialSearch {
  pub num_rollouts: usize,
}

impl SequentialSearch {
  pub fn join(&self, tree: &mut Tree, traj: &mut Trajectory, tree_policy: &mut TreePolicy, roll_policy: &mut RolloutPolicy) -> (Stone, Action) {
    let mut slow_rng = thread_rng();
    let mut rng = XorShift128PlusRng::from_seed([slow_rng.next_u64(), slow_rng.next_u64()]);
    let root_turn = tree.current_turn();
    for i in (0 .. self.num_rollouts) {
      match tree.walk(tree_policy, traj) {
        WalkResult::Terminal(node) => {
          node.borrow_mut().visit();
        }
        WalkResult::Leaf(node) => {
          node.borrow_mut().visit();
          traj.init_rollout(&node.borrow().state);
          // TODO(20151112): do rollout.
          tree.backup(tree_policy, traj);
        }
      }
    }
    // FIXME(20151112)
    (root_turn, Action::Pass)
  }
}
