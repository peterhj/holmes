use fastboard::{PosExt, Pos, Action, Stone, FastBoard, FastBoardAux, FastBoardWork};
use policy::{SearchPolicy, RolloutPolicy};
use random::{XorShift128PlusRng, arg_choose};

use statistics_avx2::array::{array_prefix_sum};
use statistics_avx2::random::{StreamRng};

use bit_set::{BitSet};
use rand::{Rng, SeedableRng, thread_rng};
use std::cell::{RefCell};
use std::collections::{HashMap};
use std::iter::{repeat};
use std::rc::{Rc, Weak};

#[derive(Clone, Copy)]
pub enum RolloutBehavior {
  SelectUniform,
  SelectGreedyK{top_k: usize},
  SampleDiscrete,
}

pub struct McTrajectory {
  pub search_pairs:   Vec<(Rc<RefCell<McNode>>, Pos, usize)>,
  pub leaf_node:      Option<Rc<RefCell<McNode>>>,

  pub rollout_moves:  Vec<Pos>,
  pub score:          Option<f32>,
  state:  FastBoard,
  work:   FastBoardWork,
}

impl McTrajectory {
  pub fn reset(&mut self) {
    self.search_pairs.clear();
    self.leaf_node = None;
    self.rollout_moves.clear();
    self.score = None;
  }

  pub fn init_leaf(&mut self, leaf_node: Rc<RefCell<McNode>>) {
    self.state.clone_from(&leaf_node.borrow().state);
    self.leaf_node = Some(leaf_node.clone());
  }
}

#[derive(Clone)]
pub struct McNode {
  pub parent_node:  Option<Weak<RefCell<McNode>>>,
  pub state:        FastBoard,
  pub aux_state:    FastBoardAux,

  pub valid_moves:  Vec<Pos>,
  pub child_nodes:  Vec<Option<Rc<RefCell<McNode>>>>,
  pub num_visits:   f32,
  pub num_trials:   Vec<f32>,
  pub succ_trials:  Vec<f32>,
}

impl McNode {
  pub fn new() -> McNode {
    McNode{
      parent_node:  None,
      state:        FastBoard::new(),
      aux_state:    FastBoardAux::new(),
      valid_moves:  Vec::with_capacity(FastBoard::BOARD_SIZE),
      child_nodes:  Vec::with_capacity(FastBoard::BOARD_SIZE),
      num_visits:   0.0,
      num_trials:   repeat(0.0).take(FastBoard::BOARD_SIZE).collect(),
      succ_trials:  repeat(0.0).take(FastBoard::BOARD_SIZE).collect(),
    }
  }

  pub fn set<R>(&mut self, parent: Option<Rc<RefCell<McNode>>>, state: &FastBoard, aux: &FastBoardAux, rng: &mut R) where R: Rng {
    self.parent_node = parent.map(|ref node| Rc::downgrade(node));
    self.state.clone_from(state);
    self.aux_state.clone_from(aux);
    self.valid_moves.clear();
    self.valid_moves.extend(
        aux.get_legal_positions(state.current_turn())
          .iter()
          .map(|p| p as Pos)
    );
    rng.shuffle(&mut self.valid_moves);
  }

  pub fn get_turn(&self) -> Stone {
    self.state.current_turn()
  }
}

pub struct McSearchTree {
  rng:  XorShift128PlusRng,

  empty_node: McNode,
  root_node:  Rc<RefCell<McNode>>,
  max_depth:  usize,

  work:       FastBoardWork,
  tmp_board:  FastBoard,
  tmp_aux:    FastBoardAux,
}

impl McSearchTree {
  pub fn new_with_root(root_state: &FastBoard, root_aux: &FastBoardAux) -> McSearchTree {
    let mut rng = XorShift128PlusRng::from_seed([thread_rng().next_u64(), thread_rng().next_u64()]);
    let empty_node = McNode::new();
    let mut root_node = empty_node.clone();
    root_node.set(None, root_state, root_aux, &mut rng);
    McSearchTree{
      rng:  rng,

      empty_node: empty_node,
      root_node:  Rc::new(RefCell::new(root_node)),
      max_depth:  1, // FIXME(20151024): for debugging.

      work:       FastBoardWork::new(),
      tmp_board:  FastBoard::new(),
      tmp_aux:    FastBoardAux::new(),
    }
  }

  pub fn walk(&mut self, search_policy: &SearchPolicy, traj: &mut McTrajectory) {
    traj.reset();
    let mut cursor_node = self.root_node.clone();
    loop {
      // TODO(20151024): select a greedy or random action the cursor node.
      if let Some((mov, j)) = arg_choose(&cursor_node.borrow().valid_moves, &mut self.rng) {
        traj.search_pairs.push((cursor_node.clone(), mov, j));
        if let Some(ref child_node) = cursor_node.child_nodes[j] {
          cursor_node = child_node.clone();
          continue;
        } else {
          // XXX: The j-th child of the cursor node does not yet exist. Either
          // return the cursor node as the leaf node or create a new leaf node.
          if cursor_node.num_visits < 2.0 {
            traj.leaf_node = Some(cursor_node);
            return;
          } else {
            // TODO(20151024): create a new leaf node.
            return;
          }
        }
      } else {
        return;
      }
    }
  }
}
