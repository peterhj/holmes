use fastboard::{PosExt, Pos, Action, Stone, FastBoard, FastBoardAux, FastBoardWork};
use policy::{SearchPolicy, RolloutPolicy};
use random::{XorShift128PlusRng, arg_choose, choose_without_replace};

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
  SelectKGreedy{top_k: usize},
  SampleDiscrete,
}

pub struct McTrajectory {
  pub search_pairs:   Vec<(Rc<RefCell<McNode>>, Pos, usize)>,
  pub leaf_node:      Option<Rc<RefCell<McNode>>>,

  state:  FastBoard,
  work:   FastBoardWork,
  pub rollout_moves:  Vec<Pos>,
  pub score:          Option<f32>,
}

impl McTrajectory {
  pub fn new() -> McTrajectory {
    McTrajectory{
      search_pairs: Vec::new(),
      leaf_node: None,
      state: FastBoard::new(),
      work: FastBoardWork::new(),
      rollout_moves: Vec::new(),
      score: None,
    }
  }

  pub fn reset(&mut self) {
    self.search_pairs.clear();
    self.leaf_node = None;
    self.state.reset();
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
  pub total_visits:   f32,
  pub num_trials:   Vec<f32>,
  pub succ_trials:  Vec<f32>,
  pub value:        Vec<f32>,
}

impl McNode {
  /*pub fn new() -> McNode {
    McNode{
      parent_node:  None,
      state:        FastBoard::new(),
      aux_state:    FastBoardAux::new(),
      valid_moves:  Vec::with_capacity(FastBoard::BOARD_SIZE),
      child_nodes:  Vec::with_capacity(FastBoard::BOARD_SIZE),
      total_visits:   0.0,
      num_trials:   repeat(0.0).take(FastBoard::BOARD_SIZE).collect(),
      succ_trials:  repeat(0.0).take(FastBoard::BOARD_SIZE).collect(),
    }
  }*/

  pub fn new<R>(parent: Option<Rc<RefCell<McNode>>>, state: FastBoard, aux: FastBoardAux, rng: &mut R) -> McNode where R: Rng {
    let mut valid_moves: Vec<_> =
        aux.get_legal_positions(state.current_turn())
          .iter()
          .map(|p| p as Pos)
          .collect();
    rng.shuffle(&mut valid_moves);
    let num_moves = valid_moves.len();
    McNode{
      parent_node:  parent.map(|ref node| Rc::downgrade(node)),
      state:        state,
      aux_state:    aux,
      valid_moves:  valid_moves,
      child_nodes:  repeat(None).take(num_moves).collect(),
      total_visits:   0.0,
      num_trials:   repeat(0.0).take(num_moves).collect(),
      succ_trials:  repeat(0.0).take(num_moves).collect(),
      value:        repeat(0.0).take(num_moves).collect(),
    }
  }

  /*pub fn set<R>(&mut self, parent: Option<Rc<RefCell<McNode>>>, state: &FastBoard, aux: &FastBoardAux, rng: &mut R) where R: Rng {
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
  }*/

  pub fn get_turn(&self) -> Stone {
    self.state.current_turn()
  }

  pub fn credit_one_arm(&mut self, j: usize, reward: f32) {
    self.num_trials[j] += 1.0;
    self.succ_trials[j] += reward;
  }
}

pub struct McSearchTree {
  rng:  XorShift128PlusRng,

  //empty_node: McNode,
  root_node:  Rc<RefCell<McNode>>,
  max_depth:  usize,

  work:       FastBoardWork,
  tmp_board:  FastBoard,
  tmp_aux:    FastBoardAux,
}

impl McSearchTree {
  pub fn new_with_root(root_state: FastBoard, root_aux: FastBoardAux) -> McSearchTree {
    let mut rng = XorShift128PlusRng::from_seed([thread_rng().next_u64(), thread_rng().next_u64()]);
    let root_node = McNode::new(None, root_state, root_aux, &mut rng);
    McSearchTree{
      rng:  rng,

      root_node:  Rc::new(RefCell::new(root_node)),
      max_depth:  30, // FIXME(20151024): for debugging.

      work:       FastBoardWork::new(),
      tmp_board:  FastBoard::new(),
      tmp_aux:    FastBoardAux::new(),
    }
  }

  pub fn set_root(&mut self, root_state: FastBoard, root_aux: FastBoardAux) {
    let new_root = McNode::new(None, root_state, root_aux, &mut self.rng);
    self.root_node = Rc::new(RefCell::new(new_root));
  }

  pub fn apply_move(&mut self, turn: Stone, pos: Pos) {
    let mut next_root = None;
    {
      let root_node = self.root_node.borrow();
      for (j, &m) in root_node.valid_moves.iter().enumerate() {
        if m == pos {
          let root_has_jth_child = root_node.child_nodes[j].is_some();
          if root_has_jth_child {
            next_root = root_node.child_nodes[j].clone();
          } else {
            let mut next_state = root_node.state.clone();
            let mut next_aux = root_node.aux_state.clone();
            next_state.play(turn, Action::Place{pos: pos}, &mut self.work, &mut Some(&mut next_aux), false);
            next_aux.update(turn, &next_state, &mut self.work, &mut self.tmp_board, &mut self.tmp_aux);
            let next_node = Rc::new(RefCell::new(McNode::new(None, next_state, next_aux, &mut self.rng)));
            next_root = Some(next_node);
          }
          break;
        }
      }
    }
    if let Some(next_root) = next_root {
      next_root.borrow_mut().parent_node = None;
      self.root_node = next_root;
    } else {
      panic!("FATAL: McSearchTree: move {:?} {:?} is not in root's legal moves!", turn, pos);
    }
  }

  pub fn walk(&mut self, search_policy: &SearchPolicy, traj: &mut McTrajectory) {
    traj.reset();
    let mut cursor_node = self.root_node.clone();
    let mut depth = 0;
    loop {
      cursor_node.borrow_mut().total_visits += 1.0;
      if depth >= self.max_depth {
        // XXX: At max depth, optionally return the cursor node as the leaf node
        // if it has valid moves.
        if !cursor_node.borrow().valid_moves.is_empty() {
          traj.init_leaf(cursor_node);
        }
        return;
      }
      // FIXME(20151024): select a greedy or random action using a policy.
      let move_pair = arg_choose(&cursor_node.borrow().valid_moves, &mut self.rng);
      if let Some((mov, j)) = move_pair {
        traj.search_pairs.push((cursor_node.clone(), mov, j));
        let cursor_has_jth_child = cursor_node.borrow().child_nodes[j].is_some();
        if cursor_has_jth_child {
          let next_cursor_node = cursor_node.borrow().child_nodes[j].clone().unwrap();
          cursor_node = next_cursor_node;
          depth += 1;
          continue;
        } else {
          // XXX: The j-th child of the cursor node does not yet exist. Either
          // return the cursor node as the leaf node or create a new leaf node.
          let cursor_total_visits = cursor_node.borrow().total_visits;
          if cursor_total_visits < 2.0 {
            traj.init_leaf(cursor_node);
            return;
          } else {
            let mut leaf_state = cursor_node.borrow().state.clone();
            let mut leaf_aux = cursor_node.borrow().aux_state.clone();
            let leaf_turn = leaf_state.current_turn();
            leaf_state.play(leaf_turn, Action::Place{pos: mov}, &mut self.work, &mut Some(&mut leaf_aux), false);
            leaf_aux.update(leaf_turn, &leaf_state, &mut self.work, &mut self.tmp_board, &mut self.tmp_aux);
            let mut leaf_node = Rc::new(RefCell::new(McNode::new(Some(cursor_node.clone()), leaf_state, leaf_aux, &mut self.rng)));
            leaf_node.borrow_mut().total_visits += 1.0;
            cursor_node.borrow_mut().child_nodes[j] = Some(leaf_node.clone());
            traj.init_leaf(leaf_node);
            return;
          }
        }
      } else {
        // XXX: No valid moves, i.e., a terminal node.
        return;
      }
    }
  }
}

pub struct McSearchProblem<'a> {
  rng:  XorShift128PlusRng,
  max_playouts:   usize,
  batch_size:     usize,
  search_policy:  &'a SearchPolicy,
  rollout_policy: &'a mut RolloutPolicy,
  tree:           &'a mut McSearchTree,
  trajs:          Vec<McTrajectory>,
}

impl<'a> McSearchProblem<'a> {
  pub fn new(max_playouts: usize, search_policy: &'a SearchPolicy, rollout_policy: &'a mut RolloutPolicy, tree: &'a mut McSearchTree) -> McSearchProblem<'a> {
    let batch_size = rollout_policy.batch_size();
    let mut trajs = Vec::with_capacity(batch_size);
    for _ in (0 .. batch_size) {
      trajs.push(McTrajectory::new());
    }
    McSearchProblem{
      rng:  XorShift128PlusRng::from_seed([thread_rng().next_u64(), thread_rng().next_u64()]),
      max_playouts:   max_playouts,
      batch_size:     batch_size,
      search_policy:  search_policy,
      rollout_policy: rollout_policy,
      tree:           tree,
      trajs:          trajs,
    }
  }

  pub fn join(mut self) -> Action {
    let batch_size = self.batch_size;
    let num_batches = (self.max_playouts + batch_size - 1) / batch_size;
    let behavior = RolloutBehavior::SelectUniform;

    for batch in (0 .. num_batches) {
      // TODO: walk tree.
      for batch_idx in (0 .. batch_size) {
        self.tree.walk(self.search_policy, &mut self.trajs[batch_idx]);
      }

      match behavior {
        RolloutBehavior::SelectUniform => {
          // TODO(20151024)
          let mut work = FastBoardWork::new();
          let mut valid_moves = self.trajs[0].leaf_node.as_ref().unwrap().borrow()
              .valid_moves.clone();
          let mut depth = 0;
          loop {
            let mut did_play = false;
            while !valid_moves.is_empty() {
              let turn = self.trajs[0].state.current_turn();
              let pos = choose_without_replace(&mut valid_moves, &mut self.rng);
              if let Some(pos) = pos {
                if self.trajs[0].state.is_legal_move_fast(turn, pos) {
                  self.trajs[0].state.play(turn, Action::Place{pos: pos}, &mut work, &mut None, false);
                  self.trajs[0].rollout_moves.push(pos);
                  did_play = true;
                  break;
                }
              }
            }
            if did_play {
              // FIXME(20151024): there are other moves which are also valid...
              valid_moves.extend(self.trajs[0].state.last_captures().iter());
            }
            if valid_moves.is_empty() {
              break;
            }
            depth += 1;
            if depth >= 210 {
              break;
            }
          }
          self.trajs[0].score = Some(self.trajs[0].state.score_fast(6.5));
        }

        RolloutBehavior::SelectKGreedy{top_k} => {
          loop {
            // TODO: prepare states.
            for batch_idx in (0 .. batch_size) {
            }

            // TODO: execute rollout policy.

            // TODO: transition states.
            for batch_idx in (0 .. batch_size) {
            }
          }
        }

        RolloutBehavior::SampleDiscrete => {
          // TODO(20151024)
          unimplemented!();
        }
      }

      // TODO: backup tree.
      for batch_idx in (0 .. batch_size) {
        self.search_policy.backup_new(&mut self.trajs[batch_idx]);
      }
    }

    self.search_policy.execute_best_new(&self.tree.root_node.borrow())
  }
}
