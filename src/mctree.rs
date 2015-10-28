use fastboard::{PosExt, Pos, Action, Stone, FastBoard, FastBoardAux, FastBoardWork};
use policy::{PriorPolicy, SearchPolicy, RolloutPolicy};
use random::{XorShift128PlusRng, arg_choose, choose_without_replace};

use statistics_avx2::array::{array_prefix_sum};
use statistics_avx2::random::{StreamRng};

use bit_set::{BitSet};
use rand::{Rng, SeedableRng, thread_rng};
use std::cell::{RefCell};
use std::cmp::{max, min};
use std::collections::{HashMap};
use std::iter::{repeat};
use std::rc::{Rc, Weak};

#[derive(Clone, Copy, Debug)]
pub enum RolloutBehavior {
  SelectUniform,
  SelectEpsGreedy{eps: f32},
  SelectKGreedy{top_k: usize},
  SampleDiscrete,
}

pub struct McTrajectory {
  pub search_pairs:   Vec<(Rc<RefCell<McNode>>, Pos, usize)>,
  pub leaf_node:      Option<Rc<RefCell<McNode>>>,

  pub rollout_moves:  Vec<(Stone, Pos)>,
  pub score:          Option<f32>,

  state:  FastBoard,
  //work:   FastBoardWork,
}

impl McTrajectory {
  pub fn new() -> McTrajectory {
    McTrajectory{
      search_pairs: Vec::new(),
      leaf_node: None,
      rollout_moves: Vec::new(),
      score: None,
      state: FastBoard::new(),
      //work: FastBoardWork::new(),
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

  pub valid_moves:      Vec<Pos>,
  pub child_nodes:      Vec<Option<Rc<RefCell<McNode>>>>,
  pub total_visits:     f32,
  pub num_trials:       Vec<f32>,
  pub num_succs:        Vec<f32>,
  pub num_trials_prior: Vec<f32>,
  pub num_succs_prior:  Vec<f32>,
  pub num_trials_rave:  Vec<f32>,
  pub num_succs_rave:   Vec<f32>,
  pub values:           Vec<f32>,
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
      num_succs:  repeat(0.0).take(FastBoard::BOARD_SIZE).collect(),
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
      valid_moves:      valid_moves,
      child_nodes:      repeat(None).take(num_moves).collect(),
      total_visits:     0.0,
      num_trials:       repeat(0.0).take(num_moves).collect(),
      num_succs:        repeat(0.0).take(num_moves).collect(),
      num_trials_prior: repeat(0.0).take(num_moves).collect(),
      num_succs_prior:  repeat(0.0).take(num_moves).collect(),
      num_trials_rave:  repeat(0.0).take(num_moves).collect(),
      num_succs_rave:   repeat(0.0).take(num_moves).collect(),
      values:           repeat(0.0).take(num_moves).collect(),
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

  /*pub fn credit_one_arm(&mut self, j: usize, reward: f32) {
    self.num_trials[j] += 1.0;
    self.num_succs[j] += reward;
  }

  pub fn credit_one_arm_rave(&mut self, j: usize, reward: f32) {
    self.num_trials_rave[j] += 1.0;
    self.num_succs_rave[j] += reward;
  }*/

  pub fn get_relative_reward(&self, outcome: f32) -> f32 {
    let turn = self.state.current_turn();
    let reward = if outcome >= 0.0 {
      if let Stone::White = turn { 1.0 } else { 0.0 }
    } else {
      if let Stone::Black = turn { 1.0 } else { 0.0 }
    };
    reward
  }

  pub fn credit_arm(&mut self, j: usize, expected_move: Pos, outcome: f32) {
    if self.valid_moves[j] != expected_move {
      println!("WARNING: thompson policy: node valid mov index {} is not {:?}!",
          j, expected_move);
      return;
    }
    let reward = self.get_relative_reward(outcome);
    self.num_trials[j] += 1.0;
    self.num_succs[j] += reward;
  }

  pub fn credit_arms_rave(&mut self, rave_mask: &[BitSet], outcome: f32) {
    let turn = self.state.current_turn();
    let turn_off = turn.offset();
    let reward = self.get_relative_reward(outcome);
    let &mut McNode{ref valid_moves,
      ref mut num_trials_rave, ref mut num_succs_rave, ..} = self;
    for (j, &mov) in valid_moves.iter().enumerate() {
      if rave_mask[turn_off].contains(&mov.idx()) {
        num_trials_rave[j] += 1.0;
        num_succs_rave[j] += reward;
      }
    }
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
      max_depth:  100, // FIXME(20151024): for debugging.

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

  pub fn walk(&mut self, search_policy: &mut SearchPolicy, traj: &mut McTrajectory) {
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
      //let move_pair = arg_choose(&cursor_node.borrow().valid_moves, &mut self.rng);
      let move_pair = search_policy.execute_search_new(&cursor_node.borrow());
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
            leaf_node.borrow_mut().total_visits = 1.0;
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
  prior_policy:   &'a mut PriorPolicy,
  search_policy:  &'a mut SearchPolicy,
  rollout_policy: &'a mut RolloutPolicy,
  tree:           &'a mut McSearchTree,
  trajs:          Vec<McTrajectory>,
}

impl<'a> McSearchProblem<'a> {
  pub fn new(max_playouts: usize, prior_policy: &'a mut PriorPolicy, search_policy: &'a mut SearchPolicy, rollout_policy: &'a mut RolloutPolicy, tree: &'a mut McSearchTree) -> McSearchProblem<'a> {
    let batch_size = rollout_policy.batch_size();
    let mut trajs = Vec::with_capacity(batch_size);
    for _ in (0 .. batch_size) {
      trajs.push(McTrajectory::new());
    }
    McSearchProblem{
      rng:  XorShift128PlusRng::from_seed([thread_rng().next_u64(), thread_rng().next_u64()]),
      max_playouts:   max_playouts,
      batch_size:     batch_size,
      prior_policy:   prior_policy,
      search_policy:  search_policy,
      rollout_policy: rollout_policy,
      tree:           tree,
      trajs:          trajs,
    }
  }

  pub fn join(mut self) -> Action {
    let batch_size = self.batch_size;
    let num_batches = (self.max_playouts + batch_size - 1) / batch_size;
    let behavior = self.rollout_policy.rollout_behavior();
    println!("DEBUG: mc search problem: num batches: {} batch size: {} behavior: {:?}",
        num_batches, batch_size, behavior);

    let mut work = FastBoardWork::new();
    let mut scores: HashMap<i32, usize> = HashMap::new();

    // TODO(20151028): If equipped with a batch prior, initialize the root.
    //self.prior_policy.preload_batch_state(0, 

    for batch in (0 .. num_batches) {
      for batch_idx in (0 .. batch_size) {
        self.tree.walk(self.search_policy, &mut self.trajs[batch_idx]);
      }

      // TODO(20151028): If equipped with a batch prior, initialize new leaf
      // nodes.

      match behavior {
        RolloutBehavior::SelectUniform => {
          // TODO(20151024)
          for batch_idx in (0 .. batch_size) {
            let mut valid_moves = self.trajs[batch_idx].leaf_node.as_ref().unwrap().borrow()
                .valid_moves.clone();
            let mut depth = 0;
            loop {
              let mut did_play = false;
              while !valid_moves.is_empty() {
                let turn = self.trajs[batch_idx].state.current_turn();
                let pos = choose_without_replace(&mut valid_moves, &mut self.rng);
                if let Some(pos) = pos {
                  if self.trajs[batch_idx].state.is_legal_move_fast(turn, pos) {
                    self.trajs[batch_idx].state.play(turn, Action::Place{pos: pos}, &mut work, &mut None, false);
                    self.trajs[batch_idx].rollout_moves.push((turn, pos));
                    did_play = true;
                    break;
                  }
                }
              }
              if did_play {
                // XXX(20151026): Not playing "under the stones"; although maybe
                // some plays which were not previously valid (due to liberties)
                // may become valid.
                //valid_moves.extend(self.trajs[batch_idx].state.last_captures().iter());
              }
              if valid_moves.is_empty() {
                break;
              }
              depth += 1;
              //if depth >= 210 {
              //if depth >= 240 {
              if depth >= 360 {
                break;
              }
            }
            // FIXME(20151026): when scoring, what komi to use?
            let score = self.trajs[batch_idx].state.score_fast(0.5);
            self.trajs[batch_idx].score = Some(score);
          }
        }

        RolloutBehavior::SelectEpsGreedy{eps} => {
          let mut valid_moves = Vec::new();
          let mut min_valid_moves_len = 1000;
          for batch_idx in (0 .. batch_size) {
            valid_moves.push(self.trajs[batch_idx].leaf_node.as_ref().unwrap().borrow()
              .valid_moves.clone());
            min_valid_moves_len = min(min_valid_moves_len, valid_moves[batch_idx].len());
          }
          let mut valid_plays = 0;
          let mut total_plays = 0;
          let mut num_term = 0;
          let mut depth = 0;
          loop {
            // TODO(20151028): epsilon-greedy part.
            for batch_idx in (0 .. batch_size) {
              self.rollout_policy.preload_batch_state(batch_idx, &self.trajs[batch_idx].state);
            }
            self.rollout_policy.execute_batch(batch_size);
            //let mut greedy_moves = Vec::new();
            //let mut greedy_stones =Vec::new();
            num_term = 0;
            for batch_idx in (0 .. batch_size) {
              if valid_moves[batch_idx].is_empty() {
                num_term += 1;
                continue;
              }
              let greedy_pos = self.rollout_policy.read_policy_argmax(batch_idx) as Pos;
              //greedy_moves.push(greedy_pos);
              //greedy_stones.push(self.trajs[batch_idx].state.get_stone(greedy_pos));
              let turn = self.trajs[batch_idx].state.current_turn();
              if self.trajs[batch_idx].state.is_legal_move_fast(turn, greedy_pos) {
                self.trajs[batch_idx].state.play(turn, Action::Place{pos: greedy_pos}, &mut work, &mut None, false);
                self.trajs[batch_idx].rollout_moves.push((turn, greedy_pos));
                valid_plays += 1;
              } else {
                while !valid_moves[batch_idx].is_empty() {
                  let pos = choose_without_replace(&mut valid_moves[batch_idx], &mut self.rng);
                  if let Some(pos) = pos {
                    if self.trajs[batch_idx].state.is_legal_move_fast(turn, pos) {
                      self.trajs[batch_idx].state.play(turn, Action::Place{pos: pos}, &mut work, &mut None, false);
                      self.trajs[batch_idx].rollout_moves.push((turn, pos));
                      break;
                    }
                  }
                }
              }
              total_plays += 1;
            }
            /*println!("DEBUG: mc search: rollout step: {} greedy moves: {:?}",
                depth, &greedy_moves);
            println!("DEBUG: mc search: rollout step: {} greedy stones: {:?}",
                depth, &greedy_stones);*/
            depth += 1;
            if depth >= 360 {
              break;
            }
            if num_term >= max((batch_size + 1) / 2, batch_size - 4) {
              break;
            }
          }
          println!("DEBUG: mc search: rollout: batch {}/{} depth {} min val moves: {} valid plays {}/{} num term: {}",
              batch, num_batches, depth, min_valid_moves_len, valid_plays, total_plays, num_term);
          for batch_idx in (0 .. batch_size) {
            let score = self.trajs[batch_idx].state.score_fast(0.5);
            self.trajs[batch_idx].score = Some(score);
          }
        }

        RolloutBehavior::SelectKGreedy{top_k} => {
          // TODO(20151024)
          unimplemented!();
        }

        RolloutBehavior::SampleDiscrete => {
          // TODO(20151024)
          unimplemented!();
        }
      }

      for batch_idx in (0 .. batch_size) {
        if let Some(score) = self.trajs[batch_idx].score {
          let int_score = (2.0 * score).round() as i32;
          if !scores.contains_key(&int_score) {
            scores.insert(int_score, 0);
          }
          *scores.get_mut(&int_score).unwrap() += 1;
        }
        self.search_policy.backup_new(&mut self.trajs[batch_idx]);
      }
    }

    println!("DEBUG: mc search problem: scores: {:?}", scores);
    println!("DEBUG: mc search problem: end search: root visits: {}", self.tree.root_node.borrow().total_visits);
    self.search_policy.execute_best_new(&self.tree.root_node.borrow())
  }
}
