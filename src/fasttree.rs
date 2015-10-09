use fastboard::{PosExt, Pos, Action, Stone, FastBoard, FastBoardAux, FastBoardWork};
use policy::{SearchPolicy, RolloutPolicy};
use random::{random_shuffle};
use table::{TranspositionTable};

use bit_set::{BitSet};
use rand::{Rng, thread_rng};
use std::collections::{HashMap};
use std::iter::{repeat};

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct Id(i64);

pub enum SearchResult {
  Terminal,
  Leaf,
}

pub struct FastSearchNode {
  pub id:           Id,
  pub inst_next:    BitSet,
  pub next:         HashMap<Pos, Id>,
  pub rev_moves:    HashMap<Pos, usize>,
  pub moves:        Vec<Pos>,
  pub state:        FastBoard,
  pub aux_state:    FastBoardAux,
  pub total_trials: f32,
  pub num_trials:   Vec<f32>,
  pub succ_ratio:   Vec<f32>,
  pub value:        Vec<f32>,
}

impl FastSearchNode {
  pub fn new(id: Id, state: FastBoard, aux_state: FastBoardAux) -> FastSearchNode {
    let mut legal_pos: Vec<Pos> = aux_state.get_legal_positions(state.current_turn()).iter()
      .map(|p| p as Pos)
      .collect();
    random_shuffle(&mut legal_pos, &mut thread_rng());
    let mut rev_moves = HashMap::new();
    for (j, &pos) in legal_pos.iter().enumerate() {
      rev_moves.insert(pos, j);
    }
    let num_moves = legal_pos.len();
    FastSearchNode{
      id:           id,
      inst_next:    BitSet::with_capacity(FastBoard::BOARD_SIZE),
      next:         HashMap::new(),
      rev_moves:    rev_moves,
      moves:        legal_pos,
      state:        state,
      aux_state:    aux_state,
      total_trials: 0.0,
      num_trials:   repeat(0.0).take(num_moves).collect(),
      succ_ratio:   repeat(0.0).take(num_moves).collect(),
      value:        repeat(0.0).take(num_moves).collect(),
    }
  }

  fn backup(&mut self, pos: Pos, x: f32, search_policy: &SearchPolicy) {
    let j = self.rev_moves[&pos];
    let total = self.total_trials + 1.0;
    let n = self.num_trials[j] + 1.0;
    let mut mean = self.succ_ratio[j];
    let delta = x - mean;
    mean = mean + delta / n;
    self.total_trials = total;
    self.num_trials[j] = n;
    self.succ_ratio[j] = mean;
    search_policy.backup(self.total_trials, &self.num_trials, &self.succ_ratio, &mut self.value);
  }

  pub fn backup_success(&mut self, pos: Pos, search_policy: &SearchPolicy) {
    self.backup(pos, 1.0, search_policy);
  }

  pub fn backup_failure(&mut self, pos: Pos, search_policy: &SearchPolicy) {
    self.backup(pos, 0.0, search_policy);
  }
}

pub struct FastSearchTree {
  counter:    Id,
  root:       Option<Id>,
  nodes:      HashMap<Id, FastSearchNode>,
  //table:      TranspositionTable,

  path_nodes: Vec<Id>,
  path_moves: Vec<Pos>,
  leaf_move:  Option<Pos>,
  leaf_out:   Option<Stone>,
  roll_moves: Vec<Pos>,
  work:       FastBoardWork,
  tmp_board:  FastBoard,
  tmp_aux:    FastBoardAux,
}

impl FastSearchTree {
  pub fn new() -> FastSearchTree {
    // TODO(20151008)
    FastSearchTree{
      counter:    Id(0),
      root:       None,
      nodes:      HashMap::new(),
      path_nodes: Vec::new(),
      path_moves: Vec::new(),
      leaf_move:  None,
      leaf_out:   None,
      roll_moves: Vec::new(),
      work:       FastBoardWork::new(),
      tmp_board:  FastBoard::new(),
      tmp_aux:    FastBoardAux::new(),
    }
  }

  fn alloc_id(&mut self) -> Id {
    let id = self.counter;
    self.counter.0 += 1;
    id
  }

  pub fn root(&mut self, state: &FastBoard, aux_state: &FastBoardAux) {
    // FIXME(20151008): currently, always reset nodes.
    self.counter.0 = 0;
    self.nodes.clear();
    let id = self.alloc_id();
    let node = FastSearchNode::new(id, state.clone(), aux_state.clone());
    self.root = Some(id);
    self.nodes.insert(id, node);
  }

  pub fn get_root(&self) -> &FastSearchNode {
    &self.nodes[&self.root.unwrap()]
  }

  pub fn expand(&mut self) {
  }

  pub fn walk(&mut self, search_policy: &SearchPolicy) -> SearchResult {
    let mut id = self.root
      .expect("FATAL: search tree missing root, can't walk!");
    self.path_nodes.clear();
    self.path_moves.clear();
    self.leaf_move  = None;
    self.leaf_out = None;
    self.path_nodes.push(id);
    loop {
      let action = search_policy.execute_search(&self.nodes[&id]);
      match action {
        Action::Resign | Action::Pass => { return SearchResult::Terminal; }
        Action::Place{pos} => {
          if self.nodes[&id].inst_next.contains(&pos.idx()) {
            id = self.nodes[&id].next[&pos];
            self.path_nodes.push(id);
            self.path_moves.push(pos);
          } else {
            let next_id = self.alloc_id();
            let mut next_state = self.nodes[&id].state.clone();
            let mut next_aux_state = self.nodes[&id].aux_state.clone();
            let turn = next_state.current_turn();
            {
              let &mut FastSearchTree{ref mut work, ref mut tmp_board, ref mut tmp_aux, ..} = self;
              next_state.play(turn, action, work, &mut Some(&mut next_aux_state), false);
              next_aux_state.update(turn, &next_state, work, tmp_board, tmp_aux);
            }
            let next_node = FastSearchNode::new(next_id, next_state, next_aux_state);
            self.nodes.insert(next_id, next_node);
            self.nodes.get_mut(&id).unwrap().inst_next.insert(pos.idx());
            self.nodes.get_mut(&id).unwrap().next.insert(pos, next_id);
            self.path_nodes.push(next_id);
            self.path_moves.push(pos);
            return SearchResult::Leaf;
          }
        }
      }
    }
  }

  pub fn simulate(&mut self, rollout_policy: &mut RolloutPolicy) {
    let id = self.path_nodes[self.path_nodes.len() - 1];
    let node = &self.nodes[&id];
    let outcome = rollout_policy.execute(&node.state, &node.aux_state);
    self.leaf_out = Some(outcome);
  }

  pub fn backup(&mut self, search_policy: &SearchPolicy) {
    let leaf_id = self.path_nodes.pop()
      .expect("FATAL: current search tree path missing leaf node, can't backup!");
    //let leaf_action = self.leaf_move
    //  .expect("FATAL: current search tree path missing leaf action, can't backup!");
    // TODO(20151009): this backup needs to be changed for RAVE policy.
    let leaf_outcome = self.leaf_out
      .expect("FATAL: current search tree path missing leaf outcome, can't backup!");
    for (&id, &pos) in self.path_nodes.iter().zip(self.path_moves.iter()).rev() {
      // TODO(20151008)
      let mut node = self.nodes.get_mut(&id).unwrap();
      let node_turn = node.state.current_turn();
      if node_turn == leaf_outcome {
        node.backup_success(pos, search_policy);
      } else {
        node.backup_failure(pos, search_policy);
      }
    }
  }
}
