use fastboard::{PosExt, Action, FastBoard, FastBoardAux, FastBoardWork};
use policy::{TreePolicy, RolloutPolicy};

use bit_set::{BitSet};
use std::collections::{HashMap};

#[derive(Clone, Copy)]
pub struct SearchEdge<E> where E: Clone + Copy + Default {
  pub next_id:  usize,
  pub action:   Action,
  pub data:     E,
}

impl<E> SearchEdge<E> where E: Clone + Copy + Default {
  pub fn new(next_id: usize, action: Action, init_q: f32) -> SearchEdge<E> {
    SearchEdge{
      next_id:  next_id,
      action:   action,
      data:     Default::default(),
    }
  }
}

pub struct SearchNode<D, E> where D: Default, E: Clone + Copy + Default {
  pub id:           usize,
  pub prev_id:      Option<usize>,
  pub state:        FastBoard,
  pub aux_state:    Option<FastBoardAux>,
  pub depth:        i32,
  pub uninst_pos:   BitSet,
  pub inst_actions: HashMap<usize, Action>,
  pub inst_childs:  HashMap<Action, SearchEdge<E>>,
  pub data:         D,
}

impl<D, E> SearchNode<D, E> where D: Default, E: Clone + Copy + Default {
  pub fn new(prev_id: Option<usize>, id: usize, state: FastBoard, aux_state: FastBoardAux, depth: i32) -> SearchNode<D, E> {
    let legal_pos = aux_state.get_legal_positions(state.current_turn()).clone();
    SearchNode{
      id:           id,
      prev_id:      prev_id,
      state:        state,
      aux_state:    Some(aux_state),
      depth:        depth,
      uninst_pos:   legal_pos,
      inst_actions: HashMap::with_capacity(1),
      inst_childs:  HashMap::with_capacity(1),
      data:         Default::default(),
    }
  }
}

pub enum TreePathResult {
  Terminal{id: usize},
  Leaf{id: usize},
}

pub struct SearchTree<P> where P: TreePolicy {
  pub nodes:    Vec<SearchNode<P::NodeData, P::EdgeData>>,
  // TODO(20151005): need a transposition table for compressing the tree.
}

impl<P> SearchTree<P> where P: TreePolicy {
  pub fn new() -> SearchTree<P> {
    SearchTree{nodes: Vec::new()}
  }

  pub fn reset(&mut self, init_state: &FastBoard, init_aux_state: &Option<FastBoardAux>, init_depth: i32, init_q: f32) {
    self.nodes.clear();
    self.expand(None, init_state.clone(), init_aux_state.clone(), init_depth, init_q);
  }

  pub fn expand(&mut self, prev: Option<(usize, Action)>, next_state: FastBoard, next_aux_state: Option<FastBoardAux>, init_depth: i32, init_q: f32) -> usize {
    let id = self.nodes.len();
    let (prev_id, next_depth) = if let Some((prev_id, action)) = prev {
      let prev_node = &mut self.nodes[prev_id];
      if let Action::Place{pos} = action {
        prev_node.uninst_pos.remove(&pos.idx());
      }
      prev_node.inst_actions.insert(id, action);
      prev_node.inst_childs.insert(action, SearchEdge::new(id, action, init_q));
      (Some(prev_id), prev_node.depth + 1)
    } else {
      (None, init_depth)
    };
    self.nodes.push(SearchNode::new(prev_id, id, next_state, next_aux_state.unwrap(), next_depth));
    id
  }

  pub fn execute_path(&mut self, max_depth: i32, tree_policy: &P, work: &mut FastBoardWork) -> TreePathResult {
    let mut id = 0;
    loop {
      let action = tree_policy.execute(&self.nodes[id]);
      match action {
        Action::Resign | Action::Pass => {
          // TODO(20151004)
          return TreePathResult::Terminal{id: id};
        }
        Action::Place{pos} => {
          if self.nodes[id].depth >= max_depth {
            return TreePathResult::Leaf{id: id};
          }
          if let Some(next_id) = self.nodes[id].inst_childs.get(&action).map(|e| e.next_id) {
            id = next_id;
          } else {
            let mut next_state = self.nodes[id].state.clone();
            let mut next_aux_state = self.nodes[id].aux_state.clone();
            // FIXME(20151006): beware semantics of current_turn(); should probably
            // manually track turn.
            let curr_turn = next_state.current_turn();
            next_state.play(curr_turn, Action::Place{pos: pos}, work, &mut next_aux_state);
            next_state.update(curr_turn, Action::Place{pos: pos}, work, &mut next_aux_state);
            let next_id = self.expand(Some((id, action)), next_state, next_aux_state, 0, 0.0);
            return TreePathResult::Leaf{id: next_id};
          }
        }
      }
    }
  }

  pub fn simulate(&mut self, action: Action, id: usize, tree_policy: &mut P, rollout_policy: &mut RolloutPolicy) {
    let result = {
      let node = &self.nodes[id];
      rollout_policy.execute_rollout(&node.state, node.depth)
    };
    tree_policy.update(result);
    let mut next_id = None;
    let mut update_id = id;
    loop {
      tree_policy.update_tree_node(self, update_id, next_id, action, result);
      match self.nodes[update_id].prev_id {
        Some(prev_id) => {
          next_id = Some(update_id);
          update_id = prev_id;
        }
        None => break,
      }
    }
  }
}
