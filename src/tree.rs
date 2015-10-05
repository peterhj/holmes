//use contains::hash_map::{OpenCopyHashMap};
use fastboard::{IntoIndex, Action, FastBoard, FastBoardAux, FastBoardWork};
use policy::{TreePolicy, PlayoutPolicy};

use bit_set::{BitSet};
use bit_vec::{BitVec};
use std::collections::{HashMap};
//use std::mem::{size_of};

#[derive(Clone, Copy)]
pub struct QValueEdge<E> where E: Clone + Copy + Default {
  pub next_id:  usize,
  pub action:   Action,
  pub q_value:  f32,
  pub data:     E,
}

impl<E> QValueEdge<E> where E: Clone + Copy + Default {
  pub fn new(next_id: usize, action: Action, init_q: f32) -> QValueEdge<E> {
    QValueEdge{
      next_id:  next_id,
      action:   action,
      q_value:  init_q,
      data:     Default::default(),
    }
  }
}

pub struct QValueNode<D, E> where D: Default, E: Clone + Copy + Default {
  pub id:           usize,
  pub prev_id:      Option<usize>,
  pub state:        FastBoard,
  pub aux_state:    Option<FastBoardAux>,
  pub depth:        i32,
  pub legal_pos:    BitSet,
  pub uninst_pos:   BitSet,
  //pub inst_actions: OpenCopyHashMap<usize, Action>,
  //pub inst_childs:  OpenCopyHashMap<Action, QValueEdge<E>>,
  pub inst_actions: HashMap<usize, Action>,
  pub inst_childs:  HashMap<Action, QValueEdge<E>>,
  pub data:         D,
}

impl<D, E> QValueNode<D, E> where D: Default, E: Clone + Copy + Default {
  pub fn new(prev_id: Option<usize>, id: usize, state: FastBoard, aux_state: FastBoardAux, depth: i32) -> QValueNode<D, E> {
    let legal_pos = aux_state.mark_legal_positions(state.current_turn(), &state);
    QValueNode{
      id:           id,
      prev_id:      prev_id,
      state:        state,
      aux_state:    Some(aux_state),
      depth:        depth,
      legal_pos:    legal_pos.clone(),
      uninst_pos:   legal_pos,
      //inst_actions: OpenCopyHashMap::with_capacity(1),
      //inst_childs:  OpenCopyHashMap::with_capacity(1),
      inst_actions: HashMap::with_capacity(1),
      inst_childs:  HashMap::with_capacity(1),
      data:         Default::default(),
    }
  }
}

pub enum TreePathResult {
  Terminal,
  Leaf,
}

pub struct QValueTree<P> where P: TreePolicy {
  pub nodes:    Vec<QValueNode<P::NodeData, P::EdgeData>>,
}

impl<P> QValueTree<P> where P: TreePolicy {
  pub fn new() -> QValueTree<P> {
    QValueTree{nodes: Vec::new()}
  }

  pub fn reset(&mut self, init_state: &FastBoard, init_aux_state: &Option<FastBoardAux>, init_depth: i32, init_q: f32) {
    self.nodes.clear();
    self.expand(None, init_state.clone(), init_aux_state.clone(), init_depth, init_q);
  }

  pub fn expand(&mut self, prev: Option<(usize, Action)>, next_state: FastBoard, next_aux_state: Option<FastBoardAux>, init_depth: i32, init_q: f32) {
    let id = self.nodes.len();
    let (prev_id, next_depth) = if let Some((prev_id, action)) = prev {
      let prev_node = &mut self.nodes[prev_id];
      if let Action::Place{pos} = action {
        prev_node.uninst_pos.remove(&pos.idx());
      }
      prev_node.inst_actions.insert(id, action);
      prev_node.inst_childs.insert(action, QValueEdge::new(id, action, init_q));
      (Some(prev_id), prev_node.depth + 1)
    } else {
      (None, init_depth)
    };
    self.nodes.push(QValueNode::new(prev_id, id, next_state, next_aux_state.unwrap(), next_depth));
  }

  pub fn execute_path(&mut self, max_depth: i32, tree_policy: &P, work: &mut FastBoardWork) -> TreePathResult {
    let mut id = 0;
    loop {
      let action = tree_policy.execute(&self.nodes[id]);
      match action {
        Action::Resign | Action::Pass => {
          // TODO
          return TreePathResult::Terminal;
        }
        Action::Place{pos} => {
          if self.nodes[id].depth >= max_depth {
            return TreePathResult::Leaf;
          }
          if let Some(next_id) = self.nodes[id].inst_childs.get(&action).map(|e| e.next_id) {
            id = next_id;
          } else {
            let mut state = self.nodes[id].state.clone();
            let mut aux_state = self.nodes[id].aux_state.clone();
            state.play_turn(Action::Place{pos: pos}, work, &mut aux_state);
            self.expand(Some((id, action)), state, aux_state, 0, 0.0);
            return TreePathResult::Leaf;
          }
        }
      }
    }
  }

  pub fn simulate(&mut self, action: Action, id: usize, tree_policy: &P, playout_policy: &mut PlayoutPolicy) {
    let result = {
      let node = &self.nodes[id];
      playout_policy.execute_playout(&node.state, node.depth)
    };
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
