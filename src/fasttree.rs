use fastboard::{PosExt, Pos, Action, Stone, FastBoard, FastBoardAux, FastBoardWork};
use policy::{SearchPolicy, RolloutPolicy};
use table::{TranspositionTable};

use bit_set::{BitSet};
use std::collections::{HashMap};

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct Id(i64);

pub struct FastSearchNode {
  pub id:           Id,
  pub prev:         Vec<Id>,
  pub inst_next:    BitSet,
  pub next:         HashMap<Pos, Id>,
  pub moves:        Vec<Pos>,
  pub state:        FastBoard,
  pub aux_state:    FastBoardAux,
  pub num_trials:   f32,
  pub e_num_trials: Vec<f32>,
  pub e_succ_ratio: Vec<f32>,
  pub value:        Vec<f32>,
}

impl FastSearchNode {
  pub fn new(parent: Option<Id>, id: Id, state: FastBoard, aux_state: FastBoardAux) -> FastSearchNode {
    // TODO(20151008)
    unimplemented!();
  }
}

pub struct FastSearchTree {
  counter:  Id,
  root:     Option<Id>,
  nodes:    HashMap<Id, FastSearchNode>,
  table:    TranspositionTable,
  work:     FastBoardWork,
}

impl FastSearchTree {
  pub fn new() -> FastSearchTree {
    // TODO(20151008)
    unimplemented!();
  }

  fn alloc_id(&mut self) -> Id {
    let id = self.counter;
    self.counter.0 += 1;
    id
  }

  pub fn root(&mut self, state: &FastBoard, aux_state: &FastBoardAux) {
    // TODO(20151008): set the root node, creating it if necessary.
  }

  pub fn expand(&mut self) {
  }

  pub fn walk(&mut self, search_policy: &SearchPolicy) {
    let mut id = self.root.expect("FATAL: search tree missing root, can't walk!");
    loop {
      let action = search_policy.execute_search(&self.nodes[&id]);
      match action {
        Action::Resign | Action::Pass => { return; }
        Action::Place{pos} => {
          if self.nodes[&id].inst_next.contains(&pos.idx()) {
            id = self.nodes[&id].next[&pos];
          } else {
            let next_id = self.alloc_id();
            let mut next_state = self.nodes[&id].state.clone();
            let mut next_aux_state = Some(self.nodes[&id].aux_state.clone());
            let turn = next_state.current_turn();
            next_state.play(turn, action, &mut self.work, &mut next_aux_state);
            next_state.update(turn, action, &mut self.work, &mut next_aux_state);
            let next_node = FastSearchNode::new(Some(id), next_id, next_state, next_aux_state.unwrap());
            self.nodes.insert(next_id, next_node);
            self.nodes.get_mut(&id).unwrap().inst_next.insert(pos.idx());
            self.nodes.get_mut(&id).unwrap().next.insert(pos, next_id);
            return;
          }
        }
      }
    }
  }

  pub fn simulate(&mut self, id: Id, rollout_policy: &mut RolloutPolicy) -> Stone {
    let node = &self.nodes[&id];
    let result = rollout_policy.execute_rollout(&node.state, &node.aux_state);
    result
  }

  pub fn backprop(&mut self) {
    loop {
      // TODO(20151008)
    }
  }
}
