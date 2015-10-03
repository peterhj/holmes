use fastboard::{Action, FastBoard};
use tree::{QValueNode, QValueEdge, QValueTree};

pub trait TreePolicy {
  type NodeData: Default;
  type EdgeData: Clone + Copy + Default;

  fn initialize_q_value(&self, state: &FastBoard) -> f32 {
    unimplemented!();
  }

  fn update_tree_node(&self, tree: &mut QValueTree<Self>, update_id: usize, leaf_action: Action, next_id: Option<usize>, result: i32) {
    unimplemented!();
  }

  fn update_q_value(&self, s: &Self::NodeData, sa: &Self::EdgeData, old_q_value: f32) -> f32 {
    unimplemented!();
  }

  //fn execute(&self, node: &QValueNode) -> 
}

#[derive(Clone, Copy, Default)]
pub struct Ucb1NodeData {
  pub n_plays:  i32,
}

#[derive(Clone, Copy, Default)]
pub struct Ucb1EdgeData {
  pub n_plays:  i32,
  pub n_wins:   i32,
}

pub struct Ucb1TreePolicy {
  pub c:      f32,
  pub bias:   f32,
  pub tuned:  bool,
}

impl TreePolicy for Ucb1TreePolicy {
  type NodeData = Ucb1NodeData;
  type EdgeData = Ucb1EdgeData;

  fn update_tree_node(&self, tree: &mut QValueTree<Self>, update_id: usize, leaf_action: Action, next_id: Option<usize>, result: i32) {
    let mut update_node = &mut tree.nodes[update_id];
    // Update node stats (n_plays).
    update_node.data.n_plays += 1;
    // Update edge stats (n_plays, n_wins, q_value).
    if let Some(next_id) = next_id {
      let old_q_value = update_node.childs.get(&next_id).unwrap().q_value;
      let mut update_edge = update_node.childs.get_mut(&next_id).unwrap();
      update_edge.data.n_plays += 1;
      if result == 1 {
        update_edge.data.n_wins += 1;
      }
      let new_q_value = self.update_q_value(
          &update_node.data,
          &update_edge.data,
          old_q_value);
      update_edge.q_value = new_q_value;
    }
  }

  fn update_q_value(&self, s: &Self::NodeData, sa: &Self::EdgeData, old_q_value: f32) -> f32 {
    let exploit_term = (sa.n_wins as f32) / (sa.n_plays as f32);
    let explore_term = match self.tuned {
      false => {
        self.c * ((s.n_plays as f32).ln() / (self.bias + sa.n_plays as f32)).sqrt()
      }
      true  => {
        // TODO
        unimplemented!();
      }
    };
    exploit_term + explore_term
  }
}

pub struct UctRaveTreePolicy {
  pub c:      f32,
  pub bias:   f32,
  pub tuned:  bool,
  pub equiv:  f32,
}

impl TreePolicy for UctRaveTreePolicy {
  type NodeData = Ucb1NodeData;
  type EdgeData = Ucb1EdgeData;

  fn update_tree_node(&self, tree: &mut QValueTree<Self>, update_id: usize, leaf_action: Action, next_id: Option<usize>, result: i32) {
    // TODO(20151003): This particular version is the reason for the strange
    // method signature. RAVE, if I understand it correctly, can create edges
    // (corresponding to the leaf action) when updating the tree.
    unimplemented!();
  }
}

pub trait PlayoutPolicy {
  fn execute_playout(&mut self, init_state: &FastBoard, init_depth: i32) -> i32 {
    unimplemented!();
  }
}

pub struct FastPlayoutPolicy {
  state:  FastBoard,
}

impl PlayoutPolicy for FastPlayoutPolicy {
  fn execute_playout(&mut self, init_state: &FastBoard, init_depth: i32) -> i32 {
    self.state.clone_from(init_state);
    // TODO
    unimplemented!();
  }
}

pub struct UniformPlayoutPolicy;

impl PlayoutPolicy for UniformPlayoutPolicy {
}
