use fastboard::{Pos, Action, FastBoard};
use tree::{SearchNode, SearchEdge, SearchTree};

pub trait TreePolicy {
  type NodeData: Default;
  type EdgeData: Clone + Copy + Default;

  /*fn initialize_q_value(&self, state: &FastBoard) -> f32 {
    unimplemented!();
  }*/

  fn reset(&mut self) {
    unimplemented!();
  }

  fn update(&mut self, result: i32) where Self: Sized {
    unimplemented!();
  }

  fn update_tree_node(&self, tree: &mut SearchTree<Self>, update_id: usize, next_id: Option<usize>, leaf_action: Action, result: i32) where Self: Sized {
    unimplemented!();
  }

  fn execute(&self, node: &SearchNode<Self::NodeData, Self::EdgeData>) -> Action {
    unimplemented!();
  }
}

pub struct BenchmarkTreePolicy;

impl BenchmarkTreePolicy {
  pub fn new() -> BenchmarkTreePolicy {
    BenchmarkTreePolicy
  }
}

impl TreePolicy for BenchmarkTreePolicy {
  type NodeData = ();
  type EdgeData = ();

  fn update_tree_node(&self, tree: &mut SearchTree<Self>, update_id: usize, next_id: Option<usize>, leaf_action: Action, result: i32) where Self: Sized {
  }

  fn execute(&self, node: &SearchNode<(), ()>) -> Action {
    for p in node.uninst_pos.iter() {
      return Action::Place{pos: p as Pos};
    }
    Action::Pass
  }
}

#[derive(Clone, Copy, Default)]
pub struct UctNodeData;

#[derive(Clone, Copy, Default)]
pub struct UctEdgeData {
  pub n_plays:  i32,
  pub n_wins:   i32,
  pub q_value:  f32,
}

#[derive(Clone, Copy)]
pub struct UctTreePolicyConfig {
  pub c:          f32,
  pub bias:       f32,
  pub tuned:      bool,
  pub max_plays:  i32,
}

#[derive(Clone, Copy)]
pub struct UctTreePolicy {
  pub config:     UctTreePolicyConfig,
  pub num_plays:  i32,
}

impl UctTreePolicy {
  pub fn new(config: UctTreePolicyConfig) -> UctTreePolicy {
    UctTreePolicy{
      config: config,
      num_plays: 0,
    }
  }

  pub fn update_q_value(&self, s: &<Self as TreePolicy>::NodeData, sa: &<Self as TreePolicy>::EdgeData, old_q_value: f32) -> f32 {
    let exploit_term = (sa.n_wins as f32) / (sa.n_plays as f32);
    let explore_term = match self.config.tuned {
      false => {
        self.config.c * ((self.num_plays as f32).ln() / (self.config.bias + sa.n_plays as f32)).sqrt()
      }
      true  => {
        // TODO
        unimplemented!();
      }
    };
    exploit_term + explore_term
  }
}

impl TreePolicy for UctTreePolicy {
  type NodeData = UctNodeData;
  type EdgeData = UctEdgeData;

  fn reset(&mut self) {
    self.num_plays = 0;
  }

  fn update(&mut self, result: i32) where Self: Sized {
    self.num_plays += 1;
  }

  fn update_tree_node(&self, tree: &mut SearchTree<Self>, update_id: usize, next_id: Option<usize>, leaf_action: Action, result: i32) where Self: Sized {
    let mut update_node = &mut tree.nodes[update_id];
    // Update node stats (n_plays).
    //update_node.data.n_plays += 1;
    // Update edge stats (n_plays, n_wins, q_value).
    if let Some(next_id) = next_id {
      let next_action = update_node.inst_actions.get(&next_id).unwrap();
      let old_q_value = update_node.inst_childs.get(&next_action).unwrap().data.q_value;
      let mut update_edge = update_node.inst_childs.get_mut(&next_action).unwrap();
      update_edge.data.n_plays += 1;
      if result == 1 {
        update_edge.data.n_wins += 1;
      }
      let new_q_value = self.update_q_value(
          &update_node.data,
          &update_edge.data,
          old_q_value);
      update_edge.data.q_value = new_q_value;
    }
  }
}

#[derive(Clone, Copy)]
pub struct UctRaveTreePolicy {
  pub c:      f32,
  pub bias:   f32,
  pub tuned:  bool,
  pub equiv:  f32,
}

impl TreePolicy for UctRaveTreePolicy {
  type NodeData = UctNodeData;
  type EdgeData = UctEdgeData;

  fn update_tree_node(&self, tree: &mut SearchTree<Self>, update_id: usize, next_id: Option<usize>, leaf_action: Action, result: i32) where Self: Sized {
    // TODO(20151003): This particular version is the reason for the strange
    // method signature. RAVE, if I understand it correctly, can create edges
    // (corresponding to the leaf action) when updating the tree.
    unimplemented!();
  }

  fn execute(&self, node: &SearchNode<Self::NodeData, Self::EdgeData>) -> Action {
    // TODO
    unimplemented!();
  }
}

pub trait RolloutPolicy {
  fn execute_rollout(&mut self, init_state: &FastBoard, init_depth: i32) -> i32 {
    unimplemented!();
  }
}

#[derive(Clone)]
pub struct QuasiUniformRolloutPolicy {
  state:  FastBoard,
}

impl QuasiUniformRolloutPolicy {
  pub fn new() -> QuasiUniformRolloutPolicy {
    QuasiUniformRolloutPolicy{
      state:  FastBoard::new(),
    }
  }
}

impl RolloutPolicy for QuasiUniformRolloutPolicy {
  fn execute_rollout(&mut self, init_state: &FastBoard, init_depth: i32) -> i32 {
    self.state.clone_from(init_state);
    // TODO
    unimplemented!();
  }
}

pub struct UniformRolloutPolicy;

impl RolloutPolicy for UniformRolloutPolicy {
}
