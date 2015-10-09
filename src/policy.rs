use fastboard::{PosExt, Pos, Action, Stone, FastBoard, FastBoardAux, FastBoardWork};
use fasttree::{FastSearchNode};
use random::{XorShift128PlusRng, random_sample_without_replace};
use tree::{SearchNode, SearchEdge, SearchTree};

use rand::{Rng, SeedableRng};

pub trait TreePolicy {
  type NodeData: Default;
  type EdgeData: Clone + Copy + Default;

  fn reset(&mut self) {
    unimplemented!();
  }

  fn update(&mut self, result: Stone) where Self: Sized {
    unimplemented!();
  }

  fn update_tree_node(&self, tree: &mut SearchTree<Self>, update_id: usize, next_id: Option<usize>, leaf_action: Action, result: Stone) where Self: Sized {
    unimplemented!();
  }

  fn execute(&self, node: &SearchNode<Self::NodeData, Self::EdgeData>) -> Action {
    unimplemented!();
  }

  fn choose_best(&self, node: &SearchNode<Self::NodeData, Self::EdgeData>) -> Action {
    unimplemented!();
  }

  fn execute_search(&self, node: &FastSearchNode) -> Action {
    unimplemented!();
  }

  fn execute_best(&self, node: &FastSearchNode) -> Action {
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

  fn update_tree_node(&self, tree: &mut SearchTree<Self>, update_id: usize, next_id: Option<usize>, leaf_action: Action, result: Stone) where Self: Sized {
  }

  fn execute(&self, node: &SearchNode<(), ()>) -> Action {
    for p in node.uninst_pos.iter() {
      return Action::Place{pos: p as Pos};
    }
    Action::Pass
  }
}

#[derive(Clone, Copy, Default)]
pub struct UctNodeData {
  pub n_trials: i32,
}

#[derive(Clone, Copy, Default)]
pub struct UctEdgeData {
  pub n_trials: i32,
  pub n_wins:   i32,
  pub q_value:  f32,
}

#[derive(Clone, Copy)]
pub struct UctTreePolicyConfig {
  pub c:          f32,
  pub bias:       f32,
  pub tuned:      bool,
  pub max_trials: i32,
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
    let exploit_term = (sa.n_wins as f32) / (sa.n_trials as f32);
    let explore_term = match self.config.tuned {
      false => {
        self.config.c * ((s.n_trials as f32).ln() / (self.config.bias + sa.n_trials as f32)).sqrt()
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

  fn update(&mut self, result: Stone) where Self: Sized {
    self.num_plays += 1;
  }

  fn update_tree_node(&self, tree: &mut SearchTree<Self>, update_id: usize, next_id: Option<usize>, leaf_action: Action, result: Stone) where Self: Sized {
    let mut update_node = &mut tree.nodes[update_id];
    // Update node stats (n_trials).
    update_node.data.n_trials += 1;
    // Update edge stats (n_trials, n_wins, q_value).
    if let Some(next_id) = next_id {
      let update_turn = update_node.state.current_turn();
      let next_action = update_node.inst_actions.get(&next_id).unwrap();
      let old_q_value = update_node.inst_childs.get(&next_action).unwrap().data.q_value;
      let mut update_edge = update_node.inst_childs.get_mut(&next_action).unwrap();
      update_edge.data.n_trials += 1;
      if update_turn == result {
        update_edge.data.n_wins += 1;
      }
      let new_q_value = self.update_q_value(
          &update_node.data,
          &update_edge.data,
          old_q_value);
      update_edge.data.q_value = new_q_value;
    }
  }

  fn execute(&self, node: &SearchNode<Self::NodeData, Self::EdgeData>) -> Action {
    // TODO(20151008): choose an arm that has not yet been picked.
    for p in node.uninst_pos.iter() {
      return Action::Place{pos: p as Pos};
    }
    // TODO(20151008): choose an arm according to q.
    //for 
    Action::Pass
  }

  fn choose_best(&self, node: &SearchNode<Self::NodeData, Self::EdgeData>) -> Action {
    // TODO(20151008): choose the arm with the most trials.
    Action::Pass
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

  fn update_tree_node(&self, tree: &mut SearchTree<Self>, update_id: usize, next_id: Option<usize>, leaf_action: Action, result: Stone) where Self: Sized {
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

pub trait SearchPolicy {
  fn reset(&mut self) {
    unimplemented!();
  }

  fn execute_search(&self, node: &FastSearchNode) -> Action {
    unimplemented!();
  }

  fn execute_best(&self, node: &FastSearchNode) -> Action {
    unimplemented!();
  }
}

/*#[derive(Clone, Copy, Default)]
pub struct ThompsonNodeData;

#[derive(Clone, Copy, Default)]
pub struct ThompsonEdgeData {
  succ_ratio: f32,
  num_trials: f32,
}

impl ThompsonEdgeData {
  fn update(&mut self, r: f32) {
    let n = self.num_trials + 1.0;
    let mut mu = self.succ_ratio;
    let delta = r - mu;
    mu = mu + delta / n;
    self.succ_ratio = mu;
    self.num_trials = n;
  }

  pub fn update_succ(&mut self) {
    self.update(1.0);
  }

  pub fn update_fail(&mut self) {
    self.update(0.0);
  }
}*/

#[derive(Clone, Copy)]
pub struct ThompsonSearchPolicyConfig {
  pub max_trials: i32,
}

#[derive(Clone, Copy)]
pub struct ThompsonSearchPolicy {
  config: ThompsonSearchPolicyConfig,
}

pub trait RolloutPolicy {
  fn execute_rollout(&mut self, init_state: &FastBoard, init_aux_state: &FastBoardAux) -> Stone {
    unimplemented!();
  }
}

#[derive(Clone)]
pub struct QuasiUniformRolloutPolicy {
  rng:        XorShift128PlusRng,
  state:      FastBoard,
  work:       FastBoardWork,
  moves:      Vec<Pos>,
  max_plies:  usize,
}

impl QuasiUniformRolloutPolicy {
  pub fn new() -> QuasiUniformRolloutPolicy {
    QuasiUniformRolloutPolicy{
      // FIXME(20151008)
      rng:        XorShift128PlusRng::from_seed([1234, 5678]),
      state:      FastBoard::new(),
      work:       FastBoardWork::new(),
      moves:      Vec::with_capacity(FastBoard::BOARD_SIZE),
      max_plies:  FastBoard::BOARD_SIZE,
    }
  }
}

impl RolloutPolicy for QuasiUniformRolloutPolicy {
  fn execute_rollout(&mut self, init_state: &FastBoard, init_aux_state: &FastBoardAux) -> Stone {
    let leaf_turn = init_state.current_turn();
    self.state.clone_from(init_state);
    // TODO(20151008): initialize moves list.
    self.moves.clear();
    self.moves.extend(init_aux_state.get_legal_positions(leaf_turn).iter()
      .map(|p| p as Pos));
    if self.moves.len() > 0 {
      'rollout:
      for _ in (0 .. self.max_plies) {
        let ply_turn = self.state.current_turn();
        loop {
          let pos = random_sample_without_replace(&mut self.moves, &mut self.rng).unwrap();
          if self.state.is_legal_move_fast(ply_turn, pos) {
            let action = Action::Place{pos: pos};
            self.state.play(ply_turn, action, &mut self.work, &mut None);
            self.state.update(ply_turn, action, &mut self.work, &mut None);
            self.moves.extend(self.state.last_captures());
            break;
          } else if self.moves.len() == 0 {
            break 'rollout;
          }
        }
      }
    }
    // TODO(20151008): set correct komi.
    // XXX: Settle ties in favor of W player.
    if self.state.score_fast(6.5) >= 0.0 {
      Stone::White
    } else {
      Stone::Black
    }
  }
}

pub struct UniformRolloutPolicy;

impl RolloutPolicy for UniformRolloutPolicy {
}
