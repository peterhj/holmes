use fastboard::{FastBoard, FastBoardAux};
use policy::{
  TreePolicy, Ucb1TreePolicy,
  PlayoutPolicy, QuasiUniformPlayoutPolicy,
};
use tree::{QValueTree};

pub type AgentTreePolicy = Ucb1TreePolicy;
pub type AgentPlayoutPolicy = QuasiUniformPlayoutPolicy;

pub struct Agent {
  current_state:  FastBoard,
  current_aux:    FastBoardAux,
  current_depth:  i32,
  tree_policy:    AgentTreePolicy,
  playout_policy: AgentPlayoutPolicy,
}

impl Agent {
  pub fn search(&self) -> SearchProblem<AgentTreePolicy, AgentPlayoutPolicy> {
    let mut tree = QValueTree::new();
    tree.reset(&self.current_state, &self.current_aux, self.current_depth, 0.0);
    SearchProblem{
      tree:           tree,
      tree_policy:    self.tree_policy.clone(),
      playout_policy: self.playout_policy.clone(),
    }
  }
}

pub struct SearchProblem<TreeP, PlayoutP> where TreeP: TreePolicy, PlayoutP: PlayoutPolicy {
  tree:           QValueTree<TreeP>,
  tree_policy:    TreeP,
  playout_policy: PlayoutP,
}
