use board::{Stone, Action};

pub mod minimax;
pub mod parallel_policies;
pub mod parallel_tree;
pub mod policies;
pub mod tree;

#[derive(Clone, Copy)]
pub struct ProgWideConfig {
  pub pwide:    bool,
  pub pwide_mu: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct SearchResult {
  pub turn:           Stone,
  pub action:         Action,
  pub expected_score: f32,
}

#[derive(Default, Debug)]
pub struct SearchStats {
  pub elapsed_ms:       i64,
  pub argmax_rank:      i32,
  pub argmax_trials:    i32,
  pub max_ply:          i32,
  pub edge_count:       i32,
  pub inner_edge_count: i32,
  pub term_edge_count:  i32,
  pub old_leaf_count:   i32,
  pub new_leaf_count:   i32,
  pub term_count:       i32,
  pub nonterm_count:    i32,
}
