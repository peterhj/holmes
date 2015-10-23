use fastboard::{PosExt, Pos, Action, Stone, FastBoard, FastBoardAux, FastBoardWork};
use policy::{SearchPolicy, RolloutPolicy};
use random::{
  XorShift128PlusRng,
  choose_without_replace, random_shuffle,
  sample_discrete_cdf, sample_discrete_cdf_scaled,
};
//use table::{TranspositionTable};

use statistics_avx2::array::{array_prefix_sum};
use statistics_avx2::random::{StreamRng};

use bit_set::{BitSet};
use rand::{Rng, SeedableRng, thread_rng};
use std::collections::{HashMap};
use std::iter::{repeat};

#[derive(Clone, Copy)]
pub enum SearchResult {
  Terminal,
  Leaf,
}

#[derive(Clone, Copy)]
pub enum ExecutionBehavior {
  Uniform,
  DiscreteDist,
}

pub struct Trajectory {
  // History.
  search_pairs:   Vec<(NodeId, Pos)>,
  //search_pairs:   Vec<(NodeId, Stone, Pos)>,
  search_result:  Option<(NodeId, SearchResult)>,
  rollout_moves:  Vec<Pos>,
  //rollout_moves:  Vec<(Stone, Pos)>,
  //outcome:        Option<Stone>,
  outcome:        f32,

  // Current rollout state.
  legal_moves:      Vec<Pos>,
  legal_moves_set:  BitSet,
  //num_legal_moves:  usize,
  scaled_pdf:   Vec<f32>,
  scaled_cdf:   Vec<f32>,
  state:            FastBoard,
  work:             FastBoardWork,
}

impl Trajectory {
  pub fn new() -> Trajectory {
    Trajectory{
      search_pairs:   Vec::new(),
      search_result:  None,
      rollout_moves:  Vec::new(),
      //outcome:        None,
      outcome:        0.0,

      state:            FastBoard::new(),
      legal_moves:      Vec::new(),
      legal_moves_set:  BitSet::with_capacity(FastBoard::BOARD_SIZE),
      //num_legal_moves:  0,
      scaled_pdf:   Vec::with_capacity(FastBoard::BOARD_SIZE),
      scaled_cdf:   Vec::with_capacity(FastBoard::BOARD_SIZE),
      work:             FastBoardWork::new(),
    }
  }

  pub fn reset_walk(&mut self) {
    self.search_pairs.clear();
    self.search_result = None;
    self.rollout_moves.clear();
    //self.outcome = None;
    self.outcome = 0.0;
  }

  pub fn reset_rollout(&mut self, init_state: &FastBoard, init_aux_state: &FastBoardAux) {
    let turn = init_state.current_turn();
    self.state.clone_from(init_state);
    self.legal_moves.clear();
    self.legal_moves.extend(init_aux_state.get_legal_positions(turn).iter().map(|p| p as Pos));
    self.legal_moves_set.clear();
    self.legal_moves_set.extend(init_aux_state.get_legal_positions(turn).iter());
    unsafe { self.scaled_pdf.set_len(FastBoard::BOARD_SIZE) };
    unsafe { self.scaled_cdf.set_len(FastBoard::BOARD_SIZE) };
    //self.num_legal_moves = self.legal_moves_set.len();
  }

  pub fn current_state(&self) -> &FastBoard {
    &self.state
  }

  /*pub fn step_rollout(&mut self, mov: Pos) {
    self.legal_moves.extend(self.state.last_captures().iter().map(|pos| pos.idx()));
    // TODO(20151018)
    unimplemented!();
  }*/

  pub fn step_rollout<R>(&mut self, batch_idx: usize, rollout_policy: &mut RolloutPolicy, rng: &mut R) -> Option<Pos>
  where R: StreamRng {
    if self.legal_moves_set.len() == 0 {
      return None;
    }
    let turn = self.state.current_turn();
    match rollout_policy.execution_behavior() {
      ExecutionBehavior::Uniform => {
        while self.legal_moves.len() > 0 {
          if let Some(j) = choose_without_replace(&mut self.legal_moves, rng) {
            if self.state.is_legal_move_fast(turn, j as Pos) {
              self.state.play(turn, Action::Place{pos: j as Pos}, &mut self.work, &mut None, false);
              self.legal_moves.extend(self.state.last_captures().iter().map(|pos| pos));
              return Some(j as Pos);
            }
          } else {
            return None;
          }
        }
      }
      ExecutionBehavior::DiscreteDist => {
        let init_j = sample_discrete_cdf(rollout_policy.read_policy_cdfs(batch_idx), rng);
        if self.state.is_legal_move_fast(turn, init_j as Pos) {
          self.state.play(turn, Action::Place{pos: init_j as Pos}, &mut self.work, &mut None, false);
          if let Some(last_pos) = self.state.last_position() {
            self.legal_moves_set.remove(&last_pos.idx());
          }
          self.legal_moves_set.extend(self.state.last_captures().iter().map(|pos| pos.idx()));
          return Some(init_j as Pos);
        } else {
          self.scaled_pdf.clone_from_slice(rollout_policy.read_policy(batch_idx));
          self.scaled_pdf[init_j] = 0.0;
          while self.legal_moves_set.len() > 0 {
            array_prefix_sum(&self.scaled_pdf, &mut self.scaled_cdf);
            let j = sample_discrete_cdf_scaled(&self.scaled_cdf, rng);
            if self.state.is_legal_move_fast(turn, j as Pos) {
              self.state.play(turn, Action::Place{pos: j as Pos}, &mut self.work, &mut None, false); 
              if let Some(last_pos) = self.state.last_position() {
                self.legal_moves_set.remove(&last_pos.idx());
              }
              self.legal_moves_set.extend(self.state.last_captures().iter().map(|pos| pos.idx()));
              return Some(j as Pos);
            } else {
              self.legal_moves_set.remove(&j);
              self.scaled_pdf[j] = 0.0;
            }
          }
        }
      }
    }
    None
  }

  pub fn end_rollout(&mut self) {
    // FIXME(20151019): use correct komi.
    /*self.outcome = if self.state.score_fast(6.5) >= 0.0 {
      Some(Stone::White)
    } else {
      Some(Stone::Black)
    };*/
    self.outcome = self.state.score_fast(6.5);
  }
}

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct NodeId(i64);

pub struct FastSearchNode {
  pub id:           NodeId,
  pub is_inst_next: BitSet,
  pub next:         HashMap<Pos, NodeId>,
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
  pub fn new<R>(id: NodeId, state: FastBoard, aux_state: FastBoardAux, rng: &mut R) -> FastSearchNode
  where R: Rng {
    let mut legal_pos: Vec<Pos> = aux_state.get_legal_positions(state.current_turn()).iter()
      .map(|p| p as Pos)
      .collect();
    //random_shuffle(&mut legal_pos, rng);
    rng.shuffle(&mut legal_pos);
    let mut rev_moves = HashMap::new();
    for (j, &pos) in legal_pos.iter().enumerate() {
      rev_moves.insert(pos, j);
    }
    let num_moves = legal_pos.len();
    FastSearchNode{
      id:           id,
      is_inst_next: BitSet::with_capacity(FastBoard::BOARD_SIZE),
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

  pub fn credit_one_arm(&mut self, pos: Pos, x: f32/*, search_policy: &SearchPolicy*/) {
    let j = self.rev_moves[&pos];
    let total = self.total_trials + 1.0;
    let n = self.num_trials[j] + 1.0;
    let mut mean = self.succ_ratio[j];
    let delta = x - mean;
    mean = mean + delta / n;
    self.total_trials = total;
    self.num_trials[j] = n;
    self.succ_ratio[j] = mean;
    //search_policy.backup(self.total_trials, &self.num_trials, &self.succ_ratio, &mut self.value);
  }

  /*pub fn backup_success(&mut self, pos: Pos, search_policy: &SearchPolicy) {
    self.backup(pos, 1.0, search_policy);
  }

  pub fn backup_failure(&mut self, pos: Pos, search_policy: &SearchPolicy) {
    self.backup(pos, 0.0, search_policy);
  }*/
}

pub struct FastSearchTree {
  rng:        XorShift128PlusRng,
  counter:    NodeId,
  root:       Option<NodeId>,
  nodes:      HashMap<NodeId, FastSearchNode>,
  //table:      TranspositionTable,

  /*path_nodes: Vec<NodeId>,
  path_moves: Vec<Pos>,
  leaf_move:  Option<Pos>,
  leaf_out:   Option<Stone>,
  roll_state: FastBoard,
  roll_moves: Vec<Pos>,*/

  work:       FastBoardWork,
  tmp_board:  FastBoard,
  tmp_aux:    FastBoardAux,
}

impl FastSearchTree {
  pub fn new() -> FastSearchTree {
    // TODO(20151008)
    FastSearchTree{
      rng:        XorShift128PlusRng::from_seed([thread_rng().next_u64(), thread_rng().next_u64()]),
      counter:    NodeId(0),
      root:       None,
      nodes:      HashMap::new(),
      /*path_nodes: Vec::new(),
      path_moves: Vec::new(),
      leaf_move:  None,
      leaf_out:   None,
      roll_state: FastBoard::new(),
      roll_moves: Vec::new(),*/
      work:       FastBoardWork::new(),
      tmp_board:  FastBoard::new(),
      tmp_aux:    FastBoardAux::new(),
    }
  }

  fn alloc_id(&mut self) -> NodeId {
    let id = self.counter;
    self.counter.0 += 1;
    id
  }

  pub fn root(&mut self, state: &FastBoard, aux_state: &FastBoardAux) {
    // FIXME(20151008): currently, always reset nodes.
    self.counter.0 = 0;
    self.nodes.clear();
    let id = self.alloc_id();
    let node = FastSearchNode::new(id, state.clone(), aux_state.clone(), &mut self.rng);
    self.root = Some(id);
    self.nodes.insert(id, node);
  }

  pub fn get_root(&self) -> &FastSearchNode {
    &self.nodes[&self.root.unwrap()]
  }

  pub fn expand(&mut self) {
    // TODO
  }

  pub fn walk(&mut self, search_policy: &SearchPolicy, traj: &mut Trajectory) -> SearchResult {
    /*self.path_nodes.clear();
    self.path_moves.clear();
    self.leaf_move  = None;
    self.leaf_out = None;*/
    traj.reset_walk();

    let mut id = self.root
      .expect("FATAL: search tree missing root, can't walk!");
    //self.path_nodes.push(id);
    loop {
      let action = search_policy.execute_search(&self.nodes[&id]);
      match action {
        Action::Resign | Action::Pass => {
          traj.search_result = Some((id, SearchResult::Terminal));
          return SearchResult::Terminal;
        }
        Action::Place{pos} => {
          if self.nodes[&id].is_inst_next.contains(&pos.idx()) {
            /*self.path_nodes.push(id);
            self.path_moves.push(pos);*/
            traj.search_pairs.push((id, pos));
            id = self.nodes[&id].next[&pos];
          } else if self.nodes[&id].total_trials < 2.0 {
            if self.nodes[&id].aux_state.get_legal_positions(self.nodes[&id].state.current_turn()).len() > 0 {
              traj.search_result = Some((id, SearchResult::Leaf));
              return SearchResult::Leaf;
            } else {
              traj.search_result = Some((id, SearchResult::Terminal));
              return SearchResult::Terminal;
            }
          } else {
            traj.search_pairs.push((id, pos));
            let next_id = self.alloc_id();
            let mut next_state = self.nodes[&id].state.clone();
            let mut next_aux_state = self.nodes[&id].aux_state.clone();
            let turn = next_state.current_turn();
            {
              let &mut FastSearchTree{ref mut work, ref mut tmp_board, ref mut tmp_aux, ..} = self;
              next_state.play(turn, action, work, &mut Some(&mut next_aux_state), false);
              next_aux_state.update(turn, &next_state, work, tmp_board, tmp_aux);
            }
            let next_node = FastSearchNode::new(next_id, next_state, next_aux_state, &mut self.rng);
            self.nodes.insert(next_id, next_node);
            self.nodes.get_mut(&id).unwrap().is_inst_next.insert(pos.idx());
            self.nodes.get_mut(&id).unwrap().next.insert(pos, next_id);
            /*self.path_nodes.push(next_id);
            self.path_moves.push(pos);*/
            traj.search_result = Some((next_id, SearchResult::Leaf));
            return SearchResult::Leaf;
          }
        }
      }
    }
  }

  /*pub fn simulate(&mut self, rollout_policy: &mut RolloutPolicy) {
    // TODO(20151019): replace this with Trajectory-related methods.
    unimplemented!();
    /*let id = self.path_nodes[self.path_nodes.len() - 1];
    let node = &self.nodes[&id];
    let outcome = rollout_policy.execute(&node.state, &node.aux_state);
    self.leaf_out = Some(outcome);*/
  }*/

  pub fn init_rollout(&mut self, traj: &mut Trajectory) {
    //let id = self.path_nodes[self.path_nodes.len() - 1];
    let id = traj.search_result.unwrap().0;
    let node = &self.nodes[&id];
    /*self.roll_moves.clear();
    self.roll_state.clone_from(&node.state);*/
    traj.reset_rollout(&node.state, &node.aux_state);
  }

  pub fn backup(&mut self, search_policy: &SearchPolicy, traj: &Trajectory) {
    // FIXME(20151019): use Trajectory to backup.
    if let Some((leaf_id, SearchResult::Leaf)) = traj.search_result {
      search_policy.backup(self.nodes.get_mut(&leaf_id).unwrap(), &[], &traj.rollout_moves, traj.outcome);
    }
    for &(id, pos) in traj.search_pairs.iter().rev() {
      // TODO(20151022)
    }

    /*let _leaf_id = self.path_nodes.pop()
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
    }*/
  }
}
