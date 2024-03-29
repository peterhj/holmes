use book::{OpeningBook};
use fastboard::{PosExt, Pos, Stone, Action, ActionStatus, FastBoard, FastBoardAux, FastBoardWork};
//use fasttree::{FastSearchTree, SearchResult, Trajectory};
use gtp_board::{Player, Coord, Vertex, RuleSystem, TimeSystem, MoveResult, UndoResult};
use mctree::{McSearchTree, McSearchProblem};
use policy::{
  PriorPolicy, NoPriorPolicy, ConvNetPriorPolicy,
  SearchPolicy, UctSearchPolicy, ThompsonRaveSearchPolicy,
  RolloutPolicy, QuasiUniformRolloutPolicy, ConvNetBatchRolloutPolicy,
};
use random::{random_shuffle};
//use search::{SearchProblem, BatchSearchProblem};

use statistics_avx2::random::{StreamRng, XorShift128PlusStreamRng};

use bit_set::{BitSet};
use rand::{Rng, thread_rng};
use std::path::{PathBuf};
use time::{Timespec, Duration, get_time};

#[derive(Default)]
pub struct AgentBuilder {
  pub board_dim:    Option<i16>,
  pub komi:         Option<f32>,
  pub rule_system:  Option<RuleSystem>,
  pub time_system:  Option<TimeSystem>,
  pub handicap:     Vec<Pos>,
  pub whoami:       Option<Stone>,
}

impl AgentBuilder {
  pub fn reset(&mut self) {
    self.board_dim = None;
    self.komi = None;
    self.rule_system = None;
    self.time_system = None;
    self.handicap.clear();
    self.whoami = None;
  }

  pub fn board_dim(&mut self, dim: usize) {
    self.board_dim = Some(dim as i16);
  }

  pub fn komi(&mut self, new_komi: f32) {
    self.komi = Some(new_komi);
  }

  pub fn place_handicap(&mut self, pos: Pos) {
    self.handicap.push(pos);
  }

  pub fn rule_system(&mut self, rule_system: RuleSystem) {
    self.rule_system = Some(rule_system);
  }

  pub fn time_system(&mut self, time_system: TimeSystem) {
    self.time_system = Some(time_system);
  }

  pub fn own_color(&mut self, stone: Stone) {
    self.whoami = Some(stone);
  }

  pub fn remaining_defaults(&mut self) {
    if self.rule_system.is_none() {
      self.rule_system = Some(RuleSystem::Chinese);
    }
    if self.time_system.is_none() {
      self.time_system = Some(TimeSystem::NoTimeLimit);
    }
  }

  pub fn build(&self) -> Agent {
    assert!(self.board_dim.unwrap() == 19);
    let mut agent = Agent::new();
    agent.config = AgentConfig{
      komi: self.komi.unwrap(),
      rule_system: self.rule_system.unwrap(),
      time_system: self.time_system.unwrap(),
      handicap: self.handicap.clone(),
      whoami: self.whoami.unwrap(),
    };
    agent.reset();
    agent
  }
}

pub struct AgentConfig {
  pub komi:         f32,
  pub rule_system:  RuleSystem,
  pub time_system:  TimeSystem,
  pub handicap:     Vec<Pos>,
  pub whoami:       Stone,
}

impl Default for AgentConfig {
  fn default() -> AgentConfig {
    AgentConfig{
      komi:         0.0,
      rule_system:  RuleSystem::Japanese,
      time_system:  TimeSystem::NoTimeLimit,
      handicap:     vec![],
      whoami:       Stone::Black,
    }
  }
}

pub struct Agent {
  valid:          bool,
  config:         AgentConfig,

  //prior_policy:   NoPriorPolicy,
  prior_policy:   ConvNetPriorPolicy,
  search_policy:  ThompsonRaveSearchPolicy,
  //rollout_policy: QuasiUniformRolloutPolicy,
  rollout_policy: ConvNetBatchRolloutPolicy,
  tree:           McSearchTree,

  opening_book:   OpeningBook,
  state_history:  Vec<(FastBoard, FastBoardAux)>,
  action_history: Vec<(Stone, Action)>,
  current_state:  FastBoard,
  current_aux:    FastBoardAux,
  current_ply:    usize,
  opponent_pass:  bool,
  work:           FastBoardWork,
  tmp_board:      FastBoard,
  tmp_aux:        FastBoardAux,
}

impl Agent {
  pub fn new() -> Agent {
    Agent{
      valid:          false,
      config:         Default::default(),

      //prior_policy:   NoPriorPolicy,
      prior_policy:   ConvNetPriorPolicy::new(),
      //search_policy:  UctSearchPolicy{c: 0.5},
      search_policy:  ThompsonRaveSearchPolicy::new(),
      //rollout_policy: QuasiUniformRolloutPolicy::new(),
      rollout_policy: ConvNetBatchRolloutPolicy::new(),
      tree:           McSearchTree::new_with_root(FastBoard::new(), FastBoardAux::new()),

      opening_book:   OpeningBook::load_fuego(&PathBuf::from("data/book-fuego.dat"), &mut thread_rng()),
      state_history:  Vec::new(),
      action_history: Vec::new(),
      current_state:  FastBoard::new(),
      current_aux:    FastBoardAux::new(),
      current_ply:    0,
      opponent_pass:  false,
      work:           FastBoardWork::new(),
      tmp_board:      FastBoard::new(),
      tmp_aux:        FastBoardAux::new(),
    }
  }

  pub fn is_valid(&self) -> bool {
    self.valid
  }

  pub fn invalidate(&mut self) {
    self.valid = false;
  }

  pub fn reset(&mut self) {
    self.valid = true;
    self.state_history.clear();
    self.action_history.clear();
    self.current_state.reset();
    self.current_aux.reset();
    self.current_ply = 0;
    self.work.reset();
    self.tree.set_root(self.current_state.clone(), self.current_aux.clone());
  }

  pub fn get_stone(&self, pos: Pos) -> Stone {
    if self.valid {
      self.current_state.get_stone(pos)
    } else {
      Stone::Empty
    }
  }

  pub fn begin_search<'a>(&'a mut self) -> McSearchProblem<'a> {
    // TODO(20151024)
    let &mut Agent{
      ref mut prior_policy,
      ref mut search_policy,
      ref mut rollout_policy,
      ref mut tree, ..} = self;
    McSearchProblem::new(10000, prior_policy, search_policy, rollout_policy, tree)

    /*let mut tree = FastSearchTree::new();
    //tree.reset(&self.current_state, &self.current_aux, self.current_ply as i32, 0.0);
    tree.root(&self.current_state, &self.current_aux);
    /*SearchProblem{
      //stop_time:      get_time() + Duration::seconds(10), // FIXME(20151006)
      rng:            XorShift128PlusStreamRng::with_rng_seed(&mut thread_rng()),
      tree:           tree,
      traj:           Trajectory::new(),
      tree_policy:    UctSearchPolicy{c: 0.5},
      rollout_policy: QuasiUniformRolloutPolicy::new(),
    }*/
    BatchSearchProblem::new(tree, self.rollout_policy.batch_size())*/
  }

  pub fn play_external(&mut self, player: Player, vertex: Vertex) -> MoveResult {
    let turn = Stone::from_player(player);
    let action = Action::from_vertex(vertex);
    self.play(turn, action, false)
  }

  pub fn play(&mut self, turn: Stone, action: Action, check_legal: bool) -> MoveResult {
    assert_eq!(self.current_state.current_turn(), turn);
    if turn == self.config.whoami.opponent() &&
        (action == Action::Resign || action == Action::Pass)
    {
      self.opponent_pass = true;
    } else {
      self.opponent_pass = false;
    }
    let &mut Agent{
      ref mut current_state, ref mut current_aux,
      ref mut work, ref mut tmp_board, ref mut tmp_aux, ..} = self;
    let result = if check_legal {
      tmp_aux.clone_from(current_aux);
      current_aux.check_move(turn, action, current_state, work, tmp_board, tmp_aux, true)
    } else {
      ActionStatus::Legal
    };
    match result {
      ActionStatus::Legal => {
        self.state_history.push((current_state.clone(), current_aux.clone()));
        self.action_history.push((turn, action));
        current_state.play(turn, action, work, &mut Some(current_aux), true);
        current_aux.update(turn, current_state, work, tmp_board, tmp_aux);
        if let Action::Place{pos} = action {
          // FIXME(20151026): drop the tree?
          self.tree = McSearchTree::new_with_root(current_state.clone(), current_aux.clone());
          //self.tree.apply_move(turn, pos);
        } else {
          // XXX(20151024): on pass/resign, all nodes are essentially invalid.
          self.tree.set_root(current_state.clone(), current_aux.clone());
        }
        self.current_ply += 1;
        MoveResult::Okay
      }
      _ => {
        MoveResult::IllegalMove
      }
    }
  }

  pub fn undo(&mut self) -> UndoResult {
    if let Some((state, aux)) = self.state_history.pop() {
      self.action_history.pop();
      self.current_state = state;
      self.current_aux = aux;
      self.current_ply -= 1;
      UndoResult::Okay
    } else {
      UndoResult::CannotUndo
    }
  }

  pub fn gen_move_external(&mut self, player: Player) -> Vertex {
    let turn = Stone::from_player(player);
    self.gen_move(turn)
  }

  pub fn gen_move(&mut self, turn: Stone) -> Vertex {
    // FIXME(20151025): heuristic for playing against gnugo;
    // if gnugo passed, we pass too.
    if turn == self.config.whoami && self.opponent_pass {
      return Action::Pass.to_vertex()
    }
    let digest = self.current_state.get_digest();
    if let Some(mut book_plays) = self.opening_book.lookup(turn, &digest).map(|ps| ps.to_vec()) {
      thread_rng().shuffle(&mut book_plays);
      let mut action = Action::Pass;
      for &pos in book_plays.iter() {
        action = Action::Place{pos: pos};
        if let MoveResult::Okay = self.play(turn, action, true) {
          break;
        } else {
          action = Action::Pass;
        }
      }
      action.to_vertex()
    } else {
      let action = self.begin_search().join();
      if let MoveResult::Okay = self.play(turn, action, true) {
        //println!("DEBUG: played {:?}, captures? {:?}", action, self.current_state.last_captures());
        action.to_vertex()
      } else {
        Action::Pass.to_vertex()
      }
    }
  }
}
