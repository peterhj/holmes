use book::{OpeningBook};
use fastboard::{PosExt, Pos, Stone, Action, ActionStatus, FastBoard, FastBoardAux, FastBoardWork};
use gtp_board::{Player, Coord, Vertex, RuleSystem, TimeSystem, MoveResult, UndoResult};
use policy::{
  TreePolicy, UctTreePolicy, UctTreePolicyConfig,
  RolloutPolicy, QuasiUniformRolloutPolicy,
};
use tree::{SearchTree};

use rand::{Rng, thread_rng};
use std::path::{PathBuf};
use time::{Timespec, Duration, get_time};

// XXX: See <http://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html#sec:fixed-handicap-placement>.
static FIXED_HANDICAP_19X19_POSITIONS: [&'static [&'static [u8]]; 10] = [
  &[],
  &[],
  &[b"D4", b"Q16"],
  &[b"D4", b"Q16", b"D16"],
  &[b"D4", b"Q16", b"D16", b"Q4"],
  &[b"D4", b"Q16", b"D16", b"Q4", b"K10"],
  &[b"D4", b"Q16", b"D16", b"Q4", b"D10", b"Q10"],
  &[b"D4", b"Q16", b"D16", b"Q4", b"D10", b"Q10", b"K10"],
  &[b"D4", b"Q16", b"D16", b"Q4", b"D10", b"Q10", b"K4", b"K16"],
  &[b"D4", b"Q16", b"D16", b"Q4", b"D10", b"Q10", b"K4", b"K16", b"K10"],
];

#[derive(Default)]
pub struct PreGame;

impl PreGame {
  pub fn fixed_handicap_positions(&self, num_stones: usize) -> Vec<Pos> {
    assert!(num_stones >= 2 && num_stones <= 9);
    let mut ps = vec![];
    for &code in FIXED_HANDICAP_19X19_POSITIONS[num_stones] {
      let coord = Coord::from_code(code);
      let pos = Pos::from_coord(coord);
      ps.push(pos);
    }
    ps
  }

  pub fn prefer_handicap_positions(&self, num_stones: usize) -> Vec<Pos> {
    self.fixed_handicap_positions(num_stones)
  }
}

#[derive(Default)]
pub struct AgentBuilder {
  board_dim:    Option<i16>,
  komi:         Option<f32>,
  rule_system:  Option<RuleSystem>,
  time_system:  Option<TimeSystem>,
  handicap:     Vec<Pos>,
  whoami:       Option<Stone>,
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
  komi:         f32,
  rule_system:  RuleSystem,
  time_system:  TimeSystem,
  handicap:     Vec<Pos>,
  whoami:       Stone,
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
  opening_book:   OpeningBook,
  state_history:  Vec<(FastBoard, Option<FastBoardAux>)>,
  action_history: Vec<(Stone, Action)>,
  current_state:  FastBoard,
  current_aux:    Option<FastBoardAux>,
  current_ply:    usize,
  work:           FastBoardWork,
}

impl Agent {
  pub fn new() -> Agent {
    Agent{
      valid:          false,
      config:         Default::default(),
      opening_book:   OpeningBook::load_fuego(&PathBuf::from("data/book-fuego.dat"), &mut thread_rng()),
      state_history:  Vec::new(),
      action_history: Vec::new(),
      current_state:  FastBoard::new(),
      current_aux:    Some(FastBoardAux::new()),
      current_ply:    0,
      work:           FastBoardWork::new(),
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
    self.current_aux.as_mut().unwrap().reset();
    self.current_ply = 0;
    self.work.reset();
  }

  pub fn get_stone(&self, pos: Pos) -> Stone {
    if self.valid {
      self.current_state.get_stone(pos)
    } else {
      Stone::Empty
    }
  }

  pub fn begin_search(&self) -> SearchProblem<UctTreePolicy, QuasiUniformRolloutPolicy> {
    let mut tree = SearchTree::new();
    tree.reset(&self.current_state, &self.current_aux, self.current_ply as i32, 0.0);
    SearchProblem{
      stop_time:      get_time() + Duration::seconds(10), // FIXME(20151006)
      tree:           tree,
      tree_policy:    UctTreePolicy::new(UctTreePolicyConfig{
        c: 0.5, bias: 0.0, tuned: false,
        max_plays: 100000,
      }),
      rollout_policy: QuasiUniformRolloutPolicy::new(),
    }
  }

  pub fn play_external(&mut self, player: Player, vertex: Vertex) -> MoveResult {
    let turn = Stone::from_player(player);
    let action = Action::from_vertex(vertex);
    self.play(turn, action)
  }

  pub fn play(&mut self, turn: Stone, action: Action) -> MoveResult {
    assert_eq!(self.current_state.current_turn(), turn);
    let &mut Agent{
      ref mut current_state, ref mut current_aux,
      ref mut work, ..} = self;
    match current_aux.as_ref().unwrap().check_move(turn, action, current_state, work) {
      ActionStatus::Legal => {
        self.state_history.push((current_state.clone(), current_aux.clone()));
        self.action_history.push((turn, action));
        current_state.play(turn, action, work, current_aux);
        current_aux.as_mut().unwrap().update(turn, current_state, work);
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
    let digest = self.current_state.get_digest();
    let action = if let Some(book_plays) = self.opening_book.lookup(turn, &digest) {
      let pos = book_plays[0];
      Action::Place{pos: pos}
    } else {
      let mut search_problem = self.begin_search();
      let action = search_problem.join();
      action
    };
    if let MoveResult::Okay = self.play(turn, action) {
      action.to_vertex()
    } else {
      self.undo();
      Action::Pass.to_vertex()
    }
  }
}

pub struct SearchProblem<TreeP, RolloutP> where TreeP: TreePolicy, RolloutP: RolloutPolicy {
  stop_time:      Timespec,
  tree:           SearchTree<TreeP>,
  tree_policy:    TreeP,
  rollout_policy: RolloutP,
}

impl<TreeP, RolloutP> SearchProblem<TreeP, RolloutP> where TreeP: TreePolicy, RolloutP: RolloutPolicy {
  pub fn join(&mut self) -> Action {
    // TODO
    unimplemented!();
  }
}
