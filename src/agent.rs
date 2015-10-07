use fastboard::{Stone, Pos, Action, ActionStatus, FastBoard, FastBoardAux, FastBoardWork};
use gtp_board::{Player, Vertex, RuleSystem, TimeSystem, MoveResult, UndoResult};
use policy::{
  TreePolicy, UctTreePolicy, UctTreePolicyConfig,
  RolloutPolicy, QuasiUniformRolloutPolicy,
};
use tree::{SearchTree};

use time::{Timespec, Duration, get_time};

pub struct PreGame;

impl PreGame {
  pub fn prefer_handicap_positions(&self, num_stones: usize) -> Vec<Pos> {
    // TODO(20151006)
    vec![]
  }
}

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
  history:        Vec<(FastBoard, Option<FastBoardAux>)>,
  current_state:  FastBoard,
  current_aux:    Option<FastBoardAux>,
  current_ply:    i32,
  work:           FastBoardWork,
}

impl Agent {
  pub fn new() -> Agent {
    Agent{
      valid:          false,
      config:         Default::default(),
      history:        Vec::new(),
      current_state:  FastBoard::new(),
      current_aux:    Some(FastBoardAux::new()),
      current_ply:    0,
      work:           FastBoardWork::new(),
    }
  }

  pub fn invalidate(&mut self) {
    self.valid = false;
  }

  pub fn is_valid(&self) -> bool {
    self.valid
  }

  pub fn get_stone(&self, pos: Pos) -> Stone {
    self.current_state.get_stone(pos)
  }

  pub fn begin_search(&self) -> SearchProblem<UctTreePolicy, QuasiUniformRolloutPolicy> {
    let mut tree = SearchTree::new();
    tree.reset(&self.current_state, &self.current_aux, self.current_ply, 0.0);
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
    let &mut Agent{ref mut current_state, ref mut current_aux, ref mut work, ..} = self;
    match current_aux.as_ref().unwrap().check_move(turn, action, current_state, work) {
      ActionStatus::Legal => {
        current_state.play(turn, action, work, current_aux);
        current_aux.as_mut().unwrap().update(turn, current_state, work);
        MoveResult::Okay
      }
      _ => {
        MoveResult::IllegalMove
      }
    }
  }

  pub fn undo(&mut self) -> UndoResult {
    if let Some((state, aux)) = self.history.pop() {
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
    let mut search_problem = self.begin_search();
    let action = search_problem.join();
    action.to_vertex()
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
