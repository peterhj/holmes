use agent::{PreGame, AgentBuilder, Agent};
use fastboard::{PosExt, Pos, Stone};
use gtp::{GtpClient, Entity};
use gtp::Entity::*;
use gtp_board::{Player, Coord, Vertex, RuleSystem, TimeSystem, MoveResult, UndoResult, dump_xcoord, dump_ycoord};

use std::str::{from_utf8};

pub struct Client {
  host:             String,
  port:             u16,
  //player:           Option<Player>,
  should_shutdown:  bool,

  pre_game:         PreGame,
  agent_builder:    AgentBuilder,
  agent:            Agent,
}

impl Client {
  pub fn new(host: String, port: u16, player: Option<Player>) -> Client {
    Client{
      host: host,
      port: port,
      //player: player,
      should_shutdown: false,
      pre_game: Default::default(),
      agent_builder: Default::default(),
      agent: Agent::new(),
    }
  }
}

impl GtpClient for Client {
  fn get_address(&self) -> (String, u16) {
    (self.host.clone(), self.port)
  }

  fn get_extensions(&self) {
  }

  fn should_shutdown(&self) -> bool {
    self.should_shutdown
  }

  fn reply_protocol_version(&mut self) -> Vec<Entity> {
    [StringEntity(b"2".to_vec())].to_vec()
  }

  fn reply_name(&mut self) -> Vec<Entity> {
    [StringEntity(b"Holmes".to_vec())].to_vec()
  }

  fn reply_version(&mut self) -> Vec<Entity> {
    [StringEntity(b"0.2".to_vec())].to_vec()
  }

  fn reply_known_command(&mut self, args: &[Vec<u8>]) -> Vec<Entity> {
    let is_known = match &args[0] as &[u8] {
      b"protocol_version"       => true,
      b"name"                   => true,
      b"version"                => true,
      b"known_command"          => true,
      b"list_commands"          => true,
      b"quit"                   => true,
      b"boardsize"              => true,
      b"clear_board"            => true,
      b"komi"                   => true,
      b"play"                   => true,
      b"genmove"                => true,
      b"undo"                   => true,
      b"time_settings"          => true,
      b"time_left"              => true,
      b"final_score"            => true,
      b"final_status_list"      => true,
      b"kgs-game_over"          => true,
      b"kgs-rules"              => true,
      b"kgs-time_settings"      => true,
      b"kgs-genmove_cleanup"    => true,
      b"loadsgf"                => true,
      b"reg_genmove"            => true,
      b"showboard"              => true,
      _ => false,
    };
    [BooleanEntity(is_known)].to_vec()
  }

  fn reply_list_commands(&mut self) -> Vec<Entity> {
    [MultilineListEntity([
      StringEntity(b"protocol_version".to_vec()),
      StringEntity(b"name".to_vec()),
      StringEntity(b"version".to_vec()),
      StringEntity(b"known_command".to_vec()),
      StringEntity(b"list_commands".to_vec()),
      StringEntity(b"quit".to_vec()),
      StringEntity(b"boardsize".to_vec()),
      StringEntity(b"clear_board".to_vec()),
      StringEntity(b"komi".to_vec()),
      StringEntity(b"play".to_vec()),
      StringEntity(b"genmove".to_vec()),
      StringEntity(b"undo".to_vec()),
      StringEntity(b"time_settings".to_vec()),
      StringEntity(b"time_left".to_vec()),
      StringEntity(b"final_score".to_vec()),
      StringEntity(b"final_status_list".to_vec()),
      StringEntity(b"kgs-time_settings".to_vec()),
      StringEntity(b"kgs-genmove_cleanup".to_vec()),
      StringEntity(b"loadsgf".to_vec()),
      StringEntity(b"reg_genmove".to_vec()),
      StringEntity(b"showboard".to_vec()),
    ].to_vec())].to_vec()
  }

  fn reply_quit(&mut self) -> Vec<Entity> {
    [].to_vec()
  }

  // Setup commands.

  fn reply_boardsize(&mut self, args: &[Vec<u8>]) -> Vec<Entity> {
    let size = match Entity::parse_int(&args[0]) {
      IntEntity(size) => size,
      _ => return vec![ErrorEntity(b"syntax error".to_vec())],
    } as usize;
    match size {
      19 => {
        self.agent_builder.board_dim(size);
        self.agent.invalidate();
      },
      _ => return vec![ErrorEntity(b"unacceptable size".to_vec())],
    }
    vec![]
  }

  fn reply_clear_board(&mut self) -> Vec<Entity> {
    // XXX(20151006): not sure about the order in which boardsize, clear, etc.
    // are called.
    //self.agent_builder.reset();
    self.agent.invalidate();
    vec![]
  }

  fn reply_komi(&mut self, args: &[Vec<u8>]) -> Vec<Entity> {
    let new_komi = match Entity::parse_float(&args[0]) {
      FloatEntity(new_komi) => new_komi,
      _ => return vec![ErrorEntity(b"syntax error".to_vec())],
    };
    self.agent_builder.komi(new_komi);
    vec![]
  }

  // Tournament setup commands.

  fn reply_fixed_handicap(&mut self, args: &[Vec<u8>]) -> Vec<Entity> {
    let num_stones = match Entity::parse_int(&args[0]) {
      IntEntity(x) => x,
      _ => return vec![ErrorEntity(b"syntax error".to_vec())],
    } as usize;
    if !(num_stones >= 2 && num_stones <= 9) {
      return vec![ErrorEntity(b"invalid number of stones".to_vec())];
    }
    // XXX: The black player places the handicap stones.
    if !self.agent.is_valid() {
      self.agent_builder.own_color(Stone::Black);
      self.agent_builder.remaining_defaults();
      self.agent = self.agent_builder.build();
    }
    // TODO(20151008): check for empty board.
    /*if self.agent.is_valid() {
      return vec![ErrorEntity(b"board not empty".to_vec())];
    }*/
    let mut vertexes = vec![];
    for &pos in self.pre_game.fixed_handicap_positions(num_stones).iter() {
      let coord = pos.to_coord();
      self.agent_builder.place_handicap(pos);
      vertexes.push(VertexEntity(Vertex::Play(coord)));
    }
    vertexes
  }

  fn reply_place_free_handicap(&mut self, args: &[Vec<u8>]) -> Vec<Entity> {
    let num_stones = match Entity::parse_int(&args[0]) {
      IntEntity(x) => x,
      _ => return vec![ErrorEntity(b"syntax error".to_vec())],
    } as usize;
    if !(num_stones >= 2 && num_stones <= 9) {
      return vec![ErrorEntity(b"invalid number of stones".to_vec())];
    }
    // XXX: The black player places the handicap stones.
    if !self.agent.is_valid() {
      self.agent_builder.own_color(Stone::Black);
      self.agent_builder.remaining_defaults();
      self.agent = self.agent_builder.build();
    }
    // TODO(20151008): check for empty board.
    /*if self.agent.is_valid() {
      return vec![ErrorEntity(b"board not empty".to_vec())];
    }*/
    let mut vertexes = vec![];
    for &pos in self.pre_game.prefer_handicap_positions(num_stones).iter() {
      let coord = pos.to_coord();
      self.agent_builder.place_handicap(pos);
      vertexes.push(VertexEntity(Vertex::Play(coord)));
    }
    vertexes
  }

  fn reply_set_free_handicap(&mut self, args: &[Vec<u8>]) -> Vec<Entity> {
    // XXX: The black player places the handicap stones.
    if !self.agent.is_valid() {
      self.agent_builder.own_color(Stone::White);
      self.agent_builder.remaining_defaults();
      self.agent = self.agent_builder.build();
    }
    // TODO(20151008): check for empty board.
    /*if self.agent.is_valid() {
      return vec![ErrorEntity(b"board not empty".to_vec())];
    }*/
    for arg in args {
      let coord = match Entity::parse_vertex(arg) {
        VertexEntity(Vertex::Play(coord)) => coord,
        _ => return vec![ErrorEntity(b"bad vertex list".to_vec())],
      };
      let pos = Pos::from_coord(coord);
      self.agent_builder.place_handicap(pos);
    }
    vec![]
  }

  // Core play commands.

  fn reply_play(&mut self, args: &[Vec<u8>]) -> Vec<Entity> {
    let (player, vertex) = match Entity::parse_move(&args[0], &args[1]) {
      MoveEntity(player, vertex) => (player, vertex),
      _ => return vec![ErrorEntity(b"syntax error".to_vec())],
    };
    // TODO(20151006): can we assume that play commands are always for the
    // opponent's moves? seems like there may be cases where this is not
    // true.
    if !self.agent.is_valid() {
      self.agent_builder.own_color(Stone::from_player(player).opponent());
      self.agent = self.agent_builder.build();
    }
    match self.agent.play_external(player, vertex) {
      MoveResult::Okay        => vec![],
      MoveResult::IllegalMove => vec![ErrorEntity(b"illegal move".to_vec())],
      _ => unreachable!(),
    }
  }

  fn reply_genmove(&mut self, args: &[Vec<u8>]) -> Vec<Entity> {
    let player = match Entity::parse_color(&args[0]) {
      ColorEntity(player) => player,
      _ => return vec![ErrorEntity(b"syntax error".to_vec())],
    };
    if !self.agent.is_valid() {
      self.agent_builder.own_color(Stone::from_player(player));
      self.agent_builder.remaining_defaults();
      self.agent = self.agent_builder.build();
    }
    let vertex = self.agent.gen_move_external(player);
    vec![VertexEntity(vertex)]
  }

  fn reply_undo(&mut self) -> Vec<Entity> {
    if !self.agent.is_valid() {
      return vec![ErrorEntity(b"cannot undo".to_vec())];
    }
    match self.agent.undo() {
      UndoResult::Okay        => vec![],
      UndoResult::CannotUndo  => vec![ErrorEntity(b"cannot undo".to_vec())],
    }
  }

  // Tournament commands.

  fn reply_time_settings(&mut self, args: &[Vec<u8>]) -> Vec<Entity> {
    let main_time = match Entity::parse_int(&args[0]) {
      IntEntity(x) => x,
      _ => return vec![ErrorEntity(b"syntax error".to_vec())],
    };
    let byo_yomi_time = match Entity::parse_int(&args[1]) {
      IntEntity(x) => x,
      _ => return vec![ErrorEntity(b"syntax error".to_vec())],
    };
    let stones = match Entity::parse_int(&args[2]) {
      IntEntity(x) => x,
      _ => return vec![ErrorEntity(b"syntax error".to_vec())],
    };
    let time_system = TimeSystem::Canadian{
      main_time_s:      main_time,
      byo_yomi_time_s:  byo_yomi_time,
      stones:           stones,
    };
    self.agent_builder.time_system(time_system);
    vec![]
  }

  fn reply_time_left(&mut self, args: &[Vec<u8>]) -> Vec<Entity> {
    let player = match Entity::parse_color(&args[0]) {
      ColorEntity(player) => player,
      _ => return vec![ErrorEntity(b"syntax error".to_vec())],
    };
    let time = match Entity::parse_int(&args[1]) {
      IntEntity(time) => time,
      _ => return vec![ErrorEntity(b"syntax error".to_vec())],
    };
    let stones = match Entity::parse_int(&args[2]) {
      IntEntity(stones) => stones,
      _ => return vec![ErrorEntity(b"syntax error".to_vec())],
    };
    // TODO(20151006)
    vec![]
  }

  fn reply_final_score(&mut self) -> Vec<Entity> {
    // FIXME(20151003)
    [StringEntity(format!("B+{}", 0.0).as_bytes().to_vec())].to_vec()
    /*let score = self.agent.score_board();
    if score > 0.0 {
      [StringEntity(format!("B+{}", score).as_slice().as_bytes().to_vec())].to_vec()
    } else {
      [StringEntity(format!("W+{}", -score).as_slice().as_bytes().to_vec())].to_vec()
    }*/
  }

  fn reply_final_status_list(&mut self, args: &[Vec<u8>]) -> Vec<Entity> {
    // TODO(20151006)
    vec![]
  }

  // KGS extensions.

  fn reply_kgs_game_over(&mut self, args: &[Vec<u8>]) -> Vec<Entity> {
    self.should_shutdown = true;
    vec![]
  }

  fn reply_kgs_rules(&mut self, args: &[Vec<u8>]) -> Vec<Entity> {
    let rule_system = match &args[0] as &[u8] {
      b"japanese"     => RuleSystem::Japanese,
      b"chinese"      => RuleSystem::Chinese,
      b"aga"          => RuleSystem::Aga,
      b"new_zealand"  => RuleSystem::NewZealand,
      x => panic!("FATAL: unknown kgs-rules rule system: '{}'", from_utf8(x).unwrap()),
    };
    self.agent_builder.rule_system(rule_system);
    vec![]
  }

  fn reply_kgs_time_settings(&mut self, args: &[Vec<u8>]) -> Vec<Entity> {
    let time_system = match &args[0] as &[u8] {
      b"none"     => {
        TimeSystem::NoTimeLimit
      }
      b"absolute" => {
        let main_time = match Entity::parse_int(&args[1]) {
          IntEntity(main_time) => main_time,
          _ => return vec![ErrorEntity(b"syntax error".to_vec())],
        };
        TimeSystem::Absolute{
          main_time_s: main_time,
        }
      }
      b"byoyomi"  => {
        let main_time = match Entity::parse_int(&args[1]) {
          IntEntity(x) => x,
          _ => return vec![ErrorEntity(b"syntax error".to_vec())],
        };
        let byo_yomi_time = match Entity::parse_int(&args[2]) {
          IntEntity(x) => x,
          _ => return vec![ErrorEntity(b"syntax error".to_vec())],
        };
        let periods = match Entity::parse_int(&args[3]) {
          IntEntity(x) => x,
          _ => return vec![ErrorEntity(b"syntax error".to_vec())],
        };
        TimeSystem::ByoYomi{
          main_time_s:      main_time,
          byo_yomi_time_s:  byo_yomi_time,
          periods:          periods,
        }
      }
      b"canadian" => {
        let main_time = match Entity::parse_int(&args[1]) {
          IntEntity(x) => x,
          _ => return vec![ErrorEntity(b"syntax error".to_vec())],
        };
        let byo_yomi_time = match Entity::parse_int(&args[2]) {
          IntEntity(x) => x,
          _ => return vec![ErrorEntity(b"syntax error".to_vec())],
        };
        let stones = match Entity::parse_int(&args[3]) {
          IntEntity(x) => x,
          _ => return vec![ErrorEntity(b"syntax error".to_vec())],
        };
        TimeSystem::Canadian{
          main_time_s:      main_time,
          byo_yomi_time_s:  byo_yomi_time,
          stones:           stones,
        }
      }
      x => return vec![ErrorEntity(b"unsupported time system".to_vec())],
    };
    self.agent_builder.time_system(time_system);
    vec![]
  }

  fn reply_kgs_genmove_cleanup(&mut self) -> Vec<Entity> {
    vec![]
  }

  // Regression commands.

  fn reply_loadsgf(&mut self) -> Vec<Entity> {
    unimplemented!();
  }

  fn reply_reg_genmove(&mut self) -> Vec<Entity> {
    unimplemented!();
  }

  // Debug commands.

  fn reply_showboard(&mut self) -> Vec<Entity> {
    let mut lines = Vec::new();
    lines.push(StringEntity(b" ".to_vec()));
    let mut line = b"   ".to_vec();
    for x in (0 .. 19u8) {
      line.extend(&dump_xcoord(x));
      line.extend(b" ");
    }
    lines.push(StringEntity(line));
    for y in (0 .. 19u8).rev() {
      let mut line = Vec::new();
      let y_label = dump_ycoord(y);
      match y_label.len() {
        1 => {
          line.push(b' ');
          line.push(y_label[0]);
        },
        2 => {
          line.extend(&y_label);
        }
        _ => unreachable!(),
      }
      line.push(b' ');
      for x in (0 .. 19u8) {
        let coord = Coord::new(x, y);
        let pos = Pos::from_coord(coord);
        let stone = self.agent.get_stone(pos);
        let stone_label = match stone {
          Stone::Black => b'X',
          Stone::White => b'O',
          Stone::Empty => match (x, y) {
            (3, 3)  | (9, 3)  | (15, 3) |
            (3, 9)  | (9, 9)  | (15, 9) |
            (3, 15) | (9, 15) | (15, 15) => b'+',
            _ => b'.',
          },
        };
        line.push(stone_label);
        line.push(b' ');
      }
      line.extend(&y_label);
      if y == 9 || y == 10 {
        line.extend(b"     ");
        if y == 9 {
          line.extend(b"BLACK (X) has captured ");
          // FIXME(20151006)
          line.extend(b"0");
          line.extend(b" stones");
        } else if y == 10 {
          line.extend(b"WHITE (O) has captured ");
          // FIXME(20151006)
          line.extend(b"0");
          line.extend(b" stones");
        }
      }
      lines.push(StringEntity(line));
    }
    let mut line = b"   ".to_vec();
    for x in (0 .. 19u8) {
      line.extend(&dump_xcoord(x));
      line.extend(b" ");
    }
    lines.push(StringEntity(line));
    [MultilineListEntity(lines)].to_vec()
  }
}
