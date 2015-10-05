use gtp::{GtpClient, Entity};
use gtp::Entity::*;
use gtp_board::{Vertex};

pub struct HolmesClient {
  host: String,
  port: u16,
}

impl GtpClient for HolmesClient {
  fn get_address(&self) -> (String, u16) {
    (self.host.clone(), self.port)
  }

  fn get_extensions(&self) {
  }

  // FIXME: change .to_vec() to .into_vec() once I figure out how to make boxed slices.

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
      _ => unreachable!(),
    };
    match size {
      19 => {
        // FIXME(20151003)
        //let rules = RuleSet::chinese();
        //let initial_board = Board::new(rules);
        //self.agent.setup_board(initial_board);
        //self.agent.setup_board();
        [].to_vec()
      },
      _ => [ErrorEntity(b"unacceptable size".to_vec())].to_vec(),
    }
  }

  fn reply_clear_board(&mut self) -> Vec<Entity> {
    [].to_vec()
  }

  fn reply_komi(&mut self, args: &[Vec<u8>]) -> Vec<Entity> {
    let new_komi = match Entity::parse_float(&args[0]) {
      FloatEntity(new_komi) => new_komi,
      _ => unreachable!(),
    };
    [].to_vec()
  }

  // Tournament setup commands.

  fn reply_fixed_handicap(&mut self) -> Vec<Entity> {
    unimplemented!();
  }

  fn reply_place_free_handicap(&mut self) -> Vec<Entity> {
    unimplemented!();
  }

  fn reply_set_free_handicap(&mut self) -> Vec<Entity> {
    unimplemented!();
  }

  // Core play commands.

  fn reply_play(&mut self, args: &[Vec<u8>]) -> Vec<Entity> {
    let (player, action) = match Entity::parse_move(&args[0], &args[1]) {
      MoveEntity(player, action) => (player, action),
      _ => unreachable!(),
    };
    // FIXME(20151003)
    vec![]
    /*let mov = Move::new(player, action);
    match self.agent.play_move(mov) {
      // Allow stupid moves so long as they are legal.
      MoveResult::LegalMove |
      MoveResult::StupidMove(_) => [].to_vec(),
      // If the move is illegal, return b"illegal move" error message.
      MoveResult::IllegalMove(_) => [ErrorEntity(b"illegal move".to_vec())].to_vec(),
    }*/
  }

  fn reply_genmove(&mut self, args: &[Vec<u8>]) -> Vec<Entity> {
    let player = match Entity::parse_color(&args[0]) {
      ColorEntity(player) => player,
      _ => unreachable!(),
    };
    // FIXME(20151003)
    // TODO: normally, return a VertexEntity; otherwise return b"resign" string.
    /*let top_moves = self.agent.choose_move(player, &mut self.rng);
    for &mov in top_moves.iter() {
      match self.agent.clone().play_move(mov) {
        MoveResult::LegalMove | MoveResult::StupidMove(_) => {
          self.agent.play_move(mov);
          return [VertexEntity(mov.action)].to_vec();
        },
        //MoveResult::StupidMove(_) => [VertexEntity(Action::Pass)].to_vec(),
        //_ => unreachable!(),
        _ => continue,
      }
    }*/
    [VertexEntity(Vertex::Pass)].to_vec()
  }

  fn reply_undo(&mut self) -> Vec<Entity> {
    // TODO: if we cannot take back the last move, return b"cannot undo" error message.
    [].to_vec()
  }

  // Tournament commands.

  fn reply_time_settings(&mut self, args: &[Vec<u8>]) -> Vec<Entity> {
    [].to_vec()
  }

  fn reply_time_left(&mut self, args: &[Vec<u8>]) -> Vec<Entity> {
    [].to_vec()
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
    unimplemented!();
  }

  // KGS extensions.

  fn reply_kgs_time_settings(&mut self, args: &[Vec<u8>]) -> Vec<Entity> {
    [].to_vec()
  }

  fn reply_kgs_genmove_cleanup(&mut self) -> Vec<Entity> {
    [].to_vec()
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
    // FIXME(20151003)
    vec![]
    //self.agent.dump_board()
  }
}
