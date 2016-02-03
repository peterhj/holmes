use agents::{Agent};
use board::{Board, RuleSet, PlayerRank, Coord, Stone, Point, Action};
use txnstate::{TxnStateConfig, TxnState};

use std::io::{stdin};

pub struct ManualAgent {
  komi:     f32,
  player:   Option<Stone>,

  history:  Vec<(TxnState, Action)>,
  state:    TxnState,
}

impl ManualAgent {
  pub fn new() -> ManualAgent {
    ManualAgent{
      komi:     0.0,
      player:   None,
      history:  vec![],
      state:    TxnState::new(
          TxnStateConfig{
            rules:  RuleSet::KgsJapanese.rules(),
            ranks:  [PlayerRank::Dan(9), PlayerRank::Dan(9)],
          },
          (),
      ),
    }
  }
}

impl Agent for ManualAgent {
  fn reset(&mut self) {
    self.komi = 7.5;
    self.player = None;

    self.history.clear();
    self.state.reset();
  }

  fn board_dim(&mut self, board_dim: usize) {
    assert_eq!(Board::DIM, board_dim);
  }

  fn komi(&mut self, komi: f32) {
    self.komi = komi;
  }

  fn player(&mut self, stone: Stone) {
    self.player = Some(Stone::Black);
  }

  fn apply_action(&mut self, turn: Stone, action: Action) {
    self.history.push((self.state.clone(), action));
    match self.state.try_action(turn, action) {
      Ok(_)   => { self.state.commit(); }
      Err(_)  => { panic!("agent tried to apply an illegal action!"); }
    }
  }

  fn undo(&mut self) {
    // FIXME(20151108): track ply; limit to how far we can undo.
    if let Some((prev_state, _)) = self.history.pop() {
      self.state.clone_from(&prev_state);
    } else {
      self.state.reset();
    }
  }

  fn act(&mut self, turn: Stone) -> Action {
    loop {
      println!("DEBUG: manual: AWAITING INPUT:");
      let mut input = String::new();
      stdin().read_line(&mut input).unwrap();
      println!("DEBUG: manual: GOT INPUT: {}", input);
      let input = input.trim();
      if input.to_lowercase() == "pass" {
        return Action::Pass;
      } else if input.to_lowercase() == "resign" {
        return Action::Resign;
      } else {
        if let Some(coord) = Coord::parse_code_str(input) {
          let point = Point::from_coord(coord);
          return Action::Place{point: point};
        } else {
          println!("DEBUG: manual: INVALID INPUT, TRY AGAIN:");
          continue;
        }
      }
    }
  }
}
