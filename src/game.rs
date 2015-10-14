use fastboard::{PosExt, Pos, Stone, Action, PlayResult, FastBoard, FastBoardAux, FastBoardWork};
use gtp_board::{Coord};
use sgf::{Sgf};

pub struct GameHistory {
  pub sgf:      Sgf,
  pub outcome:  Option<Stone>,
  pub history:  Vec<(Stone, FastBoard, Action)>,
}

impl GameHistory {
  pub fn new(sgf: Sgf) -> Option<GameHistory> {
    let outcome = {
      let result_toks: Vec<_> = sgf.result.split("+").collect();
      match result_toks[0].split_whitespace().next().unwrap() {
        "B" => {
          Some(Stone::Black)
        }
        "W" => {
          Some(Stone::White)
        }
        "?" | "Jigo" | "Left" | "Void" => {
          None
        }
        _ => {
          println!("DEBUG: unimplemented result: '{}'", sgf.result);
          None
        }
      }
    };
    let mut history = vec![];
    let mut prev_board = FastBoard::new();
    let mut prev_aux = FastBoardAux::new();
    let mut work = FastBoardWork::new();
    prev_board.reset();
    prev_aux.reset();
    for &(ref player, ref mov) in sgf.moves.iter() {
      let turn = match player as &str {
        "B" => Stone::Black,
        "W" => Stone::White,
        _ => unimplemented!(),
      };
      let action = if mov == "Pass" {
        Action::Pass
      } else {
        let coord = Coord::from_code(mov.as_bytes());
        Action::Place{pos: Pos::from_coord(coord)}
      };
      history.push((turn, prev_board.clone(), action));
      if let Action::Place{pos} = action {
        if let PlayResult::FatalError = prev_board.play(turn, action, &mut work, &mut Some(&mut prev_aux), false) {
          return None;
        }
        if !prev_board.last_move_was_legal() {
          return None;
        }
      }
    }
    Some(GameHistory{
      sgf:      sgf,
      outcome:  outcome,
      history:  history,
    })
  }
}
