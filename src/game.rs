use fastboard::{PosExt, Pos, Stone, Action, PlayResult, FastBoard, FastBoardAux, FastBoardWork};
use gtp_board::{Coord};
use sgf::{Sgf};

use array::{Array3d};

use std::iter::{repeat};
//use std::slice::bytes::{copy_memory};

pub fn freshly_extract_features(turn: Stone, state: &FastBoard, prev_state: &Option<FastBoard>) -> Array3d<u8> {
  const NUM_PLANES:       usize = 4;
  const GREEN_PLANE:      usize = 0;
  const RED_PLANE:        usize = FastBoard::BOARD_SIZE;
  const PREV_GREEN_PLANE: usize = FastBoard::BOARD_SIZE * 2;
  const PREV_RED_PLANE:   usize = FastBoard::BOARD_SIZE * 3;

  //let mut x = zeros.clone();
  let mut x: Vec<_> = repeat(0).take(FastBoard::BOARD_SIZE * NUM_PLANES).collect();
  for p in (0 .. FastBoard::BOARD_SIZE) {
    let stone = state.get_stone(p as Pos);
    if stone == turn {
      x[GREEN_PLANE + p] = 1;
    } else if stone == turn.opponent() {
      x[RED_PLANE + p] = 1;
    }
  }
  if let &Some(ref prev_state) = prev_state {
    for p in (0 .. FastBoard::BOARD_SIZE) {
      let stone = prev_state.get_stone(p as Pos);
      if stone == turn {
        x[PREV_GREEN_PLANE + p] = 1;
      } else if stone == turn.opponent() {
        x[PREV_RED_PLANE + p] = 1;
      }
    }
  }
  let frame = Array3d::with_data(x, (FastBoard::BOARD_DIM as usize, FastBoard::BOARD_DIM as usize, NUM_PLANES));
  frame
}

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

  /*pub fn feature_frame_size() -> usize {
    FastBoard::BOARD_SIZE * 4
  }*/

  pub fn extract_features(&self) -> Vec<(usize, Array3d<u8>, i32, i32)> {
    let mut features_seq = Vec::new();
    if let Some(outcome) = self.outcome {
      let mut prev_turn: Option<Stone> = None;
      let mut prev_state: Option<FastBoard> = None;
      for (_, &(turn, ref state, action)) in self.history.iter().enumerate() {
        let features = state.extract_relative_features(turn);
        let fresh_features = freshly_extract_features(turn, state, &prev_state);
        {
          let mut mismatches = Vec::new();
          for p in (0 .. FastBoard::BOARD_SIZE) {
            if features.as_slice()[p] != fresh_features.as_slice()[p] {
              mismatches.push(p);
            }
          }
          if !mismatches.is_empty() {
            println!("FATAL!");
            println!("  FastBoard features do NOT match old (green/red) features!");
            println!("  mismatched points: {:?}", mismatches);
            panic!();
          }
        }
        if let Action::Place{pos} = action {
          if prev_turn.is_none() || prev_turn.unwrap().opponent() == turn {
            let win: i32 = if outcome == turn { 1 } else { 0 };
            features_seq.push((turn.offset(), features, pos as i32, win));
          }
        }
        prev_turn = Some(turn);
        prev_state = Some(state.clone());
      }
    }
    features_seq
  }

  pub fn extract_features_old(&self) -> Vec<(Array3d<u8>, i32)> {
    const GREEN_PLANE:      usize = 0;
    const RED_PLANE:        usize = FastBoard::BOARD_SIZE;
    const PREV_GREEN_PLANE: usize = FastBoard::BOARD_SIZE * 2;
    const PREV_RED_PLANE:   usize = FastBoard::BOARD_SIZE * 3;
    const NUM_PLANES:       usize = 4;

    let zeros: Vec<_> = repeat(0).take(FastBoard::BOARD_SIZE * NUM_PLANES).collect();
    let mut features_seq = vec![];
    let mut prev_state: Option<FastBoard> = None;
    for (i, &(turn, ref state, action)) in self.history.iter().enumerate() {
      let mut x = zeros.clone();
      for p in (0 .. FastBoard::BOARD_SIZE) {
        let stone = state.get_stone(p as Pos);
        if stone == turn {
          x[GREEN_PLANE + p] = 1;
        } else if stone == turn.opponent() {
          x[RED_PLANE + p] = 1;
        }
      }
      if let Some(ref prev_state) = prev_state {
        for p in (0 .. FastBoard::BOARD_SIZE) {
          let stone = prev_state.get_stone(p as Pos);
          if stone == turn {
            x[PREV_GREEN_PLANE + p] = 1;
          } else if stone == turn.opponent() {
            x[PREV_RED_PLANE + p] = 1;
          }
        }
      }
      prev_state = Some(state.clone());
      let label = match action {
        Action::Place{pos} => pos as i32,
        _ => continue,
      };
      assert!(label >= 0 && label < 361);
      let frame = Array3d::with_data(x, (FastBoard::BOARD_DIM as usize, FastBoard::BOARD_DIM as usize, NUM_PLANES));
      features_seq.push((frame, label));
    }
    features_seq
  }
}
