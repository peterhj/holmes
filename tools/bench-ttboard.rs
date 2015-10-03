extern crate holmes;
extern crate rand;

//use holmes::board::{Piece, Coord, MoveResult};
//use holmes::board_impls::{KoRule, TtBoard};
use holmes::fastboard::{Stone, Action, FastBoard, FastBoardWork};

use rand::{Rng, thread_rng};

fn main() {
  let n = 24000;
  let mut rng = thread_rng();
  let mut valid_coords = Vec::with_capacity(361);
  for i in (0 .. 361i16) {
    valid_coords.push(i);
  }
  let mut work = FastBoardWork::new();
  let mut board = FastBoard::new();
  for idx in (0 .. n) {
    rng.shuffle(&mut valid_coords);
    board.reset();
    let mut stone = Stone::Black;
    for (i, &c) in valid_coords.iter().enumerate() {
      if i >= 200 {
        break;
      }
      board.play(stone, Action::Place{pos: c}, &mut work);
      stone = stone.opponent();
    }
    /*if idx == 0 {
      println!("DEBUG: {:?}", board);
    }*/
  }
}
