extern crate holmes;
extern crate rand;

use holmes::fastboard::{Stone, Action, FastBoard, FastBoardWork};
use holmes::random::{XorShift128PlusRng};

use rand::{Rng, SeedableRng, thread_rng};

fn main() {
  let n = 35000;
  //let seed = [thread_rng().next_u64(), thread_rng().next_u64()];
  let seed = [1234, 5678];
  let mut rng: XorShift128PlusRng = SeedableRng::from_seed(seed);
  let mut valid_coords = Vec::with_capacity(361);
  for i in (0 .. 361i16) {
    valid_coords.push(i);
  }
  rng.shuffle(&mut valid_coords);
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
      board.play(stone, Action::Place{pos: c}, &mut work, &mut None, false);
      stone = stone.opponent();
    }
    /*if idx == 0 {
      println!("DEBUG: {:?}", board);
    }*/
  }
}
