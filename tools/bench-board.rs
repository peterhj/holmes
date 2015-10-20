#![feature(slice_bytes)]

extern crate holmes;
extern crate rand;
extern crate statistics_avx2;

use holmes::fastboard::{Stone, Action, FastBoard, FastBoardWork};
use holmes::random::{XorShift128PlusRng, choose_without_replace};
use statistics_avx2::array::{array_sum};

use rand::{Rng, SeedableRng, thread_rng};
use std::slice::bytes::{copy_memory};

fn main() {
  let n = 40000;
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
  let mut final_cdf_norm = 0.0;
  let mut fake_cdf: Vec<f32> = Vec::with_capacity(361);
  unsafe { fake_cdf.set_len(361) };
  let mut features_buf: Vec<u8> = Vec::with_capacity(361 * 4 * 256);
  unsafe { features_buf.set_len(361 * 4 * 256) };
  for idx in (0 .. n) {
    let mut valid_coords = valid_coords.clone();
    rng.shuffle(&mut valid_coords);
    board.reset();
    let mut stone = Stone::Black;
    //for (i, &c) in valid_coords.iter().enumerate() {
    let mut i = 0;
    loop {
      if i >= 360 {
        break;
      }
      while valid_coords.len() > 0 {
        let p = choose_without_replace(&mut valid_coords, &mut rng);
        if let Some(p) = p {
          let pos = p as i16;
          if board.is_legal_move_fast(stone, pos) {
            board.play(stone, Action::Place{pos: pos}, &mut work, &mut None, false);
            break;
          }
        }
        let cdf_norm = array_sum(&fake_cdf);
        final_cdf_norm = cdf_norm;
      }
      if valid_coords.len() == 0 {
        break;
      }
      valid_coords.extend(board.last_captures().iter());
      let features = board.extract_features();
      let offset = (idx * 360 + i) % 256;
      copy_memory(features.as_slice(), &mut features_buf[offset * 361 * 4 .. (offset + 1) * 361 * 4]);
      stone = stone.opponent();
      i += 1;
    }
    /*if idx == 0 {
      println!("DEBUG: {:?}", board);
    }*/
  }
  println!("DEBUG: {}", final_cdf_norm);
}
