#![feature(slice_bytes)]

extern crate holmes;
extern crate rand;
//extern crate statistics_avx2;

//use holmes::fastboard::{Stone, Action, FastBoard, FastBoardWork};
use holmes::board::{RuleSet, PlayerRank, Stone, Point, Action};
use holmes::discrete::bfilter::{BFilter};
use holmes::random::{XorShift128PlusRng, choose_without_replace};
use holmes::txnstate::{TxnStateConfig, TxnState};
use holmes::txnstate::extras::{TxnStateLegalityData};
use holmes::txnstate::features::{TxnStateFeaturesData};
//use statistics_avx2::array::{array_sum};

use rand::{Rng, SeedableRng, thread_rng};
//use std::slice::bytes::{copy_memory};

fn main() {
  let n = 1000;
  //let seed = [thread_rng().next_u64(), thread_rng().next_u64()];
  let seed = [1234, 5678];
  let mut rng: XorShift128PlusRng = SeedableRng::from_seed(seed);
  let mut valid_coords = Vec::with_capacity(361);
  for i in (0 .. 361i16) {
    valid_coords.push(i);
  }
  rng.shuffle(&mut valid_coords);

  //let mut state = TxnState::new(RuleSet::KgsJapanese.rules(), ());
  let mut state = TxnState::new(
      TxnStateConfig{
        rules:  RuleSet::KgsJapanese.rules(),
        ranks:  [PlayerRank::Dan(9), PlayerRank::Dan(9)],
      },
      TxnStateLegalityData::new(false),
  );
  //let mut state = TxnState::new(RuleSet::KgsJapanese.rules(), TxnStateFeaturesData::new());

  //let mut final_cdf_norm = 0.0;
  //let mut fake_cdf: Vec<f32> = Vec::with_capacity(361);
  //unsafe { fake_cdf.set_len(361) };
  //let mut features_buf: Vec<u8> = Vec::with_capacity(361 * 4 * 256);
  //unsafe { features_buf.set_len(361 * 4 * 256) };

  let mut fake_pdf: Vec<f32> = Vec::with_capacity(361);
  unsafe { fake_pdf.set_len(361) };
  for j in (0 .. 361) {
    fake_pdf[j] = 0.01;
  }

  let mut filter = BFilter::with_capacity(361);

  let mut fake_sum = 0.0;
  for idx in (0 .. n) {
    let mut valid_coords = valid_coords.clone();
    //rng.shuffle(&mut valid_coords);
    state.reset();
    let mut stone = Stone::Black;
    //for (i, &c) in valid_coords.iter().enumerate() {
    let mut t = 0;
    loop {
      if t >= 542 {
        break;
      }

      // XXX(20151119): combination of these two result in slowdown of 7-8x
      // compared to pure quasirandom policy, or a slowdown of about 1.8x
      // compared to quasirandom w/ legality data.
      /*for j in (0 .. 361) {
        fake_sum += fake_pdf[j];
      }*/
      //filter.fill(&fake_pdf);

      while valid_coords.len() > 0 {
        /*let j = filter.sample(&mut rng).unwrap();
        filter.zero(j);*/
        let p = choose_without_replace(&mut valid_coords, &mut rng);
        if let Some(p) = p {
          let point = Point(p as i16);
          /*if board.is_legal_move_fast(stone, pos) {
            board.play(stone, Action::Place{pos: pos}, &mut work, &mut None, false);
            break;
          }*/
          if state.try_place(stone, point).is_ok() {
            state.commit();
            break;
          } else {
            state.undo();
          }
        }
        //let cdf_norm = array_sum(&fake_cdf);
        //final_cdf_norm = cdf_norm;
      }
      if valid_coords.len() == 0 {
        break;
      }
      //valid_coords.extend(board.last_captures().iter());
      /*let features = board.extract_features();
      let offset = (idx * 360 + i) % 256;
      copy_memory(features.as_slice(), &mut features_buf[offset * 361 * 4 .. (offset + 1) * 361 * 4]);*/
      stone = stone.opponent();
      t += 1;
    }
    /*if idx == 0 {
      println!("DEBUG: {:?}", board);
    }*/
  }
  //println!("DEBUG: {}", final_cdf_norm);
  println!("DEBUG: {}", fake_sum);
}
