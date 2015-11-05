extern crate arraydb;

use arraydb::{ArrayDb};

use std::env;
use std::path::{PathBuf};

fn main() {
  let args: Vec<_> = env::args().collect();
  let left_db_path = PathBuf::from(&args[1]);
  let right_db_path = PathBuf::from(&args[2]);
  let left_db = ArrayDb::open(left_db_path, true);
  let right_db = ArrayDb::open(right_db_path, true);
  let left_len = left_db.len();
  let right_len = right_db.len();
  assert!(left_len >= right_len);
  let mut num_missing = 0;
  let mut num_mismatches = 0;
  let mut left_i = 0;
  let mut right_i = 0;
  while left_i < left_len {
    if (left_i+1) % 1000 == 0 {
      println!("DEBUG: compared {} elements...", left_i+1);
    }
    let left = left_db.get(left_i);
    let right = right_db.get(right_i);
    if right.is_none() {
      num_missing += 1;
      left_i += 1;
    } else {
      let left = left.unwrap();
      let right = right.unwrap();
      let mut diff = false;
      for p in (0 .. left.len()) {
        if left[p] != right[p] {
          diff = true;
          break;
        }
      }
      if diff {
        num_mismatches += 1;
        left_i += 1;
      } else {
        left_i += 1;
        right_i += 1;
      }
    }
  }
  println!("DEBUG: num missing: {}", num_missing);
  println!("DEBUG: num mismatches: {}", num_mismatches);
  println!("DEBUG: num left to go: {}", left_len - left_i);
  println!("DEBUG: num right to go: {}", right_len - right_i);
}
