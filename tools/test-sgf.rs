extern crate holmes;

use holmes::sgf::{parse_sgf};
use std::fs::{File};
use std::io::{Read};
use std::path::{PathBuf};

fn main() {
  let path = PathBuf::from("2015-05-01-1.sgf");
  let mut file = File::open(&path).unwrap();
  let mut text = Vec::new();
  file.read_to_end(&mut text).unwrap();
  let sgf = parse_sgf(&text);
  println!("{:?}", sgf);
}
