extern crate holmes;
extern crate rustc_serialize;

use holmes::game::{GameHistory};
use holmes::sgf::{Sgf, Property, RootProperty, GameInfoProperty, parse_raw_sgf};

use rustc_serialize::json;
use std::collections::{BTreeMap};
use std::fs::{File};
use std::io::{Read, BufRead, BufReader};
use std::path::{PathBuf};

fn main() {
  let index_path = PathBuf::from("/data0/go/gogodb-preproc/filtered_index");
  let index_file = BufReader::new(File::open(&index_path).unwrap());
  let mut index_lines = Vec::new();
  for line in index_file.lines() {
    let line = line.unwrap();
    index_lines.push(line);
  }
  index_lines.sort_by(|a, b| {
    let a_toks: Vec<_> = a.rsplitn(2, '/').collect();
    let b_toks: Vec<_> = b.rsplitn(2, '/').collect();
    a_toks[0].cmp(&b_toks[0])
  });
  let sgf_paths: Vec<_> = index_lines.iter()
    .map(|line| PathBuf::from(line)).collect();
  println!("num sgf paths: {}", sgf_paths.len());

  for (i, sgf_path) in sgf_paths.iter().enumerate() {
    if (i+1) % 1000 == 0 { // || (i+1) >= 12000 {
      println!("DEBUG: {} / {}: {:?}", i+1, sgf_paths.len(), sgf_path);
    }
    let mut sgf_file = File::open(sgf_path).unwrap();
    let mut text = Vec::new();
    sgf_file.read_to_end(&mut text).unwrap();
    let raw_sgf = parse_raw_sgf(&text);

    let sgf = Sgf::from_raw(&raw_sgf);
    if let Some(game) = GameHistory::new(sgf) {
      // TODO
      let encoded = json::encode(&game.sgf).unwrap();
    }
  }
}
