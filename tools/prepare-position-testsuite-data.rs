extern crate holmes;
//extern crate leveldb_rs as leveldb;
extern crate rusqlite;
extern crate rustc_serialize;

use holmes::sgf::{Sgf, parse_raw_sgf};
//use leveldb::{DBOptions, DB};
use rusqlite::{SqliteConnection};
use rustc_serialize::json;

use std::env;
use std::fs::{File};
use std::io::{Read, BufRead, BufReader, Cursor};
use std::path::{PathBuf};

#[derive(RustcEncodable, RustcDecodable)]
pub struct SgfBlobEntry {
  pub sgf_path: String,
  pub sgf_body: String,
  pub sgf_num_moves: i64,
}

fn main() {
  let args: Vec<_> = env::args().collect();
  let filtered_index_path = PathBuf::from(&args[1]);
  let db_path = PathBuf::from(&args[2]);
  let file = File::open(&filtered_index_path).unwrap();
  let reader = BufReader::new(file);
  let mut sgf_paths = Vec::new();
  for line in reader.lines() {
    let line = line.unwrap();
    sgf_paths.push(line);
  }

  let mut db = SqliteConnection::open(&db_path).unwrap();
  db.execute(
      "CREATE TABLE sgf (
          id          INTEGER PRIMARY KEY,
          sgf_blob    VARCHAR NOT NULL,
          gnugo_blob  VARCHAR NOT NULL
      )", &[]).unwrap();

  for (i, sgf_path) in sgf_paths.iter().enumerate() {
    // TODO: load sgf
    let mut sgf_file = File::open(&PathBuf::from(sgf_path)).unwrap();
    let mut sgf_text = vec![];
    sgf_file.read_to_end(&mut sgf_text).unwrap();
    let raw_sgf = parse_raw_sgf(&sgf_text);
    let sgf = Sgf::from_raw(&raw_sgf);
    if (i+1) % 1000 == 0 {
      println!("DEBUG: {} / {} num moves: {}", i+1, sgf_paths.len(), sgf.moves.len());
    }

    let num_moves = sgf.moves.len();

    let entry = SgfBlobEntry{
      sgf_path: sgf_path.clone(),
      sgf_body: String::from_utf8(sgf_text).unwrap(),
      sgf_num_moves: num_moves as i64,
    };
    let encoded_entry = json::encode(&entry).unwrap();
    db.execute(
        "INSERT INTO sgf (id, sgf_blob, gnugo_blob)
        VALUES ($1, $2, $3)",
        &[&(i as i32), &encoded_entry, &""]).unwrap();
  }
}
