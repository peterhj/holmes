extern crate rusqlite;
extern crate rustc_serialize;
extern crate threadpool;

use rusqlite::{SqliteConnection};
use rustc_serialize::json;
use threadpool::{ThreadPool};

use std::collections::{BTreeMap};
use std::env;
use std::io::{Write};
use std::path::{PathBuf};
use std::process::{Command, Stdio};
use std::str::{from_utf8};
use std::sync::mpsc::{channel};

#[derive(RustcEncodable, RustcDecodable)]
pub struct SgfBlobEntry {
  pub sgf_path: String,
  pub sgf_body: String,
  pub sgf_num_moves: i64,
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct GnugoBlobEntry {
  pub gnugo_positions: Vec<String>,
}

fn main() {
  let args: Vec<_> = env::args().collect();
  let db_path = PathBuf::from(&args[1]);

  let mut db = SqliteConnection::open(&db_path).unwrap();
  let pool = ThreadPool::new(8);
  let (tx, rx) = channel();

  let sgf_entries = {
    let mut sgf_iter_stmt = db.prepare(
        "SELECT id, sgf_blob FROM sgf").unwrap();
    let mut sgf_iter = sgf_iter_stmt.query_map(&[], |row| {
      let i: i32 = row.get(0);
      let sgf_blob: String = row.get(1);
      let sgf_entry: SgfBlobEntry = json::decode(&sgf_blob).unwrap();
      (i, sgf_entry)
    }).unwrap();
    let sgf_entries: Vec<_> = sgf_iter.map(|sgf| {
      let (i, sgf_entry) = sgf.unwrap();
      (i, sgf_entry)
    }).collect();
    sgf_entries
  };

  let mut iter_count = 0;
  for &(i, ref sgf_entry) in sgf_entries.iter() {
    for j in (0 .. sgf_entry.sgf_num_moves + 1) {
      let tx = tx.clone();
      let sgf_body = sgf_entry.sgf_body.clone();
      pool.execute(move || {
        let mut child = Command::new("../bin/gnugo")
          .stdin(Stdio::piped())
          .stdout(Stdio::piped())
          .args(&[
            //"--infile", &sgf_entry.sgf_path as &str,
            "--infile",   "/proc/self/fd/0",
            "--until",    &format!("{}", j + 1) as &str,
            "--printsgf", "/proc/self/fd/1",
          ])
          .spawn().unwrap();
        child.stdin.as_mut().unwrap()
          .write_all(sgf_body.as_bytes()).unwrap();
        child.stdin.as_mut().unwrap()
          .flush().unwrap();
        {
          let _ = child.stdin.take();
        }
        /*child.wait().unwrap();
        let mut stdout = vec![];
        child.stdout.take().read_to_end(&mut stdout).unwrap();*/
        let output = child.wait_with_output().unwrap();
        let position_j = String::from_utf8(output.stdout).unwrap();
        tx.send((j as usize, position_j));
      });
    }
    let mut gnugo_results: Vec<_> = rx.iter()
      .take((sgf_entry.sgf_num_moves + 1) as usize)
      .collect();
    gnugo_results.sort();
    let mut gnugo_entry = GnugoBlobEntry{
      gnugo_positions: vec![],
    };
    for (j, (j_ref, position_j)) in gnugo_results.into_iter().enumerate() {
      assert_eq!(j, j_ref);
      gnugo_entry.gnugo_positions.push(position_j);
    }
    println!("DEBUG: idx {} num moves {} positions {}",
        i, sgf_entry.sgf_num_moves, gnugo_entry.gnugo_positions.len());
    let encoded_entry = json::encode(&gnugo_entry).unwrap();
    //println!("DEBUG: idx {} encoded: '{}'", i, encoded_entry);
    db.execute(
        "UPDATE sgf
        SET gnugo_blob=$1
        WHERE id=$2",
        &[&encoded_entry, &i]).unwrap();
    if (i+1) % 10 == 0 {
      println!("DEBUG: {} sgf path: {}", i+1, sgf_entry.sgf_path);
      println!("DEBUG: {} num moves: {}", i+1, sgf_entry.sgf_num_moves);
      println!("DEBUG: {} position: '{}'", i+1, &gnugo_entry.gnugo_positions[0]);
    }
    iter_count += 1;
  }
  println!("DEBUG: iterated items: {}", iter_count);
}
