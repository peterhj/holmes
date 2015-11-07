extern crate holmes;
extern crate rusqlite;
extern crate rustc_serialize;

use holmes::board::{RuleSet, Stone, Action};
use holmes::sgf::{Sgf};
use holmes::txnboard::{TxnState, TxnResult};
use rusqlite::{SqliteConnection};
use rustc_serialize::json;

use std::path::{PathBuf};
//use std::str::{from_utf8};

const DB_PATH: &'static str = "test_data/test_sgf.20.sqlite";
const EXPECTED_COUNT: usize = 20;

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

#[test]
#[should_panic]
fn test_sgf_dummy() {
  panic!();
}

#[test]
fn test_sgf_check_db_consistency() {
  let db_path = PathBuf::from(DB_PATH);
  let db = SqliteConnection::open(&db_path).unwrap();

  let expected_count = EXPECTED_COUNT;

  let mut total_count = 0;
  let mut positions_count = 0;

  {
    let mut sgf_iter_stmt = db.prepare(
        "SELECT id, sgf_blob, gnugo_blob
        FROM sgf").unwrap();
    let sgf_iter = sgf_iter_stmt.query_map(&[], |row| {
      let i: i32 = row.get(0);
      let sgf_blob: String = row.get(1);
      let gnugo_blob: String = row.get(2);
      let sgf_entry: SgfBlobEntry = json::decode(&sgf_blob)
        .ok().expect("failed to parse sgf");
      /*println!("DEBUG: id: {}", i);
      println!("DEBUG: sgf blob: '{}'", sgf_blob);
      println!("DEBUG: gnugo blob: '{}'", gnugo_blob);*/
      let gnugo_entry: GnugoBlobEntry = json::decode(&gnugo_blob)
        .ok().expect("failed to parse gnugo");
      (i, sgf_entry, gnugo_entry)
    }).unwrap();
    for sgf_row in sgf_iter {
      let (i, sgf_entry, gnugo_entry) = sgf_row.unwrap();
      total_count += 1;
      if gnugo_entry.gnugo_positions.len() == (sgf_entry.sgf_num_moves + 1) as usize {
        positions_count += 1;
      }
    }
  }

  assert_eq!(expected_count, total_count);
  assert_eq!(expected_count, positions_count);
}

#[test]
fn test_sgf_equivalence() {
  let db_path = PathBuf::from(DB_PATH);
  let db = SqliteConnection::open(&db_path).unwrap();

  {
    let mut sgf_iter_stmt = db.prepare(
        "SELECT id, sgf_blob, gnugo_blob
        FROM sgf").unwrap();
    let sgf_iter = sgf_iter_stmt.query_map(&[], |row| {
      let i: i32 = row.get(0);
      let sgf_blob: String = row.get(1);
      let gnugo_blob: String = row.get(2);
      let sgf_entry: SgfBlobEntry = json::decode(&sgf_blob)
        .ok().expect("failed to parse sgf");
      let gnugo_entry: GnugoBlobEntry = json::decode(&gnugo_blob)
        .ok().expect("failed to parse gnugo");
      (i, sgf_entry, gnugo_entry)
    }).unwrap();
    for sgf_row in sgf_iter {
      let (i, sgf_entry, gnugo_entry) = sgf_row
        .expect("failed to unwrap row!");
      let sgf = Sgf::from_text(sgf_entry.sgf_body.as_bytes());

      // FIXME(20151107): not supporting handicap placement.
      assert_eq!(0, sgf.black_pos.len());
      assert_eq!(0, sgf.white_pos.len());

      let rules = match sgf.rules.as_ref().map(|s| s as &str) {
        Some("Japanese")    => RuleSet::KgsJapanese,
        Some("Chinese")     => RuleSet::KgsChinese,
        Some("AGA")         => RuleSet::KgsAga,
        Some("New Zealand") => RuleSet::KgsNewZealand,
        _ => unimplemented!(),
      };

      let mut state = TxnState::new(());
      state.reset();
      println!("DEBUG: starting new game...");
      println!("DEBUG: board:");
      for s in state.to_debug_strings().iter() {
        println!("DEBUG:   {}", s);
      }
      for (j, &(ref turn_code, ref move_code)) in sgf.moves.iter().enumerate() {
        //println!("DEBUG: move code: '{}'", move_code);
        let turn = Stone::from_code_str(&turn_code);
        let action = Action::from_code_str(&move_code);
        println!("DEBUG: move {}: {:?} {:?}", j, turn, action);
        if let Action::Place{point} = action {
          let coord = point.to_coord();
          println!("DEBUG:   move coord: {:?}", coord);
        }
        state.try_action(turn, action);
        state.undo();
        let res = state.try_action(turn, action);
        println!("DEBUG: board:");
        for s in state.to_debug_strings().iter() {
          println!("DEBUG:   {}", s);
        }
        match res {
          Ok(_)   => state.commit(),
          Err(e)  => panic!("move should be legal! {:?}", e),
        }
      }
    }
  }
}
