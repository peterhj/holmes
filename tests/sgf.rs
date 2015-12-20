extern crate holmes;
extern crate rusqlite;
extern crate rustc_serialize;

use holmes::board::{RuleSet, Board, PlayerRank, Stone, Point, Action};
use holmes::sgf::{Sgf};
use holmes::txnstate::{TxnState, TxnResult};
use holmes::txnstate::extras::{TxnStateAllData};
use rusqlite::{SqliteConnection, SqliteOpenFlags};
use rustc_serialize::json;

use std::collections::{BTreeSet};
use std::iter::{FromIterator};
use std::path::{PathBuf};
//use std::str::{from_utf8};

const DB_PATH: &'static str = "test_data/test_sgf.1000.sqlite";
const EXPECTED_COUNT: usize = 1000;

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
  let db = SqliteConnection::open_with_flags(
      &db_path, SqliteOpenFlags::from_bits(0x01).unwrap()).unwrap();

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
fn test_sgf_correctness() {
  let db_path = PathBuf::from(DB_PATH);
  let db = SqliteConnection::open_with_flags(
      &db_path, SqliteOpenFlags::from_bits(0x01).unwrap()).unwrap();

  let expected_count = EXPECTED_COUNT;
  let mut total_count = 0;

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

      // TODO(20151107): legal moves depend on rule set.
      let ruleset = match sgf.rules.as_ref().map(|s| s as &str) {
        Some("Japanese")    => RuleSet::KgsJapanese,
        Some("Chinese")     => RuleSet::KgsChinese,
        Some("AGA")         => RuleSet::KgsAga,
        Some("New Zealand") => RuleSet::KgsNewZealand,
        _ => unimplemented!(),
      };
      assert_eq!(RuleSet::KgsJapanese, ruleset);
      //let rules = ruleset.rules();
      /*println!("DEBUG: sgf:");
      println!("{}", sgf_entry.sgf_body);*/

      let mut state = TxnState::new(
          [PlayerRank::Dan(9), PlayerRank::Dan(9)],
          ruleset.rules(),
          TxnStateAllData::new(),
      );
      state.reset();
      for (j, &(ref turn_code, ref move_code)) in sgf.moves.iter().enumerate() {
        let turn = Stone::from_code_str(&turn_code);
        let action = Action::from_code_str(&move_code);

        println!("DEBUG: move {}: {:?} {:?}", j, turn, action);
        if let Action::Place{point} = action {
          println!("DEBUG:   move coord: {:?} {:?}", turn, point.to_coord());
        }
        println!("DEBUG: gnugo repr ({}):", j);
        println!("{}", gnugo_entry.gnugo_positions[j]);
        println!("DEBUG: board ({}):", j);
        for s in state.to_debug_strings().iter() {
          println!("DEBUG:   {}", s);
        }

        // Check that the state position and legal points match the oracle's
        // (gnugo's) exactly.
        let gnugo_pos_guess = state.to_gnugo_printsgf("2015-11-07", 6.5, RuleSet::KgsJapanese);
        if gnugo_entry.gnugo_positions[j] != gnugo_pos_guess.output {
          println!("PANIC: mismatched gnugo printsgf-style positions ({}):", j);
          println!("# gnugo repr:");
          println!("{}", gnugo_entry.gnugo_positions[j]);
          println!("# our repr:");
          println!("{}", gnugo_pos_guess.output);
          println!("# state:");
          for s in state.to_debug_strings().iter() {
            println!("{}", s);
          }
          panic!();
        }

        // Check that cached illegal moves in TxnStateLegalityData are
        // consistent with the printsgf illegal moves.
        let mut valid_moves = vec![];
        state.get_data().legality.fill_legal_points(turn, &mut valid_moves);
        let valid_moves = BTreeSet::from_iter(valid_moves.into_iter());
        for p in (0 .. Board::SIZE) {
          let point = Point(p as i16);
          let mut err = false;
          if gnugo_pos_guess.illegal.contains(&point) || state.current_stone(point) != Stone::Empty {
            if valid_moves.contains(&point) {
              println!("PANIC: point ({}) should be illegal!", point.to_coord().to_string());
              err = true;
            }
          } else {
            if !valid_moves.contains(&point) {
              println!("PANIC: point ({}) should be legal!", point.to_coord().to_string());
              err = true;
            }
          }
          if err {
            println!("# brute force illegal empty points:");
            println!("{:?}", gnugo_pos_guess.illegal);
            println!("# cached valid moves:");
            println!("{:?}", valid_moves);
            println!("# state:");
            for s in state.to_debug_strings().iter() {
              println!("{}", s);
            }
            panic!();
          }
        }

        // Check that the given action is valid. Check that undo also works as
        // expected.
        let res1 = state.try_action(turn, action);
        state.undo();
        let res2 = state.try_action(turn, action);
        assert_eq!(res1, res2);
        match res2 {
          Ok(_)   => state.commit(),
          Err(e)  => {
            println!("PANIC: move should be legal! {:?}", e);
            println!("DEBUG: move {}: {:?} {:?}", j, turn, action);
            if let Action::Place{point} = action {
              let coord = point.to_coord();
              println!("DEBUG:   move coord: {:?} {:?}", turn, coord);
            }
            println!("DEBUG: board:");
            for s in state.to_debug_strings().iter() {
              println!("DEBUG:   {}", s);
            }
            panic!();
          }
        }

        // Check position features.
        for p in (0 .. Board::SIZE) {
          let point = Point::from_idx(p);
          match state.current_stone(point) {
            Stone::Black => {
              assert_eq!(1, state.get_data().features.current_feature(0, point));
              assert_eq!(0, state.get_data().features.current_feature(1, point));
              assert_eq!(1, state.get_data().libfeats.current_feature(0, point));
              assert_eq!(0, state.get_data().libfeats.current_feature(4, point));
            }
            Stone::White => {
              assert_eq!(0, state.get_data().features.current_feature(0, point));
              assert_eq!(1, state.get_data().features.current_feature(1, point));
              assert_eq!(0, state.get_data().libfeats.current_feature(0, point));
              assert_eq!(1, state.get_data().libfeats.current_feature(4, point));
            }
            Stone::Empty => {
              assert_eq!(0, state.get_data().features.current_feature(0, point));
              assert_eq!(0, state.get_data().features.current_feature(1, point));
              assert_eq!(0, state.get_data().libfeats.current_feature(0, point));
              assert_eq!(0, state.get_data().libfeats.current_feature(4, point));
            }
          }
        }

        // TODO(20151123):
        // Check liberty features.
        for p in (0 .. Board::SIZE) {
          let point = Point::from_idx(p);
          match state.current_stone(point) {
            Stone::Black => {
              let libs = state.current_libs(point);
              assert_eq!(if libs == 1 { 1 } else { 0 }, state.get_data().libfeats.current_feature(1, point));
              assert_eq!(if libs == 2 { 1 } else { 0 }, state.get_data().libfeats.current_feature(2, point));
              assert_eq!(if libs == 3 { 1 } else { 0 }, state.get_data().libfeats.current_feature(3, point));
              assert_eq!(0, state.get_data().libfeats.current_feature(5, point));
              assert_eq!(0, state.get_data().libfeats.current_feature(6, point));
              assert_eq!(0, state.get_data().libfeats.current_feature(7, point));
            }
            Stone::White => {
              let libs = state.current_libs(point);
              assert_eq!(0, state.get_data().libfeats.current_feature(1, point));
              assert_eq!(0, state.get_data().libfeats.current_feature(2, point));
              assert_eq!(0, state.get_data().libfeats.current_feature(3, point));
              assert_eq!(if libs == 1 { 1 } else { 0 }, state.get_data().libfeats.current_feature(5, point));
              assert_eq!(if libs == 2 { 1 } else { 0 }, state.get_data().libfeats.current_feature(6, point));
              assert_eq!(if libs == 3 { 1 } else { 0 }, state.get_data().libfeats.current_feature(7, point));
            }
            Stone::Empty => {
              assert_eq!(0, state.get_data().libfeats.current_feature(1, point));
              assert_eq!(0, state.get_data().libfeats.current_feature(2, point));
              assert_eq!(0, state.get_data().libfeats.current_feature(3, point));
              assert_eq!(0, state.get_data().libfeats.current_feature(5, point));
              assert_eq!(0, state.get_data().libfeats.current_feature(6, point));
              assert_eq!(0, state.get_data().libfeats.current_feature(7, point));
            }
          }
        }
      }

      total_count += 1;
    }

    assert_eq!(expected_count, total_count);
  }
}
