//#![allow(non_snake_case)]

extern crate holmes;
extern crate rand;
extern crate rusqlite;
extern crate rustc_serialize;

use holmes::board::{
  RuleSet, Board, /*PlayerRank,*/ Stone, Point, Action,
};
use holmes::sgf::{Sgf};
use holmes::txnstate::{
  TOMBSTONE, TxnStateData, TxnPosition, TxnChainsList, TxnStateConfig, TxnState,
};
//use holmes::txnstate::extras::{TxnStateAllData};
use holmes::txnstate::extras::{TxnStateLegalityData};
use holmes::txnstate::features::{
  //TxnStateLibFeaturesData,
  //TxnStateAlphaFeatsV1Data,
  //TxnStateAlphaFeatsV2Data,
  TxnStateAlphaV3FeatsData,
  TxnStateAlphaMiniV3FeatsData,
};
use rusqlite::{SqliteConnection, SqliteOpenFlags};
use rustc_serialize::json;

use rand::{Rng, thread_rng};
use std::cmp::{min};
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

#[derive(Clone)]
pub struct TxnStateTestFeatsData {
  legality:     TxnStateLegalityData,
  alpha_v3:     TxnStateAlphaV3FeatsData,
  alpha_m_v3:   TxnStateAlphaMiniV3FeatsData,
}

impl TxnStateTestFeatsData {
  pub fn new() -> TxnStateTestFeatsData {
    TxnStateTestFeatsData{
      legality:     TxnStateLegalityData::new(),
      alpha_v3:     TxnStateAlphaV3FeatsData::new(),
      alpha_m_v3:   TxnStateAlphaMiniV3FeatsData::new(),
    }
  }
}

impl TxnStateData for TxnStateTestFeatsData {
  fn reset(&mut self) {
    self.legality.reset();
    self.alpha_v3.reset();
    self.alpha_m_v3.reset();
  }

  fn update(&mut self, position: &TxnPosition, chains: &TxnChainsList, update_turn: Stone, update_action: Action) {
    self.legality.update(position, chains, update_turn, update_action);
    self.alpha_v3.update(position, chains, update_turn, update_action);
    self.alpha_m_v3.update(position, chains, update_turn, update_action);
  }
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
      let (_, sgf_entry, gnugo_entry) = sgf_row.unwrap();
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

  let mut action_space = vec![];
  action_space.push(Action::Resign);
  action_space.push(Action::Pass);
  for p in 0 .. Board::SIZE {
    action_space.push(Action::Place{point: Point::from_idx(p)});
  }

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
      let (_, sgf_entry, gnugo_entry) = sgf_row
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
          /*[PlayerRank::Dan(9), PlayerRank::Dan(9)],
          ruleset.rules(),
          TxnStateAllData::new(),*/
          TxnStateConfig::default(),
          //TxnStateAllData::new(),
          TxnStateTestFeatsData::new(),
      );
      state.reset();
      let mut actions_history = vec![];
      for (j, &(ref turn_code, ref move_code)) in sgf.moves.iter().enumerate() {
        let turn = Stone::from_code_str(&turn_code);
        let action = Action::from_code_str(&move_code);

        /*// XXX: Debugging output.
        println!("DEBUG: move {}: {:?} {:?}", j, turn, action);
        if let Action::Place{point} = action {
          println!("DEBUG:   move coord: {:?} {:?}", turn, point.to_coord());
        }
        println!("DEBUG: gnugo repr ({}):", j);
        println!("{}", gnugo_entry.gnugo_positions[j]);
        println!("DEBUG: board ({}):", j);
        for s in state.to_debug_strings().iter() {
          println!("DEBUG:   {}", s);
        }*/

        // Check that the state position and legal points match the oracle's
        // (gnugo's) exactly.
        // XXX(20151107): The database was prepared on 11/07/2015.
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
        for p in 0 .. Board::SIZE {
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

        // Check that undo also works as expected for all possible actions.
        // Also check that the given action is valid. 
        thread_rng().shuffle(&mut action_space);
        for &a in action_space.iter() {
          let _ = state.try_action(turn, a);
          state.undo();
        }
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

        // Record the action taken this turn.
        actions_history.push(action);

        // Check AlphaV3 features.
        {
          type Feats = TxnStateAlphaV3FeatsData;
          const SET: u8 = Feats::SET;
          let raw_feats = &state.get_data().alpha_v3.features;

          for p in 0 .. Board::SIZE {
            // Test for baseline ones plane.
            assert_eq!(SET, raw_feats[Feats::BASELINE_PLANE + p]);

            // FIXME(20160218): Test for center distance plane.

            // Test for stones plane.
            let point = Point::from_idx(p);
            match state.current_stone(point) {
              Stone::Black => {
                assert_eq!(SET, raw_feats[Feats::BLACK_PLANE + p]);
              }
              Stone::White => {
                assert_eq!(SET, raw_feats[Feats::WHITE_PLANE + p]);
              }
              Stone::Empty => {
                assert_eq!(SET, raw_feats[Feats::EMPTY_PLANE + p]);
              }
            }

            let head = state.chains.find_chain(point);
            if head != TOMBSTONE {
              let chain = state.chains.get_chain(head).unwrap();

              // Test for liberty counts plane.
              let libs = chain.count_libs_up_to_8();
              match libs {
                0 => { unreachable!(); }
                1 => {
                  assert_eq!(SET, raw_feats[Feats::LIBS_1_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_2_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_3_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_4_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_5_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_6_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_7_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_8_PLANE + p]);
                }
                2 => {
                  assert_eq!(0,   raw_feats[Feats::LIBS_1_PLANE + p]);
                  assert_eq!(SET, raw_feats[Feats::LIBS_2_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_3_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_4_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_5_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_6_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_7_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_8_PLANE + p]);
                }
                3 => {
                  assert_eq!(0,   raw_feats[Feats::LIBS_1_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_2_PLANE + p]);
                  assert_eq!(SET, raw_feats[Feats::LIBS_3_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_4_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_5_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_6_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_7_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_8_PLANE + p]);
                }
                4 => {
                  assert_eq!(0,   raw_feats[Feats::LIBS_1_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_2_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_3_PLANE + p]);
                  assert_eq!(SET, raw_feats[Feats::LIBS_4_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_5_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_6_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_7_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_8_PLANE + p]);
                }
                5 => {
                  assert_eq!(0,   raw_feats[Feats::LIBS_1_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_2_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_3_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_4_PLANE + p]);
                  assert_eq!(SET, raw_feats[Feats::LIBS_5_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_6_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_7_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_8_PLANE + p]);
                }
                6 => {
                  assert_eq!(0,   raw_feats[Feats::LIBS_1_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_2_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_3_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_4_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_5_PLANE + p]);
                  assert_eq!(SET, raw_feats[Feats::LIBS_6_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_7_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_8_PLANE + p]);
                }
                7 => {
                  assert_eq!(0,   raw_feats[Feats::LIBS_1_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_2_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_3_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_4_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_5_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_6_PLANE + p]);
                  assert_eq!(SET, raw_feats[Feats::LIBS_7_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_8_PLANE + p]);
                }
                8 => {
                  assert_eq!(0,   raw_feats[Feats::LIBS_1_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_2_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_3_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_4_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_5_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_6_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_7_PLANE + p]);
                  assert_eq!(SET, raw_feats[Feats::LIBS_8_PLANE + p]);
                }
                _ => { unreachable!(); }
              }

              // Test for chain sizes plane.
              let size = chain.count_length();
              match size {
                0 => { unreachable!(); }
                1 => {
                  assert_eq!(SET, raw_feats[Feats::CAPS_1_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_2_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_3_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_4_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_5_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_6_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_7_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_8_PLANE + p]);
                }
                2 => {
                  assert_eq!(0,   raw_feats[Feats::CAPS_1_PLANE + p]);
                  assert_eq!(SET, raw_feats[Feats::CAPS_2_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_3_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_4_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_5_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_6_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_7_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_8_PLANE + p]);
                }
                3 => {
                  assert_eq!(0,   raw_feats[Feats::CAPS_1_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_2_PLANE + p]);
                  assert_eq!(SET, raw_feats[Feats::CAPS_3_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_4_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_5_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_6_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_7_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_8_PLANE + p]);
                }
                4 => {
                  assert_eq!(0,   raw_feats[Feats::CAPS_1_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_2_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_3_PLANE + p]);
                  assert_eq!(SET, raw_feats[Feats::CAPS_4_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_5_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_6_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_7_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_8_PLANE + p]);
                }
                5 => {
                  assert_eq!(0,   raw_feats[Feats::CAPS_1_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_2_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_3_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_4_PLANE + p]);
                  assert_eq!(SET, raw_feats[Feats::CAPS_5_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_6_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_7_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_8_PLANE + p]);
                }
                6 => {
                  assert_eq!(0,   raw_feats[Feats::CAPS_1_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_2_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_3_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_4_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_5_PLANE + p]);
                  assert_eq!(SET, raw_feats[Feats::CAPS_6_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_7_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_8_PLANE + p]);
                }
                7 => {
                  assert_eq!(0,   raw_feats[Feats::CAPS_1_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_2_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_3_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_4_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_5_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_6_PLANE + p]);
                  assert_eq!(SET, raw_feats[Feats::CAPS_7_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_8_PLANE + p]);
                }
                _ => {
                  assert_eq!(0,   raw_feats[Feats::CAPS_1_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_2_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_3_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_4_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_5_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_6_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_7_PLANE + p]);
                  assert_eq!(SET, raw_feats[Feats::CAPS_8_PLANE + p]);
                }
              }

            } else {
              assert_eq!(0,   raw_feats[Feats::LIBS_1_PLANE + p]);
              assert_eq!(0,   raw_feats[Feats::LIBS_2_PLANE + p]);
              assert_eq!(0,   raw_feats[Feats::LIBS_3_PLANE + p]);
              assert_eq!(0,   raw_feats[Feats::LIBS_4_PLANE + p]);
              assert_eq!(0,   raw_feats[Feats::LIBS_5_PLANE + p]);
              assert_eq!(0,   raw_feats[Feats::LIBS_6_PLANE + p]);
              assert_eq!(0,   raw_feats[Feats::LIBS_7_PLANE + p]);
              assert_eq!(0,   raw_feats[Feats::LIBS_8_PLANE + p]);

              assert_eq!(0,   raw_feats[Feats::CAPS_1_PLANE + p]);
              assert_eq!(0,   raw_feats[Feats::CAPS_2_PLANE + p]);
              assert_eq!(0,   raw_feats[Feats::CAPS_3_PLANE + p]);
              assert_eq!(0,   raw_feats[Feats::CAPS_4_PLANE + p]);
              assert_eq!(0,   raw_feats[Feats::CAPS_5_PLANE + p]);
              assert_eq!(0,   raw_feats[Feats::CAPS_6_PLANE + p]);
              assert_eq!(0,   raw_feats[Feats::CAPS_7_PLANE + p]);
              assert_eq!(0,   raw_feats[Feats::CAPS_8_PLANE + p]);
            }
          }

          // FIXME(20160218): Test for player ranks.

          // XXX(20160218): Test for previous turns.
          let horizon = actions_history.len();
          for lag in 0 .. min(8, horizon) {
            match actions_history[horizon - lag - 1] {
              Action::Place{point} => {
                let plane = match lag {
                  0 => Feats::TURNS_1_PLANE,
                  1 => Feats::TURNS_2_PLANE,
                  2 => Feats::TURNS_3_PLANE,
                  3 => Feats::TURNS_4_PLANE,
                  4 => Feats::TURNS_5_PLANE,
                  5 => Feats::TURNS_6_PLANE,
                  6 => Feats::TURNS_7_PLANE,
                  7 => Feats::TURNS_8_PLANE,
                  _ => { unreachable!(); }
                };
                let p = point.idx();
                // FIXME(20160218): this test could be more precise.
                assert_eq!(SET, raw_feats[plane + p]);
              }
              _ => {}
            }
          }

          // XXX(20160218): Test for ko.
          match state.current_ko() {
            None => {
              for p in 0 .. Board::SIZE {
                assert_eq!(0,   raw_feats[Feats::KO_PLANE + p]);
              }
            }
            Some((_, ko_point)) => {
              let ko_p = ko_point.idx();
              for p in 0 .. Board::SIZE {
                if p == ko_p {
                  assert_eq!(SET, raw_feats[Feats::KO_PLANE + p]);
                } else {
                  assert_eq!(0,   raw_feats[Feats::KO_PLANE + p]);
                }
              }
            }
          }
        }

        // Check for AlphaMiniV3 features.
        {
          type Feats = TxnStateAlphaMiniV3FeatsData;
          const SET: u8 = Feats::SET;
          let raw_feats = &state.get_data().alpha_m_v3.features;

          for p in 0 .. Board::SIZE {
            // Test for baseline ones plane.
            assert_eq!(SET, raw_feats[Feats::BASELINE_PLANE + p]);

            // Test for stones plane.
            let point = Point::from_idx(p);
            match state.current_stone(point) {
              Stone::Black => {
                assert_eq!(SET, raw_feats[Feats::BLACK_PLANE + p]);
              }
              Stone::White => {
                assert_eq!(SET, raw_feats[Feats::WHITE_PLANE + p]);
              }
              Stone::Empty => {
                assert_eq!(SET, raw_feats[Feats::EMPTY_PLANE + p]);
              }
            }

            let head = state.chains.find_chain(point);
            if head != TOMBSTONE {
              let chain = state.chains.get_chain(head).unwrap();

              // Test for liberty counts plane.
              let libs = chain.count_libs_up_to_4();
              match libs {
                0 => { unreachable!(); }
                1 => {
                  assert_eq!(SET, raw_feats[Feats::LIBS_1_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_2_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_3_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_4_PLANE + p]);
                }
                2 => {
                  assert_eq!(0,   raw_feats[Feats::LIBS_1_PLANE + p]);
                  assert_eq!(SET, raw_feats[Feats::LIBS_2_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_3_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_4_PLANE + p]);
                }
                3 => {
                  assert_eq!(0,   raw_feats[Feats::LIBS_1_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_2_PLANE + p]);
                  assert_eq!(SET, raw_feats[Feats::LIBS_3_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_4_PLANE + p]);
                }
                4 => {
                  assert_eq!(0,   raw_feats[Feats::LIBS_1_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_2_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::LIBS_3_PLANE + p]);
                  assert_eq!(SET, raw_feats[Feats::LIBS_4_PLANE + p]);
                }
                _ => { unreachable!(); }
              }

              // Test for chain sizes plane.
              let size = chain.count_length();
              match size {
                0 => { unreachable!(); }
                1 => {
                  assert_eq!(SET, raw_feats[Feats::CAPS_1_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_2_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_3_PLANE + p]);
                }
                2 => {
                  assert_eq!(0,   raw_feats[Feats::CAPS_1_PLANE + p]);
                  assert_eq!(SET, raw_feats[Feats::CAPS_2_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_3_PLANE + p]);
                }
                _ => {
                  assert_eq!(0,   raw_feats[Feats::CAPS_1_PLANE + p]);
                  assert_eq!(0,   raw_feats[Feats::CAPS_2_PLANE + p]);
                  assert_eq!(SET, raw_feats[Feats::CAPS_3_PLANE + p]);
                }
              }

            } else {
              assert_eq!(0,   raw_feats[Feats::LIBS_1_PLANE + p]);
              assert_eq!(0,   raw_feats[Feats::LIBS_2_PLANE + p]);
              assert_eq!(0,   raw_feats[Feats::LIBS_3_PLANE + p]);
              assert_eq!(0,   raw_feats[Feats::LIBS_4_PLANE + p]);

              assert_eq!(0,   raw_feats[Feats::CAPS_1_PLANE + p]);
              assert_eq!(0,   raw_feats[Feats::CAPS_2_PLANE + p]);
              assert_eq!(0,   raw_feats[Feats::CAPS_3_PLANE + p]);
            }
          }

          // XXX(20160218): Test for previous turns.
          let horizon = actions_history.len();
          for lag in 0 .. min(4, horizon) {
            match actions_history[horizon - lag - 1] {
              Action::Place{point} => {
                let plane = match lag {
                  0 => Feats::TURNS_1_PLANE,
                  1 => Feats::TURNS_2_PLANE,
                  2 => Feats::TURNS_3_PLANE,
                  3 => Feats::TURNS_4_PLANE,
                  _ => { unreachable!(); }
                };
                let p = point.idx();
                // FIXME(20160218): this test could be more precise.
                assert_eq!(SET, raw_feats[plane + p]);
              }
              _ => {}
            }
          }

          // XXX(20160218): Test for ko.
          match state.current_ko() {
            None => {
              for p in 0 .. Board::SIZE {
                assert_eq!(0,   raw_feats[Feats::KO_PLANE + p]);
              }
            }
            Some((_, ko_point)) => {
              let ko_p = ko_point.idx();
              for p in 0 .. Board::SIZE {
                if p == ko_p {
                  assert_eq!(SET, raw_feats[Feats::KO_PLANE + p]);
                } else {
                  assert_eq!(0,   raw_feats[Feats::KO_PLANE + p]);
                }
              }
            }
          }
        }

        /*// Check position features.
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
              let libs = state.current_libs_up_to_3(point);
              assert_eq!(if libs == 1 { 1 } else { 0 }, state.get_data().libfeats.current_feature(1, point));
              assert_eq!(if libs == 2 { 1 } else { 0 }, state.get_data().libfeats.current_feature(2, point));
              assert_eq!(if libs == 3 { 1 } else { 0 }, state.get_data().libfeats.current_feature(3, point));
              assert_eq!(0, state.get_data().libfeats.current_feature(5, point));
              assert_eq!(0, state.get_data().libfeats.current_feature(6, point));
              assert_eq!(0, state.get_data().libfeats.current_feature(7, point));
            }
            Stone::White => {
              let libs = state.current_libs_up_to_3(point);
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
        }*/
      }

      total_count += 1;
    }

    assert_eq!(expected_count, total_count);
  }
}
