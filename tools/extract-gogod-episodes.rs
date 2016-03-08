//extern crate array;
extern crate array_new;
extern crate byteorder;
extern crate episodb;
extern crate holmes;
extern crate rustc_serialize;

use holmes::board::{RuleSet, Coord, PlayerRank, Stone, Point, Action};
use holmes::sgf::{Sgf, parse_raw_sgf};
use holmes::txnstate::{TxnStateConfig, TxnState};
use holmes::txnstate::features::{
  /*TxnStateFeaturesData,
  TxnStateLibFeaturesData,
  TxnStateExtLibFeatsData,
  TxnStateAlphaFeatsV1Data,
  TxnStateAlphaFeatsV2Data,*/
  TxnStateAlphaV3FeatsData,
  TxnStateAlphaMiniV3FeatsData,
};

//use array::{NdArrayFormat, ArrayDeserialize, ArraySerialize, Array3d};
use array_new::{NdArraySerialize, Array3d, BitArray3d};
use byteorder::{WriteBytesExt, LittleEndian};
use episodb::{EpisoDb};

use std::env;
use std::fs::{File};
use std::io::{Read, BufRead, BufReader, Cursor};
use std::iter::{repeat};
use std::mem::{size_of};
use std::path::{PathBuf};

fn main() {
  let args: Vec<_> = env::args().collect();
  let prefix = args[1].clone();
  let suffix = args[2].clone();
  let index_path = PathBuf::from(&format!("{}_index", prefix));
  let frames_db_path = PathBuf::from(&format!("{}_frames_{}.episodb", prefix, suffix));
  let action_labels_db_path = PathBuf::from(&format!("{}_labels_action_{}.episodb", prefix, suffix));
  let value_labels_db_path = PathBuf::from(&format!("{}_labels_value_{}.episodb", prefix, suffix));

  let index_file = BufReader::new(File::open(&index_path).unwrap());
  let mut index_lines = Vec::new();
  for line in index_file.lines() {
    let line = line.unwrap();
    index_lines.push(line);
  }
  let sgf_paths: Vec<_> = index_lines.iter()
    .map(|line| PathBuf::from(line)).collect();
  println!("num sgf paths: {}", sgf_paths.len());

  //let frame_sz = GameHistory::feature_frame_size();
  //let serial_frame_sz = <Array3d<u8> as ArrayDeserialize<u8, NdArrayFormat>>::serial_size((19, 19, 2));
  //let serial_frame_sz = <Array3d<u8> as ArrayDeserialize<u8, NdArrayFormat>>::serial_size((19, 19, 4));
  //let serial_frame_sz = <Array3d<u8> as ArrayDeserialize<u8, NdArrayFormat>>::serial_size((19, 19, 10));

  //let expected_dims = (19, 19, 16);
  //let expected_dims = (19, 19, 28);
  //let expected_dims = (19, 19, 37);
  //let expected_dims = (19, 19, 44);
  //let expected_dims = (19, 19, 32);
  //let expected_dims = (19, 19, 16);

  //let expected_frame_sz = <Array3d<u8> as ArrayDeserialize<u8, NdArrayFormat>>::serial_size(expected_dims);
  //let expected_frame_sz = Array3d::<u8>::serial_size(expected_dims);
  //let expected_frame_sz = BitArray3d::serial_size(expected_dims);
  //let expected_frame_sz = BitArray3d::serial_size((19, 19, 43)) + Array3d::<u8>::serial_size((19, 19, 1));
  //let expected_frame_sz = BitArray3d::serial_size((19, 19, 31)) + Array3d::<u8>::serial_size((19, 19, 1));
  //let expected_frame_sz = BitArray3d::serial_size(expected_dims);

  //type FeatsData = TxnStateAlphaV3FeatsData;
  type FeatsData = TxnStateAlphaMiniV3FeatsData;

  let expected_frame_sz = FeatsData::serial_size();

  let n = sgf_paths.len();
  let est_ep_len = 216;

  let mut frames_db = EpisoDb::create(frames_db_path, n, est_ep_len, expected_frame_sz);
  let mut action_labels_db = EpisoDb::create(action_labels_db_path, n, est_ep_len, size_of::<i32>());
  let mut value_labels_db = EpisoDb::create(value_labels_db_path, n, est_ep_len, size_of::<i32>());
/*
  let mut frames_db = EpisoDb::open_read_write(frames_db_path);
  let mut action_labels_db = EpisoDb::open_read_write(action_labels_db_path);
  let mut value_labels_db = EpisoDb::open_read_write(value_labels_db_path);
*/
  let mut num_skipped = 0;
  let mut num_positions = 0;
  for (i, sgf_path) in sgf_paths.iter().enumerate() {
    if (i+1) % 1000 == 0 {
      println!("DEBUG: {} / {}: {:?} num skipped positions: {} num positions: {}",
          i+1, sgf_paths.len(), sgf_path, num_skipped, num_positions);
    }
    let mut sgf_file = match File::open(sgf_path) {
      Ok(file) => file,
      Err(_) => {
        println!("WARNING: extract: failed to open sgf file: '{:?}'", sgf_path);
        continue;
      }
    };
    let mut text = Vec::new();
    sgf_file.read_to_end(&mut text).unwrap();
    let raw_sgf = parse_raw_sgf(&text);
    let sgf = Sgf::from_raw(&raw_sgf);

    fn parse_rank(text: &str, i: usize, sgf_path: &PathBuf) -> Vec<PlayerRank> {
      if text.contains("&") {
        let rank_texts: Vec<_> = text.split("&").collect();
        repeat(PlayerRank::Dan(1)).take(rank_texts.len()).collect()
      } else if text.contains(",") {
        let rank_texts: Vec<_> = text.split(",").collect();
        repeat(PlayerRank::Dan(1)).take(rank_texts.len()).collect()
      } else if text.contains("+") {
        let rank_texts: Vec<_> = text.split("+").collect();
        repeat(PlayerRank::Dan(1)).take(rank_texts.len()).collect()
      } else if text.contains(" to ") {
        let rank_texts: Vec<_> = text.split("to").collect();
        repeat(PlayerRank::Dan(1)).take(rank_texts.len()).collect()
      } else if text.contains("Both") {
        let rank_texts: Vec<_> = text.split("Both").collect();
        let one_rank = parse_rank(rank_texts[1].trim(), i, sgf_path);
        vec![one_rank[0], one_rank[0]]
      } else if text.contains("total") {
        // XXX(20160104): in `19x19/2005-08-00a.sgf` the rank applied to a
        // group, '250 dan total'.
        vec![PlayerRank::Dan(1), PlayerRank::Dan(1)]
      } else if text.contains("Honinbo") {
        // XXX(20160103): This is for a ridiculous case, `19x19/1982-08-00a.sgf`
        // where Black has rank 'Fukui Pref. Honinbo'. Should be before 'Hon'.
        vec![PlayerRank::Dan(9)]
      } else if text.contains("3rd") {
        let rank_texts: Vec<_> = text.splitn(2, "3rd").collect();
        parse_rank(rank_texts[1].trim(), i, sgf_path)
      } else if text.contains("Prov.") {
        let rank_texts: Vec<_> = text.splitn(2, "Prov.").collect();
        parse_rank(rank_texts[1].trim(), i, sgf_path)
      } else if text.contains("Ex") {
        let rank_texts: Vec<_> = text.splitn(2, "Ex").collect();
        parse_rank(rank_texts[1].trim(), i, sgf_path)
      } else if text.contains("Hon.") {
        let rank_texts: Vec<_> = text.splitn(2, "Hon.").collect();
        parse_rank(rank_texts[1].trim(), i, sgf_path)
      } else if text.contains("Hon") {
        let rank_texts: Vec<_> = text.splitn(2, "Hon").collect();
        parse_rank(rank_texts[1].trim(), i, sgf_path)
      } else if text.contains("Shogi") {
        let rank_texts: Vec<_> = text.splitn(2, "Shogi").collect();
        parse_rank(rank_texts[1].trim(), i, sgf_path)
      } else if text.contains("Insei") {
        vec![PlayerRank::Ama(7)]
      } else if text.contains("Holder") {
        vec![PlayerRank::Dan(9)]
      } else if text.contains("Challenger") {
        vec![PlayerRank::Dan(9)]
      } else if text.contains("Meijin") {
        // XXX(20160103): `19x19/1996-05-06a.sgf` has 'Shogi Meijin'.
        vec![PlayerRank::Dan(9)]
      } else if text.contains("ama") {
        let rank_texts: Vec<_> = text.splitn(2, "ama").collect();
        match rank_texts[0].trim() {
          ""    => vec![PlayerRank::Ama(7)],
          "8k"  => vec![PlayerRank::Kyu(8)],
          "7k"  => vec![PlayerRank::Kyu(7)],
          "6k"  => vec![PlayerRank::Kyu(6)],
          "5k"  => vec![PlayerRank::Kyu(5)],
          "4k"  => vec![PlayerRank::Kyu(4)],
          "3k"  => vec![PlayerRank::Kyu(3)],
          "2k"  => vec![PlayerRank::Kyu(2)],
          "1k"  => vec![PlayerRank::Kyu(1)],
          "1d"  => vec![PlayerRank::Ama(1)],
          "2d"  => vec![PlayerRank::Ama(2)],
          "3d"  => vec![PlayerRank::Ama(3)],
          "4d"  => vec![PlayerRank::Ama(4)],
          "5d"  => vec![PlayerRank::Ama(5)],
          "6d"  => vec![PlayerRank::Ama(6)],
          "7d"  => vec![PlayerRank::Ama(7)],
          "8d"  => vec![PlayerRank::Ama(8)],
          "9d"  => vec![PlayerRank::Ama(9)],
          x     => panic!("index: {} sgf path: {:?} unknown 'ama' rank: {}", i, sgf_path, x),
        }
      } else if text.contains("am") {
        let rank_texts: Vec<_> = text.splitn(2, "am").collect();
        match rank_texts[0].trim() {
          ""    => vec![PlayerRank::Ama(7)],
          "8k"  => vec![PlayerRank::Kyu(8)],
          "7k"  => vec![PlayerRank::Kyu(7)],
          "6k"  => vec![PlayerRank::Kyu(6)],
          "5k"  => vec![PlayerRank::Kyu(5)],
          "4k"  => vec![PlayerRank::Kyu(4)],
          "3k"  => vec![PlayerRank::Kyu(3)],
          "2k"  => vec![PlayerRank::Kyu(2)],
          "1k"  => vec![PlayerRank::Kyu(1)],
          "1d"  => vec![PlayerRank::Ama(1)],
          "2d"  => vec![PlayerRank::Ama(2)],
          "3d"  => vec![PlayerRank::Ama(3)],
          "4d"  => vec![PlayerRank::Ama(4)],
          "5d"  => vec![PlayerRank::Ama(5)],
          "6d"  => vec![PlayerRank::Ama(6)],
          "7d"  => vec![PlayerRank::Ama(7)],
          "8d"  => vec![PlayerRank::Ama(8)],
          "9d"  => vec![PlayerRank::Ama(9)],
          x     => panic!("index: {} sgf path: {:?} unknown 'am' rank: {}", i, sgf_path, x),
        }
      } else if text.contains("insei") {
        let rank_texts: Vec<_> = text.splitn(2, "insei").collect();
        match rank_texts[0].trim() {
          ""    => vec![PlayerRank::Ama(7)],
          "8k"  => vec![PlayerRank::Kyu(8)],
          "7k"  => vec![PlayerRank::Kyu(7)],
          "6k"  => vec![PlayerRank::Kyu(6)],
          "5k"  => vec![PlayerRank::Kyu(5)],
          "4k"  => vec![PlayerRank::Kyu(4)],
          "3k"  => vec![PlayerRank::Kyu(3)],
          "2k"  => vec![PlayerRank::Kyu(2)],
          "1k"  => vec![PlayerRank::Kyu(1)],
          "1d"  => vec![PlayerRank::Ama(1)],
          "2d"  => vec![PlayerRank::Ama(2)],
          "3d"  => vec![PlayerRank::Ama(3)],
          "4d"  => vec![PlayerRank::Ama(4)],
          "5d"  => vec![PlayerRank::Ama(5)],
          "6d"  => vec![PlayerRank::Ama(6)],
          "7d"  => vec![PlayerRank::Ama(7)],
          "8d"  => vec![PlayerRank::Ama(8)],
          "9d"  => vec![PlayerRank::Ama(9)],
          x     => panic!("index: {} sgf path: {:?} unknown 'insei' rank: {}", i, sgf_path, x),
        }
      } else if text.contains("prov") {
        let rank_texts: Vec<_> = text.splitn(2, "prov").collect();
        parse_rank(rank_texts[0].trim(), i, sgf_path)
      } else if text.contains("pro") {
        let rank_texts: Vec<_> = text.splitn(2, "pro").collect();
        parse_rank(rank_texts[0].trim(), i, sgf_path)
      } else if text.contains("hon.") {
        let rank_texts: Vec<_> = text.splitn(2, "hon.").collect();
        parse_rank(rank_texts[0].trim(), i, sgf_path)
      } else if text.contains("{Chinese}") {
        let rank_texts: Vec<_> = text.splitn(2, "{Chinese}").collect();
        parse_rank(rank_texts[0].trim(), i, sgf_path)
      } else if text.contains("(Keiinsha)") {
        let rank_texts: Vec<_> = text.splitn(2, "(Keiinsha)").collect();
        parse_rank(rank_texts[0].trim(), i, sgf_path)
      } else if text.is_empty() {
        vec![]
      } else {
        vec![match text {
          "?" => PlayerRank::Ama(7),
          "Ama" | "Amateur" => PlayerRank::Ama(7),
          "Insei" | "insei" => PlayerRank::Ama(7),
          "Gisung" | "Kisung" |
          "Gosei" |
          "Honinbo" |
          "Judan" | "Siptan" |
          "Kisei" |
          "Meijin" | "Mingren" | "Myungin" |
          "Oza" |
          "Tengen" | "Tianyuan" => PlayerRank::Dan(9),
          "1k" | "2k" | "3k" | "4k" | "5k" |
          "6k" | "7k" | "8k" | "9k" | "10k" |
          "11k" | "12k" | "13k" | "14k" | "15k" |
          "16k" | "17k" | "18k" | "19k" | "20k" => PlayerRank::Kyu(1),
          "1a" => PlayerRank::Ama(1),
          "2a" => PlayerRank::Ama(2),
          "3a" => PlayerRank::Ama(3),
          "4a" => PlayerRank::Ama(4),
          "5a" => PlayerRank::Ama(5),
          "6a" => PlayerRank::Ama(6),
          "7a" => PlayerRank::Ama(7),
          "8a" => PlayerRank::Ama(8),
          "9a" => PlayerRank::Ama(9),
          "1d" | "1p" => PlayerRank::Dan(1),
          "2d" | "2p" => PlayerRank::Dan(2),
          "3d" | "3p" => PlayerRank::Dan(3),
          "4d" | "4p" => PlayerRank::Dan(4),
          "5d" | "5p" => PlayerRank::Dan(5),
          "6d" | "6p" => PlayerRank::Dan(6),
          "7d" | "7p" => PlayerRank::Dan(7),
          "8d" | "8p" => PlayerRank::Dan(8),
          "9d" | "9p" => PlayerRank::Dan(9),
          // XXX(20150104): `19x19/1978-09-00l.sgf`.
          "4.3a" => PlayerRank::Ama(4),
          // XXX(20150104): `19x19/1996-01-16a.sgf`.
          "6d*" => PlayerRank::Dan(6),
          // XXX(20150104): `19x19/1994-05-07c.sgf`.
          "7d*" => PlayerRank::Dan(7),
          s => {
            panic!("index: {} sgf_path: {:?}, unimplemented player rank: \"{}\"", i, sgf_path, s);
          }
        }]
      }
    }

    /*// FIXME(20160129): disabling rank parsing for now.
    let b_rank = PlayerRank::Dan(9);
    let w_rank = PlayerRank::Dan(9);*/

    let b_ranks = parse_rank(&sgf.black_rank, i, sgf_path);
    let w_ranks = parse_rank(&sgf.white_rank, i, sgf_path);
    let ranks = match (b_ranks.len(), w_ranks.len()) {
      (1, 1) => vec![b_ranks[0], w_ranks[0]],
      (1, 0) => vec![b_ranks[0], PlayerRank::Dan(1)],
      (0, 1) => vec![PlayerRank::Dan(1), w_ranks[0]],
      (0, 0) => vec![PlayerRank::Dan(1), PlayerRank::Dan(1)],
      (x, y) => {
        //panic!("sgf_path: {:?}, unexpected number of player ranks: {:?} {:?}",
        //    sgf_path, b_ranks, w_ranks);
        println!("WARNING: extract: skipping relay game ({} {}), sgf_path: {:?}",
            x, y, sgf_path);
        continue;
      }
    };
    let b_rank = ranks[0];
    let w_rank = ranks[1];

    let outcome = {
      let result_toks: Vec<_> = sgf.result.split("+").collect();
      //println!("DEBUG: outcome tokens: {:?}", result_toks);
      if result_toks.is_empty() {
        None
      } else {
        let first_tok = result_toks[0].split_whitespace().next();
        match first_tok {
          Some("B") => {
            Some(Stone::Black)
          }
          Some("W") => {
            Some(Stone::White)
          }
          Some("?") | Some("Jigo") | Some("Left") | Some("Void") => {
            None
          }
          _ => {
            println!("WARNING: extract: unimplemented outcome: \"{}\" sgf path: '{:?}'", sgf.result, sgf_path);
            None
          }
        }
      }
    };

    if outcome.is_none() {
      continue;
    }

    let mut history = vec![];
    let mut state = TxnState::new(
        TxnStateConfig{
          rules:  RuleSet::KgsJapanese.rules(),
          ranks:  [b_rank, w_rank],
          komi:   6.5,
        },
        FeatsData::new(),
    );
    state.reset();
    for (t, &(ref player, ref mov)) in sgf.moves.iter().enumerate() {
      let turn = match player as &str {
        "B" => Stone::Black,
        "W" => Stone::White,
        _ => unimplemented!(),
      };
      // XXX(20160119): Set state turn the first time if necessary.
      if turn != state.current_turn() {
        if t == 0 {
          state.unsafe_set_current_turn(turn);
        } else {
          println!("WARNING: extract: repeated turn: sgf path: '{:?}'", sgf_path);
          history.clear();
          break;
        }
      }
      let action = match mov as &str {
        "Pass"    => Action::Pass,
        "Resign"  => Action::Resign,
        x         => Action::Place{point: Point::from_coord(Coord::from_code_str(x))},
      };
      history.push((turn, action, state.clone(), outcome));
      match state.try_action(turn, action) {
        Ok(_) => {
          state.commit();
        }
        Err(_) => {
          println!("WARNING: extract: found an illegal action: sgf path: '{:?}'", sgf_path);
          history.clear();
          break;
        }
      }
    }

    if history.is_empty() {
      continue;
    }

    frames_db.append_episode();
    action_labels_db.append_episode();
    value_labels_db.append_episode();

    for (t, &(turn, action, ref state, outcome)) in history.iter().enumerate() {
      if t >= 1 {
        // XXX(20151209): if 2 turns in a row, skip the rest of this game.
        if history[t-1].0 == turn {
          println!("WARNING: extract: duplicated turns: sgf path: '{:?}'", sgf_path);
          num_skipped += history.len() - t;
          break;
        }
      }
      let action_label = match action {
        Action::Place{point} => {
          point.0 as i32
        }
        // XXX(20160129): Trying out new label system. Resigns are 361, passes are 362.
        Action::Pass => {
          362
        }
        Action::Resign => {
          /*num_skipped += history.len() - t;
          //continue;
          break;*/
          361
        }
      };
      let value_label = if let Some(outcome) = outcome {
        if outcome == turn { 1i32 }
        else if outcome == turn.opponent() { 0i32 }
        else {
          println!("WARNING: extract: invalid outcome: sgf path: '{:?}'", sgf_path);
          /*num_skipped += 1;
          continue;*/
          unreachable!();
        }
      } else {
        /*num_skipped += 1;
        continue;*/
        unreachable!();
      };

      /*let frame_dims = state.get_data().feature_dims();
      assert_eq!(expected_dims, frame_dims);
      let mut frame_data: Vec<u8> = repeat(0).take(frame_dims.0 * frame_dims.1 * frame_dims.2).collect();
      state.get_data().extract_relative_features(turn, &mut frame_data);
      let frame = Array3d::with_data(frame_data, frame_dims);
      let raw_frame = BitArray3d::from_byte_array(&frame);

      let mut serial_frame = vec![];
      //frame.as_view().serialize(&mut serial_frame);
      raw_frame.serialize(&mut serial_frame);
      assert_eq!(expected_frame_sz, serial_frame.len());*/

      // XXX(20160220): Option 1: Single serialized bit array.
      /*let mut serial_frame: Vec<u8> = Vec::with_capacity(expected_frame_sz);
      let bit_arr = state.get_data().extract_relative_serial_array(turn);
      bit_arr.serialize(&mut serial_frame).unwrap();*/

      // XXX(20160220): Option 2: Serialized into two: bit array and byte array.
      // XXX(20160202): use custom encoding to get 2 arrays for AlphaV2 feats.
      /*let mut serial_frame: Vec<u8> = Vec::with_capacity(expected_frame_sz);
      let (bit_arr, bytes_arr) = state.get_data().extract_relative_serial_arrays(turn);
      bit_arr.serialize(&mut serial_frame).unwrap();
      bytes_arr.serialize(&mut serial_frame).unwrap();*/

      // XXX(20160223): Option 3: Let the feature data return a mystery blob.
      let serial_frame = state.get_data().extract_relative_serial_blob(turn);

      assert_eq!(serial_frame.len(), expected_frame_sz);
      frames_db.append_frame(&serial_frame);

      let mut action_label_cursor = Cursor::new(Vec::with_capacity(4));
      action_label_cursor.write_i32::<LittleEndian>(action_label).unwrap();
      let action_label_bytes = action_label_cursor.get_ref();
      assert_eq!(4, action_label_bytes.len());
      action_labels_db.append_frame(action_label_bytes);

      let mut value_label_cursor = Cursor::new(Vec::with_capacity(4));
      value_label_cursor.write_i32::<LittleEndian>(value_label).unwrap();
      let value_label_bytes = value_label_cursor.get_ref();
      assert_eq!(4, value_label_bytes.len());
      value_labels_db.append_frame(value_label_bytes);

      num_positions += 1;
    }
  }
}
