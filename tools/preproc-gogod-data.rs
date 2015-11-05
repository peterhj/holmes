extern crate array;
extern crate arraydb;
extern crate byteorder;
extern crate holmes;
extern crate rustc_serialize;

use holmes::fastboard::{Action};
use holmes::game::{GameHistory};
use holmes::sgf::{Sgf, Property, RootProperty, GameInfoProperty, parse_raw_sgf};

use array::{NdArrayFormat, ArrayDeserialize, ArraySerialize, Array3d};
use arraydb::{ArrayDb};
use byteorder::{WriteBytesExt, LittleEndian};

use rustc_serialize::json;
use std::collections::{BTreeMap};
use std::env;
use std::fs::{File, create_dir_all};
use std::io::{Read, BufRead, Write, BufReader, Cursor};
use std::mem::{size_of};
use std::path::{PathBuf};

fn main() {
  let args: Vec<_> = env::args().collect();
  let index_path = PathBuf::from(&args[1]);
  let frames_db_path = PathBuf::from(&args[2]);
  let move_labels_db_path = PathBuf::from(&args[3]);
  let value_labels_db_path = PathBuf::from(&args[4]);

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

  //let frame_sz = GameHistory::feature_frame_size();
  //let serial_frame_sz = <Array3d<u8> as ArrayDeserialize<u8, NdArrayFormat>>::serial_size((19, 19, 2));
  let serial_frame_sz = <Array3d<u8> as ArrayDeserialize<u8, NdArrayFormat>>::serial_size((19, 19, 4));
  //let serial_frame_sz = <Array3d<u8> as ArrayDeserialize<u8, NdArrayFormat>>::serial_size((19, 19, 10));

  let n = sgf_paths.len() * 210;
  let mut frames_db = ArrayDb::create(frames_db_path, n, serial_frame_sz);
  let mut move_labels_db = ArrayDb::create(move_labels_db_path, n, size_of::<i32>());
  let mut value_labels_db = ArrayDb::create(value_labels_db_path, n, size_of::<i32>());
  let mut num_skipped = 0;
  let mut num_positions = 0;
  for (i, sgf_path) in sgf_paths.iter().enumerate() {
    if (i+1) % 1000 == 0 {
      println!("DEBUG: {} / {}: {:?} num skipped games: {} num positions: {}",
          i+1, sgf_paths.len(), sgf_path, num_skipped, num_positions);
    }
    let mut sgf_file = File::open(sgf_path).unwrap();
    let mut text = Vec::new();
    sgf_file.read_to_end(&mut text).unwrap();
    let raw_sgf = parse_raw_sgf(&text);
    let sgf = Sgf::from_raw(&raw_sgf);
    if let Some(game) = GameHistory::new(sgf) {
      let features = game.extract_features();
      if features.is_empty() {
        num_skipped += 1;
      } else {
        num_positions += features.len();
      }
      for (j, &(_, ref frame, move_label, value_label)) in features.iter().enumerate() {
        assert!(move_label >= 0 && move_label < 361);
        assert!(value_label == 0 || value_label == 1);

        let mut serial_frame = Vec::new();
        frame.as_view().serialize(&mut serial_frame);
        assert_eq!(serial_frame_sz, serial_frame.len());
        frames_db.append(&serial_frame);

        let mut move_label_cursor = Cursor::new(Vec::with_capacity(4));
        move_label_cursor.write_i32::<LittleEndian>(move_label).unwrap();
        let move_label_bytes = move_label_cursor.get_ref();
        assert_eq!(4, move_label_bytes.len());
        /*if i == 0 {
          println!("DEBUG: move_label: {} move_label bytes: {:?}", move_label, move_label_bytes);
        }*/
        move_labels_db.append(move_label_bytes);

        let mut value_label_cursor = Cursor::new(Vec::with_capacity(4));
        value_label_cursor.write_i32::<LittleEndian>(value_label).unwrap();
        let value_label_bytes = value_label_cursor.get_ref();
        assert_eq!(4, value_label_bytes.len());
        /*if i == 0 {
          println!("DEBUG: value_label: {} value_label bytes: {:?}", value_label, value_label_bytes);
        }*/
        value_labels_db.append(value_label_bytes);
      }
    }
  }
}
