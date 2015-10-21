extern crate array;
extern crate arraydb;
extern crate byteorder;
extern crate holmes;
extern crate rustc_serialize;

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
  let labels_db_path = PathBuf::from(&args[3]);

  let index_file = BufReader::new(File::open(&index_path).unwrap());
  let mut index_lines = Vec::new();
  for line in index_file.lines() {
    let line = line.unwrap();
    index_lines.push(line);
  }
  /*index_lines.sort_by(|a, b| {
    let a_toks: Vec<_> = a.rsplitn(2, '/').collect();
    let b_toks: Vec<_> = b.rsplitn(2, '/').collect();
    a_toks[0].cmp(&b_toks[0])
  });*/
  let sgf_paths: Vec<_> = index_lines.iter()
    .map(|line| PathBuf::from(line)).collect();
  println!("num sgf paths: {}", sgf_paths.len());

  //let frame_sz = GameHistory::feature_frame_size();
  //let serial_frame_sz = <Array3d<u8> as ArrayDeserialize<u8, NdArrayFormat>>::serial_size((19, 19, 4));
  let serial_frame_sz = <Array3d<u8> as ArrayDeserialize<u8, NdArrayFormat>>::serial_size((19, 19, 2));
  let n = sgf_paths.len() * 200;
  let mut frames_db = ArrayDb::create(frames_db_path, n, serial_frame_sz);
  let mut labels_db = ArrayDb::create(labels_db_path, n, size_of::<i32>());
  for (i, sgf_path) in sgf_paths.iter().enumerate() {
    if (i+1) % 1000 == 0 {
      println!("DEBUG: {} / {}: {:?}", i+1, sgf_paths.len(), sgf_path);
    }
    let mut sgf_file = File::open(sgf_path).unwrap();
    let mut text = Vec::new();
    sgf_file.read_to_end(&mut text).unwrap();
    let raw_sgf = parse_raw_sgf(&text);
    let sgf = Sgf::from_raw(&raw_sgf);
    if let Some(game) = GameHistory::new(sgf) {
      let features = game.extract_features();
      for (j, &(ref frame, label)) in features.iter().enumerate() {
        assert!(label >= 0 && label < 361);
        let mut serial_frame = Vec::new();
        frame.as_view().serialize(&mut serial_frame);
        assert_eq!(serial_frame_sz, serial_frame.len());
        //frames_db.append(frame.as_view().as_slice());
        frames_db.append(&serial_frame);
        let mut label_cursor = Cursor::new(Vec::with_capacity(4));
        label_cursor.write_i32::<LittleEndian>(label).unwrap();
        let label_bytes = label_cursor.get_ref();
        assert_eq!(4, label_bytes.len());
        //if (i+1) % 10 == 0 && j == 20 {
        if i == 0 {
          println!("DEBUG: label: {} label bytes: {:?}", label, label_bytes);
        }
        labels_db.append(label_bytes);
      }
    }
  }
}
