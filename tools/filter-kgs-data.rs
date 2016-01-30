extern crate holmes;
extern crate rustc_serialize;

use holmes::sgf::{Sgf, Property, RootProperty, GameInfoProperty, parse_raw_sgf};

//use std::collections::{BTreeMap};
use std::fs::{File};
use std::io::{Read, BufRead, Write, BufReader};
use std::path::{PathBuf};

fn main() {
  let index_path = PathBuf::from("/data0/go/kgs-ugo-preproc/index");
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
  println!("sgf paths: {}", sgf_paths.len());
  //println!("{:?}", sgf_paths[0]);
  //println!("{:?}", sgf_paths[1]);
  //println!("{:?}", sgf_paths[sgf_paths.len() - 1]);

  //let mut jsonindex_path = PathBuf::from("/data0/go/kgs-ugo-preproc/index.json");
  //let mut jsonindex = File::create(&jsonindex_path).unwrap();

  //let mut filterindex_path = PathBuf::from("/data0/go/kgs-ugo-preproc/filtered_index.v3");
  let mut filterindex_path = PathBuf::from("/data0/go/kgs-ugo-preproc/filtered_index.v3a");
  let mut filterindex = File::create(&filterindex_path).unwrap();
  let mut filter_count = 0;

  let mut size19_count = 0;
  let mut japanese_count = 0;
  let mut chinese_count = 0;
  let mut new_zealand_count = 0;
  let mut aga_count = 0;
  let mut handicap_counts = Vec::new();
  let mut no_handicap_count = 0;
  let mut nohand_05komi_count = 0;
  let mut nohand_55komi_count = 0;
  let mut nohand_65komi_count = 0;
  let mut nohand_75komi_count = 0;
  for _ in (0 .. 10) {
    handicap_counts.push(0);
  }
  for (i, sgf_path) in sgf_paths.iter().enumerate() {
    if (i+1) % 1000 == 0 {
      println!("DEBUG: {} / {}: {:?}", i+1, sgf_paths.len(), sgf_path);
    }

    let mut sgf_file = File::open(sgf_path).unwrap();
    let mut text = Vec::new();
    sgf_file.read_to_end(&mut text).unwrap();
    let raw_sgf = parse_raw_sgf(&text);

    let mut has_handicap = false;
    let mut komi = None;

    let mut is_19 = false;
    let mut is_japanese = false;
    let mut is_chinese = false;

    for property in raw_sgf.nodes[0].properties.iter() {
      match property {
        &Property::Root(ref root) => match root {
          &RootProperty::BoardSize(sz) => {
            if sz == 19 {
              size19_count += 1;
              is_19 = true;
            }
          }
          _ => {}
        },
        &Property::GameInfo(ref game_info) => match game_info {
          &GameInfoProperty::Rules(ref rules) => {
            let rules = rules.to_lowercase();
            if rules == "japanese" {
              japanese_count += 1;
              is_japanese = true;
            } else if rules == "chinese" {
              chinese_count += 1;
              is_chinese = true;
            } else if rules == "nz" {
              new_zealand_count += 1;
            } else if rules == "aga" {
              aga_count += 1;
            } else {
              println!("DEBUG: unrecognized ruleset: {}", rules);
            }
          }
          &GameInfoProperty::GoKomi(k) => {
            komi = k;
          }
          &GameInfoProperty::GoHandicap(handicap) => {
            handicap_counts[handicap.unwrap() as usize] += 1;
            has_handicap = true;
          }
          _ => {}
        },
        _ => {}
      }
    }
    if !has_handicap {
      no_handicap_count += 1;
      if let Some(komi) = komi {
        if (komi - 6.5).abs() <= 0.01 {
          assert_eq!(komi, 6.5);
          nohand_65komi_count += 1;
        } else if (komi - 0.5).abs() <= 0.01 {
          assert_eq!(komi, 0.5);
          nohand_05komi_count += 1;
        } else if (komi - 5.5).abs() <= 0.01 {
          assert_eq!(komi, 5.5);
          nohand_55komi_count += 1;
        } else if (komi - 7.5).abs() <= 0.01 {
          assert_eq!(komi, 7.5);
          nohand_75komi_count += 1;
        }
      }
    }

    if is_19 && (is_japanese || is_chinese) {
      writeln!(filterindex, "{}", sgf_path.to_str().unwrap());
      filter_count += 1;
    }
  }
  println!("19x19: {}", size19_count);
  println!("J: {} C: {} NZ: {} A: {}",
      japanese_count, chinese_count, new_zealand_count, aga_count);
  println!("handicaps: {:?}", handicap_counts);
  println!("nohand: {}", no_handicap_count);
  println!("nohand 0.5 komi: {}", nohand_05komi_count);
  println!("nohand 5.5 komi: {}", nohand_55komi_count);
  println!("nohand 6.5 komi: {}", nohand_65komi_count);
  println!("nohand 7.5 komi: {}", nohand_75komi_count);
  println!("filtered: {}", filter_count);
}
