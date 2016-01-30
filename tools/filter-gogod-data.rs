extern crate holmes;
extern crate rustc_serialize;

//use holmes::game::{GameHistory};
use holmes::sgf::{Sgf, Property, RootProperty, GameInfoProperty, parse_raw_sgf};

use std::collections::{BTreeMap};
//use std::env;
use std::fs::{File};
use std::io::{Read, BufRead, Write, BufReader};
use std::path::{PathBuf};

fn main() {
  let index_path = PathBuf::from("/data0/go/gogodb_w2015-preproc/index");
  //let index_path = PathBuf::from("/data0/go/gogodb-preproc/index");
  //let index_path = PathBuf::from("/data0/go/kgs-ugo-preproc/index");
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

  let mut filterindex_path = PathBuf::from("/data0/go/gogodb_w2015-preproc/filtered_index.v3");
  //let mut filterindex_path = PathBuf::from("/data0/go/kgs-ugo-preproc/filtered_index");
  let mut filterindex = File::create(&filterindex_path).unwrap();

  let mut size19_count = 0;

  let mut japanese_count = 0;
  let mut chinese_count = 0;
  let mut new_zealand_count = 0;
  let mut aga_count = 0;
  let mut other_count = 0;

  let mut china_count = 0;

  let mut handicap_counts = BTreeMap::new();
  let mut no_handicap_count = 0;
  let mut nohand_05komi_count = 0;
  let mut nohand_275komi_count = 0;
  let mut nohand_325komi_count = 0;
  let mut nohand_375komi_count = 0;
  let mut nohand_55komi_count = 0;
  let mut nohand_65komi_count = 0;
  let mut nohand_75komi_count = 0;
  /*for _ in (0 .. 40) {
    handicap_counts.push(0);
  }*/

  let mut filter_count = 0;

  for (i, sgf_path) in sgf_paths.iter().enumerate() {
    if (i+1) % 1000 == 0 { // || (i+1) >= 12000 {
      println!("DEBUG: {} / {}: {:?}", i+1, sgf_paths.len(), sgf_path);
    }
    let mut sgf_file = File::open(sgf_path).unwrap();
    let mut text = Vec::new();
    sgf_file.read_to_end(&mut text).unwrap();
    /*if i >= 85000 {
      println!("DEBUG: {} {:?}", i, sgf_path);
    }*/
    let raw_sgf = parse_raw_sgf(&text);

    let mut has_handicap = false;
    let mut komi = None;

    let mut is_semimodern = false;
    let mut is_modern = false;
    let mut is_19 = false;
    let mut is_nohand = false;
    let mut is_goodkomi = false;
    let mut is_komi65 = false;
    let mut is_japanese = false;

    let path_toks: Vec<_> = sgf_path.file_name().unwrap()
      .to_str().unwrap().split("-").collect();
    let year: Option<i32> = path_toks[0].parse().ok();
    if let Some(year) = year {
      if year >= 1800 {
        is_semimodern = true;
      }
      if year >= 1950 {
        is_modern = true;
      }
    }

    for property in raw_sgf.nodes[0].properties.iter() {
      match property {
        &Property::Root(ref root) => match root {
          &RootProperty::BoardSize(sz) => {
            if sz == 19 {
              is_19 = true;
              size19_count += 1;
            }
          }
          _ => {}
        },
        &Property::GameInfo(ref game_info) => match game_info {
          &GameInfoProperty::Place(ref place) => {
            let place = place.to_lowercase();
            if place.contains("beijing") {
              china_count += 1;
            } else if place.contains("shanghai") {
              china_count += 1;
            }
          }
          &GameInfoProperty::Rules(ref rules) => {
            let rules = rules.to_lowercase();
            if rules == "japanese" {
              is_japanese = true;
              japanese_count += 1;
            } else if rules == "chinese" {
              chinese_count += 1;
            } else if rules == "nz" {
              new_zealand_count += 1;
            } else if rules == "aga" {
              aga_count += 1;
            } else {
              //println!("DEBUG: unrecognized ruleset: {}", rules);
              other_count += 1;
            }
          }
          &GameInfoProperty::GoKomi(k) => {
            komi = k;
          }
          &GameInfoProperty::GoHandicap(handicap) => {
            if let Some(handicap) = handicap {
              if !handicap_counts.contains_key(&handicap) {
                handicap_counts.insert(handicap, 0);
              }
              *handicap_counts.get_mut(&handicap).unwrap() += 1;
              has_handicap = true;
            }
          }
          _ => {}
        },
        _ => {}
      }
    }
    if !has_handicap {
      is_nohand = true;
      no_handicap_count += 1;
      if let Some(komi) = komi {
        if (komi - 6.5).abs() <= 0.01 {
          assert_eq!(komi, 6.5);
          is_goodkomi = true;
          is_komi65 = true;
          nohand_65komi_count += 1;
        } else if (komi - 0.5).abs() <= 0.01 {
          assert_eq!(komi, 0.5);
          nohand_05komi_count += 1;
        } else if (komi - 2.75).abs() <= 0.01 {
          assert_eq!(komi, 2.75);
          is_goodkomi = true;
          nohand_275komi_count += 1;
        } else if (komi - 3.25).abs() <= 0.01 {
          assert_eq!(komi, 3.25);
          nohand_325komi_count += 1;
        } else if (komi - 3.75).abs() <= 0.01 {
          assert_eq!(komi, 3.75);
          is_goodkomi = true;
          nohand_375komi_count += 1;
        } else if (komi - 5.5).abs() <= 0.01 {
          assert_eq!(komi, 5.5);
          is_goodkomi = true;
          nohand_55komi_count += 1;
        } else if (komi - 7.5).abs() <= 0.01 {
          assert_eq!(komi, 7.5);
          nohand_75komi_count += 1;
        }
      }
    }
    if is_semimodern && is_19 {
    //if is_modern && is_19 {
    //if is_modern && is_19 && is_nohand && is_goodkomi {
    //if is_modern && is_19 && is_nohand && is_komi65 && is_japanese { // XXX: KGS filter.
      let sgf = Sgf::from_raw(&raw_sgf);
      //if let Some(game) = GameHistory::new(sgf) {
        writeln!(filterindex, "{}", sgf_path.to_str().unwrap());
        filter_count += 1;
      //}
    }
  }
  println!("19x19: {}", size19_count);
  println!("china: {}", china_count);
  println!("J: {} C: {} NZ: {} A: {} other: {}",
      japanese_count, chinese_count, new_zealand_count, aga_count, other_count);
  println!("handicaps: {:?}", handicap_counts);
  println!("nohand: {}", no_handicap_count);
  println!("nohand 0.5 komi: {}", nohand_05komi_count);
  println!("nohand 2.75 komi: {}", nohand_275komi_count);
  println!("nohand 3.25 komi: {}", nohand_325komi_count);
  println!("nohand 3.75 komi: {}", nohand_375komi_count);
  println!("nohand 5.5 komi: {}", nohand_55komi_count);
  println!("nohand 6.5 komi: {}", nohand_65komi_count);
  println!("nohand 7.5 komi: {}", nohand_75komi_count);
  println!("filtered: {}", filter_count);
}
