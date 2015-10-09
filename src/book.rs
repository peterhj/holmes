use fastboard::{PosExt, Pos, Stone, FastBoard};
use gtp_board::{Coord};
use table::{TransTable};

use bit_set::{BitSet};
use rand::{Rng};
use std::fs::{File};
use std::io::{Read, BufRead, BufReader};
use std::path::{Path};

#[derive(Clone, Debug)]
pub struct BookEntry {
  turn:   Stone,
  digest: BitSet,
  plays:  Vec<Pos>,
}

#[derive(Clone, Debug)]
pub struct OpeningBook {
  table:    TransTable,
  entries:  Vec<BookEntry>,
}

impl OpeningBook {
  pub fn load_fuego<R>(path: &Path, rng: &mut R) -> OpeningBook where R: Rng {
    // Fuego opening books are text files, where each line follows the syntax:
    //
    //    <board_dim> [match_sequence] | [play_sequence]
    //
    // The current board is matched w/ symmetry on the book's set of configurations,
    // and one of the corresponding plays are selected.
    let file = File::open(path)
      .ok().expect("failed to open opening book file!");
    let mut reader = BufReader::new(file);
    let mut line_count = 0;
    for line in reader.lines() {
      line_count += 1;
    }
    let mut table = TransTable::new(line_count * 16, rng);
    let mut entries: Vec<BookEntry> = Vec::new();
    let file = File::open(path)
      .ok().expect("failed to open opening book file!");
    let mut reader = BufReader::new(file);
    for line in reader.lines() {
      let line = line.unwrap();
      let toks: Vec<&str> = line.split_whitespace().collect();
      if toks.len() == 0 {
        continue;
      }
      let mut board_dim: Option<usize> = None;
      board_dim = toks[0].parse().ok();
      match board_dim {
        Some(19) => {}
        _ => continue,
      }
      let mut matches = Vec::new();
      let mut plays = Vec::new();
      let mut turn = Stone::Black;
      let mut i = 1;
      while i < toks.len() {
        if toks[i] == "|" {
          i += 1;
          break;
        }
        let coord = Coord::from_code(toks[i].as_bytes());
        let pos = Pos::from_coord(coord);
        matches.push((turn, pos));
        turn = turn.opponent();
        i += 1;
      }
      while i < toks.len() {
        let coord = Coord::from_code(toks[i].as_bytes());
        let pos = Pos::from_coord(coord);
        plays.push(pos);
        i += 1;
      }
      for (f, r) in (0 .. 4).zip(0 .. 4) {
        let mut hash: u64 = 0;
        let mut digest = BitSet::with_capacity(FastBoard::BOARD_SIZE * 2);
        for &(stone, pos) in matches.iter() {
          let pos = pos.flip(f).rot(r);
          match stone {
            Stone::Black => {
              hash ^= table.key_stone(stone, pos);
              digest.insert(pos.idx() * 2);
            }
            Stone::White => {
              hash ^= table.key_stone(stone, pos);
              digest.insert(pos.idx() * 2 + 1);
            }
            _ => unreachable!(),
          }
        }
        let plays: Vec<_> = plays.iter()
          .map(|&p| p.flip(f).rot(r))
          .collect();
        let entry = BookEntry{turn: turn, digest: digest, plays: plays};
        let new_id = entries.len();
        if table.insert(hash, new_id, |cmp_id| entries[cmp_id].digest == entry.digest).is_none() {
          entries.push(entry);
        }
      }
    }
    let mut book = OpeningBook{
      table:    table,
      entries:  entries,
    };
    book
  }

  pub fn lookup(&self, turn: Stone, digest: &BitSet) -> Option<&[Pos]> {
    let mut hash: u64 = 0;
    for x in digest.iter() {
      let p = x / 2;
      let s = match x & 1 {
        0 => Stone::Black,
        1 => Stone::White,
        _ => unreachable!(),
      };
      hash ^= self.table.key_stone(s, p as Pos);
    }
    if let Some(query_id) = self.table.find(hash, |cmp_id| &self.entries[cmp_id].digest == digest) {
      if self.entries[query_id].turn == turn {
        return Some(&self.entries[query_id].plays);
      }
    }
    None
  }
}
