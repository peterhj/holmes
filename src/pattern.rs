use board::{Board, Stone, Point};
use txnstate::{TxnPosition, TxnChainsList, for_each_x8};

use byteorder::{ReadBytesExt, LittleEndian};

use std::cmp::{min};
use std::collections::{HashMap, HashSet};
use std::fs::{File};
use std::io::{Read, BufRead, BufReader};
use std::path::{Path};

/*static MASK_8_UV_COORDS: [(i8, i8); 8] = [
  (-1, -1), (0, -1), (1, -1),
  (-1, 0), (1, 0),
  (-1, 1), (0, 1), (1, 1),
];*/

static MASK_8_TRANS_MAPS: [[u8; 8]; 8] = [
  [0, 1, 2, 3, 4, 5, 6, 7],
  [2, 1, 0, 4, 3, 7, 6, 5],
  [5, 6, 7, 3, 4, 0, 1, 2],
  [7, 6, 5, 4, 3, 2, 1, 0],
  [0, 3, 5, 1, 6, 2, 4, 7],
  [2, 4, 7, 1, 6, 0, 3, 5],
  [5, 3, 0, 6, 1, 7, 4, 2],
  [7, 4, 2, 6, 1, 5, 3, 0],
];

#[derive(Clone, Copy, Eq, PartialEq, RustcDecodable, RustcEncodable)]
pub struct Pattern3x3(pub u16);

impl Pattern3x3 {
  pub fn to_invariant(self) -> InvariantPattern3x3 {
    let mut min_mask8: u16 = 0xffffffff;
    for t in 0 .. 8 {
      let mut mask8: u16 = 0;
      for i in 0 .. 8 {
        let x = (self.0 >> (2 * i)) & 0x3;
        let t_i = MASK_8_TRANS_MAPS[t][i];
        mask8 |= x << (2 * t_i);
      }
      min_mask8 = min(min_mask8, mask8);
    }
    InvariantPattern3x3(min_mask8)
  }
}

#[derive(Clone, Copy, Eq, PartialEq, RustcDecodable, RustcEncodable)]
pub struct InvariantPattern3x3(u16);

impl InvariantPattern3x3 {
  pub fn idx(self) -> u32 {
    self.0 as u32
  }
}

#[derive(Clone, Copy, Eq, PartialEq, Hash, RustcDecodable, RustcEncodable)]
pub struct LibPattern3x3 {
  pub mask: u32,
}

impl LibPattern3x3 {
  pub fn from_repr_bytes(repr_bytes: &[u8], p: usize) -> LibPattern3x3 {
    let pt = Point::from_idx(p);
    let mut mask = 0;
    for_each_x8(pt, |i, adj_pt| {
      let adj_p = adj_pt.idx();
      let mut st_mask = 0;
      if repr_bytes[adj_p] > 0 {
        st_mask = 0x1;
      } else if repr_bytes[adj_p + Board::SIZE] > 0 {
        st_mask = 0x2;
      } else if repr_bytes[adj_p + 2 * Board::SIZE] > 0 {
        st_mask = 0x3;
      }
      let mut lib_mask = 0;
      if repr_bytes[adj_p + 5 * Board::SIZE] > 0 {
        st_mask = 0x3;
      } else if repr_bytes[adj_p + 4 * Board::SIZE] > 0 {
        st_mask = 0x2;
      } else if repr_bytes[adj_p + 3 * Board::SIZE] > 0 {
        st_mask = 0x1;
      }
      mask |= st_mask << (2*i);
      mask |= lib_mask << (2*i + 16);
    });
    LibPattern3x3{mask: mask}
  }

  pub fn from_state(position: &TxnPosition, chains: &TxnChainsList, turn: Stone, point: Point) -> LibPattern3x3 {
    // FIXME(20160413)
    unimplemented!();
  }

  pub fn to_invariant(self) -> InvariantLibPattern3x3 {
    let mut min_mask: u32 = 0xffffffff;
    for t in 0 .. 8 {
      let mut mask: u32 = 0;
      for i in 0 .. 8 {
        let x = (self.mask >> (2*i)) & 0x3;
        let t_i = MASK_8_TRANS_MAPS[t][i];
        mask |= x << (2*t_i);
      }
      for i in 0 .. 8 {
        let x = (self.mask >> (2*i + 16)) & 0x3;
        let t_i = MASK_8_TRANS_MAPS[t][i];
        mask |= x << (2*t_i + 16);
      }
      min_mask = min(min_mask, mask);
    }
    InvariantLibPattern3x3{index: min_mask}
  }
}

#[derive(Clone, Copy, Eq, PartialEq, Hash, RustcDecodable, RustcEncodable)]
pub struct InvariantLibPattern3x3 {
  pub index: u32,
}

pub struct PatternDatabase {
  num_patterns: usize,
  patterns: HashSet<InvariantLibPattern3x3>,
}

impl PatternDatabase {
  pub fn open(db_path: &Path) -> PatternDatabase {
    let file = match File::open(db_path) {
      Ok(file) => file,
      Err(e) => panic!("failed to open pattern db: {:?}", e),
    };
    let mut reader = BufReader::new(file);
    let num_patterns = reader.read_u32::<LittleEndian>().unwrap() as usize;
    let mut patterns = HashSet::with_capacity(num_patterns);
    for _ in 0 .. num_patterns {
      let index = reader.read_u32::<LittleEndian>().unwrap();
      patterns.insert(InvariantLibPattern3x3{index: index});
    }
    PatternDatabase{
      num_patterns: num_patterns,
      patterns: patterns,
    }
  }
}

pub struct PatternGammaDatabase {
  num_bare_feats:   usize,
  num_pat_feats:    usize,
  bare_gammas:  Vec<f32>,
  pat_gammas:   HashMap<InvariantLibPattern3x3, f32>,
}

impl PatternGammaDatabase {
  pub fn open(db_path: &Path) -> PatternGammaDatabase {
    let file = match File::open(db_path) {
      Ok(file) => file,
      Err(e) => panic!("failed to open pattern db: {:?}", e),
    };
    let mut reader = BufReader::new(file);
    let num_bare_feats = reader.read_u32::<LittleEndian>().unwrap() as usize;
    let num_pat_feats = reader.read_u32::<LittleEndian>().unwrap() as usize;
    let mut bare_gammas = Vec::with_capacity(num_bare_feats);
    for _ in 0 .. num_bare_feats {
      let gamma = reader.read_f32::<LittleEndian>().unwrap();
      bare_gammas.push(gamma);
    }
    let mut pat_gammas = HashMap::with_capacity(num_pat_feats);
    for _ in 0 .. num_pat_feats {
      let index = reader.read_u32::<LittleEndian>().unwrap();
      let pat = InvariantLibPattern3x3{index: index};
      let gamma = reader.read_f32::<LittleEndian>().unwrap();
      pat_gammas.insert(pat, gamma);
    }
    PatternGammaDatabase{
      num_bare_feats:   num_bare_feats,
      num_pat_feats:    num_pat_feats,
      bare_gammas:  bare_gammas,
      pat_gammas:   pat_gammas,
    }
  }
}
