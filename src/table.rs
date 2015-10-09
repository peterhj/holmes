use fastboard::{PosExt, Pos, Stone, FastBoard};
use util::{ceil_power2};

use bit_vec::{BitVec};
use rand::{Rng};
use std::collections::{HashMap};
use std::iter::{repeat};

pub type TransTable = TranspositionTable;

#[derive(Clone, Debug)]
pub struct TranspositionTable {
  mask:       u64,
  keys:       Vec<u64>,
  flags:      BitVec,
  unique:     Vec<(u64, usize)>,
  collisions: HashMap<usize, Vec<(u64, usize)>>,
}

impl TranspositionTable {
  pub fn new<R>(min_capacity: usize, rng: &mut R) -> TranspositionTable where R: Rng {
    let capacity = ceil_power2(min_capacity as u64) as usize;
    assert!(capacity >= min_capacity);
    let mut keys = Vec::with_capacity(FastBoard::BOARD_SIZE * 3);
    for _ in (0 .. FastBoard::BOARD_SIZE * 3) {
      keys.push(rng.next_u64());
    }
    let mut flags = BitVec::from_elem(capacity * 2, false);
    let mut unique = Vec::with_capacity(capacity);
    //unique.extend(repeat((0, 0)).take(capacity));
    unsafe { unique.set_len(capacity) };
    let mask = capacity as u64 - 1;
    TranspositionTable{
      mask:       mask,
      keys:       keys,
      flags:      flags,
      unique:     unique,
      collisions: HashMap::new(),
    }
  }

  pub fn key_stone(&self, stone: Stone, pos: Pos) -> u64 {
    let idx = pos.idx() * 3 + stone.offset();
    self.keys[idx]
  }

  pub fn key_ko_point(&self, pos: Pos) -> u64 {
    let idx = pos.idx() * 3 + 2;
    self.keys[idx]
  }

  #[inline]
  fn index(&self, hash: u64) -> usize {
    (hash & self.mask) as usize
  }

  #[inline]
  fn contains(&self, index: usize) -> bool {
    self.flags.get(index * 2).expect("contains failed!")
  }

  #[inline]
  fn contains_many(&self, index: usize) -> bool {
    self.flags.get(index * 2 + 1).expect("contains_many failed!")
  }

  pub fn clear(&mut self) {
    self.flags.clear();
  }

  pub fn find<F>(&self, hash: u64, cmp: F) -> Option<usize> where F: Fn(usize) -> bool {
    let index = self.index(hash);
    if self.contains(index) {
      if !self.contains_many(index) {
        let (h, id) = self.unique[index];
        if h == hash && cmp(id) {
          return Some(id);
        }
      } else {
        for &(h, id) in self.collisions.get(&index).unwrap().iter() {
          if h == hash && cmp(id) {
            return Some(id);
          }
        }
      }
    }
    None
  }

  pub fn insert<F>(&mut self, hash: u64, new_id: usize, cmp: F) -> Option<usize> where F: Fn(usize) -> bool {
    let index = self.index(hash);
    if !self.contains(index) {
      self.flags.set(index * 2, true);
      self.unique[index] = (hash, new_id);
      None
    } else {
      if !self.contains_many(index) {
        let (first_hash, first_id) = self.unique[index];
        if first_hash == hash && cmp(first_id) {
          Some(first_id)
        } else {
          self.flags.set(index * 2 + 1, true);
          self.collisions.insert(index, vec![(first_hash, first_id), (hash, new_id)]);
          None
        }
      } else {
        for &(h, id) in self.collisions.get(&index).unwrap() {
          if h == hash && cmp(id) {
            return Some(id);
          }
        }
        self.collisions.get_mut(&index).expect("get_mut failed!").push((hash, new_id));
        None
      }
    }
  }
}
