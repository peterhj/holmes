use fastboard::{IntoIndex, Pos, Stone, FastBoard};
use util::{ceil_power2};

use bit_vec::{BitVec};
use rand::{Rng};
use std::collections::{HashMap};

pub struct TransTable {
  cap:        u64,
  keys:       Vec<u64>,
  flags:      BitVec,
  unique:     Vec<usize>,
  collisions: HashMap<u64, Vec<usize>>,
}

impl TransTable {
  pub fn new<R>(min_capacity: usize, rng: &mut R) -> TransTable where R: Rng {
    let capacity = ceil_power2(min_capacity as u64) as usize;
    assert!(capacity >= min_capacity);
    let mut keys = Vec::with_capacity(FastBoard::BOARD_SIZE * 3);
    for _ in (0 .. FastBoard::BOARD_SIZE * 3) {
      keys.push(rng.next_u64());
    }
    let mut flags = BitVec::from_elem(capacity * 2, false);
    let mut unique = Vec::with_capacity(capacity);
    unsafe { unique.set_len(capacity) };
    TransTable{
      cap:        capacity as u64,
      keys:       keys,
      flags:      flags,
      unique:     unique,
      collisions: HashMap::new(),
    }
  }

  pub fn key_stone(&self, pos: Pos, stone: Stone) -> u64 {
    let idx = pos.idx() * 3 + stone.offset();
    self.keys[idx]
  }

  pub fn key_ko_point(&self, pos: Pos) -> u64 {
    let idx = pos.idx() * 3 + 2;
    self.keys[idx]
  }

  #[inline]
  fn index(&self, hash: u64) -> usize {
    (hash & self.cap) as usize
  }

  #[inline]
  fn contains(&self, index: usize) -> bool {
    self.flags.get(index * 2).unwrap()
  }

  #[inline]
  fn contains_many(&self, index: usize) -> bool {
    self.flags.get(index * 2 + 1).unwrap()
  }

  pub fn clear(&mut self) {
    self.flags.clear();
  }

  pub fn find<F>(&self, hash: u64, query: F) -> Option<usize> where F: Fn(usize) -> bool {
    let index = self.index(hash);
    if self.contains(index) {
      if !self.contains_many(index) {
        let id = self.unique[index];
        if query(id) {
          return Some(id);
        }
      } else {
        for &id in self.collisions.get(&hash).unwrap().iter() {
          if query(id) {
            return Some(id);
          }
        }
      }
    }
    None
  }

  pub fn insert(&mut self, hash: u64, id: usize) {
    let index = self.index(hash);
    if !self.contains(index) {
      self.flags.set(index * 2, true);
      self.unique[index] = id;
    } else {
      if !self.contains_many(index) {
        let first_id = self.unique[index];
        self.flags.set(index * 2 + 1, true);
        self.collisions.insert(hash, vec![first_id, id]);
      } else {
        self.collisions.get_mut(&hash).unwrap().push(id);
      }
    }
  }
}
