use discrete::{DiscreteFilter};
use util::{ceil_power2};

use bit_vec::{BitVec};
use rand::{Rng};
use rand::distributions::{IndependentSample};
use rand::distributions::range::{Range};
use std::iter::{repeat};

#[derive(Clone)]
pub struct BitVec32 {
  length: usize,
  data:   Vec<u32>,
}

impl BitVec32 {
  pub fn with_capacity(n: usize) -> BitVec32 {
    let cap = (n + 32 - 1)/32;
    let mut data = Vec::with_capacity(cap);
    for _ in (0 .. cap) {
      data.push(0);
    }
    BitVec32{
      length: n,
      data:   data,
    }
  }

  pub fn as_slice(&self) -> &[u32] {
    &self.data
  }

  pub fn as_mut_slice(&mut self) -> &mut [u32] {
    &mut self.data
  }

  pub fn clear(&mut self) {
    for x in self.data.iter_mut() {
      *x = 0;
    }
  }

  pub fn get(&self, idx: usize) -> Option<bool> {
    //assert!(idx < self.length);
    if idx < self.length {
      let (word_idx, bit_idx) = (idx / 32, idx % 32);
      Some(0x0 != (0x1 & (self.data[word_idx] >> bit_idx)))
    } else {
      None
    }
  }

  pub fn set(&mut self, idx: usize, value: bool) {
    assert!(idx < self.length);
    let (word_idx, bit_idx) = (idx / 32, idx % 32);
    self.data[word_idx] &= !(0x1 << bit_idx);
    if value {
      self.data[word_idx] |= (0x1 << bit_idx);
    }
  }

  pub fn insert(&mut self, idx: usize) {
    self.set(idx, true);
  }

  pub fn remove(&mut self, idx: &usize) {
    self.set(*idx, false);
  }

  pub fn contains(&self, idx: &usize) -> bool {
    self.get(*idx).unwrap()
  }
}

#[derive(Clone)]
pub struct BHeap<T> {
  data: Vec<T>,
}

// XXX(20151112): Using 1-based binary heap array indexing convention:
// <http://www.cse.hut.fi/en/research/SVG/TRAKLA2/tutorials/heap_tutorial/taulukkona.html>

impl<T> BHeap<T> {
  pub fn with_capacity(leaf_level_n: usize, init: T) -> BHeap<T> where T: Copy {
    //let cap = 2 * ceil_power2(n as u64) as usize - 1;
    let cap = 2 * ceil_power2(leaf_level_n as u64) as usize;
    let data: Vec<_> = repeat(init).take(cap).collect();
    BHeap{
      data: data,
    }
  }

  #[inline]
  pub fn root(&self) -> usize {
    //0
    1
  }

  #[inline]
  pub fn parent(&self, idx: usize) -> usize {
    //(idx + 1) / 2 - 1
    idx / 2
  }

  #[inline]
  pub fn sibling(&self, idx: usize) -> usize {
    //idx + 2 * (idx % 2) - 1
    idx + 1 - 2 * (idx % 2)
  }

  #[inline]
  pub fn left(&self, idx: usize) -> usize {
    //2 * (idx + 1) - 1
    2 * idx
  }

  #[inline]
  pub fn right(&self, idx: usize) -> usize {
    //2 * (idx + 1)
    2 * idx + 1
  }

  #[inline]
  pub fn level(&self, level: usize) -> usize {
    //(1 << level) - 1
    1 << level
  }
}

#[derive(Clone)]
pub struct BFilter {
  max_n:    usize,
  leaf_idx: usize,
  n:        usize,
  heap:     BHeap<f32>,
  marks:    BitVec32,
  zeros:    BitVec32,
}

impl BFilter {
  pub fn with_capacity(max_n: usize) -> BFilter {
    let heap = BHeap::with_capacity(max_n, 0.0);
    //let half_cap = ceil_power2(n as u64) as usize - 1;
    //let cap = 2 * ceil_power2(n as u64) as usize - 1;
    let half_cap = ceil_power2(max_n as u64) as usize;
    let cap = 2 * ceil_power2(max_n as u64) as usize;
    let marks = BitVec32::with_capacity(2 * half_cap);
    let zeros = BitVec32::with_capacity(cap - half_cap);
    BFilter{
      max_n:    max_n,
      leaf_idx: half_cap,
      n:        0,
      heap:     heap,
      marks:    marks,
      zeros:    zeros,
    }
  }

  #[inline]
  fn mark_idx(&self, idx: usize) -> usize {
    //2 * self.heap.parent(idx) + 1 - (idx % 2)
    2 * self.heap.parent(idx) + (idx % 2)
  }

  #[inline]
  fn mark_sibling_idx(&self, idx: usize) -> usize {
    //2 * self.heap.parent(idx) + (idx % 2)
    2 * self.heap.parent(idx) + 1 - (idx % 2)
  }

  #[inline]
  fn mark_left_idx(&self, parent_idx: usize) -> usize {
    2 * parent_idx
  }

  #[inline]
  fn mark_right_idx(&self, parent_idx: usize) -> usize {
    2 * parent_idx + 1
  }

  pub fn capacity(&self) -> usize {
    self.max_n
  }

  pub fn unsafe_reset(&mut self) {
    self.n = self.max_n;
  }

  pub fn get_mut_parts(&mut self) -> (&mut [f32], &mut [u32], &mut [u32]) {
    (&mut self.heap.data, &mut self.marks.data, &mut self.zeros.data)
  }
}

impl DiscreteFilter for BFilter {
  fn reset(&mut self, xs: &[f32]) {
    assert_eq!(self.max_n, xs.len());
    self.n = self.max_n;
    self.marks.clear();
    self.zeros.clear();

    (&mut self.heap.data[self.leaf_idx .. self.leaf_idx + self.n]).clone_from_slice(xs);

    // XXX: Remaining part of heap data should always be zeroed.
    for idx in (self.leaf_idx + self.n .. self.heap.data.len()) {
      let mark_idx = self.mark_idx(idx);
      self.marks.set(mark_idx, true);
      self.zeros.set(idx - self.leaf_idx, true);
    }
    for idx in (self.heap.root() .. self.leaf_idx).rev() {
      let left_idx = self.heap.left(idx);
      let right_idx = self.heap.right(idx);
      self.heap.data[idx] = self.heap.data[left_idx] + self.heap.data[right_idx];
      if self.marks.get(self.mark_left_idx(idx)).unwrap() && self.marks.get(self.mark_right_idx(idx)).unwrap() {
        if idx > 0 {
          let mark_idx = self.mark_idx(idx);
          self.marks.set(mark_idx, true);
        }
      }
    }
  }

  fn zero(&mut self, j: usize) {
    if self.zeros.get(j).unwrap() {
      return;
    }
    self.n -= 1;
    self.zeros.set(j, true);
    if self.n == 0 {
      return;
    }

    let root = self.heap.root();
    let mut idx = self.leaf_idx + j;
    let mut do_mark = true;
    self.heap.data[idx] = 0.0;
    loop {
      if idx == root {
        break;
      }
      let parent_idx = self.heap.parent(idx);
      // XXX: Note: subtracting the zeroed out value, instead of recomputing the
      // sum, leads to roundoff errors which seriously mess up the sampling part
      // of the data structure.
      self.heap.data[parent_idx] = self.heap.data[idx] + self.heap.data[self.heap.sibling(idx)];
      if do_mark {
        let mark_idx = self.mark_idx(idx);
        let mark_sib_idx = self.mark_sibling_idx(idx);
        self.marks.set(mark_idx, true);
        if !self.marks.get(mark_sib_idx).unwrap() {
          do_mark = false;
        }
      }
      idx = parent_idx;
    }
  }

  fn sample<R: Rng>(&mut self, rng: &mut R) -> Option<usize> {
    if self.n == 0 {
      return None;
    }
    let mut idx = self.heap.root();
    loop {
      if idx >= self.leaf_idx {
        break;
      }
      let left_idx = self.heap.left(idx);
      let right_idx = self.heap.right(idx);
      match (self.marks.get(self.mark_left_idx(idx)).unwrap(), self.marks.get(self.mark_right_idx(idx)).unwrap()) {
        (false, false) => {
          let value = self.heap.data[idx];
          //assert!(value > 0.0); // XXX: Range already checks this.
          //println!("DEBUG: gen_range value: {}", value);
          if !(value > 0.0) {
            fn bfs_zero(filter: &mut BFilter, idx: usize) {
              let leaf_idx = filter.leaf_idx;
              if idx >= leaf_idx {
                filter.zero(idx - leaf_idx);
              } else {
                let left_idx = filter.heap.left(idx);
                let right_idx = filter.heap.right(idx);
                bfs_zero(filter, left_idx);
                bfs_zero(filter, right_idx);
              }
            };
            bfs_zero(self, idx);
            return self.sample(rng);
          }
          let left_value = self.heap.data[left_idx];
          let u = rng.gen_range(0.0, value);
          if u < left_value {
            idx = left_idx;
          } else {
            idx = right_idx;
          }
        }
        (false, true) => {
          idx = left_idx;
        }
        (true, false) => {
          idx = right_idx;
        }
        _ => {
          unreachable!();
        }
      }
    }
    Some(idx - self.leaf_idx)
  }
}
