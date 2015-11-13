use util::{ceil_power2};

use bit_vec::{BitVec};
use rand::{Rng};
use rand::distributions::{IndependentSample};
use rand::distributions::range::{Range};
use std::iter::{repeat};

#[derive(Clone)]
pub struct BHeap<T> {
  data: Vec<T>,
}

// XXX(20151112): Using 0-based binary heap array indexing convention.
// See following for 1-based indexing:
// <http://www.cse.hut.fi/en/research/SVG/TRAKLA2/tutorials/heap_tutorial/taulukkona.html>

impl<T> BHeap<T> {
  pub fn with_capacity(n: usize, init: T) -> BHeap<T> where T: Copy {
    let cap = 2 * ceil_power2(n as u64) as usize - 1;
    let data: Vec<_> = repeat(init).take(cap).collect();
    BHeap{
      data: data,
    }
  }

  #[inline]
  pub fn parent(&self, idx: usize) -> usize {
    (idx + 1) / 2 - 1
  }

  #[inline]
  pub fn sibling(&self, idx: usize) -> usize {
    idx + 2 * (idx % 2) - 1
  }

  #[inline]
  pub fn left(&self, idx: usize) -> usize {
    2 * (idx + 1) - 1
  }

  #[inline]
  pub fn right(&self, idx: usize) -> usize {
    2 * (idx + 1)
  }

  #[inline]
  pub fn level(&self, level: usize) -> usize {
    (1 << level) - 1
  }
}

#[derive(Clone)]
pub struct BFilter {
  max_n:    usize,
  leaf_idx: usize,
  n:        usize,
  heap:     BHeap<f32>,
  marks:    BitVec,
  zeros:    BitVec,
  init:     bool,
}

impl BFilter {
  pub fn with_capacity(n: usize) -> BFilter {
    let heap = BHeap::with_capacity(n, 0.0);
    let half_cap = ceil_power2(n as u64) as usize - 1;
    let cap = 2 * ceil_power2(n as u64) as usize - 1;
    let marks = BitVec::from_elem(2 * half_cap, false);
    let zeros = BitVec::from_elem(cap - half_cap, false);
    BFilter{
      max_n:    n,
      leaf_idx: half_cap,
      n:        n,
      heap:     heap,
      marks:    marks,
      zeros:    zeros,
      init:     false,
    }
  }

  pub fn fill(&mut self, xs: &[f32]) {
    assert_eq!(self.max_n, xs.len());
    self.n = self.max_n;
    self.marks.clear();
    self.zeros.clear();
    // XXX: Remaining part of heap data should always be zeroed.
    (&mut self.heap.data[self.leaf_idx .. self.leaf_idx + self.n]).clone_from_slice(xs);
    for idx in (self.leaf_idx + self.n .. self.heap.data.len()) {
      let parent_idx = self.heap.parent(idx);
      self.marks.set(2 * parent_idx + 1 - (idx % 2), true);
      self.zeros.set(idx - self.leaf_idx, true);
    }
    for idx in (0 .. self.leaf_idx).rev() {
      let left_idx = self.heap.left(idx);
      let right_idx = self.heap.right(idx);
      self.heap.data[idx] = self.heap.data[left_idx] + self.heap.data[right_idx];
      if self.marks.get(2 * idx).unwrap() && self.marks.get(2 * idx + 1).unwrap() {
        if idx > 0 {
          self.marks.set(2 * self.heap.parent(idx) + 1 - (idx % 2), true);
        }
      }
    }
    self.init = true;
  }

  pub fn zero(&mut self, j: usize) {
    assert!(self.init);
    if self.zeros.get(j).unwrap() {
      return;
    }
    self.n -= 1;
    self.zeros.set(j, true);
    if self.n == 0 {
      return;
    }
    let mut idx = self.leaf_idx + j;
    let mut do_mark = true;
    self.heap.data[idx] = 0.0;
    loop {
      if idx == 0 {
        break;
      }
      let parent_idx = self.heap.parent(idx);
      // XXX: Note: subtracting the zeroed out value, instead of recomputing the
      // sum, leads to roundoff errors which seriously mess up the sampling part
      // of the data structure.
      self.heap.data[parent_idx] = self.heap.data[idx] + self.heap.data[self.heap.sibling(idx)];
      if do_mark {
        self.marks.set(2 * parent_idx + 1 - (idx % 2), true);
        if !self.marks.get(2 * parent_idx + (idx % 2)).unwrap() {
          do_mark = false;
        }
      }
      idx = parent_idx;
    }
  }

  pub fn sample<R: Rng>(&mut self, rng: &mut R) -> Option<usize> {
    assert!(self.init);
    if self.n == 0 {
      return None;
    }
    let mut idx = 0;
    loop {
      if idx >= self.leaf_idx {
        break;
      }
      let left_idx = self.heap.left(idx);
      let right_idx = self.heap.right(idx);
      match (self.marks.get(2 * idx).unwrap(), self.marks.get(2 * idx + 1).unwrap()) {
        (false, false) => {
          let value = self.heap.data[idx];
          let left_value = self.heap.data[left_idx];
          /*let right_value = self.heap.data[right_idx];
          if 0.0 < value {*/
          //assert!(value > 0.0); // XXX: Range already checks this.
          let u = rng.gen_range(0.0, value);
          if u < left_value {
            idx = left_idx;
          } else {
            idx = right_idx;
          }
          /*} else {
            println!("DEBUG: idx: {} value: {} left: {} {} right: {} {}",
                idx, value, left_idx, self.heap.data[left_idx], right_idx, self.heap.data[right_idx]);
            if left_value != 0.0 {
              idx = left_idx;
            } else if right_value != 0.0 {
              idx = right_idx;
            } else {
              unreachable!();
            }
          }*/
        }
        (false, true) => {
          idx = left_idx;
        }
        (true, false) => {
          idx = right_idx;
        }
        _ => {
          /*println!("WARNING: idx: {} value: {} left: {} {} right: {} {}",
              idx, self.heap.data[idx], left_idx, self.heap.data[left_idx], right_idx, self.heap.data[right_idx]);
          println!("WARNING: heap data: {:?}", self.heap.data);*/
          unreachable!();
        }
      }
    }
    Some(idx - self.leaf_idx)
  }
}
