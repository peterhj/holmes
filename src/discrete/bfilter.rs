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
    let marks = BitVec::from_elem(half_cap, false);
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
    // XXX: Remaining part of heap data should always be zeroed.
    (&mut self.heap.data[self.leaf_idx .. self.leaf_idx + self.n]).clone_from_slice(xs);
    for idx in (0 .. self.leaf_idx).rev() {
      let left_idx = self.heap.left(idx);
      let right_idx = self.heap.right(idx);
      self.heap.data[idx] = self.heap.data[left_idx] + self.heap.data[right_idx];
    }
    self.marks.clear();
    self.zeros.clear();
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
    let prev_value = self.heap.data[idx];
    loop {
      self.heap.data[idx] -= prev_value;
      if idx == 0 {
        break;
      }
      let parent_idx = self.heap.parent(idx);
      self.marks.set(2 * parent_idx + 1 - (idx % 2), true);
      idx = parent_idx;
    }
  }

  pub fn sample<R: Rng>(&self, rng: &mut R) -> Option<usize> {
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
      let u = rng.gen_range(0.0, self.heap.data[idx]);
      let value = self.heap.data[left_idx];
      if u < value && !self.marks.get(2 * idx).unwrap() {
        idx = left_idx;
      } else if !self.marks.get(2 * idx + 1).unwrap() {
        idx = right_idx;
      } else {
        unreachable!();
      }
    }
    Some(idx - self.leaf_idx)
  }
}
