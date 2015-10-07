use contains::hash::{MemHasher, XxhMemHasher};
use util::{ceil_power2};

use bit_vec::{BitVec};
//use rand::{Rng, thread_rng};

use std::cmp::{max};
use std::iter::{repeat};

// FIXME(20150202): forcing K, V to be Copy because we are not dropping values
// properly.

#[derive(Clone)]
pub struct OpenCopyHashMap<K, V, H=XxhMemHasher>
where K: Clone + Copy + Eq + PartialEq,
      V: Clone + Copy,
      H: MemHasher
{
  hasher:   H,
  cap:      usize,
  len:      usize,
  occupied: BitVec,
  keys:     Vec<K>,
  values:   Vec<V>,
}

impl<K, V, H> OpenCopyHashMap<K, V, H>
where K: Clone + Copy + Eq + PartialEq,
      V: Clone + Copy,
      H: MemHasher
{
  pub fn new() -> OpenCopyHashMap<K, V, H> {
    OpenCopyHashMap::with_capacity(8)
  }

  pub fn with_capacity(min_cap: usize) -> OpenCopyHashMap<K, V, H> {
    let cap = ceil_power2(max(4, min_cap) as u64) as usize;
    assert!(cap >= min_cap);
    let mut keys = Vec::with_capacity(cap);
    let mut values = Vec::with_capacity(cap);
    unsafe {
      keys.set_len(cap);
      values.set_len(cap);
    }
    OpenCopyHashMap{
      hasher:   MemHasher::new(0), //thread_rng().next_u64()),
      cap:      cap,
      len:      0,
      occupied: BitVec::from_elem(cap, false),
      keys:     keys,
      values:   values,
    }
  }

  fn init_probe(&self, key: &K) -> usize {
    (self.hasher.hash(key) as usize) & (self.cap - 1)
  }

  fn next_probe(&self, p: usize) -> usize {
    (p + 1) & (self.cap - 1)
  }

  fn expand(&mut self) {
    let old_cap = self.cap;
    let new_cap = old_cap * 2;
    assert!(new_cap > old_cap);
    // TODO(20151003): move elements.
    self.occupied.extend(repeat(false).take(old_cap));
    let keys_cap = self.keys.capacity();
    if new_cap >= keys_cap {
      self.keys.reserve(new_cap - keys_cap);
      unsafe { self.keys.set_len(new_cap) };
    }
    let values_cap = self.values.capacity();
    if new_cap >= values_cap {
      self.values.reserve(new_cap - values_cap);
      unsafe { self.values.set_len(new_cap) };
    }
    self.cap = new_cap;
  }

  pub fn len(&self) -> usize {
    self.len
  }

  pub fn clear(&mut self) {
    self.len = 0;
    self.occupied.clear();
  }

  pub fn contains_key(&self, key: &K) -> bool {
    let p0 = self.init_probe(key);
    let mut p = p0;
    loop {
      if !self.occupied[p] {
        return false;
      } else if *key == self.keys[p] {
        return true;
      }
      p = self.next_probe(p);
      if p == p0 {
        return false;
      }
    }
  }

  pub fn get(&self, key: &K) -> Option<&V> {
    let p0 = self.init_probe(key);
    let mut p = p0;
    loop {
      if !self.occupied[p] {
        return None;
      } else if *key == self.keys[p] {
        return Some(&self.values[p]);
      }
      p = self.next_probe(p);
      if p == p0 {
        return None;
      }
    }
  }

  pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
    let p0 = self.init_probe(key);
    let mut p = p0;
    loop {
      if !self.occupied[p] {
        return None;
      } else if *key == self.keys[p] {
        return Some(&mut self.values[p]);
      }
      p = self.next_probe(p);
      if p == p0 {
        return None;
      }
    }
  }

  pub fn insert(&mut self, key: K, value: V) -> Option<V> {
    if 4 * self.len >= self.cap {
      self.expand();
    }
    let p0 = self.init_probe(&key);
    let mut p = p0;
    loop {
      if !self.occupied[p] {
        self.len += 1;
        self.occupied.set(p, true);
        self.keys[p] = key;
        self.values[p] = value;
        return None;
      } else if key == self.keys[p] {
        let old_value = self.values[p];
        self.values[p] = value;
        return Some(old_value);
      }
      p = self.next_probe(p);
      assert!(p != p0);
    }
  }

  pub fn each<F>(&self, mut f: F) where F: FnMut(&K, &V) {
    let mut p = 0;
    loop {
      if self.occupied[p] {
        f(&self.keys[p], &self.values[p]);
      }
      p = self.next_probe(p);
      if p == 0 {
        break;
      }
    }
  }
}

// FIXME(20150202): forcing V to be Copy because we are not dropping values
// properly.

#[derive(Clone)]
pub struct OpenCopyHashSet<V, H>
where V: Clone + Copy + Eq + PartialEq, H: MemHasher {
  hasher:   H,
  cap:      usize,
  len:      usize,
  occupied: BitVec,
  values:   Vec<V>,
}

impl<V, H=XxhMemHasher> OpenCopyHashSet<V, H>
where V: Clone + Copy + Eq + PartialEq, H: MemHasher {
  pub fn new() -> OpenCopyHashSet<V, H> {
    OpenCopyHashSet::with_capacity(8)
  }

  pub fn with_capacity(min_cap: usize) -> OpenCopyHashSet<V, H> {
    let cap = ceil_power2(min_cap as u64) as usize;
    assert!(cap >= min_cap);
    let mut values = Vec::with_capacity(cap);
    unsafe {
      values.set_len(cap);
    }
    OpenCopyHashSet{
      hasher:   MemHasher::new(0), //thread_rng().next_u64()),
      cap:      cap,
      len:      0,
      occupied: BitVec::from_elem(cap, false),
      values:   values,
    }
  }

  pub fn len(&self) -> usize {
    self.len
  }

  pub fn clear(&mut self) {
    self.len = 0;
    self.occupied.clear(); // = BitVec::from_elem(self.cap, false);
  }

  pub fn contains(&self, value: &V) -> bool {
    let p0 = (self.hasher.hash(value) as usize) & (self.cap - 1);
    let mut p = p0;
    loop {
      if !self.occupied[p] {
        return false;
      } else if value == &self.values[p] {
        return true;
      }
      p = (p + 1) & (self.cap - 1);
      if p == p0 {
        return false;
      }
    }
  }

  pub fn insert(&mut self, value: V) {
    // TODO(20151003): expand?
    let p0 = (self.hasher.hash(&value) as usize) & (self.cap - 1);
    let mut p = p0;
    loop {
      if !self.occupied[p] {
        self.len += 1;
        self.occupied.set(p, true);
        self.values[p] = value;
        return;
      }
      p = (p + 1) & (self.cap - 1);
      assert!(p != p0); // FIXME
    }
  }
}
