//use statistics_avx2::array::{array_binary_search};
//use statistics_avx2::random::{StreamRng};

use std::cmp::{min};
use std::num::Wrapping as Wrap;
use rand::{Rng, SeedableRng};
use rand::distributions::{Range};
use rand::distributions::{IndependentSample};

pub fn random_shuffle<T: Copy, R: Rng>(xs: &mut [T], rng: &mut R) {
  // Fisher-Yates shuffle.
  let n = xs.len();
  for i in 0 .. n {
    let uniform_dist = Range::new(i, n);
    let j = uniform_dist.ind_sample(rng);
    xs.swap(i, j);
  }
}

pub fn random_shuffle_once<T: Copy, R: Rng>(xs: &mut [T], i: usize, rng: &mut R) {
  // This convention of the Fisher-Yates shuffle has a "Markov" property:
  // 1. a call with i == 0 is independent of all previous calls;
  // 2. a call with i > 0 is dependent only on the previous call for each i' < i.
  // By convention, it is meant to be called with increments to i, punctuated by
  // resets to zero.
  let n = xs.len();
  let uniform_dist = Range::new(i, n);
  let j = uniform_dist.ind_sample(rng);
  xs.swap(i, j);
}

pub fn arg_choose<T: Copy, R: Rng>(xs: &[T], rng: &mut R) -> Option<(T, usize)> {
  if xs.is_empty() {
    None
  } else {
    let j = rng.gen_range(0, xs.len());
    Some((xs[j], j))
  }
}

pub fn choose_without_replace<T: Copy, R: Rng>(xs: &mut Vec<T>, rng: &mut R) -> Option<T> {
  let n = xs.len();
  if n == 0 {
    None
  } else {
    let uniform_dist = Range::new(0, n);
    let j = uniform_dist.ind_sample(rng);
    xs.swap(j, n - 1);
    xs.pop()
  }
}

pub fn sample_discrete_cdf<R>(cdf: &[f32], rng: &mut R) -> usize where R: Rng {
  // FIXME(20151207)
  unimplemented!();
  /*let u: f32 = rng.gen_range(0.0, 1.0);
  let j = array_binary_search(cdf, u);
  assert!(j < cdf.len());
  assert!(cdf[j] <= u);
  if j < cdf.len() - 1 {
    assert!(u < cdf[j+1]);
  }
  j*/
}

//pub fn sample_discrete_cdf_scaled<R>(cdf: &[f32], rng: &mut R) -> usize where R: StreamRng {
pub fn sample_discrete_cdf_scaled<R>(cdf: &[f32], rng: &mut R) -> usize where R: Rng {
  // FIXME(20151207)
  unimplemented!();
  /*let mut u = [0.0f32];
  rng.sample_uniform_f32(&mut u);
  let scale = cdf[cdf.len() - 1];
  let j = array_binary_search(cdf, scale * u[0]);
  assert!(j < cdf.len());
  assert!(cdf[j] <= u[0]);
  if j < cdf.len() - 1 {
    assert!(u[0] < cdf[j+1]);
  }
  j*/
}

//pub fn batch_sample_discrete_cdfs<R>(batch_cdf: &[f32], rng: &mut R) where R: StreamRng {
pub fn batch_sample_discrete_cdfs<R>(batch_cdf: &[f32], rng: &mut R) where R: Rng {
  // TODO(20151019)
  unimplemented!();
}

#[derive(Clone)]
pub struct XorShift128PlusRng {
  state: [u64; 2],
}

impl Rng for XorShift128PlusRng {
  fn next_u64(&mut self) -> u64 {
    let mut s1 = unsafe { *self.state.get_unchecked(0) };
    let s0 = unsafe { *self.state.get_unchecked(1) };
    unsafe { *self.state.get_unchecked_mut(0) = s0 };
    s1 ^= s1 << 23;
    s1 = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
    unsafe { *self.state.get_unchecked_mut(1) = s1 };
    (Wrap(s1) + Wrap(s0)).0
  }

  fn next_u32(&mut self) -> u32 {
    self.next_u64() as u32
  }
}

impl SeedableRng<[u64; 2]> for XorShift128PlusRng {
  fn reseed(&mut self, seed: [u64; 2]) {
    self.reseed(&seed as &[u64]);
  }

  fn from_seed(seed: [u64; 2]) -> XorShift128PlusRng {
    Self::from_seed(&seed as &[u64])
  }
}

impl<'a> SeedableRng<&'a [u64]> for XorShift128PlusRng {
  fn reseed(&mut self, seed: &'a [u64]) {
    assert!(seed.len() >= 1);
    for p in 0 .. min(2usize, seed.len()) {
      self.state[p] = seed[p];
    }
    // XXX: This increases the initial state entropy (many zeros to half zeros).
    // See Figure 4 in <http://arxiv.org/abs/1404.0390> for details.
    for _ in 0 .. 20 {
      let _ = self.next_u64();
    }
  }

  fn from_seed(seed: &'a [u64]) -> XorShift128PlusRng {
    let mut rng = XorShift128PlusRng{
      state: [0; 2],
    };
    rng.reseed(seed);
    rng
  }
}

pub struct XorShift1024StarRng {
  state: [u64; 16],
  p: usize,
}

impl Rng for XorShift1024StarRng {
  fn next_u64(&mut self) -> u64 {
    // See: <http://xorshift.di.unimi.it/xorshift1024star.c>
    // and <http://arxiv.org/abs/1402.6246>.
    let mut s0 = unsafe { *self.state.get_unchecked(self.p) };
    let p = (self.p + 1) & 0x0f;
    let mut s1 = unsafe { *self.state.get_unchecked(p) };
    s1 ^= s1 << 31;
    s1 ^= s1 >> 11;
    s0 ^= s0 >> 30;
    let s = s0 ^ s1;
    unsafe { *self.state.get_unchecked_mut(p) = s; }
    self.p = p;
    let r = s * 1181783497276652981_u64;
    r
  }

  fn next_u32(&mut self) -> u32 {
    self.next_u64() as u32
  }
}

impl<'a> SeedableRng<&'a [u64]> for XorShift1024StarRng {
  fn reseed(&mut self, seed: &'a [u64]) {
    assert!(seed.len() >= 1);
    for p in 0 .. min(16usize, seed.len()) {
      self.state[p] = seed[p];
    }
  }

  fn from_seed(seed: &'a [u64]) -> XorShift1024StarRng {
    let mut rng = XorShift1024StarRng{
      state: [0; 16],
      p: 0,
    };
    rng.reseed(seed);
    rng
  }
}
