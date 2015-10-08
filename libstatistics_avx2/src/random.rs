use libc::{size_t};
use std::cmp::{min};
use std::mem::{uninitialized};

#[repr(C)]
struct XorShift128PlusState {
  s0: [u64; 4],
  s1: [u64; 4],
}

#[link(name = "statistics_avx2_impl", kind = "static")]
extern "C" {
  fn xorshift128plus_avx2_stream32(state: *mut XorShift128PlusState, xs: *mut u32, len: size_t);
  fn xorshift128plus_avx2_uniform32(state: *mut XorShift128PlusState, xs: *mut f32, len: size_t);
  fn xorshift128plus_avx2_box_muller32(state: *mut XorShift128PlusState, mean: f32, std: f32, xs: *mut f32, len: size_t);
  fn xorshift128plus_avx2_box_muller_beta32(state: *mut XorShift128PlusState, mean: *const f32, std: *const f32, xs: *mut f32, len: size_t);
}

pub struct XorShift128PlusStreamRng {
  state: XorShift128PlusState,
}

impl XorShift128PlusStreamRng {
  pub fn seed_len() -> usize {
    8
  }

  pub fn new(seed: &[u64]) -> XorShift128PlusStreamRng {
    let mut rng = Self::new_unseeded();
    rng.seed(seed);
    rng.burn(20);
    rng
  }

  pub fn new_unseeded() -> XorShift128PlusStreamRng {
    XorShift128PlusStreamRng{
      state: XorShift128PlusState{
        s0: [0, 0, 0, 0],
        s1: [0, 0, 0, 0],
      }
    }
  }

  pub fn seed(&mut self, seed: &[u64]) {
    for i in (0 .. min(4, seed.len())) {
      self.state.s0[i] = seed[i];
    }
    for i in (4 .. min(8, seed.len())) {
      self.state.s1[i - 4] = seed[i];
    }
  }

  pub fn burn(&mut self, burn64: usize) {
    for _ in (0 .. burn64) {
      self.next_u64();
    }
  }

  pub fn next_u64(&mut self) -> u64 {
    let mut xs: [u32; 2] = unsafe { uninitialized() };
    self.sample_u32(&mut xs);
    unsafe { *(xs.as_ptr() as *const u64) }
  }

  pub fn sample_u32(&mut self, xs: &mut [u32]) {
    unsafe { xorshift128plus_avx2_stream32(&mut self.state as *mut _, xs.as_mut_ptr(), xs.len() as size_t) };
  }

  pub fn sample_uniform_f32(&mut self, xs: &mut [f32]) {
    unsafe { xorshift128plus_avx2_uniform32(&mut self.state as *mut _, xs.as_mut_ptr(), xs.len() as size_t) };
  }

  /*pub fn sample_normal_f32(&mut self, mean: f32, std: f32, xs: &mut [f32]) {
    self.sample_uniform_f32(xs);
    unsafe { xorshift128plus_avx2_box_muller32(&mut self.state as *mut _, mean, std, xs.as_mut_ptr(), xs.len() as size_t) };
  }*/

  pub fn sample_approx_beta_f32(&mut self, succ_ratio: &[f32], num_trials: &[f32], xs: &mut [f32]) {
    assert_eq!(succ_ratio.len(), xs.len());
    assert_eq!(num_trials.len(), xs.len());
    self.sample_uniform_f32(xs);
    unsafe { xorshift128plus_avx2_box_muller_beta32(
        &mut self.state as *mut _,
        succ_ratio.as_ptr(),
        num_trials.as_ptr(),
        xs.as_mut_ptr(), xs.len() as size_t,
    ) };
  }
}
