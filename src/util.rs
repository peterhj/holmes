pub fn slice_twice_mut<T>(xs: &mut [T], lo1: usize, hi1: usize, lo2: usize, hi2: usize) -> (&mut [T], &mut [T]) {
  assert!(lo1 <= hi1);
  assert!(lo2 <= hi2);
  let mut flip = false;
  let (lo1, hi1, lo2, hi2) = if (lo1 <= lo2) {
    (lo1, hi1, lo2, hi2)
  } else {
    flip = true;
    (lo2, hi2, lo1, hi1)
  };
  assert!(hi1 <= lo2, "ranges must not overlap!");
  let (mut lo_xs, mut hi_xs) = xs.split_at_mut(lo2);
  let lo_xs = &mut lo_xs[lo1 .. hi1];
  let hi_xs = &mut hi_xs[.. hi2 - lo2];
  if !flip {
    (lo_xs, hi_xs)
  } else {
    (hi_xs, lo_xs)
  }
}

#[inline]
pub fn ceil_power2(x: u64) -> u64 {
  let mut v = x;
  v -= 1;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v |= v >> 32;
  v += 1;
  v
}

#[inline]
pub fn rot_left64(x: u64, r: usize) -> u64 {
  (x << r) | (x >> (64 - r))
}

#[inline]
pub fn swizzle64(x: u64) -> u64 {
  ((x << 56) & 0xff00000000000000_u64) |
  ((x << 40) & 0x00ff000000000000_u64) |
  ((x << 24) & 0x0000ff0000000000_u64) |
  ((x << 8)  & 0x000000ff00000000_u64) |
  ((x >> 8)  & 0x00000000ff000000_u64) |
  ((x >> 24) & 0x0000000000ff0000_u64) |
  ((x >> 40) & 0x000000000000ff00_u64) |
  ((x >> 56) & 0x00000000000000ff_u64)
}

pub trait Verbosity {
  fn debug() -> bool;
}

pub struct Verbose;

impl Verbosity for Verbose {
  fn debug() -> bool { true }
}
