pub trait Verbosity {
  fn debug() -> bool;
}

pub struct Verbose;

impl Verbosity for Verbose {
  fn debug() -> bool { true }
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
