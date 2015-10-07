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
