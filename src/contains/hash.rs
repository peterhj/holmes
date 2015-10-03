use rand::{Rng, thread_rng};
//use std::intrinsics::{get_tydesc};
use std::mem::{size_of};
use std::slice::{from_raw_parts};

pub trait MemHasher: Clone {
  fn new() -> Self;
  fn hash_bytes(&self, bytes: &[u8]) -> u64;

  fn hash<T>(&self, v: &T) -> u64 where T: Sized {
    let bytes: &[u8] = unsafe {
      from_raw_parts(v as *const T as *const u8, size_of::<T>())
    };
    self.hash_bytes(bytes)
  }
}

// See: <https://github.com/Cyan4973/xxHash/blob/master/xxhash.c>.

const XXH_PRIME_1: u64 = 11400714785074694791_u64;
const XXH_PRIME_2: u64 = 14029467366897019727_u64;
const XXH_PRIME_3: u64 =  1609587929392839161_u64;
const XXH_PRIME_4: u64 =  9650029242287828579_u64;
const XXH_PRIME_5: u64 =  2870177450012600261_u64;

#[inline]
fn rot_left64(x: u64, r: usize) -> u64 {
  (x << r) | (x >> (64 - r))
}

#[inline]
fn swizzle64(x: u64) -> u64 {
  ((x << 56) & 0xff00000000000000_u64) |
  ((x << 40) & 0x00ff000000000000_u64) |
  ((x << 24) & 0x0000ff0000000000_u64) |
  ((x << 8)  & 0x000000ff00000000_u64) |
  ((x >> 8)  & 0x00000000ff000000_u64) |
  ((x >> 24) & 0x0000000000ff0000_u64) |
  ((x >> 40) & 0x000000000000ff00_u64) |
  ((x >> 56) & 0x00000000000000ff_u64)
}

#[derive(Clone)]
pub struct XxhMemHasher;

impl MemHasher for XxhMemHasher {
  fn new() -> XxhMemHasher {
    XxhMemHasher
  }

  fn hash_bytes(&self, bytes: &[u8]) -> u64 {
    let num_bytes = bytes.len();
    let num_words = num_bytes / 8;
    let num_4xwords = (num_words / 4) * 4;
    let num_trailing_bytes = num_bytes - 8 * num_words;
    let words: &[u64] = unsafe {
      from_raw_parts(bytes.as_ptr() as *const u64, num_words)
    };

    let seed = thread_rng().next_u64();
    let mut h: u64;
    let mut p: usize = 0;
    if num_words >= 4 {
      let mut v1 = seed + XXH_PRIME_1 + XXH_PRIME_2;
      let mut v2 = seed + XXH_PRIME_2;
      let mut v3 = seed;
      let mut v4 = seed - XXH_PRIME_1;
      while p < num_4xwords {
        v1 += words[p] * XXH_PRIME_2;
        p += 1;
        v1 = rot_left64(v1, 31);
        v1 *= XXH_PRIME_1;

        v2 += words[p] * XXH_PRIME_2;
        p += 1;
        v2 = rot_left64(v2, 31);
        v2 *= XXH_PRIME_1;

        v3 += words[p] * XXH_PRIME_2;
        p += 1;
        v3 = rot_left64(v3, 31);
        v3 *= XXH_PRIME_1;

        v4 += words[p] * XXH_PRIME_2;
        p += 1;
        v4 = rot_left64(v4, 31);
        v4 *= XXH_PRIME_1;
      }
      h = rot_left64(v1, 1) + rot_left64(v2, 7) + rot_left64(v3, 12) + rot_left64(v4, 18);

      v1 *= XXH_PRIME_2;
      v1 = rot_left64(v1, 31);
      v1 *= XXH_PRIME_1;
      h ^= v1;
      h = h * XXH_PRIME_1 + XXH_PRIME_4;

      v2 *= XXH_PRIME_2;
      v2 = rot_left64(v2, 31);
      v2 *= XXH_PRIME_1;
      h ^= v2;
      h = h * XXH_PRIME_1 + XXH_PRIME_4;

      v3 *= XXH_PRIME_2;
      v3 = rot_left64(v3, 31);
      v3 *= XXH_PRIME_1;
      h ^= v3;
      h = h * XXH_PRIME_1 + XXH_PRIME_4;

      v4 *= XXH_PRIME_2;
      v4 = rot_left64(v4, 31);
      v4 *= XXH_PRIME_1;
      h ^= v4;
      h = h * XXH_PRIME_1 + XXH_PRIME_4;
    } else {
      h = seed + XXH_PRIME_5;
    }

    h += num_bytes as u64;

    while p < num_words {
      p += 1;
    }

    // TODO(20150202): trailing bytes.

    h ^= h >> 33;
    h *= XXH_PRIME_2;
    h ^= h >> 29;
    h *= XXH_PRIME_2;
    h ^= h >> 32;
    h
  }
}
