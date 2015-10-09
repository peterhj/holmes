use util::{rot_left64};

//use rand::{Rng, thread_rng};
//use std::intrinsics::{get_tydesc};
use std::mem::{size_of};
use std::slice::{from_raw_parts};

pub trait MemHasher: Clone {
  fn new(seed: u64) -> Self;
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

#[derive(Clone)]
pub struct XxhMemHasher {
  seed: u64,
}

impl MemHasher for XxhMemHasher {
  fn new(seed: u64) -> XxhMemHasher {
    XxhMemHasher{seed: seed}
  }

  fn hash_bytes(&self, bytes: &[u8]) -> u64 {
    let num_bytes = bytes.len();
    let num_words = num_bytes / 8;
    let num_words_group4 = (num_words / 4) * 4;
    let trailing_ints_offset = num_words * 8;
    let num_ints = num_bytes / 4;
    // XXX: There should be at most one trailing u32.
    let num_trailing_ints = num_ints - trailing_ints_offset / 4;
    let trailing_bytes_offset = num_ints * 4;
    let words: &[u64] = unsafe {
      from_raw_parts(bytes.as_ptr() as *const u64, num_words)
    };
    let trailing_ints: &[u32] = unsafe {
      from_raw_parts(bytes.as_ptr().offset(trailing_ints_offset as isize) as *const u32, num_trailing_ints)
    };

    let seed = self.seed;
    let mut h: u64;

    let mut p: usize = 0;
    if num_words_group4 >= 4 {
      let mut v1 = seed + XXH_PRIME_1 + XXH_PRIME_2;
      let mut v2 = seed + XXH_PRIME_2;
      let mut v3 = seed;
      let mut v4 = seed - XXH_PRIME_1;
      loop {
        v1 += unsafe { *words.get_unchecked(p) } * XXH_PRIME_2;
        v1 = rot_left64(v1, 31);
        v1 *= XXH_PRIME_1;

        v2 += unsafe { *words.get_unchecked(p + 1) } * XXH_PRIME_2;
        v2 = rot_left64(v2, 31);
        v2 *= XXH_PRIME_1;

        v3 += unsafe { *words.get_unchecked(p + 2) } * XXH_PRIME_2;
        v3 = rot_left64(v3, 31);
        v3 *= XXH_PRIME_1;

        v4 += unsafe { *words.get_unchecked(p + 3) } * XXH_PRIME_2;
        v4 = rot_left64(v4, 31);
        v4 *= XXH_PRIME_1;

        p += 4;
        if p >= num_words_group4 {
          break;
        }
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
      let mut k1 = unsafe { *words.get_unchecked(p) };
      p += 1;
      k1 *= XXH_PRIME_2;
      k1 = rot_left64(k1, 31);
      k1 *= XXH_PRIME_1;
      h ^= k1;
      h = rot_left64(h, 27) * XXH_PRIME_1 + XXH_PRIME_4;
    }

    if num_trailing_ints > 0 {
      h ^= unsafe { *trailing_ints.get_unchecked(0) } as u64 * XXH_PRIME_1;
      h = rot_left64(h, 23) * XXH_PRIME_2 + XXH_PRIME_3;
    }

    let mut i = trailing_bytes_offset;
    while i < num_bytes {
      h ^= unsafe { *bytes.get_unchecked(i) } as u64 * XXH_PRIME_5;
      i += 1;
      h = rot_left64(h, 11) * XXH_PRIME_1;
    }

    h ^= h >> 33;
    h *= XXH_PRIME_2;
    h ^= h >> 29;
    h *= XXH_PRIME_3;
    h ^= h >> 32;
    h
  }
}
