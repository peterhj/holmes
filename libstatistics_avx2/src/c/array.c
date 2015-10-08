#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

// http://graphics.stanford.edu/~seander/bithacks.html#IntegerLogIEEE64Float
static const size_t LOG2_DEBRUIJN_OFFSETS[32] = {
  0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30,
  8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31
};

size_t statistics_arg_amax(const float *xs, size_t len) {
  __m256 x, m0, m1, m2, m3, m4, m, fmask;
  uint32_t bmask;
  float loc_max[8];
  float loc_x;
  size_t offset;
  float max = -INFINITY;
  size_t index = -1;
  const float *xs_start = xs;
  const float *xs_end = xs + len;
  for ( ; xs + 8 <= xs_end; xs += 8) {
    // http://stackoverflow.com/questions/9795529/how-to-find-the-horizontal-maximum-in-a-256-bit-avx-vector
    // Basically do a reduction tree.
    x = _mm256_loadu_ps(xs);
    // https://software.intel.com/en-us/node/583071
    // Permute the two 128-bit parts.
    m0 = _mm256_permute2f128_ps(x, x, 1);
    m1 = _mm256_max_ps(x, m0);
    // For each 128-bit part, permute [0,1,2,3] -> [2,3,0,1].
    m2 = _mm256_permute_ps(m1, 0x4e);
    m3 = _mm256_max_ps(m1, m2);
    // For each 128-bit part, permute [0,1,2,3] -> [1,0,3,2].
    m4 = _mm256_permute_ps(m3, 0xb1);
    m = _mm256_max_ps(m3, m4);
    _mm256_storeu_ps(loc_max, m);
    if (loc_max[0] > max) {
      // https://software.intel.com/en-us/node/582989
      // https://software.intel.com/en-us/node/583042
      // XXX: See <avxintrin.h> for cmp control flags.
      fmask = _mm256_cmp_ps(x, m, 0);
      bmask = _mm256_movemask_ps(fmask);
      bmask &= 0xff; // FIXME: probably not necessary, check docs.
      bmask |= bmask >> 1;
      bmask |= bmask >> 2;
      bmask |= bmask >> 4;
      offset = LOG2_DEBRUIJN_OFFSETS[(uint32_t)(bmask * 0x07C4ACDDU) >> 27];
      max = loc_max[0];
      index = (size_t)(xs - xs_start) + offset;
    }
  }
  for ( ; xs < xs_end; xs += 1) {
    loc_x = *xs;
    if (loc_x > max) {
      max = loc_x;
      index = (size_t)(xs - xs_start);
    }
  }
  return index;
}
