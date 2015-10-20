#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define ALWAYS_INLINE __attribute__ ((always_inline))

// http://graphics.stanford.edu/~seander/bithacks.html#IntegerLogDeBruijn
static const size_t LOG2_DEBRUIJN_OFFSETS[32] = {
  0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30,
  8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31
};

void statistics_array_fill(float c, float *xs, size_t len) {
  __m256 x = _mm256_set1_ps(c);
  float *xs_end = xs + len;
  for ( ; xs + 32 <= xs_end; xs += 32) {
    _mm256_storeu_ps(xs, x);
    _mm256_storeu_ps(xs + 8, x);
    _mm256_storeu_ps(xs + 16, x);
    _mm256_storeu_ps(xs + 24, x);
  }
  for ( ; xs + 16 <= xs_end; xs += 16) {
    _mm256_storeu_ps(xs, x);
    _mm256_storeu_ps(xs + 8, x);
  }
  for ( ; xs + 8 <= xs_end; xs += 8) {
    _mm256_storeu_ps(xs, x);
  }
  for ( ; xs < xs_end; xs += 1) {
    *xs = c;
  }
}

/*static inline float statistics_array_sum_aligned(const float *xs, size_t len) ALWAYS_INLINE;
static inline float statistics_array_sum_aligned(const float *xs, size_t len) {
  __m256 x;
  __m256 asum = _mm256_set1_ps(0.0);
  float tmp_asum[8];
  float s_asum = 0.0;
  const float *xs_align = xs + ((((intptr_t)xs) % 32) + 32 - 1) / 32 * 32;
  const float *xs_end = xs + len;
  for ( ; xs < xs_align; xs += 1) {
    s_asum += *xs;
  }
  if (xs + 8 <= xs_end) {
    asum = _mm256_load_ps(xs);
    xs += 8;
    for ( ; xs + 8 <= xs_end; xs += 8) {
      x = _mm256_load_ps(xs);
      asum = _mm256_add_ps(asum, x);
    }
  }
  _mm256_storeu_ps(tmp_asum, asum);
  s_asum += tmp_asum[0] + tmp_asum[1] + tmp_asum[2] + tmp_asum[3] + tmp_asum[4] + tmp_asum[5] + tmp_asum[6] + tmp_asum[7];
  for ( ; xs < xs_end; xs += 1) {
    s_asum += *xs;
  }
  return s_asum;
}*/

static inline float statistics_array_sum_unaligned(const float *xs, size_t len) ALWAYS_INLINE;
static inline float statistics_array_sum_unaligned(const float *xs, size_t len) {
  __m256 x;
  __m256 asum = _mm256_set1_ps(0.0);
  float tmp_asum[8];
  float s_asum = 0.0;
  const float *xs_end = xs + len;
  if (len >= 8) {
    asum = _mm256_loadu_ps(xs);
    xs += 8;
    for ( ; xs + 8 <= xs_end; xs += 8) {
      x = _mm256_loadu_ps(xs);
      asum = _mm256_add_ps(asum, x);
    }
  }
  _mm256_storeu_ps(tmp_asum, asum);
  s_asum = tmp_asum[0] + tmp_asum[1] + tmp_asum[2] + tmp_asum[3] + tmp_asum[4] + tmp_asum[5] + tmp_asum[6] + tmp_asum[7];
  for ( ; xs < xs_end; xs += 1) {
    s_asum += *xs;
  }
  return s_asum;
}

float statistics_array_sum(const float *xs, size_t len) {
  /*if (__builtin_expect(((intptr_t)xs) & 3, 0)) {
    return statistics_array_sum_aligned(xs, len);
  } else {
    return statistics_array_sum_unaligned(xs, len);
  }*/
  return statistics_array_sum_unaligned(xs, len);
}

size_t statistics_array_argmax(const float *xs, size_t len) {
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

// XXX(20151019): This version of branch-free binary search computes the
// highest 0-index of the array element less than or equal to the query.
// If the query is less than the smallest array element, return -1.
// This algorithm is suitable for sampling discrete CDFs.
// See: <https://github.com/patmorin/arraylayout/blob/master/src/sorted_array.h>.
// FIXME(20151019): check the inequalities and final offset.
size_t statistics_array_binary_search(const float *xs, size_t len, float x) {
  const float *base = xs;
  size_t n = len;
  while (n > 1) {
    size_t half = n / 2;
    __builtin_prefetch(base + half / 2, 0, 0);
    __builtin_prefetch(base + half + half / 2, 0, 0);
    base = (base[half] > x) ? base : base + half;
    n -= half;
  }
  return (base - xs) - (*base > x);
}
