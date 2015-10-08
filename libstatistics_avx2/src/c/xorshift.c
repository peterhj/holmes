#include "avx_mathfun.h"
#include <immintrin.h>
#include <stdint.h>

#define ALWAYS_INLINE __attribute__ ((always_inline))

extern float cephes_cosf(float);
extern float cephes_sinf(float);
extern float cephes_logf(float);
extern float cephes_sqrtf(float);

struct XorShift128PlusState {
  uint64_t s0[4];
  uint64_t s1[4];
};

#define XORSHIFT128PLUS_STREAM32_X8_INIT(state, s0, s1) \
  __m256i s0 = _mm256_loadu_si256((__m256i const *)state->s1); \
  __m256i s1 = _mm256_loadu_si256((__m256i const *)state->s0);

#define XORSHIFT128PLUS_STREAM32_X8_NEXT(s0, s1) \
  s0 = _mm256_xor_si256(s0, s1); \
  s1 = _mm256_xor_si256(s0, s1); \
  s0 = _mm256_xor_si256(s0, s1);

#define XORSHIFT128PLUS_STREAM32_X8_LAST(state, s0, s1) \
  _mm256_storeu_si256((__m256i *)state->s0, s0); \
  _mm256_storeu_si256((__m256i *)state->s1, s1);

#define XORSHIFT128PLUS_STREAM32_X8_BODY(s0, s1) \
  ({ \
    s1 = _mm256_xor_si256(s1, _mm256_slli_epi64(s1, 23)); \
    s1 = _mm256_xor_si256(_mm256_xor_si256(s1, s0), _mm256_xor_si256(_mm256_srli_epi64(s1, 17), _mm256_srli_epi64(s0, 26))); \
    _mm256_add_epi64(s0, s1); \
  })

#define XORSHIFT128PLUS_STREAM32_X1_BODY(state) \
  ({ \
    uint64_t r0, r1; \
    r0 = state->s1[0]; \
    r1 = state->s0[0]; \
    state->s0[0] = r0; \
    r1 ^= r1 << 23; \
    r1 = r1 ^ r0 ^ (r1 >> 17) ^ (r0 >> 26); \
    state->s1[0] = r1; \
    r0 + r1; \
  })

static inline __m256 uniform32_x8(__m256i u) ALWAYS_INLINE;
static inline __m256 uniform32_x8(__m256i u) {
  const __m256 scale = _mm256_set1_ps(1.0f/16777216.0f);
  __m256 x;
  u = _mm256_srli_epi32(u, 8);
  x = _mm256_cvtepi32_ps(u);
  x = _mm256_mul_ps(x, scale);
  return x;
}

static inline float uniform32_x1(uint32_t x) ALWAYS_INLINE;
static inline float uniform32_x1(uint32_t x) {
  return (float)(x >> 8) * (1.0f/16777216.0f);
}

static inline void box_muller_beta32_x16(__m256 u1, __m256 u2, const float *succ_ratio, const float *num_trials, __m256 *x1, __m256 *x2) ALWAYS_INLINE;
static inline void box_muller_beta32_x16(__m256 u1, __m256 u2, const float *succ_ratio, const float *num_trials, __m256 *x1, __m256 *x2) {
  const float s_two_pi = 2.0f * 3.14159265358979323846f;
  const __m256 two_pi = _mm256_set1_ps(s_two_pi);
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 minus_two = _mm256_set1_ps(-2.0f);
  __m256 mu1, mu2, sigma1, sigma2;
  __m256 radius, theta;
  __m256 costheta, sintheta;
  // https://github.com/miloyip/normaldist-benchmark/blob/master/src/boxmuller_avx.cpp
  mu1 = _mm256_loadu_ps(succ_ratio);
  mu2 = _mm256_loadu_ps(succ_ratio + 8);
  sigma1 = _mm256_sqrt_ps(_mm256_div_ps(_mm256_mul_ps(mu1, _mm256_sub_ps(one, mu1)), _mm256_loadu_ps(num_trials)));
  sigma2 = _mm256_sqrt_ps(_mm256_div_ps(_mm256_mul_ps(mu2, _mm256_sub_ps(one, mu2)), _mm256_loadu_ps(num_trials + 8)));
  u1 = _mm256_sub_ps(one, u1);
  radius = _mm256_sqrt_ps(_mm256_mul_ps(minus_two, log256_ps(u1)));
  theta = _mm256_mul_ps(two_pi, u2);
  sincos256_ps(theta, &sintheta, &costheta);
  *x1 = _mm256_add_ps(mu1, _mm256_mul_ps(sigma1, _mm256_mul_ps(radius, costheta)));
  *x2 = _mm256_add_ps(mu2, _mm256_mul_ps(sigma2, _mm256_mul_ps(radius, sintheta)));
}

static inline void box_muller_beta32_x2(float s_u1, float s_u2, const float *succ_ratio, const float *num_trials, float *x1, float *x2) ALWAYS_INLINE;
static inline void box_muller_beta32_x2(float s_u1, float s_u2, const float *succ_ratio, const float *num_trials, float *x1, float *x2) {
  const float s_two_pi = 2.0f * 3.14159265358979323846f;
  float s_mu1, s_mu2, s_sigma1, s_sigma2;
  float s_radius, s_theta;
  float s_costheta, s_sintheta;
  s_mu1 = succ_ratio[0];
  s_mu2 = succ_ratio[1];
  s_sigma1 = cephes_sqrtf(s_mu1 * (1.0f - s_mu1) / num_trials[0]);
  s_sigma2 = cephes_sqrtf(s_mu2 * (1.0f - s_mu2) / num_trials[1]);
  s_u1 = 1.0f - s_u1;
  s_radius = cephes_sqrtf(-2.0f * cephes_logf(s_u1));
  s_theta = s_two_pi * s_u2;
  s_costheta = cephes_cosf(s_theta);
  s_sintheta = cephes_sinf(s_theta);
  *x1 = s_mu1 + s_sigma1 * s_radius * s_costheta;
  *x2 = s_mu2 + s_sigma2 * s_radius * s_sintheta;
}

static inline float box_muller_beta32_x1(float s_u1, float s_u2, const float *succ_ratio, const float *num_trials) ALWAYS_INLINE;
static inline float box_muller_beta32_x1(float s_u1, float s_u2, const float *succ_ratio, const float *num_trials) {
  const float s_two_pi = 2.0f * 3.14159265358979323846f;
  float s_mu1, s_sigma1;
  float s_radius, s_theta;
  float s_costheta;
  s_mu1 = succ_ratio[0];
  s_sigma1 = cephes_sqrtf(s_mu1 * (1.0f - s_mu1) / num_trials[0]);
  s_u1 = 1.0 - s_u1;
  s_radius = cephes_sqrtf(-2.0f * cephes_logf(s_u1));
  s_theta = s_two_pi * s_u2;
  s_costheta = cephes_cosf(s_theta);
  return s_mu1 + s_sigma1 * s_radius * s_costheta;
}

void xorshift128plus_avx2_stream32(struct XorShift128PlusState *state, uint32_t *xs, size_t len) {
  uint32_t *xs_end = xs + len;
  if (!(xs + 8 <= xs_end)) {
    goto tail;
  }
  XORSHIFT128PLUS_STREAM32_X8_INIT(state, s0, s1);
  for (;;) {
    __m256i x;

    x = XORSHIFT128PLUS_STREAM32_X8_BODY(s0, s1);
    _mm256_storeu_si256((__m256i *)xs, x);

    xs += 8;
    if (xs + 8 <= xs_end) {
      XORSHIFT128PLUS_STREAM32_X8_NEXT(s0, s1);
    } else {
      XORSHIFT128PLUS_STREAM32_X8_LAST(state, s0, s1);
      break;
    }
  }
tail:
  for ( ; xs < xs_end; xs += 2) {
    uint64_t x = XORSHIFT128PLUS_STREAM32_X1_BODY(state);
    *((uint64_t *)xs) = x;
  }
  if (xs < xs_end) {
    uint64_t x = XORSHIFT128PLUS_STREAM32_X1_BODY(state);
    *xs = (uint32_t)x;
  }
}

void xorshift128plus_avx2_uniform32(struct XorShift128PlusState *state, float *xs, size_t len) {
  float *xs_end = xs + len;
  if (!(xs + 8 <= xs_end)) {
    goto tail;
  }
  XORSHIFT128PLUS_STREAM32_X8_INIT(state, s0, s1);
  for (;;) {
    __m256i r;
    __m256 x;
    r = XORSHIFT128PLUS_STREAM32_X8_BODY(s0, s1);
    x = uniform32_x8(r);
    _mm256_storeu_ps(xs, x);

    xs += 8;
    if (xs + 8 <= xs_end) {
      XORSHIFT128PLUS_STREAM32_X8_NEXT(s0, s1);
    } else {
      XORSHIFT128PLUS_STREAM32_X8_LAST(state, s0, s1);
      break;
    }
  }
tail:
  for ( ; xs < xs_end; xs += 1) {
    uint64_t x = XORSHIFT128PLUS_STREAM32_X1_BODY(state);
    *xs = uniform32_x1(x);
  }
}

void xorshift128plus_avx2_box_muller_beta32(struct XorShift128PlusState *state, const float *succ_ratio, const float *num_trials, float *xs, size_t len) {
  float *xs_end = xs + len;
  if (!(xs + 16 <= xs_end)) {
    goto tail;
  }
  XORSHIFT128PLUS_STREAM32_X8_INIT(state, s0, s1);
  for (;;) {
    __m256i r1, r2;
    __m256 u1, u2;
    __m256 x1, x2;

    r1 = XORSHIFT128PLUS_STREAM32_X8_BODY(s0, s1);
    XORSHIFT128PLUS_STREAM32_X8_NEXT(s0, s1);
    r2 = XORSHIFT128PLUS_STREAM32_X8_BODY(s0, s1);

    u1 = uniform32_x8(r1);
    u2 = uniform32_x8(r2);

    box_muller_beta32_x16(u1, u2, succ_ratio, num_trials, &x1, &x2);

    _mm256_storeu_ps(xs, x1);
    _mm256_storeu_ps(xs + 8, x2);

    succ_ratio += 16;
    num_trials += 16;
    xs += 16;
    if (xs + 16 <= xs_end) {
      XORSHIFT128PLUS_STREAM32_X8_NEXT(s0, s1);
    } else {
      XORSHIFT128PLUS_STREAM32_X8_LAST(state, s0, s1);
      break;
    }
  }
tail:
  /*for ( ; xs + 8 <= xs_end; xs += 8) {
    // TODO(20151007): SSE version.
  }*/
  for ( ; xs + 2 <= xs_end; succ_ratio += 2, num_trials += 2, xs += 2) {
    uint64_t r1 = XORSHIFT128PLUS_STREAM32_X1_BODY(state);
    uint64_t r2 = XORSHIFT128PLUS_STREAM32_X1_BODY(state);
    float u1 = uniform32_x1(r1);
    float u2 = uniform32_x1(r2);
    box_muller_beta32_x2(u1, u2, succ_ratio, num_trials, xs, xs + 1);
  }
  if (xs < xs_end) {
    uint64_t r1 = XORSHIFT128PLUS_STREAM32_X1_BODY(state);
    uint64_t r2 = XORSHIFT128PLUS_STREAM32_X1_BODY(state);
    float u1 = uniform32_x1(r1);
    float u2 = uniform32_x1(r2);
    *xs = box_muller_beta32_x1(u1, u2, succ_ratio, num_trials);
  }
}
