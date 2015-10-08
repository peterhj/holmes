#include "avx_mathfun.h"
//#include "sse_mathfun.h"
#include <immintrin.h>
#include <stdint.h>

extern float cephes_cosf(float);
extern float cephes_sinf(float);
extern float cephes_logf(float);
extern float cephes_sqrtf(float);

struct XorShift128PlusState {
  uint64_t s0[4];
  uint64_t s1[4];
};

void xorshift128plus_avx2_stream32(struct XorShift128PlusState *state, uint32_t *xs, size_t len) {
  __m256i s0 = _mm256_loadu_si256((__m256i const *)state->s1);
  __m256i s1 = _mm256_loadu_si256((__m256i const *)state->s0);
  __m256i x_write;
  uint64_t r0, r1;
  uint32_t *xs_end = xs + len;
  if (!(xs + 8 <= xs_end)) {
    goto tail;
  }
  for (;;) {
    // https://software.intel.com/en-us/node/582881
    // https://software.intel.com/en-us/node/582887
    s1 = _mm256_xor_si256(s1, _mm256_slli_epi64(s1, 23));
    s1 = _mm256_xor_si256(_mm256_xor_si256(s1, s0), _mm256_xor_si256(_mm256_srli_epi64(s1, 17), _mm256_srli_epi64(s0, 26)));

    // https://software.intel.com/en-us/node/582791
    x_write = _mm256_add_epi64(s0, s1);
    // https://software.intel.com/en-us/node/583024
    _mm256_storeu_si256((__m256i *)xs, x_write);

    xs += 8;
    if (xs + 8 <= xs_end) {
      // Swap s0 and s1.
      // https://software.intel.com/en-us/node/582825
      s0 = _mm256_xor_si256(s0, s1);
      s1 = _mm256_xor_si256(s0, s1);
      s0 = _mm256_xor_si256(s0, s1);
    } else {
      _mm256_storeu_si256((__m256i *)state->s0, s0);
      _mm256_storeu_si256((__m256i *)state->s1, s1);
      break;
    }
  }
tail:
  for ( ; xs < xs_end; xs += 1) {
    r0 = state->s1[0];
    r1 = state->s0[0];
    state->s0[0] = r0;
    r1 ^= r1 << 23;
    r1 = r1 ^ r0 ^ (r1 >> 17) ^ (r0 >> 26);
    state->s1[0] = r1;
    *xs = r0 + r1;
  }
}

void xorshift128plus_avx2_uniform32(struct XorShift128PlusState *state, float *xs, size_t len) {
  const __m256 scale = _mm256_set1_ps(1.0f/2147483648.0f);
  __m256i u;
  __m256 x;
  float *xs_end = xs + len;
  xorshift128plus_avx2_stream32(state, (uint32_t *)xs, len);
  for ( ; xs + 8 <= xs_end; xs += 8) {
    // http://doc.rust-lang.org/rand/src/rand/lib.rs.html#305
    u = _mm256_loadu_si256((__m256i const *)xs);
    u = _mm256_srli_epi32(u, 1);
    x = _mm256_cvtepi32_ps(u);
    x = _mm256_mul_ps(x, scale);
    _mm256_storeu_ps(xs, x);
  }
  for ( ; xs < xs_end; xs += 1) {
    *xs = (float)(*((uint32_t *)xs) >> 1) * (1.0f/2147483648.0f);
  }
}

void xorshift128plus_avx2_box_muller32(struct XorShift128PlusState *state, float mean, float std, float *xs, size_t len) {
  // TODO(20151007): Box-Muller transform.
  // https://github.com/miloyip/normaldist-benchmark/blob/master/src/boxmuller_avx.cpp
  const __m256 two_pi = _mm256_set1_ps(2.0f * 3.14159265358979323846f);
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 minus_two = _mm256_set1_ps(-2.0f);
  __m256 u1, u2;
  __m256 radius, theta;
  __m256 mu = _mm256_set1_ps(mean);
  __m256 sigma = _mm256_set1_ps(std);
  __m256 costheta, sintheta;
  float *xs_end = xs + len;
  for ( ; xs + 16 <= xs_end; xs += 16) {
    u1 = _mm256_loadu_ps(xs);
    u1 = _mm256_sub_ps(one, u1);
    u2 = _mm256_loadu_ps(xs + 8);
    radius = _mm256_sqrt_ps(_mm256_mul_ps(minus_two, log256_ps(u1)));
    theta = _mm256_mul_ps(two_pi, u2);
    sincos256_ps(theta, &sintheta, &costheta);
    _mm256_store_ps(xs, _mm256_add_ps(mu, _mm256_mul_ps(sigma, _mm256_mul_ps(radius, costheta))));
    _mm256_store_ps(xs + 8, _mm256_add_ps(mu, _mm256_mul_ps(sigma, _mm256_mul_ps(radius, sintheta))));
  }
  /*for ( ; xs + 8 <= xs_end; xs += 8) {
    // TODO(20151007): SSE version.
  }*/
  for ( ; xs + 2 <= xs_end; xs += 2) {
    // TODO
  }
  if (xs < xs_end) {
    // TODO
  }
}

void xorshift128plus_avx2_box_muller_beta32(struct XorShift128PlusState *state, const float *succ_ratio, const float *num_trials, float *xs, size_t len) {
  const float s_two_pi = 2.0f * 3.14159265358979323846f;
  const __m256 two_pi = _mm256_set1_ps(s_two_pi);
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 minus_two = _mm256_set1_ps(-2.0f);
  __m256 mu1, mu2, sigma1, sigma2;
  __m256 u1, u2;
  __m256 radius, theta;
  __m256 costheta, sintheta;
  float s_mu1, s_mu2, s_sigma1, s_sigma2;
  float s_u1, s_u2;
  float s_radius, s_theta;
  float s_costheta, s_sintheta;
  float *xs_end = xs + len;
  for ( ; xs + 16 <= xs_end; succ_ratio += 16, num_trials += 16, xs += 16) {
    mu1 = _mm256_loadu_ps(succ_ratio);
    mu2 = _mm256_loadu_ps(succ_ratio + 8);
    sigma1 = _mm256_sqrt_ps(_mm256_div_ps(_mm256_mul_ps(mu1, _mm256_sub_ps(one, mu1)), _mm256_loadu_ps(num_trials)));
    sigma2 = _mm256_sqrt_ps(_mm256_div_ps(_mm256_mul_ps(mu2, _mm256_sub_ps(one, mu2)), _mm256_loadu_ps(num_trials + 8)));
    u1 = _mm256_loadu_ps(xs);
    u1 = _mm256_sub_ps(one, u1);
    u2 = _mm256_loadu_ps(xs + 8);
    radius = _mm256_sqrt_ps(_mm256_mul_ps(minus_two, log256_ps(u1)));
    theta = _mm256_mul_ps(two_pi, u2);
    sincos256_ps(theta, &sintheta, &costheta);
    _mm256_store_ps(xs,     _mm256_add_ps(mu1, _mm256_mul_ps(sigma1, _mm256_mul_ps(radius, costheta))));
    _mm256_store_ps(xs + 8, _mm256_add_ps(mu2, _mm256_mul_ps(sigma2, _mm256_mul_ps(radius, sintheta))));
  }
  /*for ( ; xs + 8 <= xs_end; xs += 8) {
    // TODO(20151007): SSE version.
  }*/
  for ( ; xs + 2 <= xs_end; succ_ratio += 2, num_trials += 2, xs += 2) {
    s_mu1 = succ_ratio[0];
    s_mu2 = succ_ratio[1];
    s_sigma1 = cephes_sqrtf(s_mu1 * (1.0f - s_mu1) / num_trials[0]);
    s_sigma2 = cephes_sqrtf(s_mu2 * (1.0f - s_mu2) / num_trials[1]);
    s_u1 = 1.0f - xs[0];
    s_u2 = xs[1];
    s_radius = cephes_sqrtf(-2.0f * cephes_logf(s_u1));
    s_theta = s_two_pi * s_u2;
    s_costheta = cephes_cosf(s_theta);
    s_sintheta = cephes_sinf(s_theta);
    xs[0] = s_mu1 + s_sigma1 * s_radius * s_costheta;
    xs[1] = s_mu2 + s_sigma2 * s_radius * s_sintheta;
  }
  if (xs < xs_end) {
    // TODO
    //s_mu1 = succ_ratio[0];
    //s_sigma1 = cephes_sqrtf(s_mu1 * (1.0f - s_mu1) / num_trials[0]);
  }
}
