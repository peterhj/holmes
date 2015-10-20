use libc::{size_t};

pub fn array_zero(xs: &mut [f32]) {
  unsafe { statistics_array_fill(0.0, xs.as_mut_ptr(), xs.len() as size_t) };
}

pub fn array_fill(c: f32, xs: &mut [f32]) {
  unsafe { statistics_array_fill(c, xs.as_mut_ptr(), xs.len() as size_t) };
}

pub fn array_sum(xs: &[f32]) -> f32 {
  unsafe { statistics_array_sum(xs.as_ptr(), xs.len() as size_t) }
}

pub fn array_argmax(xs: &[f32]) -> usize {
  unsafe { statistics_array_argmax(xs.as_ptr(), xs.len() as size_t) as usize }
}

pub fn array_binary_search(xs: &[f32], query: f32) -> usize {
  unsafe { statistics_array_binary_search(xs.as_ptr(), xs.len() as size_t, query) as usize }
}

#[link(name = "statistics_avx2_impl", kind = "static")]
extern "C" {
  fn statistics_array_fill(c: f32, xs: *mut f32, len: size_t);
  fn statistics_array_sum(xs: *const f32, len: size_t) -> f32;
  fn statistics_array_argmax(xs: *const f32, len: size_t) -> size_t;
  fn statistics_array_binary_search(xs: *const f32, len: size_t, x: f32) -> size_t;
}
