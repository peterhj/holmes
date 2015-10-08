use libc::{size_t};

pub fn array_zero(xs: &mut [f32]) {
  unsafe { statistics_array_fill(0.0, xs.as_mut_ptr(), xs.len() as size_t) };
}

pub fn array_fill(c: f32, xs: &mut [f32]) {
  unsafe { statistics_array_fill(c, xs.as_mut_ptr(), xs.len() as size_t) };
}

pub fn array_argmax(xs: &[f32]) -> usize {
  unsafe { statistics_array_argmax(xs.as_ptr(), xs.len() as size_t) as usize }
}

#[link(name = "statistics_avx2_impl", kind = "static")]
extern "C" {
  fn statistics_array_fill(c: f32, xs: *mut f32, len: size_t);
  fn statistics_array_argmax(xs: *const f32, len: size_t) -> size_t;
}
