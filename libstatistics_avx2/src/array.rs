use libc::{size_t};

pub fn array_argmax(xs: &[f32]) -> usize {
  unsafe { statistics_arg_amax(xs.as_ptr(), xs.len() as size_t) as usize }
}

#[link(name = "statistics_avx2_impl", kind = "static")]
extern "C" {
  fn statistics_arg_amax(xs: *const f32, len: size_t) -> size_t;
}
