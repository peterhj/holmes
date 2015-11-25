use rand::{Rng};

pub mod bfilter;

pub trait DiscreteFilter {
  fn reset(&mut self, xs: &[f32]);
  fn zero(&mut self, j: usize);
  fn sample<R: Rng>(&mut self, rng: &mut R) -> Option<usize>;
}
