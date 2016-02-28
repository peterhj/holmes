use std::ops::{Add};

pub struct UFix64 {value: u64}

impl UFix64 {
  pub fn into_u64(self) -> u64 {
    self.value
  }
}

impl Add for UFix64 {
  type Output = UFix64;

  fn add(self, other: UFix64) -> UFix64 {
    UFix64{value: self.value + other.value}
  }
}
