//extern crate statistics_avx2;
extern crate holmes;
extern crate rand;

//use statistics_avx2::array::{array_argmax};
use holmes::array_util::{array_argmax};

use rand::{Rng, thread_rng};
use rand::distributions::{IndependentSample};
use rand::distributions::range::{Range};
use std::iter::{repeat};

#[test]
fn test_argmax() {
  let trials = 1_000_000;
  let n = 1000;
  let dist = Range::new(0.0f32, 1.0f32);

  let mut rng = thread_rng();
  let mut xs: Vec<_> = repeat(0.0f32).take(n).collect();
  assert_eq!(n, xs.len());

  for i in (0 .. trials) {
    for j in (0 .. n) {
      xs[j] = dist.ind_sample(&mut rng);
    }

    let maybe_max_j = array_argmax(&xs);
    match maybe_max_j {
      Some(max_j) => {
        for j in (0 .. n) {
          assert!(xs[max_j] >= xs[j]);
        }
      }
      None => {
        panic!("array_argmax should have found a maximum!");
      }
    }
  }
}
