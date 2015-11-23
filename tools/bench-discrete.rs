extern crate holmes;
extern crate rand;
extern crate time;

use holmes::discrete::bfilter::{BFilter};
use holmes::random::{XorShift128PlusRng};
use rand::{Rng, SeedableRng, thread_rng};
use rand::distributions::{IndependentSample};
use rand::distributions::range::{Range};
use std::iter::{repeat};
use time::{Duration, get_time};

fn main() {
  let trials = 330000;
  let n = 361;
  let range = Range::new(0.0f32, 1.0f32);
  let mut slow_rng = thread_rng();
  let mut rng = XorShift128PlusRng::from_seed([slow_rng.next_u64(), slow_rng.next_u64()]);

  let mut filter = BFilter::with_capacity(n);
  let mut xs: Vec<_> = repeat(0.0f32).take(n).collect();
    for x in xs.iter_mut() {
      *x = range.ind_sample(&mut rng);
    }

  let start_time = get_time();
  for _ in (0 .. trials) {
    for x in xs.iter_mut() {
      *x = range.ind_sample(&mut rng);
    }
    filter.fill(&xs);
    loop {
      if let Some(j) = filter.sample(&mut rng) {
        assert!(j < n);
        filter.zero(j);
        break;
      } else {
        break;
      }
    }
  }

  let end_time = get_time();
  let elapsed_ms = (end_time - start_time).num_milliseconds() as f32;
  println!("DEBUG: elapsed: {:.3} s", elapsed_ms * 0.001);
}
