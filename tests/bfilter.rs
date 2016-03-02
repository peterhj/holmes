extern crate holmes;
extern crate rand;
extern crate rng;

use holmes::discrete::{DiscreteFilter};
use holmes::discrete::bfilter::{BFilter};
use holmes::stats::{ChisquareTest};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, thread_rng};
use rand::distributions::{IndependentSample};
use rand::distributions::range::{Range};
use std::iter::{repeat};

#[test]
fn test_chisquare_once() {
  let d = 361;
  let n_samples = 361000;
  let exp_probs: Vec<_> = repeat(1.0 / d as f64).take(d).collect();

  let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());

  let range = Range::new(0, d);
  let mut obs_counts: Vec<_> = repeat(0).take(d).collect();
  for _ in 0 .. n_samples {
    let j = range.ind_sample(&mut rng);
    obs_counts[j] += 1;
  }

  let chisquare = ChisquareTest::new(0.001, d);
  println!("(1) crit value:       {:?}", chisquare.critical_value());
  println!("(1) test result:      {:?}", chisquare.test_null(&obs_counts, &exp_probs));
}

#[test]
fn test_bfilter_once() {
  let d = 361;
  let n_samples = 361000;
  let exp_probs: Vec<_> = repeat(1.0 / d as f64).take(d).collect();
  let exp_probs32: Vec<_> = repeat(1.0 / d as f32).take(d).collect();

  let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());

  let mut filter = BFilter::with_capacity(d);
  filter.reset(&exp_probs32);

  let mut obs_counts: Vec<_> = repeat(0).take(d).collect();
  for _ in 0 .. n_samples {
    if let Some(j) = filter.sample(&mut rng) {
      obs_counts[j] += 1;
    } else {
      panic!();
    }
  }

  let chisquare = ChisquareTest::new(0.001, d);
  println!("(2) crit value:       {:?}", chisquare.critical_value());
  println!("(2) test result:      {:?}", chisquare.test_null(&obs_counts, &exp_probs));
}

#[test]
fn test_bfilter_ablation() {
  let n_trials = 1000;
  let d = 361;
  let d_keep = 100;
  let n_samples = 361000;
  let chisquare = ChisquareTest::new(0.001, d_keep);
  println!("(3) critical value:   {:?}", chisquare.critical_value());

  let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
  let uniform = Range::new(0.0f64, 1.0f64);

  let mut rejections = 0;
  for _ in 0 .. n_trials {
    /*let mut exp_probs: Vec<_> = repeat(1.0 / d as f64).take(d).collect();
    let mut exp_probs32: Vec<_> = repeat(1.0 / d as f32).take(d).collect();*/
    let mut exp_probs = Vec::with_capacity(d);
    let mut exp_probs32 = Vec::with_capacity(d);
    for _ in 0 .. d {
      let p = uniform.ind_sample(&mut rng);
      exp_probs.push(p);
      exp_probs32.push(p as f32);
    }

    let mut indexes: Vec<_> = (0 .. d).collect();
    for k in d_keep .. d {
      exp_probs[k] = 0.0;
      exp_probs32[k] = 0.0;
    }
    let mut normalize = 0.0;
    let mut normalize32 = 0.0;
    for j in 0 .. d_keep {
      normalize += exp_probs[j];
      normalize32 += exp_probs32[j];
    }
    for j in 0 .. d_keep {
      exp_probs[j] /= normalize;
      exp_probs32[j] /= normalize32;
    }

    let mut filter = BFilter::with_capacity(d);
    filter.reset(&exp_probs32);

    let mut obs_counts: Vec<_> = repeat(0).take(d).collect();
    for _ in 0 .. n_samples {
      if let Some(j) = filter.sample(&mut rng) {
        assert!(j < d_keep);
        obs_counts[j] += 1;
      } else {
        panic!();
      }
    }

    //println!("(3) crit value:  {:?}", chisquare.critical_value());
    //println!("(3) test result: {:?}", chisquare.test_null(&obs_counts[ .. d_keep], &exp_probs[ .. d_keep]));
    if chisquare.test_null(&obs_counts[ .. d_keep], &exp_probs[ .. d_keep]).is_err() {
      rejections += 1;
    }
  }

  println!("(3) null rejections:  {}/{}", rejections, n_trials);
  // XXX(20160301): Ad-hoc threshold on acceptable number of rejections.
  assert!(rejections <= 3);
}
