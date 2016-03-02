use gsl::functions::{chisquare_cdf_comp};

pub fn chisquare_critical_value(p: f64, df: usize) -> f64 {
  // XXX(20160301): See: <http://www.stat.uchicago.edu/~mcpeek/software/MQLS/chisq.c>.
  if p <= 0.0 || p >= 1.0 {
    panic!();
  }
  let epsilon = 1.0e-6;
  let mut min_v = 0.0;
  let mut max_v = 1.0e5; // FIXME(20160301): or some other big number.
  let mut v = df as f64 / p.sqrt();
  while max_v - min_v > epsilon {
    if chisquare_cdf_comp(v, df as f64) < p {
      max_v = v;
    } else {
      min_v = v;
    }
    v = 0.5 * (max_v + min_v);
  }
  v
}

pub fn chisquare_test_statistic(obs_counts: &[usize], exp_probs: &[f64]) -> f64 {
  assert_eq!(obs_counts.len(), exp_probs.len());
  let d = obs_counts.len();
  let mut n = 0;
  for k in 0 .. d {
    n += obs_counts[k];
  }
  let n = n as f64;
  let mut x = 0.0;
  for k in 0 .. d {
    let y = obs_counts[k] as f64;
    let p = exp_probs[k];
    let error = y / (p * n) - 1.0;
    x += p * error * error;
  }
  x *= n;
  x
}

pub struct ChisquareTest {
  p:    f64,
  d:    usize,
  crit: f64,
}

impl ChisquareTest {
  pub fn new(p: f64, d: usize) -> ChisquareTest {
    ChisquareTest{
      p:    p,
      d:    d,
      crit: chisquare_critical_value(p, d - 1),
    }
  }

  pub fn critical_value(&self) -> f64 {
    self.crit
  }

  pub fn test_null(&self, obs_counts: &[usize], exp_probs: &[f64]) -> Result<f64, f64> {
    let x = chisquare_test_statistic(obs_counts, exp_probs);
    if x < self.crit {
      Ok(x)
    } else {
      Err(x)
    }
  }
}
