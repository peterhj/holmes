extern crate holmes;
extern crate rand;

use holmes::search::parallel_trace::omega::{
  OmegaNodeSharedData,
  OmegaLikelihoodIntegral,
  OmegaDiagonalGFactorIntegral,
  OmegaCrossGFactorIntegral,
};

use std::cell::{RefCell};
use std::rc::{Rc};

#[test]
fn test_omega() {
  let mut shared_data = OmegaNodeSharedData::new();
  shared_data.horizon = 3;
  shared_data.prior_equiv = 1.0;
  shared_data.diag_rank = 0;
  shared_data.net_succs = vec![88.0, 18.0, 10.0];
  shared_data.net_fails = vec![2.0, 2.0, 2.0];

  let shared_data = Rc::new(RefCell::new(shared_data));

  let mut likelihood_int = OmegaLikelihoodIntegral::new(shared_data.clone());
  let p = likelihood_int.calculate_likelihood();
  println!("TEST: p: {:.6}", p);

  let mut diag_int = OmegaDiagonalGFactorIntegral::new(shared_data.clone());
  let diag_g = diag_int.calculate_diagonal_g_factor();
  println!("TEST: g: {:.6}", diag_g);

  let mut cross_int = OmegaCrossGFactorIntegral::new(shared_data.clone());
  let cross_g = cross_int.calculate_cross_g_factor(1);
  println!("TEST: G_1: {:.6}", cross_g);
  let cross_g = cross_int.calculate_cross_g_factor(2);
  println!("TEST: G_2: {:.6}", cross_g);
}
