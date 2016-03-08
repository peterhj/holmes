extern crate gsl;
extern crate holmes;
extern crate rand;

use gsl::{Gsl, panic_error_handler};
use gsl::functions::{norm_inc_beta};
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

#[test]
fn test_omega_finite_1() {
  Gsl::disable_error_handler();
  //Gsl::set_error_handler(panic_error_handler);

  let mut shared_data = OmegaNodeSharedData::new();
  shared_data.horizon = 20;
  shared_data.prior_equiv = 1.0;
  shared_data.diag_rank = 2;
  shared_data.net_succs = vec![
      48.0, 24.0, 29.0, 37.0, 30.0, 37.0, 23.0, 52.0, 28.0, 43.0,
      51.0, 37.0, 45.0, 40.0, 34.0, 46.0, 78.0, 118.0, 53.0, 36.0,
  ];
  shared_data.net_fails = vec![
      10.0, 8.0, 8.0, 10.0, 8.0, 11.0, 8.0, 11.0, 9.0, 10.0,
      13.0, 9.0, 11.0, 10.0, 9.0, 10.0, 14.0, 16.0, 13.0, 10.0,
  ];

  let shared_data = Rc::new(RefCell::new(shared_data));

  let mut likelihood_int = OmegaLikelihoodIntegral::new(shared_data.clone());
  let p = likelihood_int.calculate_likelihood();
  println!("TEST (1): p: {:.6}", p);

  let mut diag_int = OmegaDiagonalGFactorIntegral::new(shared_data.clone());
  let diag_g = diag_int.calculate_diagonal_g_factor();
  println!("TEST (1): g: {:.6}", diag_g);

  let j = 17;
  //let incbeta = norm_inc_beta(0.5, shared_data.borrow().net_succs[j], shared_data.borrow().net_fails[j]);
  //println!("TEST (1): beta:  {:.6}", incbeta);
  let mut cross_int = OmegaCrossGFactorIntegral::new(shared_data.clone());
  let cross_g = cross_int.calculate_cross_g_factor(j);
  println!("TEST (1): G[{}]: {:.6}", j, cross_g);
}

#[test]
fn test_omega_finite_2() {
  Gsl::disable_error_handler();
  //Gsl::set_error_handler(panic_error_handler);

  let mut shared_data = OmegaNodeSharedData::new();
  shared_data.horizon = 20;
  shared_data.prior_equiv = 1.0;
  shared_data.diag_rank = 10;
  shared_data.net_succs = vec![
      17.499309539794922, 8.462668418884277, 310.7706604003906, 1.2225075960159302, 1.1900842189788818, 1.1789861917495728, 1.1253794431686401, 1.0573794841766357, 1.034510612487793, 1.0331456661224365, 3.0293936729431152, 1.024675965309143, 1.0146799087524414, 1.0113030672073364, 1.0083945989608765, 1.0082944631576538, 1.0080050230026245, 1.0079834461212158, 1.00404691696167, 1.0031243562698364
  ];
  shared_data.net_fails = vec![
      57.50069046020508, 31.537330627441406, 446.2293395996094, 16.77749252319336, 16.80991554260254, 17.821014404296875, 17.87462043762207, 16.94261932373047, 16.96548843383789, 16.966854095458984, 20.970605850219727, 17.975322723388672, 16.985321044921875, 16.988697052001953, 16.991605758666992, 16.99170684814453, 16.991994857788086, 17.992015838623047, 16.995952606201172, 17.996875762939453
  ];

  let shared_data = Rc::new(RefCell::new(shared_data));

  let mut likelihood_int = OmegaLikelihoodIntegral::new(shared_data.clone());
  let p = likelihood_int.calculate_likelihood();
  println!("TEST (1): p: {:.6}", p);

  let mut diag_int = OmegaDiagonalGFactorIntegral::new(shared_data.clone());
  let diag_g = diag_int.calculate_diagonal_g_factor();
  println!("TEST (1): g: {:.6}", diag_g);

  let j = 2;
  //let incbeta = norm_inc_beta(0.5, shared_data.borrow().net_succs[j], shared_data.borrow().net_fails[j]);
  //println!("TEST (1): beta:  {:.6}", incbeta);
  let mut cross_int = OmegaCrossGFactorIntegral::new(shared_data.clone());
  let cross_g = cross_int.calculate_cross_g_factor(j);
  println!("TEST (1): G[{}]: {:.6}", j, cross_g);
}
