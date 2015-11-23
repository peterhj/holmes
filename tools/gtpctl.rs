extern crate holmes;

use holmes::gtp::{GtpController};

use std::env;

fn main() {
  let args: Vec<String> = env::args().collect();
  let b_port: u16 = args[1].parse().unwrap();
  let w_port: u16 = args[2].parse().unwrap();
  GtpController::new(b_port, w_port).runloop();
}
