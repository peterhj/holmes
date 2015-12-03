extern crate holmes;

use holmes::gtp_ctrl::{GtpRefereedController};

use std::env;

fn main() {
  let args: Vec<String> = env::args().collect();
  let b_port: u16 = args[1].parse().unwrap();
  let w_port: u16 = args[2].parse().unwrap();
  let r_port: u16 = args[3].parse().unwrap();
  GtpRefereedController::new(b_port, w_port, r_port).runloop();
}
