extern crate holmes;

use holmes::gtp::{GtpController};

fn main() {
  GtpController::new().runloop();
}
