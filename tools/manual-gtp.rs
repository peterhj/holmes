extern crate getopts;
extern crate holmes;

use holmes::agents::manual::{ManualAgent};
use holmes::gtp::{GtpEngine};
use holmes::gtp_board::{Player};
use holmes::gtp_client::{Client};

use getopts::{Options};
use std::env;

fn main() {
  let args: Vec<_> = env::args().collect();
  let mut opts = Options::new();
  opts.optopt("h", "host", "host address for GTP", "host");
  opts.optopt("p", "port", "port for GTP", "port");
  let matches = match opts.parse(&args[1 ..]) {
    Ok(m) => m,
    Err(e) => panic!("FATAL: {:?}", e),
  };
  let host = matches.opt_str("h").expect("FATAL: manual: host required");
  let port: u16 = matches.opt_str("p").expect("FATAL: manual: port required")
    .parse().ok().expect("FATAL: manual: port should be an integer");
  println!("DEBUG: manual: host: {}", host);
  println!("DEBUG: manual: port: {}", port);
  let agent = ManualAgent::new();
  let client = Client::new(agent, host, port, None);
  GtpEngine::new(client).runloop();
}
