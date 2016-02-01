extern crate getopts;
extern crate holmes;

use holmes::agents::convnet_new::{ConvnetAgent};
use holmes::gtp::{GtpEngine};
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
    Err(e) => panic!("failed to parse args: {:?}", e),
  };
  let host = matches.opt_str("h").expect("FATAL: holmes: host required");
  let port: u16 = matches.opt_str("p").expect("FATAL: holmes: port required")
    .parse().ok().expect("FATAL: holmes: port should be an integer");
  println!("DEBUG: holmes: host: {}", host);
  println!("DEBUG: holmes: port: {}", port);
  let agent = ConvnetAgent::new();
  let client = Client::new(agent, host, port, None);
  GtpEngine::new(client).runloop();
}
