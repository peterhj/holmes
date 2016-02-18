extern crate getopts;
extern crate holmes;

use holmes::agents::parallel_search::{MonteCarloConfig, ParallelMonteCarloSearchAgent};
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
  let mc_cfg = MonteCarloConfig{
    /*num_rollouts:   1024,
    batch_size:     32,*/
    num_rollouts:   1024,
    batch_size:     128,
    /*num_rollouts:   2048,
    batch_size:     64,*/
    /*num_rollouts:   5120,
    batch_size:     256,*/
  };
  let agent = ParallelMonteCarloSearchAgent::new(mc_cfg, None);
  let client = Client::new(agent, host, port, None);
  GtpEngine::new(client).runloop();
}
