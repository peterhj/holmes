extern crate getopts;
extern crate holmes;

use holmes::agents::parallel_search::{ParallelMonteCarloSearchAgent};
use holmes::gtp::{GtpEngine};
use holmes::gtp_client::{Client};
use holmes::search::parallel_tree::{MonteCarloSearchConfig, TreePolicyConfig, HorizonConfig};

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
  let mc_cfg = MonteCarloSearchConfig{
    //num_rollouts:   1024,
    num_rollouts:   4096,
    //batch_size:     16,
    batch_size:     128,
  };
  let tree_cfg = TreePolicyConfig{
    horizon_cfg:    HorizonConfig::Fixed{max_horizon: 20},
    visit_thresh:   1,
    mc_scale:       1.0,
    //mc_scale:       0.125,
    prior_equiv:    16.0,
    rave:           false,
    rave_equiv:     0.0,
  };
  let agent = ParallelMonteCarloSearchAgent::new(mc_cfg, tree_cfg, None);
  let client = Client::new(agent, host, port, None);
  GtpEngine::new(client).runloop();
}
