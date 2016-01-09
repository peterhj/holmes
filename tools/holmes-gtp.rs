extern crate getopts;
extern crate holmes;

use holmes::agents::convnet::{ConvnetAgent};
use holmes::agents::parallel_search::{ParallelMonteCarloSearchAgent};
use holmes::agents::search::{SearchAgent};
use holmes::gtp::{GtpEngine};
use holmes::gtp_board::{Player};
use holmes::gtp_client::{Client};

use getopts::{Options};
use std::env;
//use std::os;

fn main() {
  let args: Vec<_> = env::args().collect();
  let mut opts = Options::new();
  opts.optopt("h", "host", "host address for GTP", "host");
  opts.optopt("p", "port", "port for GTP", "port");
  //opts.optopt("c", "color", "color", "black|white");
  let matches = match opts.parse(&args[1 ..]) {
    Ok(m) => m,
    Err(e) => panic!("FATAL: {:?}", e),
  };
  let host = matches.opt_str("h").expect("FATAL: holmes: host required");
  let port: u16 = matches.opt_str("p").expect("FATAL: holmes: port required")
    .parse().ok().expect("FATAL: holmes: port should be an integer");
  //let player_str = matches.opt_str("c").expect("FATAL: holmes: player color required");
  println!("DEBUG: holmes: host: {}", host);
  println!("DEBUG: holmes: port: {}", port);
  //println!("DEBUG: holmes: player: {}", player_str);
  /*let player = match &player_str as &str {
    "black" => Some(Player::Black),
    "white" => Some(Player::White),
    _ => panic!("FATAL: holmes: invalid player: {}", player_str),
  };*/
  //let agent = ConvnetAgent::new();
  //let agent = SearchAgent::new();
  let agent = ParallelMonteCarloSearchAgent::new(None);
  let client = Client::new(agent, host, port, None);
  GtpEngine::new(client).runloop();
}
