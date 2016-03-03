extern crate holmes;

use holmes::client::nngs::{NngsClient, NngsConfig};

fn main() {
  let config = NngsConfig{
    login:    b"guest".to_vec(),
    password: None,
  };
  let mut client = NngsClient::new("jsb.cs.uec.ac.jp".to_string(), 9696, config);
  client.run_loop();
}
