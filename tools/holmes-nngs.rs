extern crate holmes;

//use holmes::client::nngs::{NngsClient, NngsConfig};
use holmes::client::agent::{HelloAsyncAgent};
use holmes::client::nngs::{NngsOneShotClient, NngsServerConfig, NngsMatchConfig};

fn main() {
  /*let config = NngsConfig{
    login:    b"guest".to_vec(),
    password: None,
  };
  let mut client = NngsClient::new("jsb.cs.uec.ac.jp".to_string(), 9696, config);
  client.run_loop();*/

  let server_cfg = NngsServerConfig{
    host:     "jsb.cs.uec.ac.jp".to_string(),
    port:     9696,
    login:    "guest2".to_string(),
    password: None,
  };

  /*let mut client = NngsOneShotClient::new(server_cfg, None);
  client.run_loop();*/

  let mut client = NngsOneShotClient::<HelloAsyncAgent>::new(server_cfg, None);
  client.run_loop();
}
