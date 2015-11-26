use toml;

use rustc_serialize::json;
use std::collections::{BTreeMap};
use std::env;
use std::fs::{File};
use std::io::{Read};
use std::path::{PathBuf};

//thread_local!(static HYPERPARAM_INSTANCE: BTreeMap<String, json::Json> = {
thread_local!(static HYPERPARAM_INSTANCE: BTreeMap<String, toml::Value> = {
  let path = match env::var("HYPERPARAM_INSTANCE_PATH") {
    Ok(path)  => PathBuf::from(&path),
    Err(e)    => panic!("no env variable for hyperparam instance path: {:?}", e),
  };
  let mut file = match File::open(&path) {
    Ok(file)  => file,
    Err(e)    => panic!("failed to open hyperparam instance file: {:?}", e),
  };
  /*let instance_map: BTreeMap<String, json::Json> = match json::Json::from_reader(&mut file) {
    Ok(json::Json::Object(obj)) => obj,
    Ok(js)  => panic!("unexpected json variant: {:?}", js),
    Err(e)  => panic!("failed to read json object: {:?}", e),
  };
  instance_map*/
  let mut s = String::new();
  file.read_to_string(&mut s).unwrap();
  let mut parser = toml::Parser::new(&s);
  parser.parse().unwrap()
});

thread_local!(pub static UCB_C:       f32   = load_hyperparam("ucb_c"));
thread_local!(pub static PBIAS_C:     f32   = load_hyperparam("pbias_c"));
thread_local!(pub static PBIAS_EQUIV: f32   = load_hyperparam("pbias_equiv"));
thread_local!(pub static PWIDE:       bool  = load_hyperparam("pwide"));
thread_local!(pub static PWIDE_MU:    f32   = load_hyperparam("pwide_mu"));

thread_local!(pub static RAVE:        bool  = load_hyperparam("rave"));
thread_local!(pub static RAVE_EQUIV:  f32   = load_hyperparam("rave_equiv"));
thread_local!(pub static PRIOR_EQUIV: f32   = load_hyperparam("prior_equiv"));

pub trait Hyperparam {
  fn from_json(js: &json::Json) -> Option<Self> where Self: Sized;
  fn from_toml(tm: &toml::Value) -> Option<Self> where Self: Sized;
}

impl Hyperparam for bool {
  fn from_json(js: &json::Json) -> Option<bool> {
    match js {
      &json::Json::Boolean(x) => Some(x),
      _ => None,
    }
  }

  fn from_toml(tm: &toml::Value) -> Option<bool> {
    match tm {
      &toml::Value::Boolean(x) => Some(x),
      _ => None,
    }
  }
}

impl Hyperparam for f32 {
  fn from_json(js: &json::Json) -> Option<f32> {
    match js {
      &json::Json::F64(x) => Some(x as f32),
      _ => None,
    }
  }

  fn from_toml(tm: &toml::Value) -> Option<f32> {
    match tm {
      &toml::Value::Float(x) => Some(x as f32),
      _ => None,
    }
  }
}

pub fn load_hyperparam<T>(key: &str) -> T where T: Hyperparam {
  let res = HYPERPARAM_INSTANCE.with(|instance| {
    let value = match instance.get(key) {
      Some(value) => value,
      None => panic!("failed to find hyperparam: '{}'", key)
    };
    //Hyperparam::from_json(value)
    Hyperparam::from_toml(value)
  });
  match res {
    Some(x) => x,
    None => panic!("failed to load parameter: '{}'", key),
  }
}
