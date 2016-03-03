use byteorder::{ReadBytesExt, WriteBytesExt};

use std::io::{Read, BufRead, Write, BufReader};
use std::net::{TcpStream};
use std::str::{from_utf8};
use std::thread::{sleep_ms};

pub struct NngsConfig {
  pub login:    Vec<u8>,
  pub password: Option<Vec<u8>>,
}

enum PromptKind {
  Login,
  Password,
  Command,
}

pub struct NngsClient {
  host:     String,
  port:     u16,
  config:   NngsConfig,
  reader:   BufReader<TcpStream>,
  writer:   TcpStream,
  _history: Vec<Vec<u8>>,
}

impl NngsClient {
  pub fn new(host: String, port: u16, config: NngsConfig) -> NngsClient {
    let stream = match TcpStream::connect((&host as &str, port)) {
      Ok(stream) => stream,
      Err(_) => panic!("failed to connect to nngs server"),
    };
    let reader = BufReader::new(match stream.try_clone() {
      Ok(stream) => stream,
      Err(_) => panic!("failed to clone stream"),
    });
    NngsClient{
      host:     host,
      port:     port,
      config:   config,
      reader:   reader,
      writer:   stream,
      _history: Vec::with_capacity(500),
    }
  }

  pub fn run_loop(&mut self) {
    println!("DEBUG: starting run loop");
    self.login();
  }

  fn login(&mut self) {
    self.prompt_login();
    {
      let login = self.config.login.clone();
      self.submit(&login);
    }
    if self.config.password.is_some() {
      self.prompt_password();
      {
        let password = self.config.password.clone();
        self.submit(password.as_ref().unwrap());
      }
    }
    self.prompt_cmd();
    self.submit(b"quit");
    loop {
      sleep_ms(1000);
      break;
    }
  }

  fn prompt(&mut self, kind: PromptKind) {
    let (num_prefix, prompt_str) = match kind {
      PromptKind::Login     => (7,  b"Login: " as &'static [u8]),
      PromptKind::Password  => (10, b"Password: " as &'static [u8]),
      PromptKind::Command   => (3,  b"#> " as &'static [u8]),
    };
    loop {
      let mut buf = Vec::with_capacity(160);

      let mut prefix = Vec::with_capacity(num_prefix);
      for _ in 0 .. num_prefix {
        let p = self.reader.read_u8().unwrap();
        prefix.push(p);
      }

      if &prefix as &[u8] == prompt_str {
        println!("DEBUG: {}", from_utf8(&prefix).unwrap());
        return;
      }

      for k in 0 .. num_prefix {
        let p = prefix[k];
        buf.push(p);
        if p == b'\n' {
          self._history.push(buf.clone());
          buf.clear();
        }
      }

      self.reader.read_until(b'\n', &mut buf).unwrap();
      self._history.push(buf.clone());
      if let Ok(s) = from_utf8(&buf) {
        println!("DEBUG: {}", s);
      } else {
        println!("DEBUG: (unprintable)");
      }
      buf.clear();
    }
  }

  fn prompt_login(&mut self) {
    self.prompt(PromptKind::Login);
  }

  fn prompt_password(&mut self) {
    self.prompt(PromptKind::Password);
  }

  fn prompt_cmd(&mut self) {
    self.prompt(PromptKind::Command);
  }

  fn submit(&mut self, msg: &[u8]) {
    self.writer.write_all(msg).unwrap();
    self.writer.write_u8(b'\n').unwrap();
  }
}
