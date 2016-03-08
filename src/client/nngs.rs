use board::{Stone, Action, Point};
use gtp_board::{Coord};

use byteorder::{ReadBytesExt, WriteBytesExt};

use std::io::{Read, BufRead, Write, BufReader};
use std::marker::{PhantomData};
use std::net::{TcpStream};
use std::str::{from_utf8};
use std::sync::{Arc, Barrier};
use std::sync::mpsc::{Sender, Receiver, channel};
use std::thread::{JoinHandle, sleep_ms, spawn};

#[derive(Clone, Debug)]
pub struct NngsServerConfig {
  pub host:     String,
  pub port:     u16,
  pub login:    Vec<u8>,
  pub password: Option<Vec<u8>>,
}

#[derive(Clone, Debug)]
pub struct NngsMatchConfig {
  pub color:    Stone,
  pub opponent: Vec<u8>,
  pub time:     i32,
  pub byoyomi:  i32,
}

#[derive(Clone, Copy, Debug)]
pub enum AgentMsg {
  StartMatch{our_stone: Stone, board_size: i32, main_time_secs: i32, byoyomi_time_secs: i32},
  CheckTime,
  RecvTime,
  SubmitAction{turn: Stone, action: Action},
  RecvAction{move_number: i32, turn: Stone, action: Action},
}

pub trait OneShotAgent {
  fn spawn_runloop(agent_in_rx: Receiver<AgentMsg>, agent_out_tx: Sender<AgentMsg>) -> JoinHandle<()>;
}

pub struct HelloOneShotAgent;

impl OneShotAgent for HelloOneShotAgent {
  fn spawn_runloop(agent_in_rx: Receiver<AgentMsg>, agent_out_tx: Sender<AgentMsg>) -> JoinHandle<()> {
    spawn(move || {
      let mut started_match = false;
      let mut match_stone = None;
      loop {
        match agent_in_rx.recv() {
          Ok(AgentMsg::StartMatch{our_stone, ..}) => {
            started_match = true;
            match_stone = Some(our_stone);
          }
          Ok(AgentMsg::RecvTime) => {
            assert!(started_match);
          }
          Ok(AgentMsg::RecvAction{..}) => {
            assert!(started_match);
            agent_out_tx.send(AgentMsg::SubmitAction{
              turn:     match_stone.unwrap(),
              action:   Action::Pass,
            }).unwrap();
          }
          _ => {}
        }
      }
    })
  }
}

pub struct NngsOneShotClient<A> where A: OneShotAgent {
  server_cfg:   NngsServerConfig,
  match_cfg:    Option<NngsMatchConfig>,
  barrier:      Arc<Barrier>,
  agent:        JoinHandle<()>,
  bridge:       JoinHandle<()>,
  reader:       JoinHandle<()>,
  writer:       JoinHandle<()>,
  //reader:       BufReader<TcpStream>,
  //writer:       TcpStream,
  //_history:     Vec<Vec<u8>>,
  _marker:      PhantomData<A>,
}

fn submit(writer: &mut Write, msg: &[u8]) {
  writer.write_all(msg).unwrap();
  writer.write_u8(b'\n').unwrap();
}

enum InternalMsg {
  Quit,
  WriteCmd{cmd: Vec<u8>},
}

enum LoProtocol {
  Line{line: Vec<u8>},
  UnknownPrompt,        // 1
  ServerPrompt,         // 1 5
  GamePrompt,           // 1 6
  GameCleanupPrompt,    // 1 7
  GenericInfo{line: Vec<u8>},   // 9
  MoveInfo{line: Vec<u8>},      // 15
  ObserveInfo,          // 16
  ScoreInfo,            // 20
  ShoutInfo,            // 21
  GameInfo,             // 22
}

impl<A> NngsOneShotClient<A> where A: OneShotAgent {
  pub fn new(server_cfg: NngsServerConfig, match_cfg: Option<NngsMatchConfig>) -> NngsOneShotClient<A> {
    let stream = match TcpStream::connect((&server_cfg.host as &str, server_cfg.port)) {
      Ok(stream) => stream,
      Err(_) => panic!("failed to connect to nngs server"),
    };
    let reader = BufReader::new(match stream.try_clone() {
      Ok(stream) => stream,
      Err(_) => panic!("failed to clone stream"),
    });
    let barrier = Arc::new(Barrier::new(2));
    let (agent_in_tx, agent_in_rx) = channel();
    let (agent_out_tx, agent_out_rx) = channel();
    let (writer_tx, writer_rx) = channel();

    let agent_thr = A::spawn_runloop(agent_in_rx, agent_out_tx);

    let bridge_thr = {
      let agent_out_rx = agent_out_rx;
      let writer_tx = writer_tx.clone();
      spawn(move || {
        loop {
          match agent_out_rx.recv() {
            Ok(AgentMsg::CheckTime) => {
              // FIXME(20160307)
              writer_tx.send(InternalMsg::WriteCmd{cmd: b"time".to_vec()}).unwrap();
              unimplemented!();
            }
            Ok(AgentMsg::SubmitAction{turn, action}) => {
              let cmd = match action {
                Action::Resign => b"resign".to_vec(),
                Action::Pass => b"pass".to_vec(),
                Action::Place{point} => {
                  let coord = point.to_coord();
                  coord.to_bytestring()
                }
              };
              writer_tx.send(InternalMsg::WriteCmd{cmd: cmd}).unwrap();
            }
            Ok(_) => unreachable!(),
            Err(e) => {
              println!("WARNING: writer failed to recv internal msg: {:?}", e);
            }
          }
          sleep_ms(200);
        }
      })
    };

    let reader_thr = {
      //let barrier = barrier.clone();
      let agent_in_tx = agent_in_tx;
      let writer_tx = writer_tx;
      let mut reader = reader;
      spawn(move || {
        let mut history: Vec<LoProtocol> = Vec::with_capacity(1024);
        let mut prev_idx: Option<usize> = None;
        let mut idx: usize = 0;
        loop {
          let mut buf = Vec::with_capacity(160);
          match reader.read_until(b'\n', &mut buf) {
            Ok(_) => {}
            Err(e) => {
              // FIXME(20160307): gracefully shutdown.
            }
          }

          if buf.len() >= 2 && buf[0] == b'1' && buf[1] == b' ' {
            if buf.len() >= 3 && buf[2] == b'5' {
              history.push(LoProtocol::ServerPrompt);

              let mut is_match_request = false;
              let mut match_cmd = None;
              let mut match_our_stone = None;
              let mut match_board_size = None;
              let mut match_main_time = None;
              let mut match_byoyomi_time = None;

              let mut is_resigned = false;
              //let mut who_resigned = None;

              for i in prev_idx.unwrap_or(0) .. idx {
                match &history[i] {
                  &LoProtocol::GenericInfo{ref line} => {
                    let line_str = String::from_utf8_lossy(line);
                    if line_str.contains("Match") && line_str.contains("requested") {
                      is_match_request = true;
                    } else if is_match_request && line_str.contains("Use") {
                      let pre_toks: Vec<_> = line_str.splitn(2, "<").collect();
                      let suf_toks: Vec<_> = pre_toks[1].trim().splitn(2, ">").collect();
                      let tmp_match_cmd = suf_toks[0].clone();
                      let match_toks: Vec<_> = tmp_match_cmd.split_whitespace().collect();
                      assert_eq!(6, match_toks.len());
                      let our_stone = match match_toks[2] {
                        "B" | "b" => Stone::Black,
                        "W" | "w" => Stone::White,
                        _ => unreachable!(),
                      };
                      let board_size: i32 = match_toks[3].parse().unwrap();
                      assert_eq!(19, board_size);
                      let main_time: i32 = match_toks[4].parse().unwrap();
                      let byoyomi_time: i32 = match_toks[5].parse().unwrap();

                      // Set match details.
                      match_cmd = Some(tmp_match_cmd.as_bytes().to_vec());
                      match_our_stone = Some(our_stone);
                      match_board_size = Some(board_size);
                      match_main_time = Some(main_time);
                      match_byoyomi_time = Some(byoyomi_time);
                    } else if line_str.contains("has resigned the game") {
                      is_resigned = true;
                    }
                  }
                  _ => {}
                }
              }

              if is_match_request {
                // Send match details to agent channel.
                agent_in_tx.send(AgentMsg::StartMatch{
                  our_stone:    match_our_stone.unwrap(),
                  board_size:   match_board_size.unwrap(),
                  main_time_secs:       60 * match_main_time.unwrap(),
                  byoyomi_time_secs:    60 * match_byoyomi_time.unwrap(),
                }).unwrap();
                // Auto-accept match request.
                writer_tx.send(InternalMsg::WriteCmd{cmd: match_cmd.unwrap()}).unwrap();

              } else if is_resigned {
                writer_tx.send(InternalMsg::WriteCmd{cmd: b"quit".to_vec()}).unwrap();
              }

            } else if buf.len() >= 3 && buf[2] == b'6' {
              history.push(LoProtocol::GamePrompt);

              let mut is_move = false;
              let mut move_number = None;
              let mut move_turn = None;
              let mut move_action = None;

              for i in prev_idx.unwrap_or(0) .. idx {
                match &history[i] {
                  &LoProtocol::MoveInfo{ref line} => {
                    let line_str = String::from_utf8_lossy(line);
                    if !line_str.contains("Game") {
                      is_move = true;
                      let pre_toks: Vec<_> = line_str.splitn(2, "15").collect();
                      let suf_toks: Vec<_> = pre_toks[1].splitn(2, "(").collect();
                      let tmp_move_number: i32 = suf_toks[0].trim().parse().unwrap();
                      let suf2_toks: Vec<_> = suf_toks[1].splitn(2, "): ").collect();
                      let tmp_move_turn = match suf2_toks[0] {
                        "B" | "b" => Stone::Black,
                        "W" | "w" => Stone::White,
                        _ => unreachable!(),
                      };
                      let tmp_move_action = match suf2_toks[1].trim() {
                        "PASS" | "Pass" | "pass" => Action::Pass,
                        s => Action::Place{point: Point::from_coord(Coord::parse_code_str(s).unwrap())},
                      };

                      // Set move details.
                      move_number = Some(tmp_move_number);
                      move_turn = Some(tmp_move_turn);
                      move_action = Some(tmp_move_action);
                    }
                  }
                  _ => {}
                }
              }

              if is_move {
                // Send move details to agent channel.
                agent_in_tx.send(AgentMsg::RecvAction{
                  move_number:  move_number.unwrap(),
                  turn:         move_turn.unwrap(),
                  action:       move_action.unwrap(),
                }).unwrap();
              }

            } else if buf.len() >= 3 && buf[2] == b'7' {
              history.push(LoProtocol::GameCleanupPrompt);

              writer_tx.send(InternalMsg::WriteCmd{cmd: b"done".to_vec()}).unwrap();
              writer_tx.send(InternalMsg::WriteCmd{cmd: b"quit".to_vec()}).unwrap();

            } else {
              history.push(LoProtocol::UnknownPrompt);
            }
            prev_idx = Some(idx);
          } else if buf.len() >= 2 && buf[0] == b'9' && buf[1] == b' ' {
            history.push(LoProtocol::GenericInfo{line: buf});
          } else if buf.len() >= 3 && buf[0] == b'1' && buf[1] == b'5' && buf[2] == b' ' {
            history.push(LoProtocol::MoveInfo{line: buf});
          } else if buf.len() >= 3 && buf[0] == b'2' && buf[1] == b'0' && buf[2] == b' ' {
          } else if buf.len() >= 3 && buf[0] == b'2' && buf[1] == b'1' && buf[2] == b' ' {
          } else if buf.len() >= 3 && buf[0] == b'2' && buf[1] == b'2' && buf[2] == b' ' {
          } else {
            history.push(LoProtocol::Line{line: buf});
          }
          idx += 1;
          sleep_ms(200);
        }
      })
    };

    let writer_thr = {
      let server_cfg = server_cfg.clone();
      let match_cfg = match_cfg.clone();
      //let barrier = barrier.clone();
      let writer_rx = writer_rx;
      let mut writer = stream;
      spawn(move || {
        // Enter login details.
        submit(&mut writer, &server_cfg.login);
        if let Some(ref password) = server_cfg.password {
          submit(&mut writer, password);
        }

        // Set modes.
        submit(&mut writer, b"set client TRUE");
        submit(&mut writer, b"set verbose FALSE");

        // Wait for internal commands.
        loop {
          match writer_rx.recv() {
            Ok(InternalMsg::Quit) => {
              break;
            }
            Ok(InternalMsg::WriteCmd{ref cmd}) => {
              submit(&mut writer, cmd);
            }
            Err(e) => {
              println!("WARNING: writer failed to recv internal msg: {:?}", e);
            }
          }
        }

        //barrier.wait();
      })
    };

    NngsOneShotClient{
      server_cfg:   server_cfg,
      match_cfg:    match_cfg,
      barrier:  barrier,
      agent:    agent_thr,
      bridge:   bridge_thr,
      reader:   reader_thr,
      writer:   writer_thr,
      //_history: Vec::with_capacity(500),
      _marker:  PhantomData,
    }
  }

  pub fn run_loop(&mut self) {
    loop {
      sleep_ms(200);
    }

    /*let server_cfg = self.server_cfg.clone();
    let match_cfg = self.match_cfg.clone();
    self.submit(&server_cfg.login);
    if let Some(ref password) = server_cfg.password {
      self.submit(password);
    }
    self.submit(b"set client TRUE");
    self.submit(b"set verbose FALSE");
    self.read_to_first_prompt();*/
  }

  /*fn submit(&mut self, msg: &[u8]) {
    self.writer.write_all(msg).unwrap();
    self.writer.write_u8(b'\n').unwrap();
  }

  fn read_to_first_prompt(&mut self) {
    loop {
      let mut buf = Vec::with_capacity(160);

      self.reader.read_until(b'\n', &mut buf).unwrap();
      //self._history.push(buf.clone());
      /*if let Ok(s) = from_utf8(&buf) {
        print!("OUTPUT: {}", s);
      } else {
        println!("OUTPUT: (unprintable)");
      }*/
      if buf.len() >= 2 && buf[0] == b'1' && buf[1] == b' ' {
        break;
      }
      buf.clear();
    }
  }*/
}

pub struct NngsConfig {
  pub host:     String,
  pub port:     u16,
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
