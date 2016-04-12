use board::{Stone, Action, Point};
use client::agent::{AsyncAgent, AgentMsg};
use gtp_board::{Coord};

use byteorder::{ReadBytesExt, WriteBytesExt};
use toml;

use std::fs::{File};
use std::io::{Read, BufRead, Write, BufReader};
use std::marker::{PhantomData};
use std::path::{PathBuf};
use std::net::{TcpStream};
use std::str::{from_utf8};
use std::sync::{Arc, Barrier};
use std::sync::mpsc::{Sender, Receiver, channel};
use std::thread::{JoinHandle, sleep_ms, spawn};

#[derive(Clone, RustcDecodable, Debug)]
pub struct NngsServerConfig {
  pub host:     String,
  pub port:     u16,
  pub login:    String,
  pub password: Option<String>,
}

impl NngsServerConfig {
  pub fn open() -> Result<NngsServerConfig, ()> {
    let path = PathBuf::from("nngs_server.config");
    let mut file = match File::open(&path) {
      Ok(file) => file,
      Err(_) => return Err(()),
    };
    let mut buf = String::new();
    file.read_to_string(&mut buf).unwrap();
    let cfg: NngsServerConfig = match toml::decode_str(&buf) {
      Some(x) => x,
      None => return Err(()),
    };
    Ok(cfg)
  }
}

#[derive(Clone, RustcDecodable, Debug)]
pub struct NngsMatchConfig {
  pub load_save_path:   Option<String>,
  pub skip_as_black:    Option<bool>,
  pub automatch:    bool,
  pub our_stone:    Stone,
  pub opponent:     String,
  pub board_size:   i32,
  pub main_time:    i32,
  pub byoyomi_time: i32,
}

impl NngsMatchConfig {
  pub fn open() -> Result<NngsMatchConfig, ()> {
    let path = PathBuf::from("nngs_match.config");
    let mut file = match File::open(&path) {
      Ok(file) => file,
      Err(_) => return Err(()),
    };
    let mut buf = String::new();
    file.read_to_string(&mut buf).unwrap();
    let cfg: NngsMatchConfig = match toml::decode_str(&buf) {
      Some(x) => x,
      None => return Err(()),
    };
    Ok(cfg)
  }
}

pub struct NngsOneShotClient<A> where A: AsyncAgent {
  server_cfg:   NngsServerConfig,
  match_cfg:    Option<NngsMatchConfig>,
  barrier:      Arc<Barrier>,
  agent:        JoinHandle<()>,
  bridge:       JoinHandle<()>,
  reader:       JoinHandle<()>,
  writer:       JoinHandle<()>,
  _marker:      PhantomData<A>,
}

fn submit(writer: &mut Write, msg: &[u8]) {
  writer.write_all(msg).unwrap();
  writer.write_u8(b'\n').unwrap();
}

enum InternalMsg {
  Quit,
  WriteCmd{cmd: Vec<u8>},
  SetDeadStones{game_outcome: GameOutcome},
  CleanupDeadStones,
}

enum LoProtocol {
  Line{line: Vec<u8>},
  UnknownPrompt,        // 1
  ServerPrompt,         // 1 5
  GamePrompt,           // 1 6
  GameCleanupPrompt,    // 1 7
  GenericInfo{line: Vec<u8>},   // 9
  MoveInfo{line: Vec<u8>},      // 15
  ObserveInfo{line: Vec<u8>},   // 16
  ScoreInfo{line: Vec<u8>},     // 20
  ShoutInfo{line: Vec<u8>},     // 21
  GameInfo{line: Vec<u8>},      // 22
}

struct GameOutcome {
  dead_stones:  Vec<Vec<Point>>,
  live_stones:  Vec<Vec<Point>>,
  territory:    Vec<Vec<Point>>,
  outcome:      Option<Stone>,
}

impl<A> NngsOneShotClient<A> where A: AsyncAgent {
  pub fn new(server_cfg: NngsServerConfig, match_cfg: Option<NngsMatchConfig>) -> NngsOneShotClient<A> {
    let stream = match TcpStream::connect((&server_cfg.host as &str, server_cfg.port)) {
      Ok(stream) => stream,
      Err(_) => panic!("failed to connect to nngs server"),
    };
    let reader = BufReader::new(match stream.try_clone() {
      Ok(stream) => stream,
      Err(_) => panic!("failed to clone stream"),
    });
    let barrier = Arc::new(Barrier::new(4));
    let (agent_in_tx, agent_in_rx) = channel();
    let (agent_out_tx, agent_out_rx) = channel();
    let (writer_tx, writer_rx) = channel();

    let agent_thr = A::spawn_runloop(
        barrier.clone(),
        agent_in_rx,
        agent_out_tx,
        //match_cfg.map(|cfg| cfg.load_save_path.map(|p| PathBuf::from(p))),
        match match_cfg {
          None => None,
          Some(ref cfg) => cfg.load_save_path.as_ref().map(|p| PathBuf::from(p)),
        },
    );

    let bridge_thr = {
      let match_cfg = match_cfg.clone();
      let barrier = barrier.clone();
      let agent_in_tx = agent_in_tx.clone();
      let agent_out_rx = agent_out_rx;
      let writer_tx = writer_tx.clone();
      spawn(move || {
        loop {
          match agent_out_rx.recv() {
            Ok(AgentMsg::Ready) => {
              // Now that the agent is ready, if a match is specified, then
              // initiate it.
              if let Some(ref match_cfg) = match_cfg {
                let color_code = match match_cfg.our_stone {
                  Stone::Black => "B",
                  Stone::White => "W",
                  _ => unreachable!(),
                };
                let match_cmd_str = format!("match {} {} {} {} {}",
                    match_cfg.opponent,
                    color_code,
                    match_cfg.board_size,
                    match_cfg.main_time,
                    match_cfg.byoyomi_time,
                );
                let match_cmd = match_cmd_str.into_bytes();
                writer_tx.send(InternalMsg::WriteCmd{cmd: match_cmd}).unwrap();
              }
            }
            Ok(AgentMsg::AcceptMatch{passive, opponent, our_stone, board_size, main_time_secs, byoyomi_time_secs}) => {
              if passive {
                let color_code = match our_stone {
                  Stone::Black => "B",
                  Stone::White => "W",
                  _ => unreachable!(),
                };
                let match_cmd_str = format!("match {} {} {} {} {}",
                    opponent,
                    color_code,
                    board_size,
                    main_time_secs / 60,
                    byoyomi_time_secs / 60,
                );
                let match_cmd = match_cmd_str.into_bytes();
                writer_tx.send(InternalMsg::WriteCmd{cmd: match_cmd}).unwrap();
              }
              agent_in_tx.send(AgentMsg::StartMatch{
                skip_as_black:    match_cfg.as_ref().map_or(false, |cfg| cfg.skip_as_black.unwrap_or(false)),
                opponent:     opponent.clone(),
                our_stone:    our_stone,
                board_size:   board_size,
                main_time_secs:       main_time_secs,
                byoyomi_time_secs:    byoyomi_time_secs,
              }).unwrap();
            }
            Ok(AgentMsg::CheckTime) => {
              // XXX(20160308): ignore time command.
              //writer_tx.send(InternalMsg::WriteCmd{cmd: b"time".to_vec()}).unwrap();
            }
            Ok(AgentMsg::SubmitAction{
              turn, action,
              set_dead_stones,
              dead_stones, live_stones, territory, outcome,
            }) => {
              println!("DEBUG: client bridge: received agent action: {:?} {:?}", turn, action);
              if set_dead_stones {
                writer_tx.send(InternalMsg::SetDeadStones{game_outcome: GameOutcome{
                  dead_stones:    dead_stones,
                  live_stones:    live_stones,
                  territory:      territory,
                  outcome:        outcome,
                }});
              }
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
            Ok(_) => {}
            Err(e) => {
              //println!("WARNING: writer failed to recv internal msg: {:?}", e);
              break;
            }
          }
        }
        barrier.wait();
      })
    };

    let reader_thr = {
      let server_cfg = server_cfg.clone();
      let match_cfg = match_cfg.clone();
      let barrier = barrier.clone();
      let agent_in_tx = agent_in_tx;
      let writer_tx = writer_tx;
      let mut reader = reader;
      spawn(move || {
        let mut history: Vec<LoProtocol> = Vec::with_capacity(1024);
        let mut prev_idx: Option<usize> = None;
        let mut recently_finished_match = false;
        let mut idx: usize = 0;
        loop {
          let mut buf = Vec::with_capacity(160);
          match reader.read_until(b'\n', &mut buf) {
            Ok(_) => {}
            Err(e) => {
              // XXX(20160307): gracefully shutdown.
              println!("DEBUG: client: shutting down");
              break;
            }
          }
          if buf.is_empty() {
            break;
          }
          print!("DEBUG: client telnet: {}", String::from_utf8_lossy(&buf));
          /*{
            let s = String::from_utf8_lossy(&buf);
            if !s.is_empty() {
              print!("DEBUG: server: {}", s);
            }
          }*/

          if buf.len() >= 2 && buf[0] == b'1' && buf[1] == b' ' {
            if buf.len() >= 3 && buf[2] == b'5' {
              history.push(LoProtocol::ServerPrompt);
            } else if buf.len() >= 3 && buf[2] == b'6' {
              println!("DEBUG: client reader: got game prompt");
              history.push(LoProtocol::GamePrompt);
            } else if buf.len() >= 3 && buf[2] == b'7' {
              history.push(LoProtocol::GameCleanupPrompt);
            } else {
              history.push(LoProtocol::UnknownPrompt);
            }

            if buf.len() >= 3 && (buf[2] == b'5' || buf[2] == b'6') {
              let mut is_resigned = false;
              //let mut who_resigned = None;

              for i in prev_idx.unwrap_or(0) .. idx {
                match &history[i] {
                  &LoProtocol::GenericInfo{ref line} => {
                    let line_str = String::from_utf8_lossy(line);
                    if line_str.contains("has resigned the game") {
                      is_resigned = true;
                    }
                  }
                  _ => {}
                }
              }

              if is_resigned || recently_finished_match {
                agent_in_tx.send(AgentMsg::FinishMatch).unwrap();
                writer_tx.send(InternalMsg::Quit).unwrap();
              }
            }

            if buf.len() >= 3 && buf[2] == b'5' {
              let mut is_match_request = false;
              let mut is_match_accept = false;
              let mut match_cmd = None;
              let mut match_opponent = None;
              let mut match_our_stone = None;
              let mut match_board_size = None;
              let mut match_main_time_mins = None;
              let mut match_byoyomi_time_mins = None;

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
                      let opponent = match_toks[1].to_string();
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
                      match_opponent = Some(opponent);
                      match_our_stone = Some(our_stone);
                      match_board_size = Some(board_size);
                      match_main_time_mins = Some(main_time);
                      match_byoyomi_time_mins = Some(byoyomi_time);
                    } else if line_str.contains("Match") && line_str.contains("accepted") {
                      is_match_accept = true;
                    }
                  }
                  _ => {}
                }
              }

              if is_match_request {
                // XXX(20160308): Don't accept match request until the agent is ready.
                // Instead send match details to agent channel.
                agent_in_tx.send(AgentMsg::RequestMatch{
                  passive:      true,
                  opponent:     match_opponent.unwrap(),
                  our_stone:    match_our_stone.unwrap(),
                  board_size:   match_board_size.unwrap(),
                  main_time_secs:       60 * match_main_time_mins.unwrap(),
                  byoyomi_time_secs:    60 * match_byoyomi_time_mins.unwrap(),
                }).unwrap();
              } else if is_match_accept {
                if let Some(ref match_cfg) = match_cfg {
                  agent_in_tx.send(AgentMsg::RequestMatch{
                    passive:      false,
                    opponent:     match_cfg.opponent.clone(),
                    our_stone:    match_cfg.our_stone,
                    board_size:   match_cfg.board_size,
                    main_time_secs:       60 * match_cfg.main_time,
                    byoyomi_time_secs:    60 * match_cfg.byoyomi_time,
                  }).unwrap();
                }
              }
            }

            if buf.len() >= 3 && buf[2] == b'6' {
              let mut is_move = false;
              let mut move_main_time_left_s = None;
              //let mut move_byoyomi_time_left_s = None;
              let mut move_number = None;
              let mut move_turn = None;
              let mut move_action = None;

              for i in prev_idx.unwrap_or(0) .. idx {
                match &history[i] {
                  &LoProtocol::MoveInfo{ref line} => {
                    let line_str = String::from_utf8_lossy(line);
                    if line_str.contains("Game") {
                      println!("DEBUG: client reader: found game info");
                      let pre_toks: Vec<_> = line_str.splitn(2, ":").collect();
                      let pair_toks: Vec<_> = pre_toks[1].splitn(2, "vs").collect();
                      for tok in pair_toks.iter() {
                        let toks: Vec<_> = tok.trim().split_whitespace().collect();
                        let name = toks[0];
                        if name != &server_cfg.login {
                          continue;
                        }
                        let main_time_left_s: i32 = toks[2].parse().unwrap();
                        move_main_time_left_s = Some(main_time_left_s);
                      }
                    } else {
                      println!("DEBUG: client reader: found move info");
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
                // FIXME(20160308): byoyomi time left?
                println!("DEBUG: client reader: send move to agent: {:?} {:?}", move_turn, move_action);
                agent_in_tx.send(AgentMsg::RecvAction{
                  turn:         move_turn.unwrap(),
                  action:       move_action.unwrap(),
                  move_number:  move_number,
                  time_left_s:  move_main_time_left_s,
                }).unwrap();
              }
            }

            if buf.len() >= 3 && buf[2] == b'7' {
              if !recently_finished_match {
                recently_finished_match = true;
                // XXX(20160316): Cleanup dead stones before signaling we are done..
                writer_tx.send(InternalMsg::CleanupDeadStones).unwrap();
              }
            }

            prev_idx = Some(idx);
          } else if buf.len() >= 2 && buf[0] == b'9' && buf[1] == b' ' {
            history.push(LoProtocol::GenericInfo{line: buf});
          } else if buf.len() >= 3 && buf[0] == b'1' && buf[1] == b'5' && buf[2] == b' ' {
            history.push(LoProtocol::MoveInfo{line: buf});
          } else if buf.len() >= 3 && buf[0] == b'1' && buf[1] == b'6' && buf[2] == b' ' {
            history.push(LoProtocol::ObserveInfo{line: buf});
          } else if buf.len() >= 3 && buf[0] == b'2' && buf[1] == b'0' && buf[2] == b' ' {
            history.push(LoProtocol::ScoreInfo{line: buf});
          } else if buf.len() >= 3 && buf[0] == b'2' && buf[1] == b'1' && buf[2] == b' ' {
            history.push(LoProtocol::ShoutInfo{line: buf});
          } else if buf.len() >= 3 && buf[0] == b'2' && buf[1] == b'2' && buf[2] == b' ' {
            history.push(LoProtocol::GameInfo{line: buf});
          } else {
            history.push(LoProtocol::Line{line: buf});
          }
          idx += 1;
        }
        barrier.wait();
      })
    };

    let writer_thr = {
      let server_cfg = server_cfg.clone();
      let barrier = barrier.clone();
      let writer_rx = writer_rx;
      let mut writer = stream;
      spawn(move || {
        let mut saved_game_outcome = None;

        // Enter login details.
        submit(&mut writer, server_cfg.login.as_bytes());
        if let Some(ref password) = server_cfg.password {
          submit(&mut writer, password.as_bytes());
        }

        // Set modes.
        submit(&mut writer, b"set client TRUE");
        submit(&mut writer, b"set verbose FALSE");

        // Wait for internal commands.
        loop {
          match writer_rx.recv() {
            Ok(InternalMsg::Quit) => {
              submit(&mut writer, b"quit");
              break;
            }
            Ok(InternalMsg::WriteCmd{ref cmd}) => {
              submit(&mut writer, cmd);
            }
            Ok(InternalMsg::SetDeadStones{game_outcome}) => {
              saved_game_outcome = Some(game_outcome);
            }
            Ok(InternalMsg::CleanupDeadStones) => {
              if let Some(ref game_outcome) = saved_game_outcome {
                // FIXME(20160316): enter both players dead stones or just our own?
                /*for &point in game_outcome.dead_stones[0].iter().chain(game_outcome.dead_stones[1].iter()) {
                  let coord = point.to_coord();
                  submit(&mut writer, &coord.to_bytestring());
                }*/
                println!("DEBUG: client writer: CLEANUP PHASE:");
                println!("DEBUG: client writer: final outcome:  {:?}", game_outcome.outcome);
                println!("DEBUG: client writer: dead stones[B]: {:?}", game_outcome.dead_stones[0]);
                println!("DEBUG: client writer: dead stones[W]: {:?}", game_outcome.dead_stones[1]);
                println!("DEBUG: client writer: live stones[B]: {:?}", game_outcome.live_stones[0]);
                println!("DEBUG: client writer: live stones[W]: {:?}", game_outcome.live_stones[1]);
                println!("DEBUG: client writer: territory[B]:   {:?}", game_outcome.territory[0]);
                println!("DEBUG: client writer: territory[W]:   {:?}", game_outcome.territory[1]);
              }
              submit(&mut writer, b"done");
            }
            Err(e) => {
              //println!("WARNING: writer failed to recv internal msg: {:?}", e);
              break;
            }
          }
        }

        barrier.wait();
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
      _marker:  PhantomData,
    }
  }

  pub fn run_loop(&mut self) {
    self.barrier.wait();
  }

  /*pub fn run_loop(&mut self) {
    let server_cfg = self.server_cfg.clone();
    let match_cfg = self.match_cfg.clone();
    self.submit(&server_cfg.login);
    if let Some(ref password) = server_cfg.password {
      self.submit(password);
    }
    self.submit(b"set client TRUE");
    self.submit(b"set verbose FALSE");
    self.read_to_first_prompt();
  }

  fn submit(&mut self, msg: &[u8]) {
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

/*pub struct NngsConfig {
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
}*/
