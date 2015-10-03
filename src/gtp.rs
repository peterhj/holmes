use self::Entity::{
  StringEntity, IntEntity, FloatEntity, BooleanEntity,
  VertexEntity, ColorEntity, MoveEntity,
  MultilineListEntity,
  ErrorEntity,
};

use board::{Player, Coord, Action, Move};

use bufstream::{BufStream};
use byteorder::{ReadBytesExt};

use std::ascii::{AsciiExt};
use std::io::{BufRead, Write, BufReader};
use std::net::{TcpStream, TcpListener};
use std::str::{from_utf8};
use std::sync::mpsc::{Sender, Receiver, channel};
use std::thread::{spawn, sleep_ms};
//use time::{Duration};

// GTP spec:
// <http://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html>
// as well as KGS extensions:
// <http://www.weddslist.com/kgs/how/kgsGtp.html>.

// Character set:
// 1. Control characters are values 0-31 and 127. The following have special
//    meaning:
//    HT = decimal 9
//    LF = decimal 10
//    CR = decimal 13
// 2. Whitespace characters:
//    SPACE = decimal 32
//    HT = decimal 9
// 3. Newline convention is to use LF. CR must be discarded from input.

// Syntax:
// 1. Commands are the following:
//    id command_name [arguments]\n
// 2. Successful responses are the following:
//    =id result\n\n
// 3. Error messages are the following:
//    ?id error_message\n\n

// Preprocessing (note that 1-4 are needed for cmd strings, whereas only
// 1 and 3 are needed for responses/errors):
// 1. Remove all occurences of CR and other control characters, except for
//    HT and LF.
// 2. For each line with a hash sign '#', remove all text following and
//    including this character.
// 3. Convert all occurences of HT to SPACE.
// 4. Discard any empty or whitespace-only lines (i.e., ignore them).

const HT: u8 = 9;
const LF: u8 = 10;
//const CR: u8 = 13; // XXX: unused.
const SPACE: u8 = 32;

fn preproc_stage1(line: &[u8]) -> Vec<u8> {
  let mut preproc = Vec::new();
  for &byte in line.iter() {
    if byte > 31 || byte != 127 || byte == HT || byte == LF {
      preproc.push(byte);
    }
  }
  preproc
}

fn preproc_stage2(line: &[u8]) -> Vec<u8> {
  let mut preproc = Vec::new();
  let mut found_comment = false;
  for &byte in line.iter() {
    if byte == '#' as u8 {
      found_comment = true;
    }
    if !found_comment || byte == LF {
      preproc.push(byte);
    }
  }
  preproc
}

fn preproc_stage3(line: &[u8]) -> Vec<u8> {
  let mut preproc = Vec::new();
  for &byte in line.iter() {
    match byte {
      HT => preproc.push(SPACE),
      _ => preproc.push(byte),
    }
  }
  preproc
}

fn preproc_stage4(line: &[u8]) -> Result<(), ()> {
  for &byte in line.iter() {
    if (byte != HT || byte != SPACE) && byte != LF {
      return Ok(());
    }
  }
  return Err(());
}

fn preproc_command_string(command_str: &[u8]) -> Option<Vec<u8>> {
  let stage1 = preproc_stage1(command_str);
  let stage2 = preproc_stage2(&stage1);
  let stage3 = preproc_stage3(&stage2);
  match preproc_stage4(&stage3) {
    Ok(_) => Some(stage3),
    Err(_) => None,
  }
}

fn preproc_response_string(response_str: &[u8]) -> Vec<u8> {
  let stage1 = preproc_stage1(response_str);
  let stage3 = preproc_stage3(&stage1);
  stage3
}

#[derive(Clone)]
pub enum Entity {
  StringEntity(Vec<u8>),
  IntEntity(u32),
  FloatEntity(f32),
  BooleanEntity(bool),
  VertexEntity(Action),
  ColorEntity(Player),
  MoveEntity(Player, Action),
  MultilineListEntity(Vec<Entity>),
  ErrorEntity(Vec<u8>),
}

impl Entity {
  pub fn wrap_string(x: &[u8]) -> Entity {
    StringEntity(x.to_vec())
  }

  pub fn parse_int(x: &[u8]) -> Entity {
    let v: u32 = from_utf8(x)
      .ok().expect("FATAL: engine: expected int token")
      .parse()
      .ok().expect("FATAL: engine: expected u32");
    IntEntity(v)
  }

  pub fn parse_float(x: &[u8]) -> Entity {
    let v: f32 = from_utf8(x)
      .ok().expect("FATAL: engine: expected float token")
      .parse()
      .ok().expect("FATAL: engine: expected f32");
    FloatEntity(v)
  }

  pub fn parse_vertex(arg: &[u8]) -> Entity {
    let arg: Vec<u8> = arg.to_ascii_lowercase();
    //println!("DEBUG: gtp: parse vertex? {}", arg.to_string());
    if &arg == b"pass" {
      VertexEntity(Action::Pass)
    } else {
      let letter = arg[0];
      let valid_letter =
        (letter >= 'a' as u8 && letter < 'i' as u8) || 
        (letter > 'i' as u8 && letter <= 't' as u8);
      if !valid_letter {
        panic!("FATAL: engine: expected valid letter in vertex token");
      }
      let number = &arg[1 ..];
      let valid_number: u8 = from_utf8(number)
        .ok().expect("FATAL: engine: expected number in vertex token")
        .parse()
        .ok().expect("FATAL: engine: expected u8");
      let x: u8 = {
        if letter >= 'a' as u8 && letter < 'i' as u8 {
          letter - 'a' as u8
        } else if letter > 'i' as u8 && letter <= 't' as u8 {
          letter - 1 - 'a' as u8
        } else {
          unreachable!();
        }
      };
      let y = valid_number - 1;
      VertexEntity(Action::Play(Coord::new(x, y)))
    }
  }

  pub fn parse_vertex_or_resign(arg: &[u8]) -> Entity {
    let arg: Vec<u8> = arg.to_ascii_lowercase();
    //println!("DEBUG: gtp: parse vertex or resign? {}", arg.to_string());
    if &arg == b"resign" {
      StringEntity(b"resign".to_vec())
    } else {
      Entity::parse_vertex(&arg)
    }
  }

  pub fn parse_color(x: &[u8]) -> Entity {
    let x: Vec<u8> = x.to_ascii_lowercase();
    match &x as &[u8] {
      b"b" | b"black" => ColorEntity(Player::Black),
      b"w" | b"white" => ColorEntity(Player::White),
      _ => panic!("FATAL: engine: expected color token"),
    }
  }

  pub fn parse_move(tok1: &[u8], tok2: &[u8]) -> Entity {
    let color = match Entity::parse_color(tok1) {
      ColorEntity(color) => color,
      _ => unreachable!(),
    };
    let action = match Entity::parse_vertex(tok2) {
      VertexEntity(action) => action,
      _ => unreachable!(),
    };
    MoveEntity(color, action)
  }

  pub fn to_bytestring(&self) -> Vec<u8> {
    match self {
      &StringEntity(ref s) => s.clone(),
      &IntEntity(ref x) => x.to_string().as_bytes().to_vec(),
      &FloatEntity(ref x) => x.to_string().as_bytes().to_vec(),
      &BooleanEntity(ref x) => x.to_string().as_bytes().to_vec(),
      &VertexEntity(ref action) => action.to_bytestring(),
      &ColorEntity(ref player) => player.to_bytestring(),
      &MoveEntity(ref player, ref action) => {
        let mut s = Vec::new();
        s.extend(&player.to_bytestring());
        s.push(SPACE);
        s.extend(&action.to_bytestring());
        s
      },
      &MultilineListEntity(ref ents) => {
        //let ents = &ents;
        if ents.len() == 0 {
          [].to_vec()
        } else {
          let ent_str: Vec<u8> = ents[1 ..].iter()
            .map(|e| e.to_bytestring())
            .fold(ents[0].to_bytestring(), |mut s, s1| {
              {
                let mut_s = &mut s;
                mut_s.push(LF);
                mut_s.extend(&s1);
              }
              s
            });
          ent_str
        }
      },
      &ErrorEntity(ref s) => s.clone(),
    }
  }
}

struct Command {
  id: Option<u32>,
  name: Vec<u8>,
  args: Vec<Vec<u8>>,
}

fn parse_command_string(cmd_str: &[u8]) -> Command {
  //println!("DEBUG: engine: received command:");
  //print!("{}", String::from_utf8_lossy(cmd_str));
  let mut cmd_buf = BufReader::new(cmd_str);
  let mut id: Option<u32> = None;
  let name = {
    let mut tok0 = Vec::new();
    cmd_buf.read_until(SPACE, &mut tok0).ok().unwrap();
    if tok0[tok0.len()-1] == SPACE || tok0[tok0.len()-1] == LF {
      tok0.pop();
    }
    // FIXME(20141231): hacky way to test if the token is an integer.
    if tok0.len() > 0 && tok0[0] >= '0' as u8 && tok0[0] <= '9' as u8 { 
      id = from_utf8(&tok0)
        .ok().expect("FATAL: engine: id token should be an integer!")
        .parse()
        .ok();
      let mut tok1 = Vec::new();
      cmd_buf.read_until(SPACE, &mut tok1).ok().unwrap();
      if tok1[tok1.len()-1] == SPACE || tok1[tok1.len()-1] == LF {
        tok1.pop();
      }
      tok1
    } else {
      tok0
    }
  };
  let mut args = Vec::new();
  //while !cmd_buf.eof() {
  loop {
    let mut tok = Vec::new();
    match cmd_buf.read_until(SPACE, &mut tok) {
      Ok(n) => {
        if n == 0 {
          break;
        }
      }
      Err(_) => break,
    }
    if tok[tok.len()-1] == SPACE || tok[tok.len()-1] == LF {
      tok.pop();
    }
    args.push(tok);
  }
  let cmd = Command{id: id, name: name, args: args};
  cmd
}

fn create_response_string(id: Option<u32>, response: &[Entity]) -> Vec<u8> {
  let mut prefix = if response.len() >= 1 {
    match response[0] {
      ErrorEntity(_) => b"?",
      _ => b"=",
    }
  } else {
    b"="
  }.to_vec();
  match id {
    Some(id) => prefix.extend(format!("{}", id).as_bytes()),
    _ => (),
  }
  let mut res_str: Vec<u8> = response.iter()
    .map(|e| e.to_bytestring())
    .fold(prefix, |mut s, s1| {
      {
        let mut_s = &mut s;
        mut_s.push(SPACE);
        mut_s.extend(&s1);
      }
      s
    });
  res_str.extend(&[LF, LF]);
  //println!("DEBUG: engine: response str '{}'", String::from_utf8_lossy(&res_str));
  res_str
}

pub struct GtpEngine<C> {
  should_shutdown: bool,
  buffered_stream: BufStream<TcpStream>,
  client: C,
}

impl<C: GtpClient> GtpEngine<C> {
  pub fn new(client: C) -> GtpEngine<C> {
    let (host, port) = client.get_address();
    let stream = (move || {
      let mut try_count: usize = 0;
      loop {
        match TcpStream::connect((&host as &str, port)).ok() {
          Some(stream) => return stream,
          None => {
            try_count += 1;
            if try_count >= 5 {
              panic!("FATAL: engine: failed to wrap TcpStream in a BufferStream!");
            }
            sleep_ms(30);
            continue;
          },
        }
      }
    })();
    let buffered_stream = BufStream::new(stream);
    GtpEngine{
      should_shutdown: false,
      buffered_stream: buffered_stream,
      client: client,
    }
  }

  fn reply(&mut self, cmd: &Command) -> Vec<Entity> {
    let args = &cmd.args;
    match &cmd.name as &[u8] {
      b"protocol_version"       => self.client.reply_protocol_version(),
      b"name"                   => self.client.reply_name(),
      b"version"                => self.client.reply_version(),
      b"known_command"          => self.client.reply_known_command(args),
      b"list_commands"          => self.client.reply_list_commands(),
      b"quit"                   => {
        self.should_shutdown = true;
        self.client.reply_quit()
      },
      b"boardsize"              => self.client.reply_boardsize(args),
      b"clear_board"            => self.client.reply_clear_board(),
      b"komi"                   => self.client.reply_komi(args),
      b"play"                   => self.client.reply_play(args),
      b"genmove"                => self.client.reply_genmove(args),
      b"undo"                   => self.client.reply_undo(),
      b"time_settings"          => self.client.reply_time_settings(args),
      b"time_left"              => self.client.reply_time_left(args),
      b"final_score"            => self.client.reply_final_score(),
      b"final_status_list"      => self.client.reply_final_status_list(args),
      b"kgs-time_settings"      => self.client.reply_kgs_time_settings(args),
      b"kgs-genmove_cleanup"    => self.client.reply_kgs_genmove_cleanup(),
      b"loadsgf"                => self.client.reply_loadsgf(),
      b"reg_genmove"            => self.client.reply_reg_genmove(),
      b"showboard"              => self.client.reply_showboard(),
      _ => [].to_vec(), // FIXME
    }
  }

  pub fn runloop(&mut self) {
    while !self.should_shutdown {
      let mut line = Vec::new();
      match self.buffered_stream.read_until(LF, &mut line) {
        Ok(_) => {}
        Err(e) => { print!("FATAL: engine: io error: {}", e); return; }
      };
      let line = match preproc_command_string(&line) {
        Some(line) => line,
        None => continue,
      };
      let cmd = parse_command_string(&line);
      let response = self.reply(&cmd);
      let res_str = create_response_string(cmd.id, &response);
      let res_str = preproc_response_string(&res_str);
      self.buffered_stream.write(&res_str).ok();
      self.buffered_stream.flush().ok();
    }
  }
}

pub trait GtpClient {
  fn get_address(&self) -> (String, u16);
  fn get_extensions(&self);

  // Administrative commands.
  fn reply_protocol_version(&mut self) -> Vec<Entity>;
  fn reply_name(&mut self) -> Vec<Entity>;
  fn reply_version(&mut self) -> Vec<Entity>;
  fn reply_known_command(&mut self, args: &[Vec<u8>]) -> Vec<Entity>;
  fn reply_list_commands(&mut self) -> Vec<Entity>;
  fn reply_quit(&mut self) -> Vec<Entity>;

  // Setup commands.
  fn reply_boardsize(&mut self, args: &[Vec<u8>]) -> Vec<Entity>;
  fn reply_clear_board(&mut self) -> Vec<Entity>;
  fn reply_komi(&mut self, args: &[Vec<u8>]) -> Vec<Entity>;

  // Tournament setup commands.
  fn reply_fixed_handicap(&mut self) -> Vec<Entity>;
  fn reply_place_free_handicap(&mut self) -> Vec<Entity>;
  fn reply_set_free_handicap(&mut self) -> Vec<Entity>;

  // Core play commands.
  fn reply_play(&mut self, args: &[Vec<u8>]) -> Vec<Entity>;
  fn reply_genmove(&mut self, args: &[Vec<u8>]) -> Vec<Entity>;
  fn reply_undo(&mut self) -> Vec<Entity>;

  // Tournament commands.
  fn reply_time_settings(&mut self, args: &[Vec<u8>]) -> Vec<Entity>;
  fn reply_time_left(&mut self, args: &[Vec<u8>]) -> Vec<Entity>;
  fn reply_final_score(&mut self) -> Vec<Entity>;
  fn reply_final_status_list(&mut self, args: &[Vec<u8>]) -> Vec<Entity>;

  // KGS extensions.
  fn reply_kgs_time_settings(&mut self, args: &[Vec<u8>]) -> Vec<Entity>;
  fn reply_kgs_genmove_cleanup(&mut self) -> Vec<Entity>;

  // Regression commands.
  fn reply_loadsgf(&mut self) -> Vec<Entity>;
  fn reply_reg_genmove(&mut self) -> Vec<Entity>;

  // Debug commands.
  fn reply_showboard(&mut self) -> Vec<Entity>;
}

// TODO: Below is a simple GTP controller server for connecting two bots running
// on the same host.

fn create_command_string(id: u32, command: &[Entity]) -> Vec<u8> {
  //let mut prefix = Vec::new();
  let prefix = id.to_string().as_bytes().to_vec();
  let mut cmd_str: Vec<u8> = command.iter()
    .map(|e| e.to_bytestring())
    .fold(prefix, |mut s, s1| {
      {
        let mut_s = &mut s;
        mut_s.push(SPACE);
        mut_s.extend(&s1);
      }
      s
    });
  cmd_str.push(LF);
  cmd_str
}

fn write_client_command(id: u32, cmd: &[Entity], buffered_stream: &mut BufStream<TcpStream>) {
  let cmd_str = create_command_string(id, cmd);
  write_client_command_string(&cmd_str, buffered_stream);
}

fn write_client_command_string(cmd_str: &[u8], buffered_stream: &mut BufStream<TcpStream>) {
  let _ = buffered_stream.write(cmd_str);
  let _ = buffered_stream.flush();
}

fn read_client_response_string(buffered_stream: &mut BufStream<TcpStream>) -> Vec<u8> {
  let mut res_str = Vec::new();
  buffered_stream.read_until(LF, &mut res_str)
    .ok().expect("FATAL: client: failed to read stream");
  loop {
    let mut res_chunk = Vec::new();
    buffered_stream.read_until(LF, &mut res_chunk)
      .ok().expect("FATAL: client: failed to read stream");
    res_str.extend(&res_chunk);
    if res_chunk.len() == 1 && res_chunk[0] == LF {
      break;
    }
  }
  res_str
}

struct Response {
  has_error: bool,
  id: Option<u32>,
  lines: Vec<Vec<Vec<u8>>>,
}

fn parse_response_string(res_str: &[u8]) -> Response {
  //println!("DEBUG: server: parsing response:");
  //print!("{}", String::from_utf8_lossy(res_str));
  let mut res_buf = BufReader::new(res_str);
  let prefix = res_buf.read_u8().ok().unwrap();
  let has_error = match prefix as char {
    '=' => false,
    '?' => true,
    x => panic!("FATAL: server: invalid engine response prefix: '{}'", x),
  };
  let mut id: Option<u32> = None;
  {
    let mut tok0 = Vec::new();
    res_buf.read_until(SPACE, &mut tok0).ok().unwrap();
    if tok0[tok0.len()-1] == SPACE || tok0[tok0.len()-1] == LF {
      tok0.pop();
    }
    // FIXME(20141231): hacky way to test if the token is an integer.
    if tok0.len() > 0 && tok0[0] >= '0' as u8 && tok0[0] <= '9' as u8 {
      id = from_utf8(&tok0)
        .ok().expect("FATAL: server: id token should be an integer!")
        .parse()
        .ok();
    }
  }
  let mut lines = Vec::new();
  //while !res_buf.eof() {
  loop {
    //let mut line = res_buf.read_until(LF).ok().unwrap();
    let mut line = Vec::new();
    match res_buf.read_until(LF, &mut line) {
      Ok(n) => {
        if n == 0 {
          break;
        }
      }
      Err(_) => break,
    }
    if line[line.len()-1] == LF {
      line.pop();
    }
    if line.len() == 0 {
      continue;
    }
    let mut line_buf = BufReader::new(&line as &[u8]);
    let mut line_args = Vec::new();
    //while !line_buf.eof() {
    loop {
      let mut tok = Vec::new();
      match line_buf.read_until(SPACE, &mut tok) {
        Ok(n) => {
          if n == 0 {
            break;
          }
        }
        Err(_) => break,
      }
      if tok[tok.len()-1] == SPACE || tok[tok.len()-1] == LF {
        tok.pop();
      }
      if tok.len() > 0 {
        line_args.push(tok);
      }
    }
    lines.push(line_args);
  }
  let res = Response{has_error: has_error, id: id, lines: lines};
  res
}

#[derive(Copy, Clone)]
pub struct GtpId {
  _id: u32,
}

impl GtpId {
  pub fn new() -> GtpId {
    GtpId{
      _id: 0,
    }
  }

  pub fn increment(&mut self) -> u32 {
    let id = self._id;
    self._id += 1;
    id
  }

  pub fn increment_by(&mut self, count: u32) -> u32 {
    let id = self._id;
    self._id += count;
    id
  }
}

enum GtpMessage {
  Wakeup,
  AckWakeup(u32), //(Arc<RefCell<GtpId>>),
  Shutdown,
  AckShutdown(u32), //(Arc<RefCell<GtpId>>),
  Command(Vec<u8>),
  Response(Vec<u8>),
}

fn handle_client_stream(color: Player, to_ctl_tx: Sender<GtpMessage>, from_ctl_rx: Receiver<GtpMessage>, stream: TcpStream) {
  let mut buffered_stream = BufStream::new(stream);
  println!("DEBUG: client: requesting wakeup");
  to_ctl_tx.send(GtpMessage::Wakeup)
    .unwrap();
  loop {
    match from_ctl_rx.recv()
      //.ok().expect("FATAL: failed to recv from ctrl")
    {
      Err(_) => {
        println!("DEBUG: client: engine has disconnected, stopping");
        break;
      },
      Ok(GtpMessage::AckWakeup(id)) => {
        println!("DEBUG: client: received ackwakeup");

        // Read the engine's supported commands.
        write_client_command(id + 0, &[StringEntity(b"list_commands".to_vec())], &mut buffered_stream);
        let res_str = read_client_response_string(&mut buffered_stream);
        print!("DEBUG: client: read 'list_commands' response:\n{}",
          String::from_utf8_lossy(&res_str));

        // Read the engine protocol version (only GTP version 2 is supported).
        write_client_command(id + 1, &[StringEntity(b"protocol_version".to_vec())], &mut buffered_stream);
        let res_str = read_client_response_string(&mut buffered_stream);
        print!("DEBUG: client: read 'protocol_version' response:\n{}",
          String::from_utf8_lossy(&res_str));

        // Read the engine name.
        write_client_command(id + 2, &[StringEntity(b"name".to_vec())], &mut buffered_stream);
        let res_str = read_client_response_string(&mut buffered_stream);
        print!("DEBUG: client: read 'name' response:\n{}",
          String::from_utf8_lossy(&res_str));

        // Read the engine version.
        write_client_command(id + 3, &[StringEntity(b"version".to_vec())], &mut buffered_stream);
        let res_str = read_client_response_string(&mut buffered_stream);
        print!("DEBUG: client: read 'version' response:\n{}",
          String::from_utf8_lossy(&res_str));
      },
      Ok(GtpMessage::AckShutdown(id)) => {
        println!("DEBUG: client: received ackshutdown");

        // Tell the engine to quit.
        write_client_command(id, &[StringEntity(b"quit".to_vec())], &mut buffered_stream);
        let res_str = read_client_response_string(&mut buffered_stream);
        print!("DEBUG: client: read 'quit' response:\n{}",
          String::from_utf8_lossy(&res_str));

        println!("DEBUG: client: goodbye");
        break;
      },
      Ok(GtpMessage::Command(cmd_str)) => {
        // Give the engine a command and read its response.
        println!("DEBUG: client: write command/read response:");
        print!("{}", String::from_utf8_lossy(&cmd_str));
        write_client_command_string(&cmd_str, &mut buffered_stream);
        let res_str = read_client_response_string(&mut buffered_stream);
        print!("{}", String::from_utf8_lossy(&res_str));
        to_ctl_tx.send(GtpMessage::Response(res_str))
          .unwrap();
      },
      _ => panic!("FATAL: client: unexpected protocol message"),
    }
  }
}

struct ClientState {
  previous_move: Option<Move>,
}

impl ClientState {
  pub fn new() -> ClientState {
    ClientState{
      previous_move: None,
    }
  }
}

pub struct GtpController {
  black_listener: TcpListener,
  white_listener: TcpListener,
  id: GtpId,
}

impl GtpController {
  pub fn new() -> GtpController {
    //sleep(Duration::seconds(3));
    let black_listener = TcpListener::bind("127.0.0.1:6060")
      .ok().expect("FATAL: server: failed to bind address (6060)");
    let white_listener = TcpListener::bind("127.0.0.1:6061")
      .ok().expect("FATAL: server: failed to bind address (6061)");
    GtpController{
      black_listener: black_listener,
      white_listener: white_listener,
      id: GtpId::new(),
    }
  }

  fn setup_client(&mut self, client_to_ctl_rx: &Receiver<GtpMessage>, ctl_to_client_tx: &Sender<GtpMessage>) {
    // Perform GTP administrative functions.
    match client_to_ctl_rx.recv()
      .ok().expect("FATAL: TODO")
    {
      GtpMessage::Wakeup => {
        println!("DEBUG: server: acking wakeup to client");
        ctl_to_client_tx.send(GtpMessage::AckWakeup(self.id.increment_by(4)))
          .unwrap();
      },
      _ => (),
    }

    // Clear the board.
    let cmd_str = create_command_string(self.id.increment(), &[
      StringEntity(b"clear_board".to_vec()),
    ]);
    ctl_to_client_tx.send(GtpMessage::Command(cmd_str))
      .unwrap();
    match client_to_ctl_rx.recv()
      .ok().expect("FATAL: TODO")
    {
      GtpMessage::Response(res_str) => {
        let res = parse_response_string(&res_str);
        assert!(!res.has_error && res.lines.len() == 0);
      },
      _ => (),
    }

    // Set the board size.
    let cmd_str = create_command_string(self.id.increment(), &[
      // FIXME: fixed sample board size for now.
      StringEntity(b"boardsize".to_vec()),
      IntEntity(19),
    ]);
    ctl_to_client_tx.send(GtpMessage::Command(cmd_str))
      .unwrap();
    match client_to_ctl_rx.recv()
      .ok().expect("FATAL: TODO")
    {
      GtpMessage::Response(res_str) => {
        let res = parse_response_string(&res_str);
        assert!(!res.has_error && res.lines.len() == 0);
      },
      _ => (),
    }

    // Set the komi level.
    let cmd_str = create_command_string(self.id.increment(), &[
      StringEntity(b"komi".to_vec()),
      // FIXME: fixed sample komi for now.
      FloatEntity(7.5),
    ]);
    ctl_to_client_tx.send(GtpMessage::Command(cmd_str))
      .unwrap();
    match client_to_ctl_rx.recv()
      .ok().expect("FATAL: TODO")
    {
      GtpMessage::Response(res_str) => {
        let res = parse_response_string(&res_str);
        assert!(!res.has_error && res.lines.len() == 0);
      },
      _ => (),
    }

    // Set the time settings (KGS extension).
    let cmd_str = create_command_string(self.id.increment(), &[
      // FIXME: fixed sample time settings for now.
      //StringEntity(b"kgs-time_settings".to_vec()),
      // Variant ("none", "absolute", "byoyomi", or "canadian").
      //StringEntity(b"byoyomi".to_vec()),
      StringEntity(b"time_settings".to_vec()),
      // Main time in seconds.
      IntEntity(300),
      // Byo-yomi time in seconds.
      IntEntity(40),
      // Byo-yomi stones (Canadian) or periods (Japanese).
      IntEntity(1), // 5
    ]);
    ctl_to_client_tx.send(GtpMessage::Command(cmd_str))
      .unwrap();
    match client_to_ctl_rx.recv()
      .ok().expect("FATAL: TODO")
    {
      GtpMessage::Response(res_str) => {
        let res = parse_response_string(&res_str);
        assert!(!res.has_error && res.lines.len() == 0);
      },
      _ => (),
    }
  }

  fn play_client(&mut self, player: Player, client_to_ctl_rx: &Receiver<GtpMessage>, ctl_to_client_tx: &Sender<GtpMessage>) -> Move {
    let cmd_str = create_command_string(self.id.increment(), &[
      StringEntity(b"showboard".to_vec()),
    ]);
    ctl_to_client_tx.send(GtpMessage::Command(cmd_str))
      .unwrap();
    match client_to_ctl_rx.recv()
      .ok().expect("FATAL: TODO")
    {
      GtpMessage::Response(res_str) => {
        // TODO
      },
      _ => (),
    }
    let cmd_str = create_command_string(self.id.increment(), &[
      StringEntity(b"genmove".to_vec()),
      StringEntity(player.to_bytestring()),
    ]);
    ctl_to_client_tx.send(GtpMessage::Command(cmd_str))
      .unwrap();
    // FIXME(20151002): not checking for time limit!
    match client_to_ctl_rx.recv()
      .ok().expect("FATAL: failed to recv from client")
    {
      GtpMessage::Response(res_str) => {
        let res = parse_response_string(&res_str);
        match Entity::parse_vertex_or_resign(&res.lines[0][0]) {
          VertexEntity(action) => Move::new(player, action),
          StringEntity(_) => Move::new(player, Action::Resign),
          _ => unreachable!(),
        }
      },
      _ => unreachable!(),
    }
  }

  fn update_client(&mut self, current_move: Move, client_to_ctl_rx: &Receiver<GtpMessage>, ctl_to_client_tx: &Sender<GtpMessage>) -> Result<(), ()> {
    match current_move.action {
      Action::Pass | Action::Play(_) => {
        let cmd_str = create_command_string(self.id.increment(), &[
          StringEntity(b"play".to_vec()),
          MoveEntity(current_move.player, current_move.action),
        ]);
        ctl_to_client_tx.send(GtpMessage::Command(cmd_str))
          .unwrap();
        match client_to_ctl_rx.recv()
          .ok().expect("FATAL: TODO")
        {
          GtpMessage::Response(res_str) => {
            let res = parse_response_string(&res_str);
            if res.has_error { // TODO: check for string b"illegal move"?
              return Err(());
            }
          },
          _ => (),
        }
      },
      Action::Resign => (), // FIXME
    }
    Ok(())
  }

  fn shutdown_client(&mut self, client_to_ctl_rx: &Receiver<GtpMessage>, ctl_to_client_tx: &Sender<GtpMessage>) {
    let cmd_str = create_command_string(self.id.increment(), &[
      StringEntity(b"final_score".to_vec()),
    ]);
    ctl_to_client_tx.send(GtpMessage::Command(cmd_str))
      .unwrap();
    match client_to_ctl_rx.recv() {
      _ => (),
    }
    let cmd_str = create_command_string(self.id.increment(), &[
      StringEntity(b"quit".to_vec()),
    ]);
    ctl_to_client_tx.send(GtpMessage::Command(cmd_str))
      .unwrap();
    match client_to_ctl_rx.recv() {
      _ => (),
    }
  }

  pub fn runloop(&mut self) {
    loop {
      // TODO: We start the controller with game parameters.
      // Then the two engines connect to the respective ports, and the game is
      // considered to begin (after boilerplate GTP commands).
      let black_listener = self.black_listener.try_clone()
        .ok().expect("failed to clone listener!");
      let white_listener = self.white_listener.try_clone()
        .ok().expect("failed to clone listener!");
      let (black_to_ctl_tx, black_to_ctl_rx) = channel::<GtpMessage>();
      let (white_to_ctl_tx, white_to_ctl_rx) = channel::<GtpMessage>();
      let (ctl_to_black_tx, ctl_to_black_rx) = channel::<GtpMessage>();
      let (ctl_to_white_tx, ctl_to_white_rx) = channel::<GtpMessage>();
      spawn(move || {
        for stream in black_listener.incoming() {
          match stream {
            Ok(stream) => {
              handle_client_stream(Player::Black, black_to_ctl_tx, ctl_to_black_rx, stream);
              break;
            },
            Err(_) => (),
          }
        }
      });
      spawn(move || {
        for stream in white_listener.incoming() {
          match stream {
            Ok(stream) => {
              handle_client_stream(Player::White, white_to_ctl_tx, ctl_to_white_rx, stream);
              break;
            },
            Err(_) => (),
          }
        }
      });

      self.setup_client(&black_to_ctl_rx, &ctl_to_black_tx);
      self.setup_client(&white_to_ctl_rx, &ctl_to_white_tx);

      let mut black_state = ClientState::new();
      let mut white_state = ClientState::new();

      let mut current_player = Player::Black;
      loop {
        let result = match current_player {
          Player::Black => {
            let current_move = self.play_client(Player::Black, &black_to_ctl_rx, &ctl_to_black_tx);
            let result = self.update_client(current_move, &white_to_ctl_rx, &ctl_to_white_tx);
            black_state.previous_move = Some(current_move);
            result
          },
          Player::White => {
            let current_move = self.play_client(Player::White, &white_to_ctl_rx, &ctl_to_white_tx);
            let result = self.update_client(current_move, &black_to_ctl_rx, &ctl_to_black_tx);
            white_state.previous_move = Some(current_move);
            result
          },
        };
        // TODO: if current_player's opponent claims current_move is illegal,
        // then what?
        match result {
          Ok(_) => (),
          Err(_) => {
            panic!("FATAL: player {} called out an illegal move!",
              current_player.opponent().to_string());
          },
        }
        if black_state.previous_move.is_some() && white_state.previous_move.is_some() {
          match black_state.previous_move.unwrap().action {
            Action::Pass => {
              match white_state.previous_move.unwrap().action {
                Action::Pass => {
                  self.shutdown_client(&black_to_ctl_rx, &ctl_to_black_tx);
                  self.shutdown_client(&white_to_ctl_rx, &ctl_to_white_tx);
                  // FIXME(20150104): We get a "receiving on a closed channel"
                  // panic message on termination.
                  break;
                },
                _ => (),
              }
            },
            _ => (),
          }
        }
        current_player = current_player.opponent();
      }

      break; // FIXME: stop after one game.
    }
  }
}
