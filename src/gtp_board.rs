use std::str::{from_utf8};

#[derive(Clone, Copy, Debug)]
pub enum Player {
  Black,
  White,
}

impl Player {
  pub fn opponent(&self) -> Player {
    match *self {
      Player::Black => Player::White,
      Player::White => Player::Black,
    }
  }

  pub fn to_bytestring(&self) -> Vec<u8> {
    match *self {
      Player::Black => b"black".to_vec(),
      Player::White => b"white".to_vec(),
    }
  }
}

impl ToString for Player {
  fn to_string(&self) -> String {
    match *self {
      Player::Black => "Black".to_string(),
      Player::White => "White".to_string(),
    }
  }
}

pub fn dump_xcoord(x: u8) -> Vec<u8> {
  let x_label = vec![if x < 8 {
    'A' as u8 + x
  } else if x >= 8 {
    'J' as u8 + x - 8
  } else {
    unreachable!();
  }];
  x_label
}

pub fn dump_ycoord(y: u8) -> Vec<u8> {
  let y_label = (y + 1).to_string().as_bytes().to_vec();
  y_label
}

#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(packed)]
pub struct Coord {
  pub x: u8,
  pub y: u8,
}

impl Coord {
  pub fn new(x: u8, y: u8) -> Coord {
    Coord{x: x, y: y}
  }

  pub fn from_code(code: &[u8]) -> Coord {
    let raw_x = code[0];
    let x = match raw_x {
      b'A' ... b'H' => raw_x - b'A',
      b'J' ... b'T' => raw_x - b'A' - 1,
      _ => unreachable!(),
    };
    let raw_y: Option<u8> = from_utf8(&code[1 ..]).unwrap().parse().ok();
    let y = raw_y.unwrap() - 1;
    Coord{x: x, y: y}
  }

  pub fn to_bytestring(&self) -> Vec<u8> {
    let mut s = Vec::new();
    let x_label = dump_xcoord(self.x);
    let y_label = dump_ycoord(self.y);
    s.extend(&x_label);
    s.extend(&y_label);
    s
  }
}

impl ToString for Coord {
  fn to_string(&self) -> String {
    String::from_utf8(self.to_bytestring()).ok().unwrap()
  }
}

#[derive(Clone, Copy)]
pub enum Vertex {
  Resign,
  Pass,
  Play(Coord),
}

impl Vertex {
  pub fn to_bytestring(&self) -> Vec<u8> {
    match *self {
      Vertex::Resign => b"resign".to_vec(),
      Vertex::Pass => b"pass".to_vec(),
      Vertex::Play(coord) => coord.to_bytestring(),
    }
  }
}

#[derive(Clone, Copy)]
pub struct Move {
  pub player: Player,
  pub vertex: Vertex,
}

impl Move {
  pub fn new(player: Player, vertex: Vertex) -> Move {
    Move{
      player: player,
      vertex: vertex,
    }
  }
}

#[derive(Clone, Copy)]
pub enum RuleSystem {
  Japanese,
  Chinese,
  Aga,
  NewZealand,
}

#[derive(Clone, Copy)]
pub enum TimeSystem {
  // GTP v2 natively only supports Canadian byo-yomi.
  // KGS GTP extensions support no time limit, an absolute time limit,
  // byo-yomi, and Canadian byo-yomi.
  NoTimeLimit,
  Absolute{
    main_time_s:      u32,
  },
  ByoYomi{
    main_time_s:      u32,
    byo_yomi_time_s:  u32,
    periods:          u32,
  },
  Canadian{
    main_time_s:      u32,
    byo_yomi_time_s:  u32,
    stones:           u32,
  },
}

#[derive(Clone, Copy)]
pub enum MoveResult {
  Okay,
  SyntaxError,
  IllegalMove,
}

#[derive(Clone, Copy)]
pub enum UndoResult {
  Okay,
  CannotUndo,
}