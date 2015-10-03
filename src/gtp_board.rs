/*use self::MoveResult::{LegalMove, StupidMove, IllegalMove};
use self::StupidMoveReason::{SelfAtari};
use self::IllegalMoveReason::{OccupiedVertex, Suicide};*/

/*#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub enum Piece {
  Empty      = 0x0,
  BlackStone = 0x1,
  WhiteStone = 0x2,
}

impl Piece {
  pub fn opponent(&self) -> Piece {
    match *self {
      Piece::BlackStone => Piece::WhiteStone,
      Piece::WhiteStone => Piece::BlackStone,
      Piece::Empty => Piece::Empty, // FIXME: should this arm be set at all?
    }
  }
}*/

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

  /*pub fn to_piece(&self) -> Piece {
    match *self {
      Player::Black => Piece::BlackStone,
      Player::White => Piece::WhiteStone,
    }
  }*/
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
  let x_label = [if x < 8 {
    'A' as u8 + x
  } else if x >= 8 {
    'J' as u8 + x - 8
  } else {
    unreachable!();
  }].to_vec();
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

  /*pub fn from_idx(idx: u16) -> Coord {
    Coord{
      x: (idx as u8 % 19),
      y: (idx as u8 / 19),
    }
  }*/

  /*#[inline]
  pub fn is_border(&self) -> bool {
    self.x == 0 || self.x == 19u8 - 1 ||
    self.y == 0 || self.y == 19u8 - 1
  }*/

  /*#[inline]
  pub fn to_idx(&self) -> u16 {
    self.x as u16 + self.y as u16 * 19
  }*/

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

/*#[derive(Clone, Copy, Debug)]
pub enum MoveResult {
  LegalMove,
  StupidMove(StupidMoveReason),
  IllegalMove(IllegalMoveReason),
}

#[derive(Clone, Copy, Debug)]
pub enum StupidMoveReason {
  SelfAtari,
}

#[derive(Clone, Copy, Debug)]
pub enum IllegalMoveReason {
  Unknown,
  OccupiedVertex,
  Suicide,
  Ko,
  Superko,
}*/
