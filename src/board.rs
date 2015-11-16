pub use gtp_board::{Coord};

use std::str::{from_utf8};

pub struct Board;

impl Board {
  pub const HALF_PT:    Point = Point(9);
  pub const DIM_PT:     Point = Point(19);
  pub const MAX_PT:     Point = Point(361);
  pub const DIM:        usize = 19;
  pub const SIZE:       usize = 361;
}

#[derive(Clone, Copy)]
pub struct Rules {
  pub score_captures:   bool,
  pub score_stones:     bool,
  pub score_territory:  bool,
  // FIXME(20151107): score w/ handicap komi convention.
  pub handicap_komi:    HandicapKomi,
  pub ko_rule:          KoRule,
  pub suicide_rule:     SuicideRule,
}

#[derive(Clone, Copy)]
pub enum HandicapKomi {
  Zero,
  One,
  OneExceptFirst,
}

#[derive(Clone, Copy)]
pub enum KoRule {
  Ko,
  Superko,
}

#[derive(Clone, Copy)]
pub enum SuicideRule {
  Illegal,
  Allowed,
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum RuleSet {
  KgsJapanese,
  KgsChinese,
  KgsAga,
  KgsNewZealand,
}

impl RuleSet {
  pub fn rules(&self) -> Rules {
    match *self {
      RuleSet::KgsJapanese => Rules{
        score_captures:   true,
        score_stones:     false,
        score_territory:  true,
        handicap_komi:    HandicapKomi::Zero,
        ko_rule:          KoRule::Ko,
        suicide_rule:     SuicideRule::Illegal,
      },
      RuleSet::KgsChinese => Rules{
        score_captures:   false,
        score_stones:     true,
        score_territory:  true,
        handicap_komi:    HandicapKomi::One,
        ko_rule:          KoRule::Superko,
        suicide_rule:     SuicideRule::Illegal,
      },
      RuleSet::KgsAga => Rules{
        score_captures:   false,
        score_stones:     true,
        score_territory:  true,
        handicap_komi:    HandicapKomi::OneExceptFirst,
        ko_rule:          KoRule::Superko,
        suicide_rule:     SuicideRule::Illegal,
      },
      RuleSet::KgsNewZealand => Rules{
        score_captures:   false,
        score_stones:     true,
        score_territory:  true,
        handicap_komi:    HandicapKomi::One,
        ko_rule:          KoRule::Superko,
        suicide_rule:     SuicideRule::Allowed,
      },
    }
  }
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Stone {
  Black = 0,
  White = 1,
  Empty = 2,
}

impl Stone {
  pub fn from_code_str(code_str: &str) -> Stone {
    match code_str {
      "B" => Stone::Black,
      "W" => Stone::White,
      _   => unreachable!(),
    }
  }

  pub fn offset(self) -> usize {
    match self {
      Stone::Black => 0,
      Stone::White => 1,
      Stone::Empty => unreachable!(),
    }
  }

  pub fn opponent(self) -> Stone {
    match self {
      Stone::Black => Stone::White,
      Stone::White => Stone::Black,
      Stone::Empty => unreachable!(),
    }
  }
}

#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct Point(pub i16);

impl Point {
  pub fn from_idx(idx: usize) -> Point {
    Point(idx as i16)
  }

  pub fn from_coord(coord: Coord) -> Point {
    Point(coord.x as i16 + coord.y as i16 * Board::DIM_PT.0)
  }

  pub fn to_coord(self) -> Coord {
    Coord::new((self.0 % Board::DIM_PT.0) as u8, (self.0 / Board::DIM_PT.0) as u8)
  }

  #[inline]
  pub fn idx(self) -> usize {
    self.0 as usize
  }

  #[inline]
  pub fn is_edge(self) -> bool {
    let (x, y) = (self.0 % Board::DIM_PT.0, self.0 / Board::DIM_PT.0);
    (x == 0 || x == (Board::DIM_PT.0 - 1) ||
     y == 0 || y == (Board::DIM_PT.0 - 1))
  }
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Action {
  Place{point: Point},
  Pass,
  Resign,
}

impl Action {
  pub fn from_code_str(code_str: &str) -> Action {
    Action::from_code_bytes(code_str.as_bytes())
  }

  pub fn from_code_bytes(code: &[u8]) -> Action {
    if code == b"Pass" {
      Action::Pass
    } else if code == b"Resign" {
      Action::Resign
    } else {
      let raw_x = code[0];
      let x = match raw_x {
        b'A' ... b'H' => raw_x - b'A',
        b'J' ... b'T' => raw_x - b'A' - 1,
        _ => unreachable!(),
      };
      let raw_y: Option<u8> = from_utf8(&code[1 ..]).unwrap().parse().ok();
      let y = raw_y.unwrap() - 1;
      Action::Place{point: Point::from_coord(Coord{x: x, y: y})}
    }
  }
}
