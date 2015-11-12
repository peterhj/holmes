use board::{Coord, Stone, Point, Action};

pub mod convnet;
pub mod search;

pub trait Agent {
  fn reset(&mut self);
  fn board_dim(&mut self, board_dim: usize);
  fn komi(&mut self, komi: f32);
  fn player(&mut self, stone: Stone);

  fn apply_action(&mut self, turn: Stone, action: Action);
  fn undo(&mut self);
  fn act(&mut self, turn: Stone) -> Action;
}

// XXX: See <http://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html#sec:fixed-handicap-placement>.
static FIXED_HANDICAP_19X19_POSITIONS: [&'static [&'static [u8]]; 10] = [
  &[],
  &[],
  &[b"D4", b"Q16"],
  &[b"D4", b"Q16", b"D16"],
  &[b"D4", b"Q16", b"D16", b"Q4"],
  &[b"D4", b"Q16", b"D16", b"Q4", b"K10"],
  &[b"D4", b"Q16", b"D16", b"Q4", b"D10", b"Q10"],
  &[b"D4", b"Q16", b"D16", b"Q4", b"D10", b"Q10", b"K10"],
  &[b"D4", b"Q16", b"D16", b"Q4", b"D10", b"Q10", b"K4", b"K16"],
  &[b"D4", b"Q16", b"D16", b"Q4", b"D10", b"Q10", b"K4", b"K16", b"K10"],
];

#[derive(Default)]
pub struct PreGame;

impl PreGame {
  pub fn fixed_handicap_positions(&self, num_stones: usize) -> Vec<Point> {
    assert!(num_stones >= 2 && num_stones <= 9);
    let mut ps = vec![];
    for &code in FIXED_HANDICAP_19X19_POSITIONS[num_stones] {
      let coord = Coord::from_code(code);
      let point = Point::from_coord(coord);
      ps.push(point);
    }
    ps
  }

  pub fn prefer_handicap_positions(&self, num_stones: usize) -> Vec<Point> {
    self.fixed_handicap_positions(num_stones)
  }
}
