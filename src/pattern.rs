use std::cmp::{min};

/*static MASK_8_UV_COORDS: [(i8, i8); 8] = [
  (-1, -1), (0, -1), (1, -1),
  (-1, 0), (1, 0),
  (-1, 1), (0, 1), (1, 1),
];*/

static MASK_8_TRANS_MAPS: [[u8; 8]; 8] = [
  [0, 1, 2, 3, 4, 5, 6, 7],
  [2, 1, 0, 4, 3, 7, 6, 5],
  [5, 6, 7, 3, 4, 0, 1, 2],
  [7, 6, 5, 4, 3, 2, 1, 0],
  [0, 3, 5, 1, 6, 2, 4, 7],
  [2, 4, 7, 1, 6, 0, 3, 5],
  [5, 3, 0, 6, 1, 7, 4, 2],
  [7, 4, 2, 6, 1, 5, 3, 0],
];

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct Pattern3x3(pub u16);

impl Pattern3x3 {
  pub fn to_invariant(self) -> InvariantPattern3x3 {
    let mut min_mask8: u16 = 0xffffffff;
    for t in 0 .. 8 {
      let mut mask8: u16 = 0;
      for i in 0 .. 8 {
        let x = (self.0 >> (2 * i)) & 0x3;
        let t_i = MASK_8_TRANS_MAPS[t][i];
        mask8 |= x << (2 * t_i);
      }
      min_mask8 = min(min_mask8, mask8);
    }
    InvariantPattern3x3(min_mask8)
  }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct InvariantPattern3x3(u16);

impl InvariantPattern3x3 {
  pub fn idx(self) -> u32 {
    self.0 as u32
  }
}
