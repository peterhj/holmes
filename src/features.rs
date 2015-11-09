use board::{Board, Stone};
use txnstate::{TxnStateData, TxnPosition, TxnChainsList};
use util::{slice_twice_mut};

use std::iter::{repeat};
use std::slice::bytes::{copy_memory};

#[derive(Clone)]
pub struct TxnStateFeaturesData {
  features: Vec<u8>,
  time_step_offset: usize,
}

impl TxnStateFeaturesData {
  pub const BLACK_PLANE:      usize = 0;
  pub const WHITE_PLANE:      usize = 1 * Board::SIZE;
  pub const NUM_PLANES:       usize = 2;
  pub const NUM_TIME_STEPS:   usize = 2;

  pub fn new() -> TxnStateFeaturesData {
    TxnStateFeaturesData{
      features: repeat(0).take(Self::NUM_TIME_STEPS * Self::NUM_PLANES * Board::SIZE).collect(),
      time_step_offset: 0,
    }
  }

  pub fn feature_dims(&self) -> (usize, usize, usize) {
    (Board::DIM, Board::DIM, Self::NUM_TIME_STEPS * Self::NUM_PLANES)
  }

  pub fn extract_relative_features(&self, turn: Stone, dst_buf: &mut [u8]) {
    let slice_sz = Self::NUM_PLANES * Board::SIZE;
    let init_time_step = self.time_step_offset;
    let mut time_step = init_time_step;
    let mut dst_time_step = 0;
    loop {
      let src_offset = time_step * slice_sz;
      let dst_offset = dst_time_step * slice_sz;
      match turn {
        Stone::Black => {
          copy_memory(
              &self.features[src_offset .. src_offset + slice_sz],
              &mut dst_buf[dst_offset .. dst_offset + slice_sz],
          );
        }
        Stone::White => {
          copy_memory(
              &self.features[src_offset + Self::BLACK_PLANE .. src_offset + Self::BLACK_PLANE + Board::SIZE],
              &mut dst_buf[dst_offset + Self::WHITE_PLANE .. dst_offset + Self::WHITE_PLANE + Board::SIZE],
          );
          copy_memory(
              &self.features[src_offset + Self::WHITE_PLANE .. src_offset + Self::WHITE_PLANE + Board::SIZE],
              &mut dst_buf[dst_offset + Self::BLACK_PLANE .. dst_offset + Self::BLACK_PLANE + Board::SIZE],
          );
          /*copy_memory(
              &self.features[time_step * slice_sz + Self::ATARI_PLANE .. (time_step + 1) * slize_sz],
              &mut dst_buf[dst_time_step * slice_sz + Self::ATARI_PLANE .. (dst_time_step + 1) * slice_sz],
          );*/
        }
        _ => unreachable!(),
      }
      time_step = (time_step + Self::NUM_TIME_STEPS - 1) % Self::NUM_TIME_STEPS;
      dst_time_step += 1;
      if time_step == init_time_step {
        assert_eq!(Self::NUM_TIME_STEPS, dst_time_step);
        break;
      }
    }
  }
}

impl TxnStateData for TxnStateFeaturesData {
  fn reset(&mut self) {
    assert_eq!(Self::NUM_TIME_STEPS * Self::NUM_PLANES * Board::SIZE, self.features.len());
    for p in (0 .. self.features.len()) {
      self.features[p] = 0;
    }
    self.time_step_offset = 0;
  }

  fn update(&mut self, position: &TxnPosition, chains: &TxnChainsList) {
    self.time_step_offset = (self.time_step_offset + 1) % Self::NUM_TIME_STEPS;
    let slice_sz = Self::NUM_PLANES * Board::SIZE;
    let next_slice_off = self.time_step_offset;
    if Self::NUM_TIME_STEPS > 1 {
      let prev_slice_off = (self.time_step_offset + Self::NUM_TIME_STEPS - 1) % Self::NUM_TIME_STEPS;
      let (src_feats, mut dst_feats) = slice_twice_mut(
          &mut self.features,
          prev_slice_off * slice_sz, (prev_slice_off + 1) * slice_sz,
          next_slice_off * slice_sz, (next_slice_off + 1) * slice_sz,
      );
      copy_memory(src_feats, &mut dst_feats);
    }
    let next_start = next_slice_off * slice_sz;
    let next_end = next_start + slice_sz;
    {
      // TODO(20151107): iterate over placed and killed chains to update stone
      // repr.
      let mut features = &mut self.features[next_start .. next_end];
      if let Some((stone, point)) = position.last_placed {
        let p = point.idx();
        match stone {
          Stone::Black => {
            features[Self::BLACK_PLANE + p] = 1;
            features[Self::WHITE_PLANE + p] = 0;
          }
          Stone::White => {
            features[Self::BLACK_PLANE + p] = 0;
            features[Self::WHITE_PLANE + p] = 1;
          }
          Stone::Empty => { unreachable!(); }
        }
      }
      for &point in position.last_killed[0].iter() {
        let p = point.idx();
        features[Self::BLACK_PLANE + p] = 0;
        features[Self::WHITE_PLANE + p] = 0;
      }
      for &point in position.last_killed[1].iter() {
        let p = point.idx();
        features[Self::BLACK_PLANE + p] = 0;
        features[Self::WHITE_PLANE + p] = 0;
      }
      // TODO(20151107): iterate over "touched" chains to update liberty repr.
    }
  }
}

#[derive(Clone)]
pub struct TxnStateLibFeaturesData {
  features: Vec<u8>,
  time_step_offset: usize,
}

impl TxnStateLibFeaturesData {
  pub const BLACK_PLANE:      usize = 0;
  pub const WHITE_PLANE:      usize = 1 * Board::SIZE;
  pub const ATARI_PLANE:      usize = 2 * Board::SIZE;
  pub const LIVE_2_PLANE:     usize = 3 * Board::SIZE;
  pub const LIVE_3_PLANE:     usize = 4 * Board::SIZE;
  pub const NUM_PLANES:       usize = 5;
  pub const NUM_TIME_STEPS:   usize = 2;

  pub fn new() -> TxnStateLibFeaturesData {
    TxnStateLibFeaturesData{
      features: repeat(0).take(Self::NUM_TIME_STEPS * Self::NUM_PLANES * Board::SIZE).collect(),
      time_step_offset: 0,
    }
  }

  pub fn feature_dims(&self) -> (usize, usize, usize) {
    (Board::DIM, Board::DIM, Self::NUM_TIME_STEPS * Self::NUM_PLANES)
  }

  pub fn extract_relative_features(&self, turn: Stone, dst_buf: &mut [u8]) {
    let slice_sz = Self::NUM_PLANES * Board::SIZE;
    let init_time_step = self.time_step_offset;
    let mut time_step = init_time_step;
    let mut dst_time_step = 0;
    loop {
      let src_offset = time_step * slice_sz;
      let dst_offset = dst_time_step * slice_sz;
      match turn {
        Stone::Black => {
          copy_memory(
              &self.features[src_offset .. src_offset + slice_sz],
              &mut dst_buf[dst_offset .. dst_offset + slice_sz],
          );
        }
        Stone::White => {
          copy_memory(
              &self.features[src_offset + Self::BLACK_PLANE .. src_offset + Self::BLACK_PLANE + Board::SIZE],
              &mut dst_buf[dst_offset + Self::WHITE_PLANE .. dst_offset + Self::WHITE_PLANE + Board::SIZE],
          );
          copy_memory(
              &self.features[src_offset + Self::WHITE_PLANE .. src_offset + Self::WHITE_PLANE + Board::SIZE],
              &mut dst_buf[dst_offset + Self::BLACK_PLANE .. dst_offset + Self::BLACK_PLANE + Board::SIZE],
          );
          copy_memory(
              &self.features[src_offset + Self::ATARI_PLANE .. src_offset + slice_sz],
              &mut dst_buf[dst_offset + Self::ATARI_PLANE .. dst_offset + slice_sz],
          );
        }
        _ => unreachable!(),
      }
      time_step = (time_step + Self::NUM_TIME_STEPS - 1) % Self::NUM_TIME_STEPS;
      dst_time_step += 1;
      if time_step == init_time_step {
        assert_eq!(Self::NUM_TIME_STEPS, dst_time_step);
        break;
      }
    }
  }
}
