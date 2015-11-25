use board::{Board, Stone, Point, Action};
use txnstate::{
  TxnStateData, TxnState, TxnPosition, TxnChainsList,
  for_each_adjacent,
};
use txnstate::extras::{TxnStateLegalityData, TxnStateNodeData};
use util::{slice_twice_mut};

use bit_set::{BitSet};
use std::cell::{RefCell};
use std::iter::{repeat};
use std::slice::bytes::{copy_memory};

#[derive(Clone)]
pub struct TxnStateFeaturesData {
  //compute_diff:       bool,
  time_step:  usize,
  features:   Vec<u8>,
  /*feature_diff_vals:  Vec<f32>,
  feature_diff_idxs:  Vec<i32>,*/
}

impl TxnStateFeaturesData {
  pub const BLACK_PLANE:      usize = 0;
  pub const WHITE_PLANE:      usize = 1 * Board::SIZE;
  pub const NUM_PLANES:       usize = 2;
  pub const NUM_TIME_STEPS:   usize = 2;

  pub fn new(/*compute_diff: bool*/) -> TxnStateFeaturesData {
    TxnStateFeaturesData{
      //compute_diff:       compute_diff,
      time_step:  0,
      features:   repeat(0).take(Self::NUM_TIME_STEPS * Self::NUM_PLANES * Board::SIZE).collect(),
      /*feature_diff_vals:  Vec::with_capacity(4),
      feature_diff_idxs:  Vec::with_capacity(4),*/
    }
  }

  pub fn current_feature(&self, plane_idx: usize, point: Point) -> u8 {
    let slice_sz = Self::NUM_PLANES * Board::SIZE;
    let time_step = self.time_step;
    let offset = time_step * slice_sz;
    self.features[offset + plane_idx * Board::SIZE + point.idx()]
  }

  pub fn feature_dims(&self) -> (usize, usize, usize) {
    (Board::DIM, Board::DIM, Self::NUM_TIME_STEPS * Self::NUM_PLANES)
  }

  pub fn extract_relative_features(&self, turn: Stone, dst_buf: &mut [u8]) {
    let slice_sz = Self::NUM_PLANES * Board::SIZE;
    let init_time_step = self.time_step;
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
    self.time_step = 0;
    assert_eq!(Self::NUM_TIME_STEPS * Self::NUM_PLANES * Board::SIZE, self.features.len());
    for p in (0 .. self.features.len()) {
      self.features[p] = 0;
    }
  }

  fn update(&mut self, position: &TxnPosition, chains: &TxnChainsList, update_turn: Stone, update_action: Action) {
    // XXX: Before anything else, increment the time step.
    self.time_step = (self.time_step + 1) % Self::NUM_TIME_STEPS;

    let slice_sz = Self::NUM_PLANES * Board::SIZE;
    // FIXME(20151123): following only works when there are 2 time steps.
    let next_slice_off = self.time_step;
    if Self::NUM_TIME_STEPS > 1 {
      let prev_slice_off = (self.time_step + Self::NUM_TIME_STEPS - 1) % Self::NUM_TIME_STEPS;
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
            /*if self.compute_diff {
              if features[Self::BLACK_PLANE + p] != 1 {
                self.feature_diff_vals.push(1.0);
                self.feature_diff_idxs.push(next_start + Self::BLACK_PLANE + p);
              }
              if features[Self::WHITE_PLANE + p] != 0 {
                self.feature_diff_vals.push(-1.0);
                self.feature_diff_idxs.push(next_start + Self::BLACK_PLANE + p);
              }
            }*/
            features[Self::BLACK_PLANE + p] = 1;
            features[Self::WHITE_PLANE + p] = 0;
          }
          Stone::White => {
            /*if self.compute_diff {
              if features[Self::BLACK_PLANE + p] != 0 {
                self.feature_diff_vals.push(-1.0);
                self.feature_diff_idxs.push(next_start + Self::BLACK_PLANE + p);
              }
              if features[Self::WHITE_PLANE + p] != 1 {
                self.feature_diff_vals.push(1.0);
                self.feature_diff_idxs.push(next_start + Self::BLACK_PLANE + p);
              }
            }*/
            features[Self::BLACK_PLANE + p] = 0;
            features[Self::WHITE_PLANE + p] = 1;
          }
          Stone::Empty => { unreachable!(); }
        }
      }
      for &point in position.last_killed[0].iter().chain(position.last_killed[1].iter()) {
        let p = point.idx();
        /*if self.compute_diff {
          if features[Self::BLACK_PLANE + p] != 1 {
            self.feature_diff_vals.push(1.0);
            self.feature_diff_idxs.push(next_start + Self::BLACK_PLANE + p);
          }
          if features[Self::WHITE_PLANE + p] != 0 {
            self.feature_diff_vals.push(-1.0);
            self.feature_diff_idxs.push(next_start + Self::BLACK_PLANE + p);
          }
        }*/
        features[Self::BLACK_PLANE + p] = 0;
        features[Self::WHITE_PLANE + p] = 0;
      }
    }
  }
}

#[derive(Clone)]
pub struct TxnStateLibFeaturesData {
  time_step:  usize,
  features:   Vec<u8>,

  tmp_mark:   BitSet,
}

impl TxnStateLibFeaturesData {
  pub const BLACK_PLANE:        usize = 0;
  pub const BLACK_ATARI_PLANE:  usize = 1 * Board::SIZE;
  pub const BLACK_LIVE_2_PLANE: usize = 2 * Board::SIZE;
  pub const BLACK_LIVE_3_PLANE: usize = 3 * Board::SIZE;
  pub const WHITE_PLANE:        usize = 4 * Board::SIZE;
  pub const WHITE_ATARI_PLANE:  usize = 5 * Board::SIZE;
  pub const WHITE_LIVE_2_PLANE: usize = 6 * Board::SIZE;
  pub const WHITE_LIVE_3_PLANE: usize = 7 * Board::SIZE;
  pub const NUM_PLANES:         usize = 8;
  pub const NUM_TIME_STEPS:     usize = 2;

  pub fn new() -> TxnStateLibFeaturesData {
    TxnStateLibFeaturesData{
      time_step: 0,
      features: repeat(0).take(Self::NUM_TIME_STEPS * Self::NUM_PLANES * Board::SIZE).collect(),
      tmp_mark: BitSet::with_capacity(Board::SIZE),
    }
  }

  pub fn current_feature(&self, plane_idx: usize, point: Point) -> u8 {
    let slice_sz = Self::NUM_PLANES * Board::SIZE;
    let time_step = self.time_step;
    let offset = time_step * slice_sz;
    self.features[offset + plane_idx * Board::SIZE + point.idx()]
  }

  pub fn feature_dims(&self) -> (usize, usize, usize) {
    (Board::DIM, Board::DIM, Self::NUM_TIME_STEPS * Self::NUM_PLANES)
  }

  pub fn extract_relative_features(&self, turn: Stone, dst_buf: &mut [u8]) {
    let slice_sz = Self::NUM_PLANES * Board::SIZE;
    let half_slice_sz = slice_sz / 2;
    let init_time_step = self.time_step;
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
              &self.features[src_offset + Self::BLACK_PLANE .. src_offset + Self::BLACK_PLANE + half_slice_sz],
              &mut dst_buf[dst_offset + Self::WHITE_PLANE .. dst_offset + Self::WHITE_PLANE + half_slice_sz],
          );
          copy_memory(
              &self.features[src_offset + Self::WHITE_PLANE .. src_offset + Self::WHITE_PLANE + half_slice_sz],
              &mut dst_buf[dst_offset + Self::BLACK_PLANE .. dst_offset + Self::BLACK_PLANE + half_slice_sz],
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

  fn update_point(tmp_mark: &mut BitSet, features: &mut [u8], position: &TxnPosition, chains: &TxnChainsList, point: Point) {
    let p = point.idx();
    if tmp_mark.contains(&p) {
      return;
    }
    let stone = position.stones[p];
    if stone != Stone::Empty {
      let bw_value: u8 = match stone {
        Stone::Black => 0,
        Stone::White => 1,
        _ => unreachable!(),
      };
      let head = chains.find_chain(point);
      //assert!(head != TOMBSTONE);
      let chain = chains.get_chain(head).unwrap();
      match chain.approx_count_libs_up_to_3() {
        0 => { unreachable!(); }
        1 => {
          features[Self::BLACK_ATARI_PLANE + p]  = 1 - bw_value;
          features[Self::BLACK_LIVE_2_PLANE + p] = 0;
          features[Self::BLACK_LIVE_3_PLANE + p] = 0;
          features[Self::WHITE_ATARI_PLANE + p]  = bw_value;
          features[Self::WHITE_LIVE_2_PLANE + p] = 0;
          features[Self::WHITE_LIVE_3_PLANE + p] = 0;
        }
        2 => {
          features[Self::BLACK_ATARI_PLANE + p]  = 0;
          features[Self::BLACK_LIVE_2_PLANE + p] = 1 - bw_value;
          features[Self::BLACK_LIVE_3_PLANE + p] = 0;
          features[Self::WHITE_ATARI_PLANE + p]  = 0;
          features[Self::WHITE_LIVE_2_PLANE + p] = bw_value;
          features[Self::WHITE_LIVE_3_PLANE + p] = 0;
        }
        3 => {
          features[Self::BLACK_ATARI_PLANE + p]  = 0;
          features[Self::BLACK_LIVE_2_PLANE + p] = 0;
          features[Self::BLACK_LIVE_3_PLANE + p] = 1 - bw_value;
          features[Self::WHITE_ATARI_PLANE + p]  = 0;
          features[Self::WHITE_LIVE_2_PLANE + p] = 0;
          features[Self::WHITE_LIVE_3_PLANE + p] = bw_value;
        }
        _ => { unreachable!(); }
      }
    } else {
      features[Self::BLACK_ATARI_PLANE + p]  = 0;
      features[Self::BLACK_LIVE_2_PLANE + p] = 0;
      features[Self::BLACK_LIVE_3_PLANE + p] = 0;
      features[Self::WHITE_ATARI_PLANE + p]  = 0;
      features[Self::WHITE_LIVE_2_PLANE + p] = 0;
      features[Self::WHITE_LIVE_3_PLANE + p] = 0;
    }
    tmp_mark.insert(p);
  }
}

impl TxnStateData for TxnStateLibFeaturesData {
  fn reset(&mut self) {
    self.time_step = 0;
    assert_eq!(Self::NUM_TIME_STEPS * Self::NUM_PLANES * Board::SIZE, self.features.len());
    for p in (0 .. self.features.len()) {
      self.features[p] = 0;
    }
  }

  fn update(&mut self, position: &TxnPosition, chains: &TxnChainsList, update_turn: Stone, update_action: Action) {
    // XXX: Before anything else, increment the time step.
    self.time_step = (self.time_step + 1) % Self::NUM_TIME_STEPS;

    // TODO(20151124): for sparse features, how to compute diffs efficiently?

    let slice_sz = Self::NUM_PLANES * Board::SIZE;
    // FIXME(20151123): following only works when there are 2 time steps.
    let next_slice_idx = self.time_step;
    if Self::NUM_TIME_STEPS > 1 {
      let prev_slice_idx = (self.time_step + Self::NUM_TIME_STEPS - 1) % Self::NUM_TIME_STEPS;
      let (src_feats, mut dst_feats) = slice_twice_mut(
          &mut self.features,
          prev_slice_idx * slice_sz, (prev_slice_idx + 1) * slice_sz,
          next_slice_idx * slice_sz, (next_slice_idx + 1) * slice_sz,
      );
      copy_memory(src_feats, &mut dst_feats);
    }

    self.tmp_mark.clear();

    let next_start = next_slice_idx * slice_sz;
    let next_end = next_start + slice_sz;
    {
      // XXX(20151107): Iterate over placed and killed chains to update stone
      // repr.
      let &mut TxnStateLibFeaturesData{ref mut features, ref mut tmp_mark, .. } = self;
      let mut features = &mut features[next_start .. next_end];
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
      for &kill_point in position.last_killed[0].iter().chain(position.last_killed[1].iter()) {
        let p = kill_point.idx();
        features[Self::BLACK_PLANE + p] = 0;
        features[Self::WHITE_PLANE + p] = 0;
      }

      // XXX(20151123): Iterate over placed, adjacent, ko, captured stones, and
      // pseudolibs; see TxnStateLegalityData.
      if let Some((_, place_point)) = position.last_placed {
        Self::update_point(tmp_mark, features, position, chains, place_point);
        for_each_adjacent(place_point, |adj_point| {
          Self::update_point(tmp_mark, features, position, chains, adj_point);
        });
      }
      if let Some((_, ko_point)) = position.ko {
        Self::update_point(tmp_mark, features, position, chains, ko_point);
      }
      if let Some((_, prev_ko_point)) = position.prev_ko {
        Self::update_point(tmp_mark, features, position, chains, prev_ko_point);
      }
      for &kill_point in position.last_killed[0].iter().chain(position.last_killed[1].iter()) {
        Self::update_point(tmp_mark, features, position, chains, kill_point);
      }
      for &head in chains.last_mut_heads.iter() {
        let mut is_valid_head = false;
        if let Some(chain) = chains.get_chain(head) {
          is_valid_head = true;
          /*for &lib in chain.ps_libs.iter() {
            Self::update_point(tmp_mark, features, position, chains, lib);
          }*/
        }
        if is_valid_head {
          chains.iter_chain(head, |ch_pt| {
            Self::update_point(tmp_mark, features, position, chains, ch_pt);
          });
        }
      }
    }
  }
}
