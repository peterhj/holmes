use board::{Board, PlayerRank, Stone, Point, Action};
use txnstate::{
  TxnStateData, TxnState, TxnPosition, TxnChainsList,
  for_each_adjacent, for_each_diagonal, check_good_move_fast,
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
    for p in 0 .. self.features.len() {
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
      match chain.count_libs_up_to_3() {
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
    for p in 0 .. self.features.len() {
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

#[derive(Clone)]
pub struct TxnStateExtLibFeatsData {
  features:     Vec<u8>,
  prev_moves:   Vec<Option<(Stone, Point)>>,
  black_rank:   Option<PlayerRank>,
  white_rank:   Option<PlayerRank>,
  tmp_mark:     BitSet,
}

impl TxnStateExtLibFeatsData {
  pub const SET: u8 = 255;

  pub const NUM_PLANES:         usize = 28;
  pub const NUM_COLOR_PLANES:   usize = 12;
  pub const NUM_GRAY_PLANES:    usize = 4;

  pub const BLACK:              usize = 0 * Board::SIZE;
  pub const WHITE:              usize = 12 * Board::SIZE;
  pub const GRAY:               usize = 24 * Board::SIZE;

  pub const STONE_PLANE:        usize = 0 * Board::SIZE;
  pub const ATARI_PLANE:        usize = 1 * Board::SIZE;
  pub const AIR_2_PLANE:        usize = 2 * Board::SIZE;
  pub const AIR_3_PLANE:        usize = 3 * Board::SIZE;
  pub const AIR_4_PLANE:        usize = 4 * Board::SIZE;
  pub const KO_PLANE:           usize = 5 * Board::SIZE;
  pub const RANK_DAN_1_2_PLANE: usize = 6 * Board::SIZE;
  pub const RANK_DAN_3_4_PLANE: usize = 7 * Board::SIZE;
  pub const RANK_DAN_5_6_PLANE: usize = 8 * Board::SIZE;
  pub const RANK_PRO_1_3_PLANE: usize = 9 * Board::SIZE;
  pub const RANK_PRO_4_6_PLANE: usize = 10 * Board::SIZE;
  pub const RANK_PRO_7_9_PLANE: usize = 11 * Board::SIZE;

  pub const PREV_1_PLANE:       usize = 0 * Board::SIZE;
  pub const PREV_2_PLANE:       usize = 1 * Board::SIZE;
  pub const PREV_3_PLANE:       usize = 2 * Board::SIZE;
  pub const BASELINE_PLANE:     usize = 3 * Board::SIZE;

  pub fn new() -> TxnStateExtLibFeatsData {
    let feats = TxnStateExtLibFeatsData{
      features:     repeat(0).take(Self::NUM_PLANES * Board::SIZE).collect(),
      prev_moves:   vec![None, None, None],
      black_rank:   None,
      white_rank:   None,
      tmp_mark:     BitSet::with_capacity(Board::SIZE),
    };
    assert_eq!(
        feats.features.len(),
        Self::NUM_PLANES * Board::SIZE);
    assert_eq!(
        Self::GRAY + Self::NUM_GRAY_PLANES * Board::SIZE,
        Self::NUM_PLANES * Board::SIZE);
    feats
  }

  pub fn current_feature(&self, plane_idx: usize, point: Point) -> u8 {
    self.features[plane_idx * Board::SIZE + point.idx()]
  }

  pub fn feature_dims(&self) -> (usize, usize, usize) {
    (Board::DIM, Board::DIM, Self::NUM_PLANES)
  }

  pub fn extract_relative_features(&self, turn: Stone, dst_buf: &mut [u8]) {
    let half_slice_sz = Self::NUM_COLOR_PLANES * Board::SIZE;
    match turn {
      Stone::Black => {
        copy_memory(
            &self.features,
            dst_buf,
        );
      }
      Stone::White => {
        copy_memory(
            &self.features[Self::BLACK .. Self::BLACK + half_slice_sz],
            &mut dst_buf[Self::WHITE .. Self::WHITE + half_slice_sz],
        );
        copy_memory(
            &self.features[Self::WHITE .. Self::WHITE + half_slice_sz],
            &mut dst_buf[Self::BLACK .. Self::BLACK + half_slice_sz],
        );
        copy_memory(
            &self.features[Self::GRAY .. ],
            &mut dst_buf[Self::GRAY .. ],
        );
      }
      _ => unreachable!(),
    }
  }

  fn update_point_libs(tmp_mark: &mut BitSet, features: &mut [u8], position: &TxnPosition, chains: &TxnChainsList, point: Point) {
    let p = point.idx();
    if tmp_mark.contains(&p) {
      return;
    }
    let stone = position.stones[p];
    if stone != Stone::Empty {
      let bw_value: u8 = match stone {
        Stone::Black => 0,
        Stone::White => Self::SET,
        _ => unreachable!(),
      };
      let head = chains.find_chain(point);
      //assert!(head != TOMBSTONE);
      let chain = chains.get_chain(head).unwrap();
      match chain.count_libs_up_to_4() {
        0 => { unreachable!(); }
        1 => {
          features[Self::BLACK + Self::ATARI_PLANE + p] = Self::SET - bw_value;
          features[Self::BLACK + Self::AIR_2_PLANE + p] = 0;
          features[Self::BLACK + Self::AIR_3_PLANE + p] = 0;
          features[Self::BLACK + Self::AIR_4_PLANE + p] = 0;
          features[Self::WHITE + Self::ATARI_PLANE + p] = bw_value;
          features[Self::WHITE + Self::AIR_2_PLANE + p] = 0;
          features[Self::WHITE + Self::AIR_3_PLANE + p] = 0;
          features[Self::WHITE + Self::AIR_4_PLANE + p] = 0;
        }
        2 => {
          features[Self::BLACK + Self::ATARI_PLANE + p] = 0;
          features[Self::BLACK + Self::AIR_2_PLANE + p] = Self::SET - bw_value;
          features[Self::BLACK + Self::AIR_3_PLANE + p] = 0;
          features[Self::BLACK + Self::AIR_4_PLANE + p] = 0;
          features[Self::WHITE + Self::ATARI_PLANE + p] = 0;
          features[Self::WHITE + Self::AIR_2_PLANE + p] = bw_value;
          features[Self::WHITE + Self::AIR_3_PLANE + p] = 0;
          features[Self::WHITE + Self::AIR_4_PLANE + p] = 0;
        }
        3 => {
          features[Self::BLACK + Self::ATARI_PLANE + p] = 0;
          features[Self::BLACK + Self::AIR_2_PLANE + p] = 0;
          features[Self::BLACK + Self::AIR_3_PLANE + p] = Self::SET - bw_value;
          features[Self::BLACK + Self::AIR_4_PLANE + p] = 0;
          features[Self::WHITE + Self::ATARI_PLANE + p] = 0;
          features[Self::WHITE + Self::AIR_2_PLANE + p] = 0;
          features[Self::WHITE + Self::AIR_3_PLANE + p] = bw_value;
          features[Self::WHITE + Self::AIR_4_PLANE + p] = 0;
        }
        4 => {
          features[Self::BLACK + Self::ATARI_PLANE + p] = 0;
          features[Self::BLACK + Self::AIR_2_PLANE + p] = 0;
          features[Self::BLACK + Self::AIR_3_PLANE + p] = 0;
          features[Self::BLACK + Self::AIR_4_PLANE + p] = Self::SET - bw_value;
          features[Self::WHITE + Self::ATARI_PLANE + p] = 0;
          features[Self::WHITE + Self::AIR_2_PLANE + p] = 0;
          features[Self::WHITE + Self::AIR_3_PLANE + p] = 0;
          features[Self::WHITE + Self::AIR_4_PLANE + p] = bw_value;
        }
        _ => { unreachable!(); }
      }
    } else {
      features[Self::BLACK + Self::ATARI_PLANE + p] = 0;
      features[Self::BLACK + Self::AIR_2_PLANE + p] = 0;
      features[Self::BLACK + Self::AIR_3_PLANE + p] = 0;
      features[Self::BLACK + Self::AIR_4_PLANE + p] = 0;
      features[Self::WHITE + Self::ATARI_PLANE + p] = 0;
      features[Self::WHITE + Self::AIR_2_PLANE + p] = 0;
      features[Self::WHITE + Self::AIR_3_PLANE + p] = 0;
      features[Self::WHITE + Self::AIR_4_PLANE + p] = 0;
    }
    tmp_mark.insert(p);
  }

  fn update_point_ko(features: &mut [u8], position: &TxnPosition, ko_turn: Stone, ko_point: Point, is_ko: bool) {
    let p = ko_point.idx();
    match ko_turn {
      Stone::Black => {
        if is_ko {
          features[Self::BLACK + Self::KO_PLANE + p] = Self::SET;
          features[Self::WHITE + Self::KO_PLANE + p] = 0;
        } else {
          features[Self::BLACK + Self::KO_PLANE + p] = 0;
          features[Self::WHITE + Self::KO_PLANE + p] = 0;
        }
      }
      Stone::White => {
        if is_ko {
          features[Self::BLACK + Self::KO_PLANE + p] = 0;
          features[Self::WHITE + Self::KO_PLANE + p] = Self::SET;
        } else {
          features[Self::BLACK + Self::KO_PLANE + p] = 0;
          features[Self::WHITE + Self::KO_PLANE + p] = 0;
        }
      }
      _ => {}
    }
  }

  fn update_points_rank(features: &mut [u8], position: &TxnPosition, turn: Stone, rank: PlayerRank) {
    let level = match rank {
      PlayerRank::Kyu(_) => 0,
      PlayerRank::AmaDan(1) |
      PlayerRank::AmaDan(2) => 1,
      PlayerRank::AmaDan(3) |
      PlayerRank::AmaDan(4) => 2,
      PlayerRank::AmaDan(5) |
      PlayerRank::AmaDan(6) |
      PlayerRank::AmaDan(7) |
      PlayerRank::AmaDan(8) |
      PlayerRank::AmaDan(9) => 3,
      PlayerRank::Dan(1) |
      PlayerRank::Dan(2) |
      PlayerRank::Dan(3) => 4,
      PlayerRank::Dan(4) |
      PlayerRank::Dan(5) |
      PlayerRank::Dan(6) => 5,
      PlayerRank::Dan(7) |
      PlayerRank::Dan(8) |
      PlayerRank::Dan(9) => 6,
      PlayerRank::AmaDan(_) | PlayerRank::Dan(_) => unimplemented!(),
    };
    let offset = match turn {
      Stone::Black => Self::BLACK,
      Stone::White => Self::WHITE,
      _ => unimplemented!(),
    };
    for p in 0 .. Board::SIZE {
      if level >= 1 {
        features[offset + Self::RANK_DAN_1_2_PLANE + p] = Self::SET;
      }
      if level >= 2 {
        features[offset + Self::RANK_DAN_3_4_PLANE + p] = Self::SET;
      }
      if level >= 3 {
        features[offset + Self::RANK_DAN_5_6_PLANE + p] = Self::SET;
      }
      if level >= 4 {
        features[offset + Self::RANK_PRO_1_3_PLANE + p] = Self::SET;
      }
      if level >= 5 {
        features[offset + Self::RANK_PRO_4_6_PLANE + p] = Self::SET;
      }
      if level >= 6 {
        features[offset + Self::RANK_PRO_7_9_PLANE + p] = Self::SET;
      }
    }
  }
}

impl TxnStateData for TxnStateExtLibFeatsData {
  fn reset(&mut self) {
    for p in 0 .. Self::GRAY {
      self.features[p] = 0;
    }
    for p in Self::GRAY .. Self::GRAY + Self::BASELINE_PLANE {
      self.features[p] = 0;
    }
    for p in Self::GRAY + Self::BASELINE_PLANE .. Self::GRAY + Self::NUM_GRAY_PLANES * Board::SIZE {
      self.features[p] = Self::SET;
    }
    self.prev_moves[0] = None;
    self.prev_moves[1] = None;
    self.prev_moves[2] = None;
    self.black_rank = None;
    self.white_rank = None;
  }

  fn update(&mut self, position: &TxnPosition, chains: &TxnChainsList, update_turn: Stone, update_action: Action) {
    // TODO(20151124): for sparse features, how to compute diffs efficiently?

    {
      // XXX(20151107): Iterate over placed and killed chains to update stone
      // repr.
      let &mut TxnStateExtLibFeatsData{
        ref mut features, ref mut tmp_mark, .. } = self;

      // Update stone features.
      if let Some((stone, point)) = position.last_placed {
        let p = point.idx();
        match stone {
          Stone::Black => {
            features[Self::BLACK + Self::STONE_PLANE + p] = Self::SET;
            features[Self::WHITE + Self::STONE_PLANE + p] = 0;
          }
          Stone::White => {
            features[Self::BLACK + Self::STONE_PLANE + p] = 0;
            features[Self::WHITE + Self::STONE_PLANE + p] = Self::SET;
          }
          Stone::Empty => { unreachable!(); }
        }
      }
      for &kill_point in position.last_killed[0].iter().chain(position.last_killed[1].iter()) {
        let p = kill_point.idx();
        features[Self::BLACK + Self::STONE_PLANE + p] = 0;
        features[Self::WHITE + Self::STONE_PLANE + p] = 0;
      }

      // XXX(20151123): Update liberty features by iterating over placed,
      // adjacent, ko, captured stones, and pseudolibs; see TxnStateLegalityData.
      tmp_mark.clear();
      if let Some((_, place_point)) = position.last_placed {
        Self::update_point_libs(tmp_mark, features, position, chains, place_point);
        for_each_adjacent(place_point, |adj_point| {
          Self::update_point_libs(tmp_mark, features, position, chains, adj_point);
        });
      }
      if let Some((_, ko_point)) = position.ko {
        Self::update_point_libs(tmp_mark, features, position, chains, ko_point);
      }
      if let Some((_, prev_ko_point)) = position.prev_ko {
        Self::update_point_libs(tmp_mark, features, position, chains, prev_ko_point);
      }
      for &kill_point in position.last_killed[0].iter().chain(position.last_killed[1].iter()) {
        Self::update_point_libs(tmp_mark, features, position, chains, kill_point);
      }
      for &head in chains.last_mut_heads.iter() {
        let mut is_valid_head = false;
        if let Some(chain) = chains.get_chain(head) {
          is_valid_head = true;
        }
        if is_valid_head {
          chains.iter_chain(head, |ch_pt| {
            Self::update_point_libs(tmp_mark, features, position, chains, ch_pt);
          });
        }
      }

      // Update the ko features.
      if let Some((prev_ko_turn, prev_ko_point)) = position.prev_ko {
        Self::update_point_ko(features, position, prev_ko_turn, prev_ko_point, false);
      }
      if let Some((ko_turn, ko_point)) = position.ko {
        Self::update_point_ko(features, position, ko_turn, ko_point, true);
      }

      // XXX(20160102): Update previous moves.
      match update_action {
        Action::Place{point} => {
          let p = point.idx();
          features[Self::GRAY + Self::PREV_1_PLANE + p] = Self::SET;
        }
        _ => {}
      }
      match self.prev_moves[0] {
        Some((_, prev_point)) => {
          let p = prev_point.idx();
          features[Self::GRAY + Self::PREV_1_PLANE + p] = 0;
          features[Self::GRAY + Self::PREV_2_PLANE + p] = Self::SET;
        }
        None => {}
      }
      match self.prev_moves[1] {
        Some((_, prev_point)) => {
          let p = prev_point.idx();
          features[Self::GRAY + Self::PREV_2_PLANE + p] = 0;
          features[Self::GRAY + Self::PREV_3_PLANE + p] = Self::SET;
        }
        None => {}
      }
      match self.prev_moves[2] {
        Some((_, prev_point)) => {
          let p = prev_point.idx();
          features[Self::GRAY + Self::PREV_3_PLANE + p] = 0;
        }
        None => {}
      }
      self.prev_moves[2] = self.prev_moves[1];
      self.prev_moves[1] = self.prev_moves[0];
      self.prev_moves[0] = match update_action {
        Action::Place{point} => Some((update_turn, point)),
        _ => None,
      };

      // XXX(20160103): Update player rank features.
      match self.black_rank {
        None => {
          Self::update_points_rank(features, position, update_turn, position.ranks[0]);
          self.black_rank = Some(position.ranks[0]);
        }
        Some(rank) => {
          if rank != position.ranks[0] {
            Self::update_points_rank(features, position, update_turn, position.ranks[0]);
            self.black_rank = Some(position.ranks[0]);
          }
        }
      }
      match self.white_rank {
        None => {
          Self::update_points_rank(features, position, update_turn, position.ranks[1]);
          self.white_rank = Some(position.ranks[1]);
        }
        Some(rank) => {
          if rank != position.ranks[1] {
            Self::update_points_rank(features, position, update_turn, position.ranks[1]);
            self.white_rank = Some(position.ranks[1]);
          }
        }
      }
    }
  }
}

#[derive(Clone)]
pub struct TxnStateAlphaFeatsV1Data {
  features:     Vec<u8>,
  prev_moves:   Vec<Option<(Stone, Point)>>,
  /*black_rank:   Option<PlayerRank>,
  white_rank:   Option<PlayerRank>,*/
  tmp_mark:     BitSet,
}

impl TxnStateAlphaFeatsV1Data {
  pub const SET: u8 = 1;

  pub const NUM_PLANES:         usize = 37;

  pub const BASELINE_PLANE:     usize = 0 * Board::SIZE;
  pub const EMPTY_PLANE:        usize = 1 * Board::SIZE;
  pub const BLACK_PLANE:        usize = 2 * Board::SIZE;
  pub const WHITE_PLANE:        usize = 3 * Board::SIZE;
  pub const KO_PLANE:           usize = 4 * Board::SIZE;
  pub const TURNS_1_PLANE:      usize = 5 * Board::SIZE;
  pub const TURNS_2_PLANE:      usize = 6 * Board::SIZE;
  pub const TURNS_3_PLANE:      usize = 7 * Board::SIZE;
  pub const TURNS_4_PLANE:      usize = 8 * Board::SIZE;
  pub const TURNS_5_PLANE:      usize = 9 * Board::SIZE;
  pub const TURNS_6_PLANE:      usize = 10 * Board::SIZE;
  pub const TURNS_7_PLANE:      usize = 11 * Board::SIZE;
  pub const TURNS_8_PLANE:      usize = 12 * Board::SIZE;
  pub const LIBS_1_PLANE:       usize = 13 * Board::SIZE;
  pub const LIBS_2_PLANE:       usize = 14 * Board::SIZE;
  pub const LIBS_3_PLANE:       usize = 15 * Board::SIZE;
  pub const LIBS_4_PLANE:       usize = 16 * Board::SIZE;
  pub const LIBS_5_PLANE:       usize = 17 * Board::SIZE;
  pub const LIBS_6_PLANE:       usize = 18 * Board::SIZE;
  pub const LIBS_7_PLANE:       usize = 19 * Board::SIZE;
  pub const LIBS_8_PLANE:       usize = 20 * Board::SIZE;
  pub const CSUIS_1_PLANE:      usize = 21 * Board::SIZE;
  pub const CSUIS_2_PLANE:      usize = 22 * Board::SIZE;
  pub const CSUIS_3_PLANE:      usize = 23 * Board::SIZE;
  pub const CSUIS_4_PLANE:      usize = 24 * Board::SIZE;
  pub const CSUIS_5_PLANE:      usize = 25 * Board::SIZE;
  pub const CSUIS_6_PLANE:      usize = 26 * Board::SIZE;
  pub const CSUIS_7_PLANE:      usize = 27 * Board::SIZE;
  pub const CSUIS_8_PLANE:      usize = 28 * Board::SIZE;
  pub const CCAPS_1_PLANE:      usize = 29 * Board::SIZE;
  pub const CCAPS_2_PLANE:      usize = 30 * Board::SIZE;
  pub const CCAPS_3_PLANE:      usize = 31 * Board::SIZE;
  pub const CCAPS_4_PLANE:      usize = 32 * Board::SIZE;
  pub const CCAPS_5_PLANE:      usize = 33 * Board::SIZE;
  pub const CCAPS_6_PLANE:      usize = 34 * Board::SIZE;
  pub const CCAPS_7_PLANE:      usize = 35 * Board::SIZE;
  pub const CCAPS_8_PLANE:      usize = 36 * Board::SIZE;

  pub fn new() -> TxnStateAlphaFeatsV1Data {
    let feats = TxnStateAlphaFeatsV1Data{
      features:     repeat(0).take(Self::NUM_PLANES * Board::SIZE).collect(),
      prev_moves:   vec![None, None, None, None, None, None, None, None],
      tmp_mark:     BitSet::with_capacity(Board::SIZE),
    };
    assert_eq!(
        feats.features.len(),
        Self::NUM_PLANES * Board::SIZE);
    feats
  }

  pub fn current_feature(&self, plane_idx: usize, point: Point) -> u8 {
    self.features[plane_idx * Board::SIZE + point.idx()]
  }

  pub fn feature_dims(&self) -> (usize, usize, usize) {
    (Board::DIM, Board::DIM, Self::NUM_PLANES)
  }

  pub fn extract_relative_features(&self, turn: Stone, dst_buf: &mut [u8]) {
    match turn {
      Stone::Black => {
        copy_memory(
            &self.features,
            dst_buf,
        );
      }
      Stone::White => {
        copy_memory(
            &self.features[ .. Self::BLACK_PLANE],
            &mut dst_buf[ .. Self::BLACK_PLANE],
        );
        copy_memory(
            &self.features[Self::BLACK_PLANE .. Self::WHITE_PLANE],
            &mut dst_buf[Self::WHITE_PLANE .. Self::KO_PLANE],
        );
        copy_memory(
            &self.features[Self::WHITE_PLANE .. Self::KO_PLANE],
            &mut dst_buf[Self::BLACK_PLANE .. Self::WHITE_PLANE],
        );
        copy_memory(
            &self.features[Self::KO_PLANE .. ],
            &mut dst_buf[Self::KO_PLANE .. ],
        );
      }
      _ => unreachable!(),
    }
  }

  fn update_point_turns(features: &mut [u8], point: Point, turns: usize) {
    let p = point.idx();
    match turns {
      1 => {
        features[Self::TURNS_1_PLANE + p] = Self::SET;
      }
      2 => {
        features[Self::TURNS_1_PLANE + p] = 0;
        features[Self::TURNS_2_PLANE + p] = Self::SET;
      }
      3 => {
        features[Self::TURNS_2_PLANE + p] = 0;
        features[Self::TURNS_3_PLANE + p] = Self::SET;
      }
      4 => {
        features[Self::TURNS_3_PLANE + p] = 0;
        features[Self::TURNS_4_PLANE + p] = Self::SET;
      }
      5 => {
        features[Self::TURNS_4_PLANE + p] = 0;
        features[Self::TURNS_5_PLANE + p] = Self::SET;
      }
      6 => {
        features[Self::TURNS_5_PLANE + p] = 0;
        features[Self::TURNS_6_PLANE + p] = Self::SET;
      }
      7 => {
        features[Self::TURNS_6_PLANE + p] = 0;
        features[Self::TURNS_7_PLANE + p] = Self::SET;
      }
      8 => {
        features[Self::TURNS_7_PLANE + p] = 0;
        features[Self::TURNS_8_PLANE + p] = Self::SET;
      }
      /*9 => {
        features[Self::TURNS_8_PLANE + p] = 0;
      }*/
      _ => unreachable!(),
    }
  }

  fn update_point_libs(tmp_mark: &mut BitSet, features: &mut [u8], position: &TxnPosition, chains: &TxnChainsList, point: Point, turn: Stone) {
    let p = point.idx();
    if tmp_mark.contains(&p) {
      return;
    }
    let stone = position.stones[p];
    if stone != Stone::Empty {
      let head = chains.find_chain(point);
      //assert!(head != TOMBSTONE);
      let chain = chains.get_chain(head).unwrap();

      // Liberty features.
      let mut is_atari = false;
      match chain.count_libs_up_to_8() {
        0 => { unreachable!(); }
        1 => {
          is_atari = true;
          features[Self::LIBS_1_PLANE + p] = Self::SET;
          features[Self::LIBS_2_PLANE + p] = 0;
          features[Self::LIBS_3_PLANE + p] = 0;
          features[Self::LIBS_4_PLANE + p] = 0;
          features[Self::LIBS_5_PLANE + p] = 0;
          features[Self::LIBS_6_PLANE + p] = 0;
          features[Self::LIBS_7_PLANE + p] = 0;
          features[Self::LIBS_8_PLANE + p] = 0;
        }
        2 => {
          features[Self::LIBS_1_PLANE + p] = 0;
          features[Self::LIBS_2_PLANE + p] = Self::SET;
          features[Self::LIBS_3_PLANE + p] = 0;
          features[Self::LIBS_4_PLANE + p] = 0;
          features[Self::LIBS_5_PLANE + p] = 0;
          features[Self::LIBS_6_PLANE + p] = 0;
          features[Self::LIBS_7_PLANE + p] = 0;
          features[Self::LIBS_8_PLANE + p] = 0;
        }
        3 => {
          features[Self::LIBS_1_PLANE + p] = 0;
          features[Self::LIBS_2_PLANE + p] = 0;
          features[Self::LIBS_3_PLANE + p] = Self::SET;
          features[Self::LIBS_4_PLANE + p] = 0;
          features[Self::LIBS_5_PLANE + p] = 0;
          features[Self::LIBS_6_PLANE + p] = 0;
          features[Self::LIBS_7_PLANE + p] = 0;
          features[Self::LIBS_8_PLANE + p] = 0;
        }
        4 => {
          features[Self::LIBS_1_PLANE + p] = 0;
          features[Self::LIBS_2_PLANE + p] = 0;
          features[Self::LIBS_3_PLANE + p] = 0;
          features[Self::LIBS_4_PLANE + p] = Self::SET;
          features[Self::LIBS_5_PLANE + p] = 0;
          features[Self::LIBS_6_PLANE + p] = 0;
          features[Self::LIBS_7_PLANE + p] = 0;
          features[Self::LIBS_8_PLANE + p] = 0;
        }
        5 => {
          features[Self::LIBS_1_PLANE + p] = 0;
          features[Self::LIBS_2_PLANE + p] = 0;
          features[Self::LIBS_3_PLANE + p] = 0;
          features[Self::LIBS_4_PLANE + p] = 0;
          features[Self::LIBS_5_PLANE + p] = Self::SET;
          features[Self::LIBS_6_PLANE + p] = 0;
          features[Self::LIBS_7_PLANE + p] = 0;
          features[Self::LIBS_8_PLANE + p] = 0;
        }
        6 => {
          features[Self::LIBS_1_PLANE + p] = 0;
          features[Self::LIBS_2_PLANE + p] = 0;
          features[Self::LIBS_3_PLANE + p] = 0;
          features[Self::LIBS_4_PLANE + p] = 0;
          features[Self::LIBS_5_PLANE + p] = 0;
          features[Self::LIBS_6_PLANE + p] = Self::SET;
          features[Self::LIBS_7_PLANE + p] = 0;
          features[Self::LIBS_8_PLANE + p] = 0;
        }
        7 => {
          features[Self::LIBS_1_PLANE + p] = 0;
          features[Self::LIBS_2_PLANE + p] = 0;
          features[Self::LIBS_3_PLANE + p] = 0;
          features[Self::LIBS_4_PLANE + p] = 0;
          features[Self::LIBS_5_PLANE + p] = 0;
          features[Self::LIBS_6_PLANE + p] = 0;
          features[Self::LIBS_7_PLANE + p] = Self::SET;
          features[Self::LIBS_8_PLANE + p] = 0;
        }
        8 => {
          features[Self::LIBS_1_PLANE + p] = 0;
          features[Self::LIBS_2_PLANE + p] = 0;
          features[Self::LIBS_3_PLANE + p] = 0;
          features[Self::LIBS_4_PLANE + p] = 0;
          features[Self::LIBS_5_PLANE + p] = 0;
          features[Self::LIBS_6_PLANE + p] = 0;
          features[Self::LIBS_7_PLANE + p] = 0;
          features[Self::LIBS_8_PLANE + p] = Self::SET;
        }
        _ => { unreachable!(); }
      }

      // Suicide features.
      if stone == turn {
        if is_atari {
          match chain.count_length() {
            0 => unreachable!(),
            1 => {
              features[Self::CSUIS_1_PLANE + p] = Self::SET;
              features[Self::CSUIS_2_PLANE + p] = 0;
              features[Self::CSUIS_3_PLANE + p] = 0;
              features[Self::CSUIS_4_PLANE + p] = 0;
              features[Self::CSUIS_5_PLANE + p] = 0;
              features[Self::CSUIS_6_PLANE + p] = 0;
              features[Self::CSUIS_7_PLANE + p] = 0;
              features[Self::CSUIS_8_PLANE + p] = 0;
            }
            2 => {
              features[Self::CSUIS_1_PLANE + p] = 0;
              features[Self::CSUIS_2_PLANE + p] = Self::SET;
              features[Self::CSUIS_3_PLANE + p] = 0;
              features[Self::CSUIS_4_PLANE + p] = 0;
              features[Self::CSUIS_5_PLANE + p] = 0;
              features[Self::CSUIS_6_PLANE + p] = 0;
              features[Self::CSUIS_7_PLANE + p] = 0;
              features[Self::CSUIS_8_PLANE + p] = 0;
            }
            3 => {
              features[Self::CSUIS_1_PLANE + p] = 0;
              features[Self::CSUIS_2_PLANE + p] = 0;
              features[Self::CSUIS_3_PLANE + p] = Self::SET;
              features[Self::CSUIS_4_PLANE + p] = 0;
              features[Self::CSUIS_5_PLANE + p] = 0;
              features[Self::CSUIS_6_PLANE + p] = 0;
              features[Self::CSUIS_7_PLANE + p] = 0;
              features[Self::CSUIS_8_PLANE + p] = 0;
            }
            4 => {
              features[Self::CSUIS_1_PLANE + p] = 0;
              features[Self::CSUIS_2_PLANE + p] = 0;
              features[Self::CSUIS_3_PLANE + p] = 0;
              features[Self::CSUIS_4_PLANE + p] = Self::SET;
              features[Self::CSUIS_5_PLANE + p] = 0;
              features[Self::CSUIS_6_PLANE + p] = 0;
              features[Self::CSUIS_7_PLANE + p] = 0;
              features[Self::CSUIS_8_PLANE + p] = 0;
            }
            5 => {
              features[Self::CSUIS_1_PLANE + p] = 0;
              features[Self::CSUIS_2_PLANE + p] = 0;
              features[Self::CSUIS_3_PLANE + p] = 0;
              features[Self::CSUIS_4_PLANE + p] = 0;
              features[Self::CSUIS_5_PLANE + p] = Self::SET;
              features[Self::CSUIS_6_PLANE + p] = 0;
              features[Self::CSUIS_7_PLANE + p] = 0;
              features[Self::CSUIS_8_PLANE + p] = 0;
            }
            6 => {
              features[Self::CSUIS_1_PLANE + p] = 0;
              features[Self::CSUIS_2_PLANE + p] = 0;
              features[Self::CSUIS_3_PLANE + p] = 0;
              features[Self::CSUIS_4_PLANE + p] = 0;
              features[Self::CSUIS_5_PLANE + p] = 0;
              features[Self::CSUIS_6_PLANE + p] = Self::SET;
              features[Self::CSUIS_7_PLANE + p] = 0;
              features[Self::CSUIS_8_PLANE + p] = 0;
            }
            7 => {
              features[Self::CSUIS_1_PLANE + p] = 0;
              features[Self::CSUIS_2_PLANE + p] = 0;
              features[Self::CSUIS_3_PLANE + p] = 0;
              features[Self::CSUIS_4_PLANE + p] = 0;
              features[Self::CSUIS_5_PLANE + p] = 0;
              features[Self::CSUIS_6_PLANE + p] = 0;
              features[Self::CSUIS_7_PLANE + p] = Self::SET;
              features[Self::CSUIS_8_PLANE + p] = 0;
            }
            n => {
              features[Self::CSUIS_1_PLANE + p] = 0;
              features[Self::CSUIS_2_PLANE + p] = 0;
              features[Self::CSUIS_3_PLANE + p] = 0;
              features[Self::CSUIS_4_PLANE + p] = 0;
              features[Self::CSUIS_5_PLANE + p] = 0;
              features[Self::CSUIS_6_PLANE + p] = 0;
              features[Self::CSUIS_7_PLANE + p] = 0;
              features[Self::CSUIS_8_PLANE + p] = Self::SET;
            }
          }
        } else {
          features[Self::CSUIS_1_PLANE + p] = 0;
          features[Self::CSUIS_2_PLANE + p] = 0;
          features[Self::CSUIS_3_PLANE + p] = 0;
          features[Self::CSUIS_4_PLANE + p] = 0;
          features[Self::CSUIS_5_PLANE + p] = 0;
          features[Self::CSUIS_6_PLANE + p] = 0;
          features[Self::CSUIS_7_PLANE + p] = 0;
          features[Self::CSUIS_8_PLANE + p] = 0;
        }
      }

      // Capture features.
      if stone == turn.opponent() {
        if is_atari {
          match chain.count_length() {
            0 => unreachable!(),
            1 => {
              features[Self::CCAPS_1_PLANE + p] = Self::SET;
              features[Self::CCAPS_2_PLANE + p] = 0;
              features[Self::CCAPS_3_PLANE + p] = 0;
              features[Self::CCAPS_4_PLANE + p] = 0;
              features[Self::CCAPS_5_PLANE + p] = 0;
              features[Self::CCAPS_6_PLANE + p] = 0;
              features[Self::CCAPS_7_PLANE + p] = 0;
              features[Self::CCAPS_8_PLANE + p] = 0;
            }
            2 => {
              features[Self::CCAPS_1_PLANE + p] = 0;
              features[Self::CCAPS_2_PLANE + p] = Self::SET;
              features[Self::CCAPS_3_PLANE + p] = 0;
              features[Self::CCAPS_4_PLANE + p] = 0;
              features[Self::CCAPS_5_PLANE + p] = 0;
              features[Self::CCAPS_6_PLANE + p] = 0;
              features[Self::CCAPS_7_PLANE + p] = 0;
              features[Self::CCAPS_8_PLANE + p] = 0;
            }
            3 => {
              features[Self::CCAPS_1_PLANE + p] = 0;
              features[Self::CCAPS_2_PLANE + p] = 0;
              features[Self::CCAPS_3_PLANE + p] = Self::SET;
              features[Self::CCAPS_4_PLANE + p] = 0;
              features[Self::CCAPS_5_PLANE + p] = 0;
              features[Self::CCAPS_6_PLANE + p] = 0;
              features[Self::CCAPS_7_PLANE + p] = 0;
              features[Self::CCAPS_8_PLANE + p] = 0;
            }
            4 => {
              features[Self::CCAPS_1_PLANE + p] = 0;
              features[Self::CCAPS_2_PLANE + p] = 0;
              features[Self::CCAPS_3_PLANE + p] = 0;
              features[Self::CCAPS_4_PLANE + p] = Self::SET;
              features[Self::CCAPS_5_PLANE + p] = 0;
              features[Self::CCAPS_6_PLANE + p] = 0;
              features[Self::CCAPS_7_PLANE + p] = 0;
              features[Self::CCAPS_8_PLANE + p] = 0;
            }
            5 => {
              features[Self::CCAPS_1_PLANE + p] = 0;
              features[Self::CCAPS_2_PLANE + p] = 0;
              features[Self::CCAPS_3_PLANE + p] = 0;
              features[Self::CCAPS_4_PLANE + p] = 0;
              features[Self::CCAPS_5_PLANE + p] = Self::SET;
              features[Self::CCAPS_6_PLANE + p] = 0;
              features[Self::CCAPS_7_PLANE + p] = 0;
              features[Self::CCAPS_8_PLANE + p] = 0;
            }
            6 => {
              features[Self::CCAPS_1_PLANE + p] = 0;
              features[Self::CCAPS_2_PLANE + p] = 0;
              features[Self::CCAPS_3_PLANE + p] = 0;
              features[Self::CCAPS_4_PLANE + p] = 0;
              features[Self::CCAPS_5_PLANE + p] = 0;
              features[Self::CCAPS_6_PLANE + p] = Self::SET;
              features[Self::CCAPS_7_PLANE + p] = 0;
              features[Self::CCAPS_8_PLANE + p] = 0;
            }
            7 => {
              features[Self::CCAPS_1_PLANE + p] = 0;
              features[Self::CCAPS_2_PLANE + p] = 0;
              features[Self::CCAPS_3_PLANE + p] = 0;
              features[Self::CCAPS_4_PLANE + p] = 0;
              features[Self::CCAPS_5_PLANE + p] = 0;
              features[Self::CCAPS_6_PLANE + p] = 0;
              features[Self::CCAPS_7_PLANE + p] = Self::SET;
              features[Self::CCAPS_8_PLANE + p] = 0;
            }
            n => {
              features[Self::CCAPS_1_PLANE + p] = 0;
              features[Self::CCAPS_2_PLANE + p] = 0;
              features[Self::CCAPS_3_PLANE + p] = 0;
              features[Self::CCAPS_4_PLANE + p] = 0;
              features[Self::CCAPS_5_PLANE + p] = 0;
              features[Self::CCAPS_6_PLANE + p] = 0;
              features[Self::CCAPS_7_PLANE + p] = 0;
              features[Self::CCAPS_8_PLANE + p] = Self::SET;
            }
          }
        } else {
          features[Self::CCAPS_1_PLANE + p] = 0;
          features[Self::CCAPS_2_PLANE + p] = 0;
          features[Self::CCAPS_3_PLANE + p] = 0;
          features[Self::CCAPS_4_PLANE + p] = 0;
          features[Self::CCAPS_5_PLANE + p] = 0;
          features[Self::CCAPS_6_PLANE + p] = 0;
          features[Self::CCAPS_7_PLANE + p] = 0;
          features[Self::CCAPS_8_PLANE + p] = 0;
        }
      }

    } else {
      features[Self::LIBS_1_PLANE + p] = 0;
      features[Self::LIBS_2_PLANE + p] = 0;
      features[Self::LIBS_3_PLANE + p] = 0;
      features[Self::LIBS_4_PLANE + p] = 0;
      features[Self::LIBS_5_PLANE + p] = 0;
      features[Self::LIBS_6_PLANE + p] = 0;
      features[Self::LIBS_7_PLANE + p] = 0;
      features[Self::LIBS_8_PLANE + p] = 0;
      features[Self::CSUIS_1_PLANE + p] = 0;
      features[Self::CSUIS_2_PLANE + p] = 0;
      features[Self::CSUIS_3_PLANE + p] = 0;
      features[Self::CSUIS_4_PLANE + p] = 0;
      features[Self::CSUIS_5_PLANE + p] = 0;
      features[Self::CSUIS_6_PLANE + p] = 0;
      features[Self::CSUIS_7_PLANE + p] = 0;
      features[Self::CSUIS_8_PLANE + p] = 0;
      features[Self::CCAPS_1_PLANE + p] = 0;
      features[Self::CCAPS_2_PLANE + p] = 0;
      features[Self::CCAPS_3_PLANE + p] = 0;
      features[Self::CCAPS_4_PLANE + p] = 0;
      features[Self::CCAPS_5_PLANE + p] = 0;
      features[Self::CCAPS_6_PLANE + p] = 0;
      features[Self::CCAPS_7_PLANE + p] = 0;
      features[Self::CCAPS_8_PLANE + p] = 0;
    }
    tmp_mark.insert(p);
  }

  fn update_point_ko(features: &mut [u8], position: &TxnPosition, ko_turn: Stone, ko_point: Point, is_ko: bool) {
    let p = ko_point.idx();
    if is_ko {
      features[Self::KO_PLANE + p] = Self::SET;
    } else {
      features[Self::KO_PLANE + p] = 0;
    }
  }
}

impl TxnStateData for TxnStateAlphaFeatsV1Data {
  fn reset(&mut self) {
    for p in 0 .. Self::BLACK_PLANE {
      self.features[p] = Self::SET;
    }
    for p in Self::BLACK_PLANE .. Self::NUM_PLANES * Board::SIZE {
      self.features[p] = 0;
    }
    self.prev_moves[0] = None;
    self.prev_moves[1] = None;
    self.prev_moves[2] = None;
    self.prev_moves[3] = None;
    self.prev_moves[4] = None;
    self.prev_moves[5] = None;
    self.prev_moves[6] = None;
    self.prev_moves[7] = None;
  }

  fn update(&mut self, position: &TxnPosition, chains: &TxnChainsList, update_turn: Stone, update_action: Action) {
    {
      // XXX(20151107): Iterate over placed and killed chains to update stone
      // repr.
      let &mut TxnStateAlphaFeatsV1Data{
        ref mut features, ref mut tmp_mark, .. } = self;

      // Update stone features.
      if let Some((stone, point)) = position.last_placed {
        let p = point.idx();
        match stone {
          Stone::Black => {
            features[Self::EMPTY_PLANE + p] = 0;
            features[Self::BLACK_PLANE + p] = Self::SET;
            features[Self::WHITE_PLANE + p] = 0;
          }
          Stone::White => {
            features[Self::EMPTY_PLANE + p] = 0;
            features[Self::BLACK_PLANE + p] = 0;
            features[Self::WHITE_PLANE + p] = Self::SET;
          }
          Stone::Empty => { unreachable!(); }
        }
      }
      for &kill_point in position.last_killed[0].iter().chain(position.last_killed[1].iter()) {
        let p = kill_point.idx();
        features[Self::EMPTY_PLANE + p] = Self::SET;
        features[Self::BLACK_PLANE + p] = 0;
        features[Self::WHITE_PLANE + p] = 0;
      }

      // XXX(20160102): Update previous moves.
      match self.prev_moves[7] {
        Some((_, prev_point)) => {
          // XXX(20160129): Do not erase the oldest moves.
          //Self::update_point_turns(features, prev_point, 9);
        }
        None => {}
      }
      match self.prev_moves[6] {
        Some((_, prev_point)) => {
          Self::update_point_turns(features, prev_point, 8);
        }
        None => {}
      }
      match self.prev_moves[5] {
        Some((_, prev_point)) => {
          Self::update_point_turns(features, prev_point, 7);
        }
        None => {}
      }
      match self.prev_moves[4] {
        Some((_, prev_point)) => {
          Self::update_point_turns(features, prev_point, 6);
        }
        None => {}
      }
      match self.prev_moves[3] {
        Some((_, prev_point)) => {
          Self::update_point_turns(features, prev_point, 5);
        }
        None => {}
      }
      match self.prev_moves[2] {
        Some((_, prev_point)) => {
          Self::update_point_turns(features, prev_point, 4);
        }
        None => {}
      }
      match self.prev_moves[1] {
        Some((_, prev_point)) => {
          Self::update_point_turns(features, prev_point, 3);
        }
        None => {}
      }
      match self.prev_moves[0] {
        Some((_, prev_point)) => {
          Self::update_point_turns(features, prev_point, 2);
        }
        None => {}
      }
      match update_action {
        Action::Place{point} => {
          Self::update_point_turns(features, point, 1);
        }
        _ => {}
      }
      self.prev_moves[7] = self.prev_moves[6];
      self.prev_moves[6] = self.prev_moves[5];
      self.prev_moves[5] = self.prev_moves[4];
      self.prev_moves[4] = self.prev_moves[3];
      self.prev_moves[3] = self.prev_moves[2];
      self.prev_moves[2] = self.prev_moves[1];
      self.prev_moves[1] = self.prev_moves[0];
      self.prev_moves[0] = match update_action {
        Action::Place{point} => Some((update_turn, point)),
        _ => None,
      };

      // XXX(20160127): Update liberty, capture, and suicide features by
      // iterating over placed, adjacent, ko, captured stones, and pseudolibs;
      // see TxnStateLegalityData.
      tmp_mark.clear();
      if let Some((_, place_point)) = position.last_placed {
        Self::update_point_libs(tmp_mark, features, position, chains, place_point, update_turn);
        for_each_adjacent(place_point, |adj_point| {
          Self::update_point_libs(tmp_mark, features, position, chains, adj_point, update_turn);
        });
      }
      if let Some((_, ko_point)) = position.ko {
        Self::update_point_libs(tmp_mark, features, position, chains, ko_point, update_turn);
      }
      if let Some((_, prev_ko_point)) = position.prev_ko {
        Self::update_point_libs(tmp_mark, features, position, chains, prev_ko_point, update_turn);
      }
      for &kill_point in position.last_killed[0].iter().chain(position.last_killed[1].iter()) {
        Self::update_point_libs(tmp_mark, features, position, chains, kill_point, update_turn);
      }
      for &head in chains.last_mut_heads.iter() {
        let mut is_valid_head = false;
        if let Some(chain) = chains.get_chain(head) {
          is_valid_head = true;
        }
        if is_valid_head {
          chains.iter_chain(head, |ch_pt| {
            Self::update_point_libs(tmp_mark, features, position, chains, ch_pt, update_turn);
          });
        }
      }

      // Update the ko features.
      if let Some((prev_ko_turn, prev_ko_point)) = position.prev_ko {
        Self::update_point_ko(features, position, prev_ko_turn, prev_ko_point, false);
      }
      if let Some((ko_turn, ko_point)) = position.ko {
        Self::update_point_ko(features, position, ko_turn, ko_point, true);
      }
    }
  }
}
