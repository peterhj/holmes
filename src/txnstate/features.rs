use board::{Board, PlayerRank, Stone, Point, Action};
use txnstate::{
  TxnStateData, TxnState, TxnPosition, TxnChainsList,
  for_each_adjacent, for_each_diagonal, check_good_move_fast,
};
use txnstate::extras::{TxnStateLegalityData, TxnStateNodeData};
use util::{slice_twice_mut};

use array_new::{NdArraySerialize, Array3d, BitArray3d};

use bit_set::{BitSet};
use std::cell::{RefCell};
use std::iter::{repeat};
use std::slice::bytes::{copy_memory};

pub trait TxnStateFeatures: TxnStateData {
  fn extract_relative_features(&self, turn: Stone, dst_buf: &mut [u8]);
}

#[derive(Clone, RustcDecodable, RustcEncodable)]
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

#[derive(Clone, RustcDecodable, RustcEncodable)]
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
    if tmp_mark.contains(p) {
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

#[derive(Clone, RustcDecodable, RustcEncodable)]
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
    if tmp_mark.contains(p) {
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
      PlayerRank::Ama(1) |
      PlayerRank::Ama(2) => 1,
      PlayerRank::Ama(3) |
      PlayerRank::Ama(4) => 2,
      PlayerRank::Ama(5) |
      PlayerRank::Ama(6) |
      PlayerRank::Ama(7) |
      PlayerRank::Ama(8) |
      PlayerRank::Ama(9) => 3,
      PlayerRank::Dan(1) |
      PlayerRank::Dan(2) |
      PlayerRank::Dan(3) => 4,
      PlayerRank::Dan(4) |
      PlayerRank::Dan(5) |
      PlayerRank::Dan(6) => 5,
      PlayerRank::Dan(7) |
      PlayerRank::Dan(8) |
      PlayerRank::Dan(9) => 6,
      PlayerRank::Ama(_) | PlayerRank::Dan(_) => unimplemented!(),
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

#[derive(Clone, RustcDecodable, RustcEncodable)]
pub struct TxnStateAlphaFeatsV1Data {
  features:     Vec<u8>,
  prev_moves:   Vec<Option<(Stone, Point)>>,
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
    if tmp_mark.contains(p) {
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

#[derive(Clone, RustcDecodable, RustcEncodable)]
pub struct TxnStateAlphaFeatsV2Data {
  //pub legality: TxnStateLegalityData,

  features:     Vec<u8>,
  prev_moves:   Vec<Option<(Stone, Point)>>,
  //init_ranks:   bool,
  black_rank:   Option<PlayerRank>,
  white_rank:   Option<PlayerRank>,
  tmp_mark:     BitSet,
}

impl TxnStateAlphaFeatsV2Data {
  // XXX(20160202): Notes on V2 of the Alpha-like features:
  // - Instead of raw one-hot encoding, use "isotonic" encoding where for a
  //   feature category (e.g., number of turns), if the category has value `k`,
  //   then set the feature for all k' <= k.
  // - Turns since features are "sticky".
  // - While both friend and opponent ranks are stored, only encode the opponent
  //   rank when extracting features (like TZ features).
  // - Encode a static "center distance" mask (like TZ features). This requires
  //   dense values (SET = 255).
  // - Encode in a sparse part (BitArray3d) and a dense part (Array3d<u8>).

  pub const SET: u8 = 255;

  pub const NUM_PLANES:         usize = 50;
  pub const NUM_EXTRACT_PLANES: usize = 44;
  pub const NUM_SPARSE_PLANES:  usize = 43;
  pub const NUM_DENSE_PLANES:   usize = 1;

  pub const BASELINE_PLANE:     usize = 0 * Board::SIZE;
  pub const EMPTY_PLANE:        usize = 1 * Board::SIZE;
  pub const BLACK_PLANE:        usize = 2 * Board::SIZE;
  pub const B_CAPS_1_PLANE:     usize = 3 * Board::SIZE;
  pub const B_CAPS_2_PLANE:     usize = 4 * Board::SIZE;
  pub const B_CAPS_3_PLANE:     usize = 5 * Board::SIZE;
  pub const B_CAPS_4_PLANE:     usize = 6 * Board::SIZE;
  pub const B_CAPS_5_PLANE:     usize = 7 * Board::SIZE;
  pub const B_CAPS_6_PLANE:     usize = 8 * Board::SIZE;
  pub const B_CAPS_7_PLANE:     usize = 9 * Board::SIZE;
  pub const B_CAPS_8_PLANE:     usize = 10 * Board::SIZE;
  pub const B_RANK_1A_2A_PLANE: usize = 11 * Board::SIZE;
  pub const B_RANK_3A_4A_PLANE: usize = 12 * Board::SIZE;
  pub const B_RANK_5A_7A_PLANE: usize = 13 * Board::SIZE;
  pub const B_RANK_1P_3P_PLANE: usize = 14 * Board::SIZE;
  pub const B_RANK_4P_6P_PLANE: usize = 15 * Board::SIZE;
  pub const B_RANK_7P_9P_PLANE: usize = 16 * Board::SIZE;
  pub const WHITE_PLANE:        usize = 17 * Board::SIZE;
  pub const W_CAPS_1_PLANE:     usize = 18 * Board::SIZE;
  pub const W_CAPS_2_PLANE:     usize = 19 * Board::SIZE;
  pub const W_CAPS_3_PLANE:     usize = 20 * Board::SIZE;
  pub const W_CAPS_4_PLANE:     usize = 21 * Board::SIZE;
  pub const W_CAPS_5_PLANE:     usize = 22 * Board::SIZE;
  pub const W_CAPS_6_PLANE:     usize = 23 * Board::SIZE;
  pub const W_CAPS_7_PLANE:     usize = 24 * Board::SIZE;
  pub const W_CAPS_8_PLANE:     usize = 25 * Board::SIZE;
  pub const W_RANK_1A_2A_PLANE: usize = 26 * Board::SIZE;
  pub const W_RANK_3A_4A_PLANE: usize = 27 * Board::SIZE;
  pub const W_RANK_5A_7A_PLANE: usize = 28 * Board::SIZE;
  pub const W_RANK_1P_3P_PLANE: usize = 29 * Board::SIZE;
  pub const W_RANK_4P_6P_PLANE: usize = 30 * Board::SIZE;
  pub const W_RANK_7P_9P_PLANE: usize = 31 * Board::SIZE;
  pub const TURNS_1_PLANE:      usize = 32 * Board::SIZE;
  pub const TURNS_2_PLANE:      usize = 33 * Board::SIZE;
  pub const TURNS_3_PLANE:      usize = 34 * Board::SIZE;
  pub const TURNS_4_PLANE:      usize = 35 * Board::SIZE;
  pub const TURNS_5_PLANE:      usize = 36 * Board::SIZE;
  pub const TURNS_6_PLANE:      usize = 37 * Board::SIZE;
  pub const TURNS_7_PLANE:      usize = 38 * Board::SIZE;
  pub const TURNS_8_PLANE:      usize = 39 * Board::SIZE;
  pub const LIBS_1_PLANE:       usize = 40 * Board::SIZE;
  pub const LIBS_2_PLANE:       usize = 41 * Board::SIZE;
  pub const LIBS_3_PLANE:       usize = 42 * Board::SIZE;
  pub const LIBS_4_PLANE:       usize = 43 * Board::SIZE;
  pub const LIBS_5_PLANE:       usize = 44 * Board::SIZE;
  pub const LIBS_6_PLANE:       usize = 45 * Board::SIZE;
  pub const LIBS_7_PLANE:       usize = 46 * Board::SIZE;
  pub const LIBS_8_PLANE:       usize = 47 * Board::SIZE;
  pub const KO_PLANE:           usize = 48 * Board::SIZE;
  pub const CENTER_PLANE:       usize = 49 * Board::SIZE;

  pub const BASELINE_EXT_PLANE:     usize = 0 * Board::SIZE;
  pub const EMPTY_EXT_PLANE:        usize = 1 * Board::SIZE;
  pub const FRIEND_EXT_PLANE:       usize = 2 * Board::SIZE;
  pub const F_CAPS_1_EXT_PLANE:     usize = 3 * Board::SIZE;
  pub const F_CAPS_2_EXT_PLANE:     usize = 4 * Board::SIZE;
  pub const F_CAPS_3_EXT_PLANE:     usize = 5 * Board::SIZE;
  pub const F_CAPS_4_EXT_PLANE:     usize = 6 * Board::SIZE;
  pub const F_CAPS_5_EXT_PLANE:     usize = 7 * Board::SIZE;
  pub const F_CAPS_6_EXT_PLANE:     usize = 8 * Board::SIZE;
  pub const F_CAPS_7_EXT_PLANE:     usize = 9 * Board::SIZE;
  pub const F_CAPS_8_EXT_PLANE:     usize = 10 * Board::SIZE;
  pub const ENEMY_EXT_PLANE:        usize = 11 * Board::SIZE;
  pub const E_CAPS_1_EXT_PLANE:     usize = 12 * Board::SIZE;
  pub const E_CAPS_2_EXT_PLANE:     usize = 13 * Board::SIZE;
  pub const E_CAPS_3_EXT_PLANE:     usize = 14 * Board::SIZE;
  pub const E_CAPS_4_EXT_PLANE:     usize = 15 * Board::SIZE;
  pub const E_CAPS_5_EXT_PLANE:     usize = 16 * Board::SIZE;
  pub const E_CAPS_6_EXT_PLANE:     usize = 17 * Board::SIZE;
  pub const E_CAPS_7_EXT_PLANE:     usize = 18 * Board::SIZE;
  pub const E_CAPS_8_EXT_PLANE:     usize = 19 * Board::SIZE;
  pub const E_RANK_1A_2A_EXT_PLANE: usize = 20 * Board::SIZE;
  pub const E_RANK_3A_4A_EXT_PLANE: usize = 21 * Board::SIZE;
  pub const E_RANK_5A_7A_EXT_PLANE: usize = 22 * Board::SIZE;
  pub const E_RANK_1P_3P_EXT_PLANE: usize = 23 * Board::SIZE;
  pub const E_RANK_4P_6P_EXT_PLANE: usize = 24 * Board::SIZE;
  pub const E_RANK_7P_9P_EXT_PLANE: usize = 25 * Board::SIZE;
  pub const TURNS_1_EXT_PLANE:      usize = 26 * Board::SIZE;
  pub const TURNS_2_EXT_PLANE:      usize = 27 * Board::SIZE;
  pub const TURNS_3_EXT_PLANE:      usize = 28 * Board::SIZE;
  pub const TURNS_4_EXT_PLANE:      usize = 29 * Board::SIZE;
  pub const TURNS_5_EXT_PLANE:      usize = 30 * Board::SIZE;
  pub const TURNS_6_EXT_PLANE:      usize = 31 * Board::SIZE;
  pub const TURNS_7_EXT_PLANE:      usize = 32 * Board::SIZE;
  pub const TURNS_8_EXT_PLANE:      usize = 33 * Board::SIZE;
  pub const LIBS_1_EXT_PLANE:       usize = 34 * Board::SIZE;
  pub const LIBS_2_EXT_PLANE:       usize = 35 * Board::SIZE;
  pub const LIBS_3_EXT_PLANE:       usize = 36 * Board::SIZE;
  pub const LIBS_4_EXT_PLANE:       usize = 37 * Board::SIZE;
  pub const LIBS_5_EXT_PLANE:       usize = 38 * Board::SIZE;
  pub const LIBS_6_EXT_PLANE:       usize = 39 * Board::SIZE;
  pub const LIBS_7_EXT_PLANE:       usize = 40 * Board::SIZE;
  pub const LIBS_8_EXT_PLANE:       usize = 41 * Board::SIZE;
  pub const KO_EXT_PLANE:           usize = 42 * Board::SIZE;
  pub const CENTER_EXT_PLANE:       usize = 43 * Board::SIZE;

  /*pub const TURNS_1_PLANE:      usize = 4 * Board::SIZE;
  pub const TURNS_2_PLANE:      usize = 5 * Board::SIZE;
  pub const TURNS_3_PLANE:      usize = 6 * Board::SIZE;
  pub const TURNS_4_PLANE:      usize = 7 * Board::SIZE;
  pub const TURNS_5_PLANE:      usize = 8 * Board::SIZE;
  pub const TURNS_6_PLANE:      usize = 9 * Board::SIZE;
  pub const TURNS_7_PLANE:      usize = 10 * Board::SIZE;
  pub const TURNS_8_PLANE:      usize = 11 * Board::SIZE;
  pub const LIBS_1_PLANE:       usize = 12 * Board::SIZE;
  pub const LIBS_2_PLANE:       usize = 13 * Board::SIZE;
  pub const LIBS_3_PLANE:       usize = 14 * Board::SIZE;
  pub const LIBS_4_PLANE:       usize = 15 * Board::SIZE;
  pub const LIBS_5_PLANE:       usize = 16 * Board::SIZE;
  pub const LIBS_6_PLANE:       usize = 17 * Board::SIZE;
  pub const LIBS_7_PLANE:       usize = 18 * Board::SIZE;
  pub const LIBS_8_PLANE:       usize = 19 * Board::SIZE;
  pub const CSUIS_1_PLANE:      usize = 20 * Board::SIZE;
  pub const CSUIS_2_PLANE:      usize = 21 * Board::SIZE;
  pub const CSUIS_3_PLANE:      usize = 22 * Board::SIZE;
  pub const CSUIS_4_PLANE:      usize = 23 * Board::SIZE;
  pub const CSUIS_5_PLANE:      usize = 24 * Board::SIZE;
  pub const CSUIS_6_PLANE:      usize = 25 * Board::SIZE;
  pub const CSUIS_7_PLANE:      usize = 26 * Board::SIZE;
  pub const CSUIS_8_PLANE:      usize = 27 * Board::SIZE;
  pub const CCAPS_1_PLANE:      usize = 28 * Board::SIZE;
  pub const CCAPS_2_PLANE:      usize = 29 * Board::SIZE;
  pub const CCAPS_3_PLANE:      usize = 30 * Board::SIZE;
  pub const CCAPS_4_PLANE:      usize = 31 * Board::SIZE;
  pub const CCAPS_5_PLANE:      usize = 32 * Board::SIZE;
  pub const CCAPS_6_PLANE:      usize = 33 * Board::SIZE;
  pub const CCAPS_7_PLANE:      usize = 34 * Board::SIZE;
  pub const CCAPS_8_PLANE:      usize = 35 * Board::SIZE;
  pub const KO_PLANE:           usize = 36 * Board::SIZE;*/

  pub fn new() -> TxnStateAlphaFeatsV2Data {
    let feats = TxnStateAlphaFeatsV2Data{
      //legality:     TxnStateLegalityData::new(),
      features:     repeat(0).take(Self::NUM_PLANES * Board::SIZE).collect(),
      prev_moves:   vec![None, None, None, None, None, None, None, None],
      //init_ranks:   false,
      black_rank:   None,
      white_rank:   None,
      tmp_mark:     BitSet::with_capacity(Board::SIZE),
    };
    assert_eq!(feats.features.len(), Self::NUM_PLANES * Board::SIZE);
    feats
  }

  pub fn current_feature(&self, plane_idx: usize, point: Point) -> u8 {
    self.features[plane_idx * Board::SIZE + point.idx()]
  }

  pub fn feature_dims(&self) -> (usize, usize, usize) {
    // XXX(20160202): Smaller than the actual size because we only use the
    // opponent ranks.
    (Board::DIM, Board::DIM, Self::NUM_EXTRACT_PLANES)
  }

  pub fn extract_relative_serial_arrays(&self, turn: Stone) -> (BitArray3d, Array3d<u8>) {
    let mut buf: Vec<u8> = Vec::with_capacity(Self::NUM_EXTRACT_PLANES * Board::SIZE);
    unsafe { buf.set_len(Self::NUM_EXTRACT_PLANES * Board::SIZE) };
    self.extract_relative_features(turn, &mut buf);

    let mut bit_buf: Vec<u8> = Vec::with_capacity(Self::NUM_SPARSE_PLANES * Board::SIZE);
    unsafe { bit_buf.set_len(Self::NUM_SPARSE_PLANES * Board::SIZE) };
    copy_memory(
        &buf[ .. Self::NUM_SPARSE_PLANES * Board::SIZE],
        &mut bit_buf,
    );

    let mut bytes_buf: Vec<u8> = Vec::with_capacity(Self::NUM_DENSE_PLANES * Board::SIZE);
    unsafe { bytes_buf.set_len(Self::NUM_DENSE_PLANES * Board::SIZE) };
    copy_memory(
        &buf[Self::NUM_SPARSE_PLANES * Board::SIZE .. ],
        &mut bytes_buf,
    );

    let bit_arr = BitArray3d::from_byte_array(&Array3d::with_data(bit_buf, (Board::DIM, Board::DIM, Self::NUM_SPARSE_PLANES)));
    let bytes_arr = Array3d::with_data(bytes_buf, (Board::DIM, Board::DIM, Self::NUM_DENSE_PLANES));

    (bit_arr, bytes_arr)
  }

  pub fn extract_relative_features(&self, turn: Stone, dst_buf: &mut [u8]) {
    assert_eq!(dst_buf.len(), Self::NUM_EXTRACT_PLANES * Board::SIZE);
    match turn {
      Stone::Black => {
        /*copy_memory(
            &self.features[ .. Self::B_RANK_1A_2A_PLANE],
            &mut dst_buf[ .. Self::B_RANK_1A_2A_PLANE],
        );
        copy_memory(
            &self.features[Self::W_RANK_1A_2A_PLANE .. ],
            &mut dst_buf[Self::B_RANK_1A_2A_PLANE .. ],
        );*/
        copy_memory(
            &self.features[ .. Self::BLACK_PLANE],
            &mut dst_buf[ .. Self::FRIEND_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::BLACK_PLANE .. Self::B_RANK_1A_2A_PLANE],
            &mut dst_buf[Self::FRIEND_EXT_PLANE .. Self::ENEMY_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::WHITE_PLANE .. Self::W_RANK_1A_2A_PLANE],
            &mut dst_buf[Self::ENEMY_EXT_PLANE .. Self::E_RANK_1A_2A_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::W_RANK_1A_2A_PLANE .. Self::TURNS_1_PLANE],
            &mut dst_buf[Self::E_RANK_1A_2A_EXT_PLANE .. Self::TURNS_1_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::TURNS_1_PLANE .. ],
            &mut dst_buf[Self::TURNS_1_EXT_PLANE .. ],
        );
      }
      Stone::White => {
        copy_memory(
            &self.features[ .. Self::BLACK_PLANE],
            &mut dst_buf[ .. Self::FRIEND_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::WHITE_PLANE .. Self::W_RANK_1A_2A_PLANE],
            &mut dst_buf[Self::FRIEND_EXT_PLANE .. Self::ENEMY_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::BLACK_PLANE .. Self::B_RANK_1A_2A_PLANE],
            &mut dst_buf[Self::ENEMY_EXT_PLANE .. Self::E_RANK_1A_2A_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::B_RANK_1A_2A_PLANE .. Self::WHITE_PLANE],
            &mut dst_buf[Self::E_RANK_1A_2A_EXT_PLANE .. Self::TURNS_1_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::TURNS_1_PLANE .. ],
            &mut dst_buf[Self::TURNS_1_EXT_PLANE .. ],
        );
        /*copy_memory(
            &self.features[ .. Self::BLACK_PLANE],
            &mut dst_buf[ .. Self::BLACK_PLANE],
        );
        copy_memory(
            &self.features[Self::WHITE_PLANE .. Self::TURNS_1_PLANE],
            &mut dst_buf[Self::BLACK_PLANE .. Self::WHITE_PLANE],
        );
        copy_memory(
            &self.features[Self::BLACK_PLANE .. Self::WHITE_PLANE],
            &mut dst_buf[Self::WHITE_PLANE .. Self::TURNS_1_PLANE],
        );
        copy_memory(
            &self.features[Self::TURNS_1_PLANE .. Self::W_RANK_1A_2A_PLANE],
            &mut dst_buf[Self::TURNS_1_PLANE .. Self::W_RANK_1A_2A_PLANE],
        );
        copy_memory(
            &self.features[Self::CENTER_PLANE .. ],
            &mut dst_buf[Self::W_RANK_1A_2A_PLANE .. ],
        );*/
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
        //features[Self::TURNS_1_PLANE + p] = 0;
        features[Self::TURNS_2_PLANE + p] = Self::SET;
      }
      3 => {
        //features[Self::TURNS_2_PLANE + p] = 0;
        features[Self::TURNS_3_PLANE + p] = Self::SET;
      }
      4 => {
        //features[Self::TURNS_3_PLANE + p] = 0;
        features[Self::TURNS_4_PLANE + p] = Self::SET;
      }
      5 => {
        //features[Self::TURNS_4_PLANE + p] = 0;
        features[Self::TURNS_5_PLANE + p] = Self::SET;
      }
      6 => {
        //features[Self::TURNS_5_PLANE + p] = 0;
        features[Self::TURNS_6_PLANE + p] = Self::SET;
      }
      7 => {
        //features[Self::TURNS_6_PLANE + p] = 0;
        features[Self::TURNS_7_PLANE + p] = Self::SET;
      }
      8 => {
        //features[Self::TURNS_7_PLANE + p] = 0;
        features[Self::TURNS_8_PLANE + p] = Self::SET;
      }
      9 => {
        //features[Self::TURNS_8_PLANE + p] = 0;
      }
      _ => unreachable!(),
    }
  }

  fn update_point_libs(tmp_mark: &mut BitSet, features: &mut [u8], position: &TxnPosition, chains: &TxnChainsList, point: Point, turn: Stone) {
    let p = point.idx();
    if tmp_mark.contains(p) {
      return;
    }
    let stone = position.stones[p];
    if stone != Stone::Empty {
      let head = chains.find_chain(point);
      //assert!(head != TOMBSTONE);
      let chain = chains.get_chain(head).unwrap();

      // Liberty features.
      //let mut is_atari = false;
      match chain.count_libs_up_to_8() {
        0 => { unreachable!(); }
        1 => {
          //is_atari = true;
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

      /*// Suicide features.
      if stone == turn {
        if is_atari {*/
      match stone {
        Stone::Black => {
          match chain.count_length() {
            0 => unreachable!(),
            1 => {
              features[Self::B_CAPS_1_PLANE + p] = Self::SET;
              features[Self::B_CAPS_2_PLANE + p] = 0;
              features[Self::B_CAPS_3_PLANE + p] = 0;
              features[Self::B_CAPS_4_PLANE + p] = 0;
              features[Self::B_CAPS_5_PLANE + p] = 0;
              features[Self::B_CAPS_6_PLANE + p] = 0;
              features[Self::B_CAPS_7_PLANE + p] = 0;
              features[Self::B_CAPS_8_PLANE + p] = 0;
            }
            2 => {
              features[Self::B_CAPS_1_PLANE + p] = 0;
              features[Self::B_CAPS_2_PLANE + p] = Self::SET;
              features[Self::B_CAPS_3_PLANE + p] = 0;
              features[Self::B_CAPS_4_PLANE + p] = 0;
              features[Self::B_CAPS_5_PLANE + p] = 0;
              features[Self::B_CAPS_6_PLANE + p] = 0;
              features[Self::B_CAPS_7_PLANE + p] = 0;
              features[Self::B_CAPS_8_PLANE + p] = 0;
            }
            3 => {
              features[Self::B_CAPS_1_PLANE + p] = 0;
              features[Self::B_CAPS_2_PLANE + p] = 0;
              features[Self::B_CAPS_3_PLANE + p] = Self::SET;
              features[Self::B_CAPS_4_PLANE + p] = 0;
              features[Self::B_CAPS_5_PLANE + p] = 0;
              features[Self::B_CAPS_6_PLANE + p] = 0;
              features[Self::B_CAPS_7_PLANE + p] = 0;
              features[Self::B_CAPS_8_PLANE + p] = 0;
            }
            4 => {
              features[Self::B_CAPS_1_PLANE + p] = 0;
              features[Self::B_CAPS_2_PLANE + p] = 0;
              features[Self::B_CAPS_3_PLANE + p] = 0;
              features[Self::B_CAPS_4_PLANE + p] = Self::SET;
              features[Self::B_CAPS_5_PLANE + p] = 0;
              features[Self::B_CAPS_6_PLANE + p] = 0;
              features[Self::B_CAPS_7_PLANE + p] = 0;
              features[Self::B_CAPS_8_PLANE + p] = 0;
            }
            5 => {
              features[Self::B_CAPS_1_PLANE + p] = 0;
              features[Self::B_CAPS_2_PLANE + p] = 0;
              features[Self::B_CAPS_3_PLANE + p] = 0;
              features[Self::B_CAPS_4_PLANE + p] = 0;
              features[Self::B_CAPS_5_PLANE + p] = Self::SET;
              features[Self::B_CAPS_6_PLANE + p] = 0;
              features[Self::B_CAPS_7_PLANE + p] = 0;
              features[Self::B_CAPS_8_PLANE + p] = 0;
            }
            6 => {
              features[Self::B_CAPS_1_PLANE + p] = 0;
              features[Self::B_CAPS_2_PLANE + p] = 0;
              features[Self::B_CAPS_3_PLANE + p] = 0;
              features[Self::B_CAPS_4_PLANE + p] = 0;
              features[Self::B_CAPS_5_PLANE + p] = 0;
              features[Self::B_CAPS_6_PLANE + p] = Self::SET;
              features[Self::B_CAPS_7_PLANE + p] = 0;
              features[Self::B_CAPS_8_PLANE + p] = 0;
            }
            7 => {
              features[Self::B_CAPS_1_PLANE + p] = 0;
              features[Self::B_CAPS_2_PLANE + p] = 0;
              features[Self::B_CAPS_3_PLANE + p] = 0;
              features[Self::B_CAPS_4_PLANE + p] = 0;
              features[Self::B_CAPS_5_PLANE + p] = 0;
              features[Self::B_CAPS_6_PLANE + p] = 0;
              features[Self::B_CAPS_7_PLANE + p] = Self::SET;
              features[Self::B_CAPS_8_PLANE + p] = 0;
            }
            n => {
              features[Self::B_CAPS_1_PLANE + p] = 0;
              features[Self::B_CAPS_2_PLANE + p] = 0;
              features[Self::B_CAPS_3_PLANE + p] = 0;
              features[Self::B_CAPS_4_PLANE + p] = 0;
              features[Self::B_CAPS_5_PLANE + p] = 0;
              features[Self::B_CAPS_6_PLANE + p] = 0;
              features[Self::B_CAPS_7_PLANE + p] = 0;
              features[Self::B_CAPS_8_PLANE + p] = Self::SET;
            }
          }
        }
        /*} else {
          features[Self::B_CAPS_1_PLANE + p] = 0;
          features[Self::B_CAPS_2_PLANE + p] = 0;
          features[Self::B_CAPS_3_PLANE + p] = 0;
          features[Self::B_CAPS_4_PLANE + p] = 0;
          features[Self::B_CAPS_5_PLANE + p] = 0;
          features[Self::B_CAPS_6_PLANE + p] = 0;
          features[Self::B_CAPS_7_PLANE + p] = 0;
          features[Self::B_CAPS_8_PLANE + p] = 0;
        }
      }

      // Capture features.
      if stone == turn.opponent() {
        if is_atari {*/
        Stone::White => {
          match chain.count_length() {
            0 => unreachable!(),
            1 => {
              features[Self::W_CAPS_1_PLANE + p] = Self::SET;
              features[Self::W_CAPS_2_PLANE + p] = 0;
              features[Self::W_CAPS_3_PLANE + p] = 0;
              features[Self::W_CAPS_4_PLANE + p] = 0;
              features[Self::W_CAPS_5_PLANE + p] = 0;
              features[Self::W_CAPS_6_PLANE + p] = 0;
              features[Self::W_CAPS_7_PLANE + p] = 0;
              features[Self::W_CAPS_8_PLANE + p] = 0;
            }
            2 => {
              features[Self::W_CAPS_1_PLANE + p] = 0;
              features[Self::W_CAPS_2_PLANE + p] = Self::SET;
              features[Self::W_CAPS_3_PLANE + p] = 0;
              features[Self::W_CAPS_4_PLANE + p] = 0;
              features[Self::W_CAPS_5_PLANE + p] = 0;
              features[Self::W_CAPS_6_PLANE + p] = 0;
              features[Self::W_CAPS_7_PLANE + p] = 0;
              features[Self::W_CAPS_8_PLANE + p] = 0;
            }
            3 => {
              features[Self::W_CAPS_1_PLANE + p] = 0;
              features[Self::W_CAPS_2_PLANE + p] = 0;
              features[Self::W_CAPS_3_PLANE + p] = Self::SET;
              features[Self::W_CAPS_4_PLANE + p] = 0;
              features[Self::W_CAPS_5_PLANE + p] = 0;
              features[Self::W_CAPS_6_PLANE + p] = 0;
              features[Self::W_CAPS_7_PLANE + p] = 0;
              features[Self::W_CAPS_8_PLANE + p] = 0;
            }
            4 => {
              features[Self::W_CAPS_1_PLANE + p] = 0;
              features[Self::W_CAPS_2_PLANE + p] = 0;
              features[Self::W_CAPS_3_PLANE + p] = 0;
              features[Self::W_CAPS_4_PLANE + p] = Self::SET;
              features[Self::W_CAPS_5_PLANE + p] = 0;
              features[Self::W_CAPS_6_PLANE + p] = 0;
              features[Self::W_CAPS_7_PLANE + p] = 0;
              features[Self::W_CAPS_8_PLANE + p] = 0;
            }
            5 => {
              features[Self::W_CAPS_1_PLANE + p] = 0;
              features[Self::W_CAPS_2_PLANE + p] = 0;
              features[Self::W_CAPS_3_PLANE + p] = 0;
              features[Self::W_CAPS_4_PLANE + p] = 0;
              features[Self::W_CAPS_5_PLANE + p] = Self::SET;
              features[Self::W_CAPS_6_PLANE + p] = 0;
              features[Self::W_CAPS_7_PLANE + p] = 0;
              features[Self::W_CAPS_8_PLANE + p] = 0;
            }
            6 => {
              features[Self::W_CAPS_1_PLANE + p] = 0;
              features[Self::W_CAPS_2_PLANE + p] = 0;
              features[Self::W_CAPS_3_PLANE + p] = 0;
              features[Self::W_CAPS_4_PLANE + p] = 0;
              features[Self::W_CAPS_5_PLANE + p] = 0;
              features[Self::W_CAPS_6_PLANE + p] = Self::SET;
              features[Self::W_CAPS_7_PLANE + p] = 0;
              features[Self::W_CAPS_8_PLANE + p] = 0;
            }
            7 => {
              features[Self::W_CAPS_1_PLANE + p] = 0;
              features[Self::W_CAPS_2_PLANE + p] = 0;
              features[Self::W_CAPS_3_PLANE + p] = 0;
              features[Self::W_CAPS_4_PLANE + p] = 0;
              features[Self::W_CAPS_5_PLANE + p] = 0;
              features[Self::W_CAPS_6_PLANE + p] = 0;
              features[Self::W_CAPS_7_PLANE + p] = Self::SET;
              features[Self::W_CAPS_8_PLANE + p] = 0;
            }
            n => {
              features[Self::W_CAPS_1_PLANE + p] = 0;
              features[Self::W_CAPS_2_PLANE + p] = 0;
              features[Self::W_CAPS_3_PLANE + p] = 0;
              features[Self::W_CAPS_4_PLANE + p] = 0;
              features[Self::W_CAPS_5_PLANE + p] = 0;
              features[Self::W_CAPS_6_PLANE + p] = 0;
              features[Self::W_CAPS_7_PLANE + p] = 0;
              features[Self::W_CAPS_8_PLANE + p] = Self::SET;
            }
          }
        }

        _ => unreachable!(),
      }
        /*} else {
          features[Self::W_CAPS_1_PLANE + p] = 0;
          features[Self::W_CAPS_2_PLANE + p] = 0;
          features[Self::W_CAPS_3_PLANE + p] = 0;
          features[Self::W_CAPS_4_PLANE + p] = 0;
          features[Self::W_CAPS_5_PLANE + p] = 0;
          features[Self::W_CAPS_6_PLANE + p] = 0;
          features[Self::W_CAPS_7_PLANE + p] = 0;
          features[Self::W_CAPS_8_PLANE + p] = 0;
        }
      }*/

    } else {
      features[Self::LIBS_1_PLANE + p] = 0;
      features[Self::LIBS_2_PLANE + p] = 0;
      features[Self::LIBS_3_PLANE + p] = 0;
      features[Self::LIBS_4_PLANE + p] = 0;
      features[Self::LIBS_5_PLANE + p] = 0;
      features[Self::LIBS_6_PLANE + p] = 0;
      features[Self::LIBS_7_PLANE + p] = 0;
      features[Self::LIBS_8_PLANE + p] = 0;
      features[Self::B_CAPS_1_PLANE + p] = 0;
      features[Self::B_CAPS_2_PLANE + p] = 0;
      features[Self::B_CAPS_3_PLANE + p] = 0;
      features[Self::B_CAPS_4_PLANE + p] = 0;
      features[Self::B_CAPS_5_PLANE + p] = 0;
      features[Self::B_CAPS_6_PLANE + p] = 0;
      features[Self::B_CAPS_7_PLANE + p] = 0;
      features[Self::B_CAPS_8_PLANE + p] = 0;
      features[Self::W_CAPS_1_PLANE + p] = 0;
      features[Self::W_CAPS_2_PLANE + p] = 0;
      features[Self::W_CAPS_3_PLANE + p] = 0;
      features[Self::W_CAPS_4_PLANE + p] = 0;
      features[Self::W_CAPS_5_PLANE + p] = 0;
      features[Self::W_CAPS_6_PLANE + p] = 0;
      features[Self::W_CAPS_7_PLANE + p] = 0;
      features[Self::W_CAPS_8_PLANE + p] = 0;
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

  fn update_points_rank(features: &mut [u8], position: &TxnPosition, turn: Stone, rank: PlayerRank) {
    let max_plane = match (turn, rank) {
      (Stone::Black, PlayerRank::Kyu(_)) => None,
      (Stone::Black, PlayerRank::Ama(1)) |
      (Stone::Black, PlayerRank::Ama(2)) => Some(Self::B_RANK_1A_2A_PLANE),
      (Stone::Black, PlayerRank::Ama(3)) |
      (Stone::Black, PlayerRank::Ama(4)) => Some(Self::B_RANK_3A_4A_PLANE),
      (Stone::Black, PlayerRank::Ama(_)) => Some(Self::B_RANK_5A_7A_PLANE),
      (Stone::Black, PlayerRank::Dan(1)) |
      (Stone::Black, PlayerRank::Dan(2)) |
      (Stone::Black, PlayerRank::Dan(3)) => Some(Self::B_RANK_1P_3P_PLANE),
      (Stone::Black, PlayerRank::Dan(4)) |
      (Stone::Black, PlayerRank::Dan(5)) |
      (Stone::Black, PlayerRank::Dan(6)) => Some(Self::B_RANK_4P_6P_PLANE),
      (Stone::Black, PlayerRank::Dan(7)) |
      (Stone::Black, PlayerRank::Dan(8)) |
      (Stone::Black, PlayerRank::Dan(9)) => Some(Self::B_RANK_7P_9P_PLANE),
      (Stone::Black, _) => panic!("invalid rank for black player: {:?}", rank),
      (Stone::White, PlayerRank::Kyu(_)) => None,
      (Stone::White, PlayerRank::Ama(1)) |
      (Stone::White, PlayerRank::Ama(2)) => Some(Self::W_RANK_1A_2A_PLANE),
      (Stone::White, PlayerRank::Ama(3)) |
      (Stone::White, PlayerRank::Ama(4)) => Some(Self::W_RANK_3A_4A_PLANE),
      (Stone::White, PlayerRank::Ama(_)) => Some(Self::W_RANK_5A_7A_PLANE),
      (Stone::White, PlayerRank::Dan(1)) |
      (Stone::White, PlayerRank::Dan(2)) |
      (Stone::White, PlayerRank::Dan(3)) => Some(Self::W_RANK_1P_3P_PLANE),
      (Stone::White, PlayerRank::Dan(4)) |
      (Stone::White, PlayerRank::Dan(5)) |
      (Stone::White, PlayerRank::Dan(6)) => Some(Self::W_RANK_4P_6P_PLANE),
      (Stone::White, PlayerRank::Dan(7)) |
      (Stone::White, PlayerRank::Dan(8)) |
      (Stone::White, PlayerRank::Dan(9)) => Some(Self::W_RANK_7P_9P_PLANE),
      (Stone::White, _) => panic!("invalid rank for white player: {:?}", rank),
      _ => unreachable!(),
    };
    match turn {
      Stone::Black => {
        for &plane in &[
          Self::B_RANK_1A_2A_PLANE,
          Self::B_RANK_3A_4A_PLANE,
          Self::B_RANK_5A_7A_PLANE,
          Self::B_RANK_1P_3P_PLANE,
          Self::B_RANK_4P_6P_PLANE,
          Self::B_RANK_7P_9P_PLANE,
        ] {
          if let Some(max_plane) = max_plane {
            if plane <= max_plane {
              for p in 0 .. Board::SIZE {
                features[plane + p] = Self::SET;
              }
            }
          }
        }
      }
      Stone::White => {
        for &plane in &[
          Self::W_RANK_1A_2A_PLANE,
          Self::W_RANK_3A_4A_PLANE,
          Self::W_RANK_5A_7A_PLANE,
          Self::W_RANK_1P_3P_PLANE,
          Self::W_RANK_4P_6P_PLANE,
          Self::W_RANK_7P_9P_PLANE,
        ] {
          if let Some(max_plane) = max_plane {
            if plane <= max_plane {
              for p in 0 .. Board::SIZE {
                features[plane + p] = Self::SET;
              }
            }
          }
        }
      }
      _ => unreachable!(),
    }
  }
}

impl TxnStateData for TxnStateAlphaFeatsV2Data {
  fn reset(&mut self) {
    // FIXME(20160202): technically, should have ranks when the board is empty,
    // so would need to pass TxnPosition, but it is not really necessary.
    //self.legality.reset();
    for p in 0 .. Self::BLACK_PLANE {
      self.features[p] = Self::SET;
    }
    for p in Self::BLACK_PLANE .. Self::CENTER_PLANE {
      self.features[p] = 0;
    }
    assert_eq!(Self::CENTER_PLANE + Board::SIZE, Self::NUM_PLANES * Board::SIZE);
    for p in 0 .. Board::SIZE {
      let point = Point::from_idx(p);
      let coord = point.to_coord();
      let (u, v) = ((coord.x as i8 - Board::HALF as i8) as f32, (coord.y as i8 - Board::HALF as i8) as f32);
      let distance = (u * u + v * v).sqrt();
      self.features[Self::CENTER_PLANE + p] = (Self::SET as f32 * (-0.5 * distance).exp()).round() as u8;
    }
    //println!("DEBUG: features: center: {:?}", &self.features[Self::CENTER_PLANE .. ]);
    self.prev_moves[0] = None;
    self.prev_moves[1] = None;
    self.prev_moves[2] = None;
    self.prev_moves[3] = None;
    self.prev_moves[4] = None;
    self.prev_moves[5] = None;
    self.prev_moves[6] = None;
    self.prev_moves[7] = None;
    //self.init_ranks = false;
    self.black_rank = None;
    self.white_rank = None;
  }

  fn update(&mut self, position: &TxnPosition, chains: &TxnChainsList, update_turn: Stone, update_action: Action) {
    //self.legality.update(position, chains, update_turn, update_action);

    // FIXME(20160201): use legality data for "good" move feature.

    {
      // XXX(20151107): Iterate over placed and killed chains to update stone
      // repr.
      let &mut TxnStateAlphaFeatsV2Data{
        //ref legality,
        ref mut features,
        ref mut tmp_mark, .. } = self;

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
      /*match self.prev_moves[7] {
        Some((_, prev_point)) => {
          Self::update_point_turns(features, prev_point, 9);
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
      }*/
      for t in (0 .. 8).rev() {
        match self.prev_moves[t] {
          Some((_, prev_point)) => {
            Self::update_point_turns(features, prev_point, t + 2);
          }
          None => {}
        }
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

      // XXX(20160202): Update player rank features.
      match self.black_rank {
        None => {
          let rank = position.ranks[0];
          Self::update_points_rank(features, position, Stone::Black, rank);
          self.black_rank = Some(rank);
        }
        Some(rank) => {
          assert_eq!(rank, position.ranks[0]);
        }
      }
      match self.white_rank {
        None => {
          let rank = position.ranks[1];
          Self::update_points_rank(features, position, Stone::White, rank);
          self.white_rank = Some(rank);
        }
        Some(rank) => {
          assert_eq!(rank, position.ranks[1]);
        }
      }
    }
  }
}

#[derive(Clone, RustcDecodable, RustcEncodable)]
pub struct TxnStateAlphaV3FeatsData {
  pub features: Vec<u8>,
  prev_moves:   Vec<Option<(Stone, Point)>>,
  black_rank:   Option<PlayerRank>,
  white_rank:   Option<PlayerRank>,
  tmp_mark:     BitSet,
}

impl TxnStateAlphaV3FeatsData {
  pub const SET: u8 = 255;

  pub const NUM_PLANES:         usize = 34;
  pub const NUM_EXTRACT_PLANES: usize = 32;
  pub const NUM_SPARSE_PLANES:  usize = 31;
  pub const NUM_DENSE_PLANES:   usize = 1;

  pub const BASELINE_PLANE:     usize = 0 * Board::SIZE;
  pub const EMPTY_PLANE:        usize = 1 * Board::SIZE;
  pub const BLACK_PLANE:        usize = 2 * Board::SIZE;
  pub const B_RANK_AMA_PLANE:   usize = 3 * Board::SIZE;
  pub const B_RANK_PRO_PLANE:   usize = 4 * Board::SIZE;
  pub const WHITE_PLANE:        usize = 5 * Board::SIZE;
  pub const W_RANK_AMA_PLANE:   usize = 6 * Board::SIZE;
  pub const W_RANK_PRO_PLANE:   usize = 7 * Board::SIZE;
  pub const TURNS_1_PLANE:      usize = 8 * Board::SIZE;
  pub const TURNS_2_PLANE:      usize = 9 * Board::SIZE;
  pub const TURNS_3_PLANE:      usize = 10 * Board::SIZE;
  pub const TURNS_4_PLANE:      usize = 11 * Board::SIZE;
  pub const TURNS_5_PLANE:      usize = 12 * Board::SIZE;
  pub const TURNS_6_PLANE:      usize = 13 * Board::SIZE;
  pub const TURNS_7_PLANE:      usize = 14 * Board::SIZE;
  pub const TURNS_8_PLANE:      usize = 15 * Board::SIZE;
  pub const LIBS_1_PLANE:       usize = 16 * Board::SIZE;
  pub const LIBS_2_PLANE:       usize = 17 * Board::SIZE;
  pub const LIBS_3_PLANE:       usize = 18 * Board::SIZE;
  pub const LIBS_4_PLANE:       usize = 19 * Board::SIZE;
  pub const LIBS_5_PLANE:       usize = 20 * Board::SIZE;
  pub const LIBS_6_PLANE:       usize = 21 * Board::SIZE;
  pub const LIBS_7_PLANE:       usize = 22 * Board::SIZE;
  pub const LIBS_8_PLANE:       usize = 23 * Board::SIZE;
  pub const CAPS_1_PLANE:       usize = 24 * Board::SIZE;
  pub const CAPS_2_PLANE:       usize = 25 * Board::SIZE;
  pub const CAPS_3_PLANE:       usize = 26 * Board::SIZE;
  pub const CAPS_4_PLANE:       usize = 27 * Board::SIZE;
  pub const CAPS_5_PLANE:       usize = 28 * Board::SIZE;
  pub const CAPS_6_PLANE:       usize = 29 * Board::SIZE;
  pub const CAPS_7_PLANE:       usize = 30 * Board::SIZE;
  pub const CAPS_8_PLANE:       usize = 31 * Board::SIZE;
  pub const KO_PLANE:           usize = 32 * Board::SIZE;
  pub const CENTER_PLANE:       usize = 33 * Board::SIZE;

  pub const BASELINE_EXT_PLANE:     usize = 0 * Board::SIZE;
  pub const EMPTY_EXT_PLANE:        usize = 1 * Board::SIZE;
  pub const FRIEND_EXT_PLANE:       usize = 2 * Board::SIZE;
  pub const ENEMY_EXT_PLANE:        usize = 3 * Board::SIZE;
  pub const E_RANK_AMA_EXT_PLANE:   usize = 4 * Board::SIZE;
  pub const E_RANK_PRO_EXT_PLANE:   usize = 5 * Board::SIZE;
  pub const TURNS_1_EXT_PLANE:      usize = 6 * Board::SIZE;
  pub const TURNS_2_EXT_PLANE:      usize = 7 * Board::SIZE;
  pub const TURNS_3_EXT_PLANE:      usize = 8 * Board::SIZE;
  pub const TURNS_4_EXT_PLANE:      usize = 9 * Board::SIZE;
  pub const TURNS_5_EXT_PLANE:      usize = 10 * Board::SIZE;
  pub const TURNS_6_EXT_PLANE:      usize = 11 * Board::SIZE;
  pub const TURNS_7_EXT_PLANE:      usize = 12 * Board::SIZE;
  pub const TURNS_8_EXT_PLANE:      usize = 13 * Board::SIZE;
  pub const LIBS_1_EXT_PLANE:       usize = 14 * Board::SIZE;
  pub const LIBS_2_EXT_PLANE:       usize = 15 * Board::SIZE;
  pub const LIBS_3_EXT_PLANE:       usize = 16 * Board::SIZE;
  pub const LIBS_4_EXT_PLANE:       usize = 17 * Board::SIZE;
  pub const LIBS_5_EXT_PLANE:       usize = 18 * Board::SIZE;
  pub const LIBS_6_EXT_PLANE:       usize = 19 * Board::SIZE;
  pub const LIBS_7_EXT_PLANE:       usize = 20 * Board::SIZE;
  pub const LIBS_8_EXT_PLANE:       usize = 21 * Board::SIZE;
  pub const CAPS_1_EXT_PLANE:       usize = 22 * Board::SIZE;
  pub const CAPS_2_EXT_PLANE:       usize = 23 * Board::SIZE;
  pub const CAPS_3_EXT_PLANE:       usize = 24 * Board::SIZE;
  pub const CAPS_4_EXT_PLANE:       usize = 25 * Board::SIZE;
  pub const CAPS_5_EXT_PLANE:       usize = 26 * Board::SIZE;
  pub const CAPS_6_EXT_PLANE:       usize = 27 * Board::SIZE;
  pub const CAPS_7_EXT_PLANE:       usize = 28 * Board::SIZE;
  pub const CAPS_8_EXT_PLANE:       usize = 29 * Board::SIZE;
  pub const KO_EXT_PLANE:           usize = 30 * Board::SIZE;
  pub const CENTER_EXT_PLANE:       usize = 31 * Board::SIZE;

  pub fn new() -> TxnStateAlphaV3FeatsData {
    let feats = TxnStateAlphaV3FeatsData{
      features:     repeat(0).take(Self::NUM_PLANES * Board::SIZE).collect(),
      prev_moves:   vec![None, None, None, None, None, None, None, None],
      black_rank:   None,
      white_rank:   None,
      tmp_mark:     BitSet::with_capacity(Board::SIZE),
    };
    assert_eq!(feats.features.len(), Self::NUM_PLANES * Board::SIZE);
    feats
  }

  pub fn serial_size() -> usize {
    BitArray3d::serial_size((19, 19, 31)) + Array3d::<u8>::serial_size((19, 19, 1))
  }

  pub fn extract_relative_serial_blob(&self, turn: Stone) -> Vec<u8> {
    let mut serial_frame: Vec<u8> = Vec::with_capacity(Self::serial_size());
    let (bit_arr, bytes_arr) = self.extract_relative_serial_arrays(turn);
    bit_arr.serialize(&mut serial_frame).unwrap();
    bytes_arr.serialize(&mut serial_frame).unwrap();
    serial_frame
  }

  pub fn extract_relative_serial_arrays(&self, turn: Stone) -> (BitArray3d, Array3d<u8>) {
    let mut buf: Vec<u8> = Vec::with_capacity(Self::NUM_EXTRACT_PLANES * Board::SIZE);
    unsafe { buf.set_len(Self::NUM_EXTRACT_PLANES * Board::SIZE) };
    self.extract_relative_features(turn, &mut buf);

    let mut bit_buf: Vec<u8> = Vec::with_capacity(Self::NUM_SPARSE_PLANES * Board::SIZE);
    unsafe { bit_buf.set_len(Self::NUM_SPARSE_PLANES * Board::SIZE) };
    copy_memory(
        &buf[ .. Self::NUM_SPARSE_PLANES * Board::SIZE],
        &mut bit_buf,
    );

    let mut bytes_buf: Vec<u8> = Vec::with_capacity(Self::NUM_DENSE_PLANES * Board::SIZE);
    unsafe { bytes_buf.set_len(Self::NUM_DENSE_PLANES * Board::SIZE) };
    copy_memory(
        &buf[Self::NUM_SPARSE_PLANES * Board::SIZE .. ],
        &mut bytes_buf,
    );

    let bit_arr = BitArray3d::from_byte_array(&Array3d::with_data(bit_buf, (Board::DIM, Board::DIM, Self::NUM_SPARSE_PLANES)));
    let bytes_arr = Array3d::with_data(bytes_buf, (Board::DIM, Board::DIM, Self::NUM_DENSE_PLANES));

    (bit_arr, bytes_arr)
  }

  pub fn extract_relative_features(&self, turn: Stone, dst_buf: &mut [u8]) {
    assert_eq!(dst_buf.len(), Self::NUM_EXTRACT_PLANES * Board::SIZE);
    match turn {
      Stone::Black => {
        /*copy_memory(
            &self.features[ .. Self::BLACK_PLANE],
            &mut dst_buf[ .. Self::FRIEND_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::BLACK_PLANE .. Self::B_RANK_AMA_PLANE],
            &mut dst_buf[Self::FRIEND_EXT_PLANE .. Self::ENEMY_EXT_PLANE],
        );*/
        copy_memory(
            &self.features[ .. Self::B_RANK_AMA_PLANE],
            &mut dst_buf[ .. Self::ENEMY_EXT_PLANE],
        );
        /*copy_memory(
            &self.features[Self::WHITE_PLANE .. Self::W_RANK_AMA_PLANE],
            &mut dst_buf[Self::ENEMY_EXT_PLANE .. Self::E_RANK_AMA_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::W_RANK_AMA_PLANE .. Self::TURNS_1_PLANE],
            &mut dst_buf[Self::E_RANK_AMA_EXT_PLANE .. Self::TURNS_1_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::TURNS_1_PLANE .. ],
            &mut dst_buf[Self::TURNS_1_EXT_PLANE .. ],
        );*/
        copy_memory(
            &self.features[Self::WHITE_PLANE .. ],
            &mut dst_buf[Self::ENEMY_EXT_PLANE .. ],
        );
      }
      Stone::White => {
        copy_memory(
            &self.features[ .. Self::BLACK_PLANE],
            &mut dst_buf[ .. Self::FRIEND_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::WHITE_PLANE .. Self::W_RANK_AMA_PLANE],
            &mut dst_buf[Self::FRIEND_EXT_PLANE .. Self::ENEMY_EXT_PLANE],
        );
        /*copy_memory(
            &self.features[Self::BLACK_PLANE .. Self::B_RANK_AMA_PLANE],
            &mut dst_buf[Self::ENEMY_EXT_PLANE .. Self::E_RANK_AMA_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::B_RANK_AMA_PLANE .. Self::WHITE_PLANE],
            &mut dst_buf[Self::E_RANK_AMA_EXT_PLANE .. Self::TURNS_1_EXT_PLANE],
        );*/
        copy_memory(
            &self.features[Self::BLACK_PLANE .. Self::WHITE_PLANE],
            &mut dst_buf[Self::ENEMY_EXT_PLANE .. Self::TURNS_1_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::TURNS_1_PLANE .. ],
            &mut dst_buf[Self::TURNS_1_EXT_PLANE .. ],
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
        features[Self::TURNS_2_PLANE + p] = Self::SET;
      }
      3 => {
        features[Self::TURNS_3_PLANE + p] = Self::SET;
      }
      4 => {
        features[Self::TURNS_4_PLANE + p] = Self::SET;
      }
      5 => {
        features[Self::TURNS_5_PLANE + p] = Self::SET;
      }
      6 => {
        features[Self::TURNS_6_PLANE + p] = Self::SET;
      }
      7 => {
        features[Self::TURNS_7_PLANE + p] = Self::SET;
      }
      8 => {
        features[Self::TURNS_8_PLANE + p] = Self::SET;
      }
      9 => {
        // Do nothing.
      }
      _ => unreachable!(),
    }
  }

  fn update_point_libs(tmp_mark: &mut BitSet, features: &mut [u8], position: &TxnPosition, chains: &TxnChainsList, point: Point, turn: Stone) {
    let p = point.idx();
    if tmp_mark.contains(p) {
      return;
    }

    let stone = position.stones[p];
    if stone != Stone::Empty {
      let head = chains.find_chain(point);
      //assert!(head != TOMBSTONE);
      let chain = chains.get_chain(head).unwrap();

      // Liberty features.
      match chain.count_libs_up_to_8() {
        0 => { unreachable!(); }
        1 => {
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
          features[Self::LIBS_1_PLANE + p] = Self::SET;
          features[Self::LIBS_2_PLANE + p] = Self::SET;
          features[Self::LIBS_3_PLANE + p] = 0;
          features[Self::LIBS_4_PLANE + p] = 0;
          features[Self::LIBS_5_PLANE + p] = 0;
          features[Self::LIBS_6_PLANE + p] = 0;
          features[Self::LIBS_7_PLANE + p] = 0;
          features[Self::LIBS_8_PLANE + p] = 0;
        }
        3 => {
          features[Self::LIBS_1_PLANE + p] = Self::SET;
          features[Self::LIBS_2_PLANE + p] = Self::SET;
          features[Self::LIBS_3_PLANE + p] = Self::SET;
          features[Self::LIBS_4_PLANE + p] = 0;
          features[Self::LIBS_5_PLANE + p] = 0;
          features[Self::LIBS_6_PLANE + p] = 0;
          features[Self::LIBS_7_PLANE + p] = 0;
          features[Self::LIBS_8_PLANE + p] = 0;
        }
        4 => {
          features[Self::LIBS_1_PLANE + p] = Self::SET;
          features[Self::LIBS_2_PLANE + p] = Self::SET;
          features[Self::LIBS_3_PLANE + p] = Self::SET;
          features[Self::LIBS_4_PLANE + p] = Self::SET;
          features[Self::LIBS_5_PLANE + p] = 0;
          features[Self::LIBS_6_PLANE + p] = 0;
          features[Self::LIBS_7_PLANE + p] = 0;
          features[Self::LIBS_8_PLANE + p] = 0;
        }
        5 => {
          features[Self::LIBS_1_PLANE + p] = Self::SET;
          features[Self::LIBS_2_PLANE + p] = Self::SET;
          features[Self::LIBS_3_PLANE + p] = Self::SET;
          features[Self::LIBS_4_PLANE + p] = Self::SET;
          features[Self::LIBS_5_PLANE + p] = Self::SET;
          features[Self::LIBS_6_PLANE + p] = 0;
          features[Self::LIBS_7_PLANE + p] = 0;
          features[Self::LIBS_8_PLANE + p] = 0;
        }
        6 => {
          features[Self::LIBS_1_PLANE + p] = Self::SET;
          features[Self::LIBS_2_PLANE + p] = Self::SET;
          features[Self::LIBS_3_PLANE + p] = Self::SET;
          features[Self::LIBS_4_PLANE + p] = Self::SET;
          features[Self::LIBS_5_PLANE + p] = Self::SET;
          features[Self::LIBS_6_PLANE + p] = Self::SET;
          features[Self::LIBS_7_PLANE + p] = 0;
          features[Self::LIBS_8_PLANE + p] = 0;
        }
        7 => {
          features[Self::LIBS_1_PLANE + p] = Self::SET;
          features[Self::LIBS_2_PLANE + p] = Self::SET;
          features[Self::LIBS_3_PLANE + p] = Self::SET;
          features[Self::LIBS_4_PLANE + p] = Self::SET;
          features[Self::LIBS_5_PLANE + p] = Self::SET;
          features[Self::LIBS_6_PLANE + p] = Self::SET;
          features[Self::LIBS_7_PLANE + p] = Self::SET;
          features[Self::LIBS_8_PLANE + p] = 0;
        }
        8 => {
          features[Self::LIBS_1_PLANE + p] = Self::SET;
          features[Self::LIBS_2_PLANE + p] = Self::SET;
          features[Self::LIBS_3_PLANE + p] = Self::SET;
          features[Self::LIBS_4_PLANE + p] = Self::SET;
          features[Self::LIBS_5_PLANE + p] = Self::SET;
          features[Self::LIBS_6_PLANE + p] = Self::SET;
          features[Self::LIBS_7_PLANE + p] = Self::SET;
          features[Self::LIBS_8_PLANE + p] = Self::SET;
        }
        _ => { unreachable!(); }
      }

      // Chain size features.
      match chain.count_length() {
        0 => { unreachable!(); }
        1 => {
          features[Self::CAPS_1_PLANE + p] = Self::SET;
          features[Self::CAPS_2_PLANE + p] = 0;
          features[Self::CAPS_3_PLANE + p] = 0;
          features[Self::CAPS_4_PLANE + p] = 0;
          features[Self::CAPS_5_PLANE + p] = 0;
          features[Self::CAPS_6_PLANE + p] = 0;
          features[Self::CAPS_7_PLANE + p] = 0;
          features[Self::CAPS_8_PLANE + p] = 0;
        }
        2 => {
          features[Self::CAPS_1_PLANE + p] = Self::SET;
          features[Self::CAPS_2_PLANE + p] = Self::SET;
          features[Self::CAPS_3_PLANE + p] = 0;
          features[Self::CAPS_4_PLANE + p] = 0;
          features[Self::CAPS_5_PLANE + p] = 0;
          features[Self::CAPS_6_PLANE + p] = 0;
          features[Self::CAPS_7_PLANE + p] = 0;
          features[Self::CAPS_8_PLANE + p] = 0;
        }
        3 => {
          features[Self::CAPS_1_PLANE + p] = Self::SET;
          features[Self::CAPS_2_PLANE + p] = Self::SET;
          features[Self::CAPS_3_PLANE + p] = Self::SET;
          features[Self::CAPS_4_PLANE + p] = 0;
          features[Self::CAPS_5_PLANE + p] = 0;
          features[Self::CAPS_6_PLANE + p] = 0;
          features[Self::CAPS_7_PLANE + p] = 0;
          features[Self::CAPS_8_PLANE + p] = 0;
        }
        4 => {
          features[Self::CAPS_1_PLANE + p] = Self::SET;
          features[Self::CAPS_2_PLANE + p] = Self::SET;
          features[Self::CAPS_3_PLANE + p] = Self::SET;
          features[Self::CAPS_4_PLANE + p] = Self::SET;
          features[Self::CAPS_5_PLANE + p] = 0;
          features[Self::CAPS_6_PLANE + p] = 0;
          features[Self::CAPS_7_PLANE + p] = 0;
          features[Self::CAPS_8_PLANE + p] = 0;
        }
        5 => {
          features[Self::CAPS_1_PLANE + p] = Self::SET;
          features[Self::CAPS_2_PLANE + p] = Self::SET;
          features[Self::CAPS_3_PLANE + p] = Self::SET;
          features[Self::CAPS_4_PLANE + p] = Self::SET;
          features[Self::CAPS_5_PLANE + p] = Self::SET;
          features[Self::CAPS_6_PLANE + p] = 0;
          features[Self::CAPS_7_PLANE + p] = 0;
          features[Self::CAPS_8_PLANE + p] = 0;
        }
        6 => {
          features[Self::CAPS_1_PLANE + p] = Self::SET;
          features[Self::CAPS_2_PLANE + p] = Self::SET;
          features[Self::CAPS_3_PLANE + p] = Self::SET;
          features[Self::CAPS_4_PLANE + p] = Self::SET;
          features[Self::CAPS_5_PLANE + p] = Self::SET;
          features[Self::CAPS_6_PLANE + p] = Self::SET;
          features[Self::CAPS_7_PLANE + p] = 0;
          features[Self::CAPS_8_PLANE + p] = 0;
        }
        7 => {
          features[Self::CAPS_1_PLANE + p] = Self::SET;
          features[Self::CAPS_2_PLANE + p] = Self::SET;
          features[Self::CAPS_3_PLANE + p] = Self::SET;
          features[Self::CAPS_4_PLANE + p] = Self::SET;
          features[Self::CAPS_5_PLANE + p] = Self::SET;
          features[Self::CAPS_6_PLANE + p] = Self::SET;
          features[Self::CAPS_7_PLANE + p] = Self::SET;
          features[Self::CAPS_8_PLANE + p] = 0;
        }
        n => {
          features[Self::CAPS_1_PLANE + p] = Self::SET;
          features[Self::CAPS_2_PLANE + p] = Self::SET;
          features[Self::CAPS_3_PLANE + p] = Self::SET;
          features[Self::CAPS_4_PLANE + p] = Self::SET;
          features[Self::CAPS_5_PLANE + p] = Self::SET;
          features[Self::CAPS_6_PLANE + p] = Self::SET;
          features[Self::CAPS_7_PLANE + p] = Self::SET;
          features[Self::CAPS_8_PLANE + p] = Self::SET;
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

      features[Self::CAPS_1_PLANE + p] = 0;
      features[Self::CAPS_2_PLANE + p] = 0;
      features[Self::CAPS_3_PLANE + p] = 0;
      features[Self::CAPS_4_PLANE + p] = 0;
      features[Self::CAPS_5_PLANE + p] = 0;
      features[Self::CAPS_6_PLANE + p] = 0;
      features[Self::CAPS_7_PLANE + p] = 0;
      features[Self::CAPS_8_PLANE + p] = 0;
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

  fn update_points_rank(features: &mut [u8], position: &TxnPosition, turn: Stone, rank: PlayerRank) {
    let max_plane = match (turn, rank) {
      (Stone::Black, PlayerRank::Kyu(_)) => None,
      (Stone::Black, PlayerRank::Ama(_)) => Some(Self::B_RANK_AMA_PLANE),
      (Stone::Black, PlayerRank::Dan(_)) => Some(Self::B_RANK_PRO_PLANE),
      //(Stone::Black, _) => panic!("invalid rank for black player: {:?}", rank),
      (Stone::White, PlayerRank::Kyu(_)) => None,
      (Stone::White, PlayerRank::Ama(_)) => Some(Self::W_RANK_AMA_PLANE),
      (Stone::White, PlayerRank::Dan(_)) => Some(Self::W_RANK_PRO_PLANE),
      //(Stone::White, _) => panic!("invalid rank for white player: {:?}", rank),
      _ => unreachable!(),
    };
    match turn {
      Stone::Black => {
        for &plane in &[
          Self::B_RANK_AMA_PLANE,
          Self::B_RANK_PRO_PLANE,
        ] {
          if let Some(max_plane) = max_plane {
            if plane <= max_plane {
              for p in 0 .. Board::SIZE {
                features[plane + p] = Self::SET;
              }
            }
          }
        }
      }
      Stone::White => {
        for &plane in &[
          Self::W_RANK_AMA_PLANE,
          Self::W_RANK_PRO_PLANE,
        ] {
          if let Some(max_plane) = max_plane {
            if plane <= max_plane {
              for p in 0 .. Board::SIZE {
                features[plane + p] = Self::SET;
              }
            }
          }
        }
      }
      _ => unreachable!(),
    }
  }
}

impl TxnStateData for TxnStateAlphaV3FeatsData {
  fn reset(&mut self) {
    for p in 0 .. Self::BLACK_PLANE {
      self.features[p] = Self::SET;
    }
    for p in Self::BLACK_PLANE .. Self::CENTER_PLANE {
      self.features[p] = 0;
    }
    assert_eq!(Self::CENTER_PLANE + Board::SIZE, Self::NUM_PLANES * Board::SIZE);
    for p in 0 .. Board::SIZE {
      let point = Point::from_idx(p);
      let coord = point.to_coord();
      let (u, v) = ((coord.x as i8 - Board::HALF as i8) as f32, (coord.y as i8 - Board::HALF as i8) as f32);
      let distance = (u * u + v * v).sqrt();
      self.features[Self::CENTER_PLANE + p] = (Self::SET as f32 * (-0.5 * distance).exp()).round() as u8;
    }
    self.prev_moves[0] = None;
    self.prev_moves[1] = None;
    self.prev_moves[2] = None;
    self.prev_moves[3] = None;
    self.prev_moves[4] = None;
    self.prev_moves[5] = None;
    self.prev_moves[6] = None;
    self.prev_moves[7] = None;
    self.black_rank = None;
    self.white_rank = None;
  }

  fn update(&mut self, position: &TxnPosition, chains: &TxnChainsList, update_turn: Stone, update_action: Action) {
    {
      // XXX(20151107): Iterate over placed and killed chains to update stone
      // repr.
      let &mut TxnStateAlphaV3FeatsData{
        ref mut features,
        ref mut tmp_mark, .. } = self;

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
      for t in (0 .. 8).rev() {
        match self.prev_moves[t] {
          Some((_, prev_point)) => {
            Self::update_point_turns(features, prev_point, t + 2);
          }
          None => {}
        }
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
      if let Some((_, prev_ko_point)) = position.prev_ko {
        Self::update_point_libs(tmp_mark, features, position, chains, prev_ko_point, update_turn);
      }
      if let Some((_, ko_point)) = position.ko {
        Self::update_point_libs(tmp_mark, features, position, chains, ko_point, update_turn);
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

      // XXX(20160202): Update player rank features.
      match self.black_rank {
        None => {
          let rank = position.ranks[0];
          Self::update_points_rank(features, position, Stone::Black, rank);
          self.black_rank = Some(rank);
        }
        Some(rank) => {
          assert_eq!(rank, position.ranks[0]);
        }
      }
      match self.white_rank {
        None => {
          let rank = position.ranks[1];
          Self::update_points_rank(features, position, Stone::White, rank);
          self.white_rank = Some(rank);
        }
        Some(rank) => {
          assert_eq!(rank, position.ranks[1]);
        }
      }
    }
  }
}

#[derive(Clone, RustcDecodable, RustcEncodable)]
pub struct TxnStateAlphaMiniV3FeatsData {
  pub features: Vec<u8>,
  prev_moves:   Vec<Option<(Stone, Point)>>,
  tmp_mark:     BitSet,
}

impl TxnStateAlphaMiniV3FeatsData {
  pub const SET: u8 = 255;

  pub const NUM_PLANES:         usize = 16;
  pub const NUM_EXTRACT_PLANES: usize = 16;

  pub const BASELINE_PLANE:     usize = 0 * Board::SIZE;
  pub const EMPTY_PLANE:        usize = 1 * Board::SIZE;
  pub const BLACK_PLANE:        usize = 2 * Board::SIZE;
  pub const WHITE_PLANE:        usize = 3 * Board::SIZE;
  pub const TURNS_1_PLANE:      usize = 4 * Board::SIZE;
  pub const TURNS_2_PLANE:      usize = 5 * Board::SIZE;
  pub const TURNS_3_PLANE:      usize = 6 * Board::SIZE;
  pub const TURNS_4_PLANE:      usize = 7 * Board::SIZE;
  pub const LIBS_1_PLANE:       usize = 8 * Board::SIZE;
  pub const LIBS_2_PLANE:       usize = 9 * Board::SIZE;
  pub const LIBS_3_PLANE:       usize = 10 * Board::SIZE;
  pub const LIBS_4_PLANE:       usize = 11 * Board::SIZE;
  pub const CAPS_1_PLANE:       usize = 12 * Board::SIZE;
  pub const CAPS_2_PLANE:       usize = 13 * Board::SIZE;
  pub const CAPS_3_PLANE:       usize = 14 * Board::SIZE;
  pub const KO_PLANE:           usize = 15 * Board::SIZE;

  pub const BASELINE_EXT_PLANE:     usize = 0 * Board::SIZE;
  pub const EMPTY_EXT_PLANE:        usize = 1 * Board::SIZE;
  pub const FRIEND_EXT_PLANE:       usize = 2 * Board::SIZE;
  pub const ENEMY_EXT_PLANE:        usize = 3 * Board::SIZE;
  pub const TURNS_1_EXT_PLANE:      usize = 4 * Board::SIZE;
  pub const TURNS_2_EXT_PLANE:      usize = 5 * Board::SIZE;
  pub const TURNS_3_EXT_PLANE:      usize = 6 * Board::SIZE;
  pub const TURNS_4_EXT_PLANE:      usize = 7 * Board::SIZE;
  pub const LIBS_1_EXT_PLANE:       usize = 8 * Board::SIZE;
  pub const LIBS_2_EXT_PLANE:       usize = 9 * Board::SIZE;
  pub const LIBS_3_EXT_PLANE:       usize = 10 * Board::SIZE;
  pub const LIBS_4_EXT_PLANE:       usize = 11 * Board::SIZE;
  pub const CAPS_1_EXT_PLANE:       usize = 12 * Board::SIZE;
  pub const CAPS_2_EXT_PLANE:       usize = 13 * Board::SIZE;
  pub const CAPS_3_EXT_PLANE:       usize = 14 * Board::SIZE;
  pub const KO_EXT_PLANE:           usize = 15 * Board::SIZE;

  pub fn new() -> TxnStateAlphaMiniV3FeatsData {
    let feats = TxnStateAlphaMiniV3FeatsData{
      features:     repeat(0).take(Self::NUM_PLANES * Board::SIZE).collect(),
      prev_moves:   vec![None, None, None, None],
      tmp_mark:     BitSet::with_capacity(Board::SIZE),
    };
    assert_eq!(feats.features.len(), Self::NUM_PLANES * Board::SIZE);
    feats
  }

  pub fn from_src_feats(src: &TxnStateAlphaV3FeatsData) -> TxnStateAlphaMiniV3FeatsData {
    type SrcFeats = TxnStateAlphaV3FeatsData;
    let plane = Board::SIZE;
    let pairs = [
        (SrcFeats::BASELINE_PLANE,  Self::BASELINE_PLANE),
        (SrcFeats::EMPTY_PLANE,     Self::EMPTY_PLANE),
        (SrcFeats::BLACK_PLANE,     Self::BLACK_PLANE),
        (SrcFeats::WHITE_PLANE,     Self::WHITE_PLANE),
        (SrcFeats::TURNS_1_PLANE,   Self::TURNS_1_PLANE),
        (SrcFeats::TURNS_2_PLANE,   Self::TURNS_2_PLANE),
        (SrcFeats::TURNS_3_PLANE,   Self::TURNS_3_PLANE),
        (SrcFeats::TURNS_4_PLANE,   Self::TURNS_4_PLANE),
        (SrcFeats::LIBS_1_PLANE,    Self::LIBS_1_PLANE),
        (SrcFeats::LIBS_2_PLANE,    Self::LIBS_2_PLANE),
        (SrcFeats::LIBS_3_PLANE,    Self::LIBS_3_PLANE),
        (SrcFeats::LIBS_4_PLANE,    Self::LIBS_4_PLANE),
        (SrcFeats::CAPS_1_PLANE,    Self::CAPS_1_PLANE),
        (SrcFeats::CAPS_2_PLANE,    Self::CAPS_2_PLANE),
        (SrcFeats::CAPS_3_PLANE,    Self::CAPS_3_PLANE),
        (SrcFeats::KO_PLANE,        Self::KO_PLANE),
    ];
    let mut dst = TxnStateAlphaMiniV3FeatsData::new();
    for &(src_offset, dst_offset) in pairs.iter() {
      copy_memory(
              &src.features[src_offset .. src_offset + plane],
          &mut dst.features[dst_offset .. dst_offset + plane],
      );
    }
    dst.prev_moves[0] = src.prev_moves[0];
    dst.prev_moves[1] = src.prev_moves[1];
    dst.prev_moves[2] = src.prev_moves[2];
    dst.prev_moves[3] = src.prev_moves[3];
    dst
  }

  pub fn serial_size() -> usize {
    BitArray3d::serial_size((19, 19, 16))
  }

  pub fn extract_relative_serial_blob(&self, turn: Stone) -> Vec<u8> {
    let mut serial_frame: Vec<u8> = Vec::with_capacity(Self::serial_size());
    let bit_arr = self.extract_relative_serial_array(turn);
    bit_arr.serialize(&mut serial_frame).unwrap();
    serial_frame
  }

  pub fn extract_relative_serial_array(&self, turn: Stone) -> BitArray3d {
    let mut buf: Vec<u8> = Vec::with_capacity(Self::NUM_EXTRACT_PLANES * Board::SIZE);
    unsafe { buf.set_len(Self::NUM_EXTRACT_PLANES * Board::SIZE) };
    self.extract_relative_features(turn, &mut buf);

    let mut bit_buf: Vec<u8> = Vec::with_capacity(Self::NUM_EXTRACT_PLANES * Board::SIZE);
    unsafe { bit_buf.set_len(Self::NUM_EXTRACT_PLANES * Board::SIZE) };
    copy_memory(
        &buf[ .. Self::NUM_EXTRACT_PLANES * Board::SIZE],
        &mut bit_buf,
    );

    let bit_arr = BitArray3d::from_byte_array(&Array3d::with_data(bit_buf, (Board::DIM, Board::DIM, Self::NUM_EXTRACT_PLANES)));

    bit_arr
  }

  pub fn extract_relative_features(&self, turn: Stone, dst_buf: &mut [u8]) {
    assert_eq!(dst_buf.len(), Self::NUM_EXTRACT_PLANES * Board::SIZE);
    match turn {
      Stone::Black => {
        /*copy_memory(
            &self.features[ .. Self::BLACK_PLANE],
            &mut dst_buf[ .. Self::FRIEND_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::BLACK_PLANE .. Self::TURNS_1_PLANE],
            &mut dst_buf[Self::FRIEND_EXT_PLANE .. Self::TURNS_1_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::TURNS_1_PLANE .. ],
            &mut dst_buf[Self::TURNS_1_EXT_PLANE .. ],
        );*/
        copy_memory(
            &self.features,
            dst_buf,
        );
      }
      Stone::White => {
        copy_memory(
            &self.features[ .. Self::BLACK_PLANE],
            &mut dst_buf[ .. Self::FRIEND_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::WHITE_PLANE .. Self::TURNS_1_PLANE],
            &mut dst_buf[Self::FRIEND_EXT_PLANE .. Self::ENEMY_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::BLACK_PLANE .. Self::WHITE_PLANE],
            &mut dst_buf[Self::ENEMY_EXT_PLANE .. Self::TURNS_1_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::TURNS_1_PLANE .. ],
            &mut dst_buf[Self::TURNS_1_EXT_PLANE .. ],
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
        features[Self::TURNS_2_PLANE + p] = Self::SET;
      }
      3 => {
        features[Self::TURNS_3_PLANE + p] = Self::SET;
      }
      4 => {
        features[Self::TURNS_4_PLANE + p] = Self::SET;
      }
      5 => {
        // Do nothing.
      }
      _ => unreachable!(),
    }
  }

  fn update_point_libs(tmp_mark: &mut BitSet, features: &mut [u8], position: &TxnPosition, chains: &TxnChainsList, point: Point, turn: Stone) {
    let p = point.idx();
    if tmp_mark.contains(p) {
      return;
    }

    let stone = position.stones[p];
    if stone != Stone::Empty {
      let head = chains.find_chain(point);
      //assert!(head != TOMBSTONE);
      let chain = chains.get_chain(head).unwrap();

      // Liberty features.
      match chain.count_libs_up_to_4() {
        0 => { unreachable!(); }
        1 => {
          features[Self::LIBS_1_PLANE + p] = Self::SET;
          features[Self::LIBS_2_PLANE + p] = 0;
          features[Self::LIBS_3_PLANE + p] = 0;
          features[Self::LIBS_4_PLANE + p] = 0;
        }
        2 => {
          features[Self::LIBS_1_PLANE + p] = Self::SET;
          features[Self::LIBS_2_PLANE + p] = Self::SET;
          features[Self::LIBS_3_PLANE + p] = 0;
          features[Self::LIBS_4_PLANE + p] = 0;
        }
        3 => {
          features[Self::LIBS_1_PLANE + p] = Self::SET;
          features[Self::LIBS_2_PLANE + p] = Self::SET;
          features[Self::LIBS_3_PLANE + p] = Self::SET;
          features[Self::LIBS_4_PLANE + p] = 0;
        }
        4 => {
          features[Self::LIBS_1_PLANE + p] = Self::SET;
          features[Self::LIBS_2_PLANE + p] = Self::SET;
          features[Self::LIBS_3_PLANE + p] = Self::SET;
          features[Self::LIBS_4_PLANE + p] = Self::SET;
        }
        _ => { unreachable!(); }
      }

      // Chain size features.
      match chain.count_length() {
        0 => { unreachable!(); }
        1 => {
          features[Self::CAPS_1_PLANE + p] = Self::SET;
          features[Self::CAPS_2_PLANE + p] = 0;
          features[Self::CAPS_3_PLANE + p] = 0;
        }
        2 => {
          features[Self::CAPS_1_PLANE + p] = Self::SET;
          features[Self::CAPS_2_PLANE + p] = Self::SET;
          features[Self::CAPS_3_PLANE + p] = 0;
        }
        n => {
          features[Self::CAPS_1_PLANE + p] = Self::SET;
          features[Self::CAPS_2_PLANE + p] = Self::SET;
          features[Self::CAPS_3_PLANE + p] = Self::SET;
        }
      }

    } else {
      features[Self::LIBS_1_PLANE + p] = 0;
      features[Self::LIBS_2_PLANE + p] = 0;
      features[Self::LIBS_3_PLANE + p] = 0;
      features[Self::LIBS_4_PLANE + p] = 0;

      features[Self::CAPS_1_PLANE + p] = 0;
      features[Self::CAPS_2_PLANE + p] = 0;
      features[Self::CAPS_3_PLANE + p] = 0;
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

impl TxnStateData for TxnStateAlphaMiniV3FeatsData {
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
  }

  fn update(&mut self, position: &TxnPosition, chains: &TxnChainsList, update_turn: Stone, update_action: Action) {
    {
      // XXX(20151107): Iterate over placed and killed chains to update stone
      // repr.
      let &mut TxnStateAlphaMiniV3FeatsData{
        ref mut features,
        ref mut tmp_mark, .. } = self;

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
      for t in (0 .. 4).rev() {
        match self.prev_moves[t] {
          Some((_, prev_point)) => {
            Self::update_point_turns(features, prev_point, t + 2);
          }
          None => {}
        }
      }
      match update_action {
        Action::Place{point} => {
          Self::update_point_turns(features, point, 1);
        }
        _ => {}
      }
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
      if let Some((_, prev_ko_point)) = position.prev_ko {
        Self::update_point_libs(tmp_mark, features, position, chains, prev_ko_point, update_turn);
      }
      if let Some((_, ko_point)) = position.ko {
        Self::update_point_libs(tmp_mark, features, position, chains, ko_point, update_turn);
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

#[derive(Clone, RustcDecodable, RustcEncodable)]
pub struct TxnStateAlphaPatternV3FeatsData {
  pub features: Vec<u8>,
  prev_moves:   Vec<Option<(Stone, Point)>>,
  tmp_mark:     BitSet,
}

impl TxnStateAlphaPatternV3FeatsData {
  pub const SET: u8 = 255;

  pub const NUM_PLANES:         usize = 8;
  pub const NUM_EXTRACT_PLANES: usize = 8;

  pub const EMPTY_PLANE:        usize = 0 * Board::SIZE;
  pub const BLACK_PLANE:        usize = 1 * Board::SIZE;
  pub const WHITE_PLANE:        usize = 2 * Board::SIZE;
  pub const TURNS_1_PLANE:      usize = 3 * Board::SIZE;
  pub const LIBS_1_PLANE:       usize = 4 * Board::SIZE;
  pub const LIBS_2_PLANE:       usize = 5 * Board::SIZE;
  pub const LIBS_3_PLANE:       usize = 6 * Board::SIZE;
  pub const KO_PLANE:           usize = 7 * Board::SIZE;

  pub const EMPTY_EXT_PLANE:    usize = 0 * Board::SIZE;
  pub const FRIEND_EXT_PLANE:   usize = 1 * Board::SIZE;
  pub const ENEMY_EXT_PLANE:    usize = 2 * Board::SIZE;
  pub const TURNS_1_EXT_PLANE:  usize = 3 * Board::SIZE;
  pub const LIBS_1_EXT_PLANE:   usize = 4 * Board::SIZE;
  pub const LIBS_2_EXT_PLANE:   usize = 5 * Board::SIZE;
  pub const LIBS_3_EXT_PLANE:   usize = 6 * Board::SIZE;
  pub const KO_EXT_PLANE:       usize = 7 * Board::SIZE;

  pub fn new() -> TxnStateAlphaPatternV3FeatsData {
    let feats = TxnStateAlphaPatternV3FeatsData{
      features:     repeat(0).take(Self::NUM_PLANES * Board::SIZE).collect(),
      prev_moves:   vec![None],
      tmp_mark:     BitSet::with_capacity(Board::SIZE),
    };
    assert_eq!(feats.features.len(), Self::NUM_PLANES * Board::SIZE);
    feats
  }

  pub fn from_src_feats(src: &TxnStateAlphaV3FeatsData) -> TxnStateAlphaPatternV3FeatsData {
    type SrcFeats = TxnStateAlphaV3FeatsData;
    let plane = Board::SIZE;
    let pairs = [
        (SrcFeats::EMPTY_PLANE,     Self::EMPTY_PLANE),
        (SrcFeats::BLACK_PLANE,     Self::BLACK_PLANE),
        (SrcFeats::WHITE_PLANE,     Self::WHITE_PLANE),
        (SrcFeats::TURNS_1_PLANE,   Self::TURNS_1_PLANE),
        (SrcFeats::LIBS_1_PLANE,    Self::LIBS_1_PLANE),
        (SrcFeats::LIBS_2_PLANE,    Self::LIBS_2_PLANE),
        (SrcFeats::LIBS_3_PLANE,    Self::LIBS_3_PLANE),
        (SrcFeats::KO_PLANE,        Self::KO_PLANE),
    ];
    let mut dst = TxnStateAlphaPatternV3FeatsData::new();
    for &(src_offset, dst_offset) in pairs.iter() {
      copy_memory(
              &src.features[src_offset .. src_offset + plane],
          &mut dst.features[dst_offset .. dst_offset + plane],
      );
    }
    dst.prev_moves[0] = src.prev_moves[0];
    dst
  }

  pub fn serial_size() -> usize {
    BitArray3d::serial_size((19, 19, 8))
  }

  pub fn extract_relative_serial_blob(&self, turn: Stone) -> Vec<u8> {
    let mut serial_frame: Vec<u8> = Vec::with_capacity(Self::serial_size());
    let bit_arr = self.extract_relative_serial_array(turn);
    bit_arr.serialize(&mut serial_frame).unwrap();
    serial_frame
  }

  pub fn extract_relative_serial_array(&self, turn: Stone) -> BitArray3d {
    let mut buf: Vec<u8> = Vec::with_capacity(Self::NUM_EXTRACT_PLANES * Board::SIZE);
    unsafe { buf.set_len(Self::NUM_EXTRACT_PLANES * Board::SIZE) };
    self.extract_relative_features(turn, &mut buf);

    let mut bit_buf: Vec<u8> = Vec::with_capacity(Self::NUM_EXTRACT_PLANES * Board::SIZE);
    unsafe { bit_buf.set_len(Self::NUM_EXTRACT_PLANES * Board::SIZE) };
    copy_memory(
        &buf[ .. Self::NUM_EXTRACT_PLANES * Board::SIZE],
        &mut bit_buf,
    );

    let bit_arr = BitArray3d::from_byte_array(&Array3d::with_data(bit_buf, (Board::DIM, Board::DIM, Self::NUM_EXTRACT_PLANES)));

    bit_arr
  }

  pub fn extract_relative_features(&self, turn: Stone, dst_buf: &mut [u8]) {
    assert_eq!(dst_buf.len(), Self::NUM_EXTRACT_PLANES * Board::SIZE);
    match turn {
      Stone::Black => {
        /*copy_memory(
            &self.features[ .. Self::BLACK_PLANE],
            &mut dst_buf[ .. Self::FRIEND_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::BLACK_PLANE .. Self::TURNS_1_PLANE],
            &mut dst_buf[Self::FRIEND_EXT_PLANE .. Self::TURNS_1_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::TURNS_1_PLANE .. ],
            &mut dst_buf[Self::TURNS_1_EXT_PLANE .. ],
        );*/
        copy_memory(
            &self.features,
            dst_buf,
        );
      }
      Stone::White => {
        copy_memory(
            &self.features[ .. Self::BLACK_PLANE],
            &mut dst_buf[ .. Self::FRIEND_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::WHITE_PLANE .. Self::TURNS_1_PLANE],
            &mut dst_buf[Self::FRIEND_EXT_PLANE .. Self::ENEMY_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::BLACK_PLANE .. Self::WHITE_PLANE],
            &mut dst_buf[Self::ENEMY_EXT_PLANE .. Self::TURNS_1_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::TURNS_1_PLANE .. ],
            &mut dst_buf[Self::TURNS_1_EXT_PLANE .. ],
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
      }
      _ => unreachable!(),
    }
  }

  fn update_point_libs(tmp_mark: &mut BitSet, features: &mut [u8], position: &TxnPosition, chains: &TxnChainsList, point: Point, turn: Stone) {
    let p = point.idx();
    if tmp_mark.contains(p) {
      return;
    }

    let stone = position.stones[p];
    if stone != Stone::Empty {
      let head = chains.find_chain(point);
      //assert!(head != TOMBSTONE);
      let chain = chains.get_chain(head).unwrap();

      // Liberty features.
      match chain.count_libs_up_to_3() {
        0 => { unreachable!(); }
        1 => {
          features[Self::LIBS_1_PLANE + p] = Self::SET;
          features[Self::LIBS_2_PLANE + p] = 0;
          features[Self::LIBS_3_PLANE + p] = 0;
        }
        2 => {
          features[Self::LIBS_1_PLANE + p] = Self::SET;
          features[Self::LIBS_2_PLANE + p] = Self::SET;
          features[Self::LIBS_3_PLANE + p] = 0;
        }
        3 => {
          features[Self::LIBS_1_PLANE + p] = Self::SET;
          features[Self::LIBS_2_PLANE + p] = Self::SET;
          features[Self::LIBS_3_PLANE + p] = Self::SET;
        }
        _ => { unreachable!(); }
      }

    } else {
      features[Self::LIBS_1_PLANE + p] = 0;
      features[Self::LIBS_2_PLANE + p] = 0;
      features[Self::LIBS_3_PLANE + p] = 0;
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

impl TxnStateData for TxnStateAlphaPatternV3FeatsData {
  fn reset(&mut self) {
    for p in 0 .. Self::BLACK_PLANE {
      self.features[p] = Self::SET;
    }
    for p in Self::BLACK_PLANE .. Self::NUM_PLANES * Board::SIZE {
      self.features[p] = 0;
    }
    self.prev_moves[0] = None;
  }

  fn update(&mut self, position: &TxnPosition, chains: &TxnChainsList, update_turn: Stone, update_action: Action) {
    {
      // XXX(20151107): Iterate over placed and killed chains to update stone
      // repr.
      let &mut TxnStateAlphaPatternV3FeatsData{
        ref mut features,
        ref mut tmp_mark, .. } = self;

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
      for t in (0 .. 1).rev() {
        match self.prev_moves[t] {
          Some((_, prev_point)) => {
            Self::update_point_turns(features, prev_point, t + 2);
          }
          None => {}
        }
      }
      match update_action {
        Action::Place{point} => {
          Self::update_point_turns(features, point, 1);
        }
        _ => {}
      }
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
      if let Some((_, prev_ko_point)) = position.prev_ko {
        Self::update_point_libs(tmp_mark, features, position, chains, prev_ko_point, update_turn);
      }
      if let Some((_, ko_point)) = position.ko {
        Self::update_point_libs(tmp_mark, features, position, chains, ko_point, update_turn);
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

#[derive(Clone, RustcDecodable, RustcEncodable)]
pub struct TxnStateAlphaPatternNoKoV3FeatsData {
  pub features: Vec<u8>,
  prev_moves:   Vec<Option<(Stone, Point)>>,
  tmp_mark:     BitSet,
}

impl TxnStateAlphaPatternNoKoV3FeatsData {
  pub const SET: u8 = 255;

  pub const NUM_PLANES:         usize = 7;
  pub const NUM_EXTRACT_PLANES: usize = 7;

  pub const EMPTY_PLANE:        usize = 0 * Board::SIZE;
  pub const BLACK_PLANE:        usize = 1 * Board::SIZE;
  pub const WHITE_PLANE:        usize = 2 * Board::SIZE;
  pub const LIBS_1_PLANE:       usize = 3 * Board::SIZE;
  pub const LIBS_2_PLANE:       usize = 4 * Board::SIZE;
  pub const LIBS_3_PLANE:       usize = 5 * Board::SIZE;
  pub const TURNS_1_PLANE:      usize = 6 * Board::SIZE;

  pub const EMPTY_EXT_PLANE:    usize = 0 * Board::SIZE;
  pub const FRIEND_EXT_PLANE:   usize = 1 * Board::SIZE;
  pub const ENEMY_EXT_PLANE:    usize = 2 * Board::SIZE;
  pub const LIBS_1_EXT_PLANE:   usize = 3 * Board::SIZE;
  pub const LIBS_2_EXT_PLANE:   usize = 4 * Board::SIZE;
  pub const LIBS_3_EXT_PLANE:   usize = 5 * Board::SIZE;
  pub const TURNS_1_EXT_PLANE:  usize = 6 * Board::SIZE;

  pub fn new() -> TxnStateAlphaPatternNoKoV3FeatsData {
    let feats = TxnStateAlphaPatternNoKoV3FeatsData{
      features:     repeat(0).take(Self::NUM_PLANES * Board::SIZE).collect(),
      prev_moves:   vec![None],
      tmp_mark:     BitSet::with_capacity(Board::SIZE),
    };
    assert_eq!(feats.features.len(), Self::NUM_PLANES * Board::SIZE);
    feats
  }

  pub fn from_src_feats(src: &TxnStateAlphaV3FeatsData) -> TxnStateAlphaPatternNoKoV3FeatsData {
    type SrcFeats = TxnStateAlphaV3FeatsData;
    let plane = Board::SIZE;
    let pairs = [
        (SrcFeats::EMPTY_PLANE,     Self::EMPTY_PLANE),
        (SrcFeats::BLACK_PLANE,     Self::BLACK_PLANE),
        (SrcFeats::WHITE_PLANE,     Self::WHITE_PLANE),
        (SrcFeats::LIBS_1_PLANE,    Self::LIBS_1_PLANE),
        (SrcFeats::LIBS_2_PLANE,    Self::LIBS_2_PLANE),
        (SrcFeats::LIBS_3_PLANE,    Self::LIBS_3_PLANE),
        (SrcFeats::TURNS_1_PLANE,   Self::TURNS_1_PLANE),
    ];
    let mut dst = TxnStateAlphaPatternNoKoV3FeatsData::new();
    for &(src_offset, dst_offset) in pairs.iter() {
      copy_memory(
              &src.features[src_offset .. src_offset + plane],
          &mut dst.features[dst_offset .. dst_offset + plane],
      );
    }
    dst.prev_moves[0] = src.prev_moves[0];
    dst
  }

  pub fn serial_size() -> usize {
    BitArray3d::serial_size((19, 19, 7))
  }

  pub fn extract_relative_serial_blob(&self, turn: Stone) -> Vec<u8> {
    let mut serial_frame: Vec<u8> = Vec::with_capacity(Self::serial_size());
    let bit_arr = self.extract_relative_serial_array(turn);
    bit_arr.serialize(&mut serial_frame).unwrap();
    serial_frame
  }

  pub fn extract_relative_serial_array(&self, turn: Stone) -> BitArray3d {
    let mut buf: Vec<u8> = Vec::with_capacity(Self::NUM_EXTRACT_PLANES * Board::SIZE);
    unsafe { buf.set_len(Self::NUM_EXTRACT_PLANES * Board::SIZE) };
    self.extract_relative_features(turn, &mut buf);

    let mut bit_buf: Vec<u8> = Vec::with_capacity(Self::NUM_EXTRACT_PLANES * Board::SIZE);
    unsafe { bit_buf.set_len(Self::NUM_EXTRACT_PLANES * Board::SIZE) };
    copy_memory(
        &buf[ .. Self::NUM_EXTRACT_PLANES * Board::SIZE],
        &mut bit_buf,
    );

    let bit_arr = BitArray3d::from_byte_array(&Array3d::with_data(bit_buf, (Board::DIM, Board::DIM, Self::NUM_EXTRACT_PLANES)));

    bit_arr
  }

  pub fn extract_relative_features(&self, turn: Stone, dst_buf: &mut [u8]) {
    assert_eq!(dst_buf.len(), Self::NUM_EXTRACT_PLANES * Board::SIZE);
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
            &mut dst_buf[ .. Self::FRIEND_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::WHITE_PLANE .. Self::LIBS_1_PLANE],
            &mut dst_buf[Self::FRIEND_EXT_PLANE .. Self::ENEMY_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::BLACK_PLANE .. Self::WHITE_PLANE],
            &mut dst_buf[Self::ENEMY_EXT_PLANE .. Self::LIBS_1_EXT_PLANE],
        );
        copy_memory(
            &self.features[Self::LIBS_1_PLANE .. ],
            &mut dst_buf[Self::LIBS_1_EXT_PLANE .. ],
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
      }
      _ => unreachable!(),
    }
  }

  fn update_point_libs(tmp_mark: &mut BitSet, features: &mut [u8], position: &TxnPosition, chains: &TxnChainsList, point: Point, turn: Stone) {
    let p = point.idx();
    if tmp_mark.contains(p) {
      return;
    }

    let stone = position.stones[p];
    if stone != Stone::Empty {
      let head = chains.find_chain(point);
      //assert!(head != TOMBSTONE);
      let chain = chains.get_chain(head).unwrap();

      // Liberty features.
      match chain.count_libs_up_to_3() {
        0 => { unreachable!(); }
        1 => {
          features[Self::LIBS_1_PLANE + p] = Self::SET;
          features[Self::LIBS_2_PLANE + p] = 0;
          features[Self::LIBS_3_PLANE + p] = 0;
        }
        2 => {
          features[Self::LIBS_1_PLANE + p] = Self::SET;
          features[Self::LIBS_2_PLANE + p] = Self::SET;
          features[Self::LIBS_3_PLANE + p] = 0;
        }
        3 => {
          features[Self::LIBS_1_PLANE + p] = Self::SET;
          features[Self::LIBS_2_PLANE + p] = Self::SET;
          features[Self::LIBS_3_PLANE + p] = Self::SET;
        }
        _ => { unreachable!(); }
      }

    } else {
      features[Self::LIBS_1_PLANE + p] = 0;
      features[Self::LIBS_2_PLANE + p] = 0;
      features[Self::LIBS_3_PLANE + p] = 0;
    }

    tmp_mark.insert(p);
  }
}

impl TxnStateData for TxnStateAlphaPatternNoKoV3FeatsData {
  fn reset(&mut self) {
    for p in 0 .. Self::BLACK_PLANE {
      self.features[p] = Self::SET;
    }
    for p in Self::BLACK_PLANE .. Self::NUM_PLANES * Board::SIZE {
      self.features[p] = 0;
    }
    self.prev_moves[0] = None;
  }

  fn update(&mut self, position: &TxnPosition, chains: &TxnChainsList, update_turn: Stone, update_action: Action) {
    {
      // XXX(20151107): Iterate over placed and killed chains to update stone
      // repr.
      let &mut TxnStateAlphaPatternNoKoV3FeatsData{
        ref mut features,
        ref mut tmp_mark, .. } = self;

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
      for t in (0 .. 1).rev() {
        match self.prev_moves[t] {
          Some((_, prev_point)) => {
            Self::update_point_turns(features, prev_point, t + 2);
          }
          None => {}
        }
      }
      match update_action {
        Action::Place{point} => {
          Self::update_point_turns(features, point, 1);
        }
        _ => {}
      }
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
      if let Some((_, prev_ko_point)) = position.prev_ko {
        Self::update_point_libs(tmp_mark, features, position, chains, prev_ko_point, update_turn);
      }
      if let Some((_, ko_point)) = position.ko {
        Self::update_point_libs(tmp_mark, features, position, chains, ko_point, update_turn);
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
    }
  }
}
