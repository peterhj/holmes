use board::{Board, Rules, RuleSet, Coord, Stone, Point, Action};
use txnstate::{
  TxnStateData, TxnState, TxnPosition, TxnChainsList,
  for_each_adjacent, check_illegal_move_simple,
};
use txnstate::features::{TxnStateFeaturesData};

use bit_set::{BitSet};
use bit_vec::{BitVec};

#[derive(Clone)]
pub struct TxnStateLegalityData {
  cached_legal_moves: Vec<BitSet>,
  test_state:         TxnState,
}

impl TxnStateLegalityData {
  pub fn new() -> TxnStateLegalityData {
    TxnStateLegalityData{
      cached_legal_moves: vec![
        BitSet::from_bit_vec(BitVec::from_elem(Board::SIZE, true)),
        BitSet::from_bit_vec(BitVec::from_elem(Board::SIZE, true)),
      ],
      test_state: TxnState::new(RuleSet::KgsJapanese.rules(), ()),
    }
  }

  pub fn from_node_state(state: &TxnState<TxnStateNodeData>) -> TxnStateLegalityData {
    state.get_data().legality.clone()
  }

  pub fn for_each_legal_point<F>(&self, turn: Stone, mut f: F) where F: FnMut(Point) {
    for p in self.cached_legal_moves[turn.offset()].iter() {
      let point = Point::from_idx(p);
      f(point);
    }
  }

  pub fn fill_legal_points(&self, turn: Stone, points: &mut Vec<Point>) {
    let turn_off = turn.offset();
    for p in self.cached_legal_moves[turn_off].iter() {
      points.push(Point(p as i16));
    }
  }

  fn update_point(&mut self, position: &TxnPosition, chains: &TxnChainsList, point: Point) {
    for &turn in [Stone::Black, Stone::White].iter() {
      let turn_off = turn.offset();
      if check_illegal_move_simple(position, chains, turn, point).is_some() {
        self.cached_legal_moves[turn_off].remove(&point.idx());
      } else {
        if self.test_state.try_place(turn, point).is_err() {
          self.cached_legal_moves[turn_off].remove(&point.idx());
        } else {
          self.cached_legal_moves[turn_off].insert(point.idx());
        }
        self.test_state.undo();
      }
    }
  }
}

impl TxnStateData for TxnStateLegalityData {
  fn reset(&mut self) {
    self.cached_legal_moves[0] = BitSet::from_bit_vec(BitVec::from_elem(Board::SIZE, true));
    self.cached_legal_moves[1] = BitSet::from_bit_vec(BitVec::from_elem(Board::SIZE, true));
    self.test_state.reset();
  }

  fn update(&mut self, position: &TxnPosition, chains: &TxnChainsList, update_turn: Stone, update_action: Action) {
    let prev_ko = self.test_state.current_ko();
    match self.test_state.try_action(update_turn, update_action) {
      Ok(_) => { self.test_state.commit(); }
      Err(_) => { panic!("TxnStateLegalityData encountered an illegal update!"); }
    }

    // TODO(20151112)
    // Update the points:
    // - the last placed stone and adjacent stones
    // - the current and previous ko points
    // - captured stones (now empty points)
    // - pseudolibs of chains adjacent to last placed stone
    // - pseudolibs of chains adjacent to captured stones
    // The last two could really use a `last_mut_heads` field in chains list.
    if let Some((_, place_point)) = position.last_placed {
      self.update_point(position, chains, place_point);
      for_each_adjacent(place_point, |adj_point| {
        self.update_point(position, chains, adj_point);
      });
    }
    if let Some((_, ko_point)) = position.ko {
      self.update_point(position, chains, ko_point);
    }
    if let Some((_, prev_ko_point)) = prev_ko {
      self.update_point(position, chains, prev_ko_point);
    }
    for &kill_point in position.last_killed[0].iter().chain(position.last_killed[1].iter()) {
      self.update_point(position, chains, kill_point);
    }
    for &head in chains.last_mut_heads.iter() {
      if let Some(chain) = chains.get_chain(head) {
        for &lib in chain.ps_libs.iter() {
          self.update_point(position, chains, lib);
        }
      }
    }
  }
}

#[derive(Clone)]
pub struct TxnStateNodeData {
  pub features: TxnStateFeaturesData,
  pub legality: TxnStateLegalityData,
}

impl TxnStateNodeData {
  pub fn new() -> TxnStateNodeData {
    TxnStateNodeData{
      features: TxnStateFeaturesData::new(),
      legality: TxnStateLegalityData::new(),
    }
  }
}

impl TxnStateData for TxnStateNodeData {
  fn reset(&mut self) {
    self.features.reset();
    self.legality.reset();
  }

  fn update(&mut self, position: &TxnPosition, chains: &TxnChainsList, update_turn: Stone, update_action: Action) {
    self.features.update(position, chains, update_turn, update_action);
    self.legality.update(position, chains, update_turn, update_action);
  }
}

