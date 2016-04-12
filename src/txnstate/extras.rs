use board::{Board, Rules, RuleSet, Coord, PlayerRank, Stone, Point, Action};
use txnstate::{
  TxnStateData, TxnStateConfig, TxnState, TxnPosition, TxnChainsList,
  for_each_adjacent,
  check_illegal_move_simple,
  check_legal_move_simple,
  check_good_move_fast,
};
use txnstate::features::{
  TxnStateFeaturesData,
  TxnStateLibFeaturesData,
  TxnStateAlphaFeatsV2Data,
  TxnStateAlphaV3FeatsData,
  TxnStateAlphaMiniV3FeatsData,
};

use bit_set::{BitSet};
use bit_vec::{BitVec};

pub fn for_each_touched_empty<F>(position: &TxnPosition, chains: &TxnChainsList, mut update: F)
where F: FnMut(&TxnPosition, &TxnChainsList, Point) {
  if let Some((_, place_point)) = position.last_placed {
    update(position, chains, place_point);
    for_each_adjacent(place_point, |adj_point| {
      update(position, chains, adj_point);
    });
  }
  if let Some((_, ko_point)) = position.ko {
    update(position, chains, ko_point);
  }
  if let Some((_, prev_ko_point)) = position.prev_ko {
    update(position, chains, prev_ko_point);
  }
  for &kill_point in position.last_killed[0].iter().chain(position.last_killed[1].iter()) {
    update(position, chains, kill_point);
  }
  for &head in chains.last_mut_heads.iter() {
    if let Some(chain) = chains.get_chain(head) {
      for &lib in chain.ps_libs.iter() {
        update(position, chains, lib);
      }
    }
  }
}

#[derive(Clone, RustcDecodable, RustcEncodable)]
pub struct TxnStateLegalityData {
  strict:       bool,

  cached_legal_moves: Vec<BitSet>,
  test_state:   TxnState,

  tmp_mark:     BitSet,
}

impl TxnStateLegalityData {
  pub fn new(strict: bool) -> TxnStateLegalityData {
    TxnStateLegalityData{
      strict:   strict,
      cached_legal_moves: vec![
        BitSet::from_bit_vec(BitVec::from_elem(Board::SIZE, true)),
        BitSet::from_bit_vec(BitVec::from_elem(Board::SIZE, true)),
      ],
      test_state: TxnState::new(TxnStateConfig::default(), ()),
      tmp_mark: BitSet::with_capacity(Board::SIZE),
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

  pub fn get_legal_points(&self, turn: Stone) -> &BitSet {
    let turn_off = turn.offset();
    &self.cached_legal_moves[turn_off]
  }

  pub fn legal_points(&self, turn: Stone) -> BitSet {
    let turn_off = turn.offset();
    self.cached_legal_moves[turn_off].clone()
  }

  pub fn extract_legal_mask(&self, turn: Stone, mask: &mut [u8]) {
    assert_eq!(Board::SIZE, mask.len());
    let turn_off = turn.offset();
    for (i, elem) in self.cached_legal_moves[turn_off].get_ref().iter().enumerate() {
      match elem {
        false => mask[i] = 0,
        true  => mask[i] = 1,
      }
    }
  }

  fn update_point(&mut self, position: &TxnPosition, chains: &TxnChainsList, point: Point) {
    let p = point.idx();
    if self.tmp_mark.contains(p) {
      return;
    }
    for &turn in [Stone::Black, Stone::White].iter() {
      let turn_off = turn.offset();
      // XXX(20160222): Previously, this was a "strict" test; i.e., only truly
      // illegal moves were filtered out. Now, we also filter out "bad" moves,
      // i.e., ones that fill in the current player's eye.
      match check_legal_move_simple(position, chains, turn, point) {
        Some(Ok(())) => {
          if !self.strict {
            if check_good_move_fast(position, chains, turn, point) {
              self.cached_legal_moves[turn_off].insert(p);
            } else {
              self.cached_legal_moves[turn_off].remove(p);
            }
          } else {
            self.cached_legal_moves[turn_off].insert(p);
          }
        }
        Some(Err(_)) => {
          self.cached_legal_moves[turn_off].remove(p);
        }
        None => {
          if !self.strict {
            if check_good_move_fast(position, chains, turn, point) {
              if self.test_state.try_place(turn, point).is_err() {
                self.cached_legal_moves[turn_off].remove(p);
              } else {
                self.cached_legal_moves[turn_off].insert(p);
              }
              self.test_state.undo();
            } else {
              self.cached_legal_moves[turn_off].remove(p);
            }
          } else {
            if self.test_state.try_place(turn, point).is_err() {
              self.cached_legal_moves[turn_off].remove(p);
            } else {
              self.cached_legal_moves[turn_off].insert(p);
            }
            self.test_state.undo();
          }
        }
      }
    }
    self.tmp_mark.insert(p);
  }
}

impl TxnStateData for TxnStateLegalityData {
  fn reset(&mut self) {
    self.cached_legal_moves[0] = BitSet::from_bit_vec(BitVec::from_elem(Board::SIZE, true));
    self.cached_legal_moves[1] = BitSet::from_bit_vec(BitVec::from_elem(Board::SIZE, true));
    self.test_state.reset();
  }

  fn update(&mut self, position: &TxnPosition, chains: &TxnChainsList, update_turn: Stone, update_action: Action) {
    //let prev_ko = self.test_state.current_ko();
    match self.test_state.try_action(update_turn, update_action) {
      Ok(_) => { self.test_state.commit(); }
      Err(_) => { panic!("TxnStateLegalityData encountered an illegal update!"); }
    }
    //assert_eq!(prev_ko, position.prev_ko);
    self.tmp_mark.clear();

    // TODO(20151112)
    // Update the points:
    // - the last placed stone and adjacent stones
    // - the current and previous ko points
    // - captured stones (now empty points)
    // - pseudolibs of chains adjacent to last placed stone
    // - pseudolibs of chains adjacent to captured stones
    // The last two could really use a `last_mut_heads` field in chains list.

    /*if let Some((_, place_point)) = position.last_placed {
      self.update_point(position, chains, place_point);
      for_each_adjacent(place_point, |adj_point| {
        self.update_point(position, chains, adj_point);
      });
    }
    if let Some((_, ko_point)) = position.ko {
      self.update_point(position, chains, ko_point);
    }
    //if let Some((_, prev_ko_point)) = prev_ko {
    if let Some((_, prev_ko_point)) = position.prev_ko {
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
    }*/
    for_each_touched_empty(position, chains, |position, chains, pt| {
      self.update_point(position, chains, pt);
    });
  }
}

#[derive(Clone, RustcDecodable, RustcEncodable)]
//pub struct TxnStateNodeData<Feats=TxnStateLibFeaturesData> where Feats: TxnStateData + Clone {
pub struct TxnStateNodeData {
  //pub features: TxnStateFeaturesData,
  //pub features: TxnStateLibFeaturesData,
  //pub features: TxnStateAlphaFeatsV2Data,
  //pub features: TxnStateAlphaV3FeatsData,
  pub features: TxnStateAlphaMiniV3FeatsData,
  pub legality: TxnStateLegalityData,
}

impl TxnStateNodeData {
  pub fn new() -> TxnStateNodeData {
    TxnStateNodeData{
      //features: TxnStateFeaturesData::new(),
      //features: TxnStateLibFeaturesData::new(),
      //features: TxnStateAlphaFeatsV2Data::new(),
      //features: TxnStateAlphaV3FeatsData::new(),
      features: TxnStateAlphaMiniV3FeatsData::new(),
      legality: TxnStateLegalityData::new(false),
    }
  }
}

//impl<Feats> TxnStateData for TxnStateNodeData<Feats> where Feats: TxnStateData {
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

#[derive(Clone, RustcDecodable, RustcEncodable)]
pub struct TxnStateRolloutData {
  //pub features: TxnStateLibFeaturesData,
  //pub features: TxnStateAlphaFeatsV2Data,
  pub features: TxnStateAlphaMiniV3FeatsData,
}

impl TxnStateRolloutData {
  pub fn new() -> TxnStateRolloutData {
    TxnStateRolloutData{
      //features: TxnStateLibFeaturesData::new(),
      //features: TxnStateAlphaFeatsV2Data::new(),
      features: TxnStateAlphaMiniV3FeatsData::new(),
    }
  }

  /*pub fn with_features(features: TxnStateAlphaFeatsV2Data) -> TxnStateRolloutData {
    TxnStateRolloutData{
      features: features,
    }
  }*/

  pub fn from_node_data(node_data: &TxnStateNodeData) -> TxnStateRolloutData {
    TxnStateRolloutData{
      features: node_data.features.clone(),
      //features: TxnStateAlphaMiniV3FeatsData::from_src_feats(&node_data.features),
    }
  }
}

impl TxnStateData for TxnStateRolloutData {
  fn reset(&mut self) {
    self.features.reset();
  }

  fn update(&mut self, position: &TxnPosition, chains: &TxnChainsList, update_turn: Stone, update_action: Action) {
    self.features.update(position, chains, update_turn, update_action);
  }
}

#[derive(Clone, RustcDecodable, RustcEncodable)]
pub struct TxnStateRolloutLegalityData {
  pub features: TxnStateAlphaMiniV3FeatsData,
  pub legality: TxnStateLegalityData,
}

impl TxnStateRolloutLegalityData {
  pub fn new() -> TxnStateRolloutLegalityData {
    TxnStateRolloutLegalityData{
      features: TxnStateAlphaMiniV3FeatsData::new(),
      legality: TxnStateLegalityData::new(false),
    }
  }
}

impl TxnStateData for TxnStateRolloutLegalityData {
  fn reset(&mut self) {
    self.features.reset();
    self.legality.reset();
  }

  fn update(&mut self, position: &TxnPosition, chains: &TxnChainsList, update_turn: Stone, update_action: Action) {
    self.features.update(position, chains, update_turn, update_action);
    self.legality.update(position, chains, update_turn, update_action);
  }
}

/*#[derive(Clone)]
pub struct TxnStateAllData {
  pub features: TxnStateFeaturesData,
  pub libfeats: TxnStateLibFeaturesData,
  pub legality: TxnStateLegalityData,
}

impl TxnStateAllData {
  pub fn new() -> TxnStateAllData {
    TxnStateAllData{
      features: TxnStateFeaturesData::new(),
      libfeats: TxnStateLibFeaturesData::new(),
      legality: TxnStateLegalityData::new(),
    }
  }
}

impl TxnStateData for TxnStateAllData {
  fn reset(&mut self) {
    self.features.reset();
    self.libfeats.reset();
    self.legality.reset();
  }

  fn update(&mut self, position: &TxnPosition, chains: &TxnChainsList, update_turn: Stone, update_action: Action) {
    self.features.update(position, chains, update_turn, update_action);
    self.libfeats.update(position, chains, update_turn, update_action);
    self.legality.update(position, chains, update_turn, update_action);
  }
}*/
