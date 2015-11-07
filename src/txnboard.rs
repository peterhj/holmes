use board::{Coord, Board, Stone, Point, Action};
use gtp_board::{dump_xcoord};

use bit_set::{BitSet};
use std::iter::{repeat};
use std::slice::bytes::{copy_memory};

pub const TOMBSTONE:  Point = Point(-1);

pub const BLACK_PLANE:      usize = 0;
pub const WHITE_PLANE:      usize = 1 * Board::SIZE;
/*pub const ATARI_PLANE:      usize = 2 * Board::SIZE;
pub const LIVE_2_PLANE:     usize = 3 * Board::SIZE;
pub const LIVE_3_PLANE:     usize = 4 * Board::SIZE;*/
pub const FEAT_PLANES:      usize = 2;
//pub const FEAT_PLANES:      usize = 5;
pub const FEAT_TIME_STEPS:  usize = 2;

pub fn for_each_adjacent<F>(point: Point, mut f: F) where F: FnMut(Point) {
  let (x, y) = (point.0 % Board::DIM_PT.0, point.0 / Board::DIM_PT.0);
  let upper = Board::DIM_PT.0 - 1;
  if x >= 1 {
    f(Point(point.0 - 1));
  }
  if y >= 1 {
    f(Point(point.0 - Board::DIM_PT.0));
  }
  if x < upper {
    f(Point(point.0 + 1));
  }
  if y < upper {
    f(Point(point.0 + Board::DIM_PT.0));
  }
}

pub fn for_each_diagonal<F>(point: Point, mut f: F) where F: FnMut(Point) {
  let (x, y) = (point.0 % Board::DIM_PT.0, point.0 / Board::DIM_PT.0);
  let upper = Board::DIM_PT.0 - 1;
  if x >= 1 && y >= 1 {
    f(Point(point.0 - 1 - Board::DIM_PT.0));
  }
  if x >= 1 && y < upper {
    f(Point(point.0 - 1 + Board::DIM_PT.0));
  }
  if x < upper && y >= 1 {
    f(Point(point.0 + 1 - Board::DIM_PT.0));
  }
  if x < upper && y < upper {
    f(Point(point.0 + 1 + Board::DIM_PT.0));
  }
}

pub type TxnResult = Result<(), TxnStatus>;

#[derive(Clone, Copy, Debug)]
pub enum TxnStatusIllegalReason {
  NotEmpty,
  Suicide,
  Ko,
}

#[derive(Clone, Copy, Debug)]
pub enum TxnStatus {
  Illegal,
  IllegalReason(TxnStatusIllegalReason),
}

pub trait TxnStateData {
  fn reset(&mut self) {
    // Do nothing.
  }

  fn update(&mut self, position: &TxnPosition, chains: &TxnChainsList) {
    // Do nothing.
  }

  fn commit(&mut self) {
    // Do nothing.
  }

  fn undo(&mut self) {
    // Do nothing.
  }
}

impl TxnStateData for () {
}

pub fn slice_twice_mut<T>(xs: &mut [T], lo1: usize, hi1: usize, lo2: usize, hi2: usize) -> (&mut [T], &mut [T]) {
  let mut flip = false;
  let (lo1, hi1, lo2, hi2) = if (lo1 <= lo2) {
    (lo1, hi1, lo2, hi2)
  } else {
    flip = true;
    (lo2, hi2, lo1, hi1)
  };
  assert!(hi1 <= lo2, "ranges must not overlap!");
  let (mut lo_xs, hi_xs) = xs.split_at_mut(lo2);
  let (lo_xs, _) = lo_xs.split_at_mut(hi1 - lo1);
  if !flip {
    (lo_xs, hi_xs)
  } else {
    (hi_xs, lo_xs)
  }
}

#[derive(Clone)]
pub struct TxnStateLightData {
  features: Vec<u8>,
  time_step_offset: usize,
}

impl TxnStateLightData {
  pub fn new() -> TxnStateLightData {
    TxnStateLightData{
      features: repeat(0).take(FEAT_TIME_STEPS * FEAT_PLANES * Board::SIZE).collect(),
      time_step_offset: 0,
    }
  }
}

impl TxnStateData for TxnStateLightData {
  fn reset(&mut self) {
    for p in (0 .. FEAT_TIME_STEPS * FEAT_PLANES * Board::SIZE) {
      self.features[p] = 0;
    }
    self.time_step_offset = 0;
  }

  fn update(&mut self, position: &TxnPosition, chains: &TxnChainsList) {
    let slice_sz = FEAT_PLANES * Board::SIZE;
    let next_slice_off = self.time_step_offset;
    if FEAT_TIME_STEPS > 1 {
      let prev_slice_off = (self.time_step_offset + FEAT_TIME_STEPS - 1) % FEAT_TIME_STEPS;
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
            features[BLACK_PLANE + p] = 1;
            features[WHITE_PLANE + p] = 0;
          }
          Stone::White => {
            features[BLACK_PLANE + p] = 0;
            features[WHITE_PLANE + p] = 1;
          }
          Stone::Empty => { unreachable!(); }
        }
      }
      for &(stone, point) in position.last_killed.iter() {
        let p = point.idx();
        match stone {
          Stone::Black => {
            features[BLACK_PLANE + p] = 1;
            features[WHITE_PLANE + p] = 0;
          }
          Stone::White => {
            features[BLACK_PLANE + p] = 0;
            features[WHITE_PLANE + p] = 1;
          }
          Stone::Empty => { unreachable!(); }
        }
      }
      // TODO(20151107): iterate over "touched" chains to update liberty repr.
    }
    self.time_step_offset = (self.time_step_offset + 1) % FEAT_TIME_STEPS;
  }
}

#[derive(Clone)]
pub struct TxnStateHeavyData {
  cached_illegal_moves: Vec<BitSet>,
}

impl TxnStateHeavyData {
  pub fn new() -> TxnStateHeavyData {
    TxnStateHeavyData{
      cached_illegal_moves: vec![
        BitSet::with_capacity(Board::SIZE),
        BitSet::with_capacity(Board::SIZE),
      ],
    }
  }
}

impl TxnStateData for TxnStateHeavyData {
  fn reset(&mut self) {
    self.cached_illegal_moves[0].clear();
    self.cached_illegal_moves[1].clear();
  }

  fn update(&mut self, position: &TxnPosition, chains: &TxnChainsList) {
    // TODO(20151106)
  }

  fn undo(&mut self) {
    panic!("FATAL: TxnStateHeavyData does not support txn undo!");
  }
}

#[derive(Clone, Debug)]
pub struct TxnPosition {
  turn:         Stone,
  stones:       Vec<Stone>,
  ko:           Option<(Stone, Point)>,
  // FIXME(20151107): semantics of when `last_placed` and `last_killed` are
  // valid is unclear; currently used as tmp variables.
  last_placed:  Option<(Stone, Point)>,
  last_killed:  Vec<(Stone, Point)>,
}

#[derive(Clone, Debug)]
pub struct TxnPositionProposal {
  // Fields to directly overwrite upon commit.
  resigned:     Option<Stone>,
  next_turn:    Stone,
  prev_turn:    Stone,
  next_ko:      Option<Point>,
  prev_ko:      Option<(Stone, Point)>,
  ko_candidate: Option<Point>,

  // For reverting stateful data structures.
  place:        Option<(Stone, Point)>,
  prev_place:   (Stone, Point),
}

#[derive(Clone, Debug)]
pub struct Chain {
  ps_libs:  Vec<Point>,
  size:     u16,

  diff_insert_ps_libs: Vec<Point>,
  diff_remove_ps_libs: Vec<Point>,
  save_size: u16,
}

impl Chain {
  pub fn new() -> Chain {
    Chain{
      ps_libs: Vec::with_capacity(4),
      size: 0,
      diff_insert_ps_libs: Vec::with_capacity(4),
      diff_remove_ps_libs: Vec::with_capacity(4),
      save_size: 0,
    }
  }

  /// Returns 0 or 1.
  pub fn count_pseudolibs(&self) -> usize {
    if self.ps_libs.is_empty() {
      0
    } else {
      1
    }
  }

  /// Returns 0, 1, or 2.
  pub fn approx_count_libs(&self) -> usize {
    let mut prev_lib = None;
    for &lib in self.ps_libs.iter() {
      match prev_lib {
        None => prev_lib = Some(lib),
        Some(prev_lib) => if prev_lib != lib {
          return 2;
        },
      }
    }
    match prev_lib {
      None => 0,
      Some(_) => 1,
    }
  }

  /*/// Returns 0, 1, 2, or 3.
  pub fn approx_count_libs_3(&self) -> usize {
    // TODO(20151105)
    unimplemented!();
  }*/

  pub fn insert_pseudolib(&mut self, point: Point) {
    self.diff_insert_ps_libs.push(point);
    self.ps_libs.push(point);
  }

  pub fn remove_pseudolib(&mut self, point: Point) {
    self.diff_remove_ps_libs.push(point);
    for i in (0 .. self.ps_libs.len()).rev() {
      if point == self.ps_libs[i] {
        self.ps_libs.swap_remove(i);
      }
    }
  }

  pub fn checkpoint(&mut self) {
    self.save_size = self.size;
    self.diff_insert_ps_libs.clear();
    self.diff_remove_ps_libs.clear();
  }

  pub fn rollback(&mut self) {
    self.size = self.save_size;
    let &mut Chain{
      ref mut ps_libs, ref mut diff_insert_ps_libs, ref mut diff_remove_ps_libs, .. } = self;
    if diff_insert_ps_libs.is_empty() && diff_remove_ps_libs.is_empty() {
      return;
    }
    ps_libs.sort();
    diff_insert_ps_libs.sort();
    {
      // `i` is the search head, `diff_j` is the pattern head.
      let mut i = 0;
      let mut diff_j = 0;
      //let mut prev_lib = None;
      while i < ps_libs.len() {
        // Increment the pattern head to match the search head.
        while diff_j < diff_insert_ps_libs.len() && ps_libs[i] > diff_insert_ps_libs[diff_j] {
          diff_j += 1;
        }
        if diff_j >= diff_insert_ps_libs.len() {
          break;
        }
        /*// Mark duplicates as "dead".
        if let Some(prev) = prev_lib {
          if prev == ps_libs[i] {
            ps_libs[i] = TOMBSTONE;
          } else {
            prev_lib = Some(ps_libs[i]);
          }
        } else {
          prev_lib = Some(ps_libs[i]);
        }*/
        // Mark inserted libs as "dead".
        if ps_libs[i] == diff_insert_ps_libs[diff_j] {
          ps_libs[i] = TOMBSTONE;
        }
        i += 1;
      }
    }
    // Remove "dead" libs in reverse order. This should result in a partially
    // ordered list.
    for i in (0 .. ps_libs.len()).rev() {
      if ps_libs[i] == TOMBSTONE {
        ps_libs.swap_remove(i);
      }
    }
    // Add back removed libs.
    for &lib in diff_remove_ps_libs.iter() {
      ps_libs.push(lib);
    }
    diff_insert_ps_libs.clear();
    diff_remove_ps_libs.clear();
  }
}

#[derive(Clone, Debug)]
enum ChainOp {
  Create{head: Point},
  ExtendPoint{head: Point, point: Point},
  ExtendUnion{old_head: Point, new_head: Point, old_chain: Box<Chain>},
  Kill{head: Point, cap_ptrs: Vec<(Point, Point)>, cap_chain: Box<Chain>},
}

#[derive(Clone, Debug)]
pub struct TxnChainsList {
  chains: Vec<Option<Box<Chain>>>,
  roots:  Vec<Point>,
  links:  Vec<Point>,

  mut_chains: Vec<Point>,
  ops:        Vec<ChainOp>,
}

impl TxnChainsList {
  pub fn new() -> TxnChainsList {
    let nils: Vec<Point> = repeat(TOMBSTONE).take(Board::SIZE).collect();
    TxnChainsList{
      chains:     repeat(None).take(Board::SIZE).collect(),
      roots:      nils.clone(),
      links:      nils,
      mut_chains: Vec::with_capacity(4),
      ops:        Vec::with_capacity(4),
    }
  }

  pub fn reset(&mut self) {
    for p in (0 .. Board::SIZE) {
      self.chains[p] = None;
      self.roots[p] = TOMBSTONE;
      self.links[p] = TOMBSTONE;
    }
    self.mut_chains.clear();
    self.ops.clear();
  }

  pub fn find_chain(&self, point: Point) -> Point {
    let mut next_point = point;
    while next_point != TOMBSTONE && next_point != self.roots[next_point.idx()] {
      next_point = self.roots[next_point.idx()];
    }
    next_point
  }

  pub fn get_chain(&self, head: Point) -> Option<&Chain> {
    self.chains[head.idx()].as_ref().map(|chain| &**chain)
  }

  pub fn get_chain_mut(&mut self, head: Point) -> Option<&mut Chain> {
    self.mut_chains.push(head);
    self.chains[head.idx()].as_mut().map(|mut chain| &mut **chain)
  }

  fn insert_chain(&mut self, head: Point, chain: Box<Chain>) {
    let h = head.idx();
    assert!(self.chains[h].is_none());
    self.chains[h] = Some(chain);
  }

  fn take_chain(&mut self, head: Point) -> Option<Box<Chain>> {
    let chain = self.chains[head.idx()].take();
    chain
  }

  pub fn iter_chain<F>(&self, head: Point, mut f: F)
  where F: FnMut(Point) {
    let mut next_link = head;
    loop {
      next_link = self.links[next_link.idx()];
      f(next_link);
      if next_link == head || next_link == TOMBSTONE {
        break;
      }
    }
  }

  pub fn iter_chain_mut<F>(&mut self, head: Point, mut f: F)
  where F: FnMut(&mut TxnChainsList, Point) {
    let mut next_link = head;
    loop {
      next_link = self.links[next_link.idx()];
      f(self, next_link);
      if next_link == head || next_link == TOMBSTONE {
        break;
      }
    }
  }

  pub fn create_chain(&mut self, head: Point, position: &TxnPosition) {
    let h = head.idx();
    self.roots[h] = head;
    self.links[h] = head;
    let mut chain = Chain::new();
    chain.size = 1;
    for_each_adjacent(head, |adj_point| {
      let adj_stone = position.stones[adj_point.idx()];
      if let Stone::Empty = adj_stone {
        chain.insert_pseudolib(adj_point);
      }
    });
    self.insert_chain(head, Box::new(chain));
    self.mut_chains.push(head);
    self.ops.push(ChainOp::Create{head: head});
  }

  fn link_chain(&mut self, head: Point, point: Point, position: &TxnPosition) {
    let h = head.idx();
    let pt = point.idx();
    self.roots[pt] = head;
    let prev_tail = self.links[h];
    self.links[pt] = prev_tail;
    self.links[h] = point;
    {
      let mut chain = self.get_chain_mut(head).unwrap();
      chain.size += 1;
      chain.remove_pseudolib(point);
      for_each_adjacent(point, |adj_point| {
        let adj_stone = position.stones[adj_point.idx()];
        if let Stone::Empty = adj_stone {
          chain.insert_pseudolib(adj_point);
        }
      });
    }
    self.ops.push(ChainOp::ExtendPoint{head: head, point: point});
  }

  fn union_chains(&mut self, head1: Point, head2: Point) {
    match self.get_chain(head1) {
      Some(_) => {}
      None => panic!("union_chains: failed to get head1: {:?}", head1),
    };
    match self.get_chain(head2) {
      Some(_) => {}
      None => panic!("union_chains: failed to get head2: {:?}", head2),
    };
    let size1 = self.get_chain(head1).unwrap().size;
    let size2 = self.get_chain(head2).unwrap().size;
    let (new_head, old_head, old_size) = if size1 >= size2 {
      (head1, head2, size2)
    } else {
      (head2, head1, size1)
    };
    let new_h = new_head.idx();
    let old_h = old_head.idx();
    self.roots[old_h] = new_head;
    // XXX: The new head -> old tail, and the old head -> new tail.
    let new_tail = self.links[new_h];
    let old_tail = self.links[old_h];
    self.links[new_h] = old_tail;
    self.links[old_h] = new_tail;
    let old_chain = self.take_chain(old_head).unwrap();
    {
      let mut new_chain = self.get_chain_mut(new_head).unwrap();
      new_chain.diff_insert_ps_libs.extend(old_chain.ps_libs.iter());
      new_chain.ps_libs.extend(old_chain.ps_libs.iter());
      new_chain.size += old_size;
    }
    self.ops.push(ChainOp::ExtendUnion{old_head: old_head, new_head: new_head, old_chain: old_chain});
  }

  pub fn extend_chain(&mut self, head: Point, point: Point, position: &TxnPosition) {
    println!("DEBUG: extend_chain: extend {:?} with {:?}", head, point);
    let other_head = self.find_chain(point);
    if other_head == TOMBSTONE {
      println!("DEBUG:   extend_chain: extend point");
      self.link_chain(head, point, position);
    } else if head != other_head {
      println!("DEBUG:   extend_chain: extend union {:?} <- {:?}", other_head, head);
      self.union_chains(head, other_head);
    }
  }

  pub fn kill_chain(&mut self, head: Point) -> usize {
    let cap_chain = self.take_chain(head).unwrap();
    let cap_size = cap_chain.size;
    // FIXME(20151107): instead of clobbering the roots and links eagerly,
    // instead just clobber the roots, while preserving the links in case we
    // need to undo.
    let mut cap_ptrs = Vec::with_capacity(cap_size as usize);
    let mut next_point = head;
    loop {
      let next_next_point = self.links[next_point.idx()];
      let ch_p = next_point.idx();
      cap_ptrs.push((next_point, self.links[ch_p]));
      self.roots[ch_p] = TOMBSTONE;
      self.links[ch_p] = TOMBSTONE;
      next_point = next_next_point;
      if next_point == head || next_point == TOMBSTONE {
        break;
      }
    }
    self.ops.push(ChainOp::Kill{head: head, cap_ptrs: cap_ptrs, cap_chain: cap_chain});
    cap_size as usize
  }

  pub fn commit(&mut self) {
    {
      let &mut TxnChainsList{
        ref mut_chains, ref mut chains, .. } = self;
      for &head in mut_chains.iter() {
        // XXX: Some modified chains were killed during txn, necessitating the
        // option check.
        if let Some(mut chain) = chains[head.idx()].as_mut().map(|ch| &mut **ch) {
          chain.checkpoint();
        }
      }
    }
    self.mut_chains.clear();
    self.ops.clear();
  }

  pub fn undo(&mut self) {
    // XXX: Roll back the linked-union-find data structure.
    let ops: Vec<_> = self.ops.drain(..).collect();
    for op in ops.into_iter().rev() {
      match op {
        ChainOp::Create{head} => {
          let created_chain = self.take_chain(head).unwrap();
        }
        ChainOp::ExtendPoint{head, point} => {
          // XXX: Want to excise `point` from the list.
          //
          // Before:
          //     ... <- `prev_tail` <- `point` <- `head`
          // After:
          //     ... <- `prev_tail` <- `head`
          self.roots[point.idx()] = TOMBSTONE;
          let prev_tail = self.links[point.idx()];
          assert_eq!(point, self.links[head.idx()]);
          self.links[head.idx()] = prev_tail;
          self.links[point.idx()] = TOMBSTONE;
        }
        ChainOp::ExtendUnion{old_head, new_head, old_chain} => {
          // XXX: Want to split the list in two b/w `old_head` and `new_head`.
          //
          // Before:
          //     ... <- `new_tail` <- `old_head` <- ... <- `old_tail` <- `new_head`
          // After:
          //     ... <- `new_tail` <- `new_head`
          //     ... <- `old_tail` <- `old_head`
          self.roots[old_head.idx()] = old_head;
          let new_tail = self.links[old_head.idx()];
          let old_tail = self.links[new_head.idx()];
          self.links[old_head.idx()] = old_tail;
          self.links[new_head.idx()] = new_tail;
          self.insert_chain(old_head, old_chain);
        }
        ChainOp::Kill{head, cap_ptrs, cap_chain} => {
          for &(ch_point, ch_link) in cap_ptrs.iter() {
            let ch_p = ch_point.idx();
            // XXX: Quick-find optimization.
            self.roots[ch_p] = head; // instead of `ch_root`.
            self.links[ch_p] = ch_link;
          }
          self.insert_chain(head, cap_chain);
        }
      }
    }
    // XXX: Roll back the chain metadata (pseudolibs, size).
    {
      let &mut TxnChainsList{
        ref mut_chains, ref mut chains, .. } = self;
      for &head in mut_chains.iter() {
        // XXX: Some modified chains were created during txn and have since been
        // rolled back, necessitating the option check.
        if let Some(mut chain) = chains[head.idx()].as_mut().map(|ch| &mut **ch) {
          chain.rollback();
        }
      }
    }
    self.mut_chains.clear();
    self.ops.clear();
  }
}

/*#[derive(Clone, Debug)]
pub struct TxnStateScratch {
  nothing: usize,
}*/

/// Transactional board state.
///
/// Some various properties we would like TxnState to have:
/// - support 1-step undo
/// - the board position and chains list should be in a consistent state before
///   entering a txn and after committing a txn
/// - the TxnState holds auxiliary extra data (e.g., useful for search nodes vs.
///   rollouts)
#[derive(Clone)]
pub struct TxnState<Data=()> where Data: TxnStateData + Clone {
  in_soft_txn:  bool,
  in_txn:       bool,
  txn_mut:      bool,

  // Data structures totally owned and managed by the board struct.
  resigned:     Option<Stone>,
  passed:       [bool; 2],
  position:     TxnPosition,
  proposal:     TxnPositionProposal,

  // Complex data structures that contain their own support for txns.
  chains:       TxnChainsList,
  data:         Data,
}

impl<Data> TxnState<Data> where Data: TxnStateData + Clone {
  pub fn new(data: Data) -> TxnState<Data> {
    let mut stones: Vec<Stone> = repeat(Stone::Empty).take(Board::SIZE).collect();
    TxnState{
      in_soft_txn: false,
      in_txn: false,
      txn_mut: false,
      resigned: None,
      passed: [false, false],
      position: TxnPosition{
        turn:         Stone::Black,
        stones:       stones,
        ko:           None,
        last_placed:  None,
        last_killed:  vec![],
      },
      proposal: TxnPositionProposal{
        resigned: None,
        next_turn: Stone::Black,
        prev_turn: Stone::Black,
        next_ko: None,
        prev_ko: None,
        ko_candidate: None,
        place: None,
        prev_place: (Stone::Empty, TOMBSTONE),
      },
      chains: TxnChainsList::new(),
      data: data,
    }
  }

  pub fn reset(&mut self) {
    // TODO(20151105)
    self.in_soft_txn = false;
    self.in_txn = false;
    self.txn_mut = false;
    self.resigned = None;
    self.passed[0] = false;
    self.passed[1] = false;
    self.position.turn = Stone::Black;
    self.position.ko = None;
    for p in (0 .. Board::SIZE) {
      self.position.stones[p] = Stone::Empty;
    }
    self.chains.reset();
    self.data.reset();
  }

  pub fn shrink_clone(&self) -> TxnState<()> {
    assert!(!self.in_txn);
    TxnState{
      in_soft_txn:  false,
      in_txn:   false,
      txn_mut:  false,
      resigned: self.resigned.clone(),
      passed:   self.passed.clone(),
      position: self.position.clone(),
      proposal: self.proposal.clone(),
      chains:   self.chains.clone(),
      data:     (),
    }
  }

  pub fn is_terminal(&self) -> bool {
    // FIXME(20151106): some rulesets require white to pass first?
    self.resigned.is_some() || (self.passed[0] && self.passed[1])
  }

  pub fn current_turn(&self) -> Stone {
    self.position.turn
  }

  pub fn extract_absolute_features() {
    // TODO(20151105)
    unimplemented!();
  }

  pub fn extract_relative_features() {
    // TODO(20151105)
    unimplemented!();
  }

  pub fn iter_legal_moves_accurate<F>(&mut self, turn: Stone, /*scratch: &mut TxnStateScratch,*/ mut f: F) where F: FnMut(Point) {
    for p in (0 .. Board::SIZE as i16) {
      let point = Point(p);
      if self.check_illegal_move_simple(turn, point).is_some() {
        continue;
      }
      match self.try_place(turn, point) {
        Ok(_) => {
          f(point);
        }
        _ => {}
      }
      self.undo();
    }
  }

  pub fn is_eyelike(&self, stone: Stone, point: Point) -> bool {
    let mut eyeish = true;
    for_each_adjacent(point, |adj_point| {
      if self.position.stones[adj_point.idx()] != stone {
        eyeish = false;
      }
    });
    if !eyeish {
      return false;
    }
    // XXX: a.k.a. the "2/4 rule".
    let opp_stone = stone.opponent();
    let mut false_count = if point.is_edge() { 1 } else { 0 };
    for_each_diagonal(point, |diag_point| {
      if self.position.stones[diag_point.idx()] == opp_stone {
        false_count += 1;
      }
    });
    false_count < 2
  }

  pub fn check_illegal_move_simple(&self, turn: Stone, point: Point) -> Option<TxnStatusIllegalReason> {
    let p = point.idx();

    // Illegal to place a stone on top of an existing stone.
    if self.position.stones[p] != Stone::Empty {
      return Some(TxnStatusIllegalReason::NotEmpty);
    }

    // Illegal to suicide (place in opponent's eye).
    // FIXME(20151107): the "eyelike" or "2/4" definition can be overly strict
    // in rare cases.
    if self.is_eyelike(turn.opponent(), point) {
      return Some(TxnStatusIllegalReason::Suicide);
    }

    // Illegal to place at ko point.
    if let Some((ko_turn, ko_point)) = self.position.ko {
      if ko_turn == turn && ko_point == point {
        return Some(TxnStatusIllegalReason::Ko);
      }
    }

    None
  }

  pub fn check_ko_candidate(&self, place_turn: Stone, place_point: Point, ko_point: Point) -> bool {
    let opp_turn = place_turn.opponent();
    let mut is_ko = true;
    for_each_adjacent(ko_point, |adj_ko_point| {
      let adj_ko_stone = self.position.stones[adj_ko_point.idx()];
      if adj_ko_stone != Stone::Empty {
        let adj_ko_head = self.chains.find_chain(adj_ko_point);
        let adj_ko_chain = self.chains.get_chain(adj_ko_head).unwrap();
        let adj_ko_libs = adj_ko_chain.approx_count_libs();
        if adj_ko_point == place_point {
          let adj_ko_size = adj_ko_chain.size;
          match (adj_ko_libs, adj_ko_size) {
            (1, 1) => {}
            (1, _) | (2, _) => { is_ko = false; }
            _ => { unreachable!(); }
          }
        } else {
          match adj_ko_libs {
            1 => { is_ko = false; }
            2 => {}
            _ => { unreachable!(); }
          }
        }
      }
    });
    is_ko
  }

  pub fn try_action(&mut self, turn: Stone, action: Action/*, scratch: &mut TxnStateScratch*/) -> TxnResult {
    match action {
      Action::Place{point} => {
        self.try_place(turn, point/*, scratch*/)
      }
      Action::Pass => {
        self.in_soft_txn = true;
        self.proposal.resigned = None;
        self.proposal.next_turn = turn.opponent();
        self.proposal.next_ko = None;
        Ok(())
      }
      Action::Resign => {
        self.in_soft_txn = true;
        self.proposal.resigned = Some(turn);
        self.proposal.next_turn = turn.opponent();
        self.proposal.next_ko = None;
        Ok(())
      }
    }
  }

  fn merge_chains(&mut self, turn: Stone, place_point: Point) {
    let mut extended = false;
    for_each_adjacent(place_point, |adj_point| {
      let adj_stone = self.position.stones[adj_point.idx()];
      if adj_stone == turn {
        let adj_head = self.chains.find_chain(adj_point);
        assert!(TOMBSTONE != adj_head);
        // Update adjacent own chain pseudoliberties.
        {
          let mut adj_chain = self.chains.get_chain_mut(adj_head).unwrap();
          adj_chain.remove_pseudolib(place_point);
        }
        // Extend adjacent own chain, merging with another chain if necessary.
        let &mut TxnState{
          ref position, ref mut chains, .. } = self;
        chains.extend_chain(adj_head, place_point, position);
        println!("DEBUG: txnstate: extend chain: {:?} {:?} <- {:?}", turn, place_point, adj_head);
        extended = true;
      }
    });
    if !extended {
      // If necessary, create a new chain.
      let &mut TxnState{
        ref position, ref mut chains, .. } = self;
      chains.create_chain(place_point, position);
      println!("DEBUG: txnstate: create chain: {:?} {:?}", turn, place_point);
    }
  }

  fn capture_chains(&mut self, captor: Stone, opponent: Stone, place_point: Point, set_ko: bool) -> usize {
    assert!(Stone::Empty != opponent);
    let place_head = self.chains.find_chain(place_point);
    let mut total_num_captured_stones = 0;
    for_each_adjacent(place_point, |adj_point| {
      let adj_stone = self.position.stones[adj_point.idx()];
      if adj_stone == opponent {
        let adj_head = self.chains.find_chain(adj_point);
        assert!(TOMBSTONE != adj_head);
        // Update adjacent opponent chain pseudoliberties.
        if adj_head != place_head {
          let mut adj_chain = self.chains.get_chain_mut(adj_head).unwrap();
          adj_chain.remove_pseudolib(place_point);
          println!("DEBUG: capture_chains: remove lib {:?} from {:?}", place_point, adj_head);
        }
        let adj_chain_libs = self.chains.get_chain(adj_head).unwrap().count_pseudolibs();
        if adj_chain_libs == 0 {
          println!("DEBUG: txnstate: capture chain {:?} by {:?}", adj_head, place_point);
          // Clear captured opponent chain stones from position, and add
          // captured opponent chain to own chains' liberties.
          let &mut TxnState{
            ref mut position, ref mut chains, .. } = self;
          let mut num_cap_stones = 0;
          chains.iter_chain_mut(adj_head, |chains, ch_point| {
            let cap_stone = position.stones[ch_point.idx()];
            position.stones[ch_point.idx()] = Stone::Empty;
            position.last_killed.push((cap_stone, ch_point));
            num_cap_stones += 1;
            for_each_adjacent(ch_point, |adj_ch_point| {
              let adj_ch_stone = position.stones[adj_ch_point.idx()];
              if adj_ch_stone == captor {
                let adj_ch_head = chains.find_chain(adj_ch_point);
                let mut adj_ch_chain = chains.get_chain_mut(adj_ch_head).unwrap();
                adj_ch_chain.insert_pseudolib(ch_point);
              }
            });
          });
          // Kill captured opponent chain.
          let num_cap_chain = chains.kill_chain(adj_head);
          assert_eq!(num_cap_stones, num_cap_chain);
          if set_ko {
            if num_cap_stones == 1 {
              self.proposal.ko_candidate = Some(adj_head);
            }
          }
          total_num_captured_stones += num_cap_stones;
        }
      }
    });
    if set_ko {
      if total_num_captured_stones > 1 {
        self.proposal.ko_candidate = None;
      }
    }
    total_num_captured_stones
  }

  pub fn try_place(&mut self, turn: Stone, place_point: Point/*, scratch: &mut TxnStateScratch*/) -> TxnResult {
    self.in_txn = true;

    // First, try to terminate txn w/out mutating state.
    self.txn_mut = false;

    // Do not allow an empty turn.
    if Stone::Empty == turn {
      return Err(TxnStatus::Illegal);
    }
    // TODO(20151105): Allow placements out of turn, but somehow warn about it?
    let place_p = place_point.idx();
    // Do not allow simple illegal moves.
    if let Some(reason) = self.check_illegal_move_simple(turn, place_point) {
      return Err(TxnStatus::IllegalReason(reason));
    }

    // Then, enable mutating state (and full undos).
    self.txn_mut = true;

    // Following Tromp-Taylor evaluation order:

    // 1. Place the stone.
    self.proposal.next_turn = turn.opponent();
    self.proposal.prev_turn = self.position.turn;
    self.proposal.next_ko = None;
    self.proposal.prev_ko = self.position.ko;
    self.proposal.ko_candidate = None;
    self.proposal.place = Some((turn, place_point));
    self.proposal.prev_place = (self.position.stones[place_p], place_point);
    self.position.turn = turn;
    self.position.stones[place_p] = turn;
    self.position.last_placed = Some((turn, place_point));
    self.position.last_killed.clear();
    self.merge_chains(turn, place_point);

    // 2. Capture opponent chains.
    let opp_turn = turn.opponent();
    self.capture_chains(turn, opp_turn, place_point, true);

    // 3. Suicide own chains.
    let num_suicided_stones = self.capture_chains(opp_turn, turn, place_point, false);

    if let Some(ko_candidate) = self.proposal.ko_candidate {
      if self.check_ko_candidate(turn, place_point, ko_candidate) {
        self.proposal.next_ko = Some(ko_candidate);
      }
    }

    if num_suicided_stones > 0 {
      Err(TxnStatus::IllegalReason(TxnStatusIllegalReason::Suicide))
    } else {
      Ok(())
    }
  }

  pub fn commit(&mut self) {
    // Committing the position consists only of rolling forward the current turn
    // and the current ko point.
    self.position.turn = self.proposal.next_turn;
    self.position.ko = self.proposal.next_ko.map(|ko_point| (self.proposal.next_turn, ko_point));

    if self.in_soft_txn {
      self.resigned = self.proposal.resigned;
    } else {
      assert!(self.in_txn);

      // Commit changes to the chains list. This performs checkpointing per chain.
      self.chains.commit();

      // Commit changes to the extra data. Typically, .update() does all the work,
      // and .commit() does nothing..
      self.data.update(&self.position, &self.chains);
      self.data.commit();

      self.position.last_placed = None;
      self.position.last_killed.clear();
    }

    self.txn_mut = false;
    self.in_txn = false;
    self.in_soft_txn = false;
  }

  pub fn undo(&mut self) {
    if self.in_soft_txn {
      // Do nothing.
    } else {
      assert!(self.in_txn);

      // Undo everything in reverse order.
      if self.txn_mut {
        // First undo changes in the extra data. This should be either a no-op or
        // a panic (if undo is not supported).
        self.data.undo();

        // Next undo changes in the chains list.
        self.chains.undo();

        // Finally roll back the position using saved values in the proposal.
        self.position.turn = self.proposal.prev_turn;
        self.position.ko = self.proposal.prev_ko;
        let (prev_place_stone, prev_place_point) = self.proposal.prev_place;
        self.position.stones[prev_place_point.idx()] = prev_place_stone;
        self.position.last_placed = None;
        for &(cap_stone, cap_point) in self.position.last_killed.iter() {
          self.position.stones[cap_point.idx()] = cap_stone;
        }
        self.position.last_killed.clear();
      }
    }

    self.txn_mut = false;
    self.in_txn = false;
    self.in_soft_txn = false;
  }

  pub fn to_debug_strings(&self) -> Vec<String> {
    let mut strs = vec![];
    // TODO(20151107)
    for col in (0 .. Board::DIM + 2) {
      let mut s = String::new();
      if col == 0 || col == Board::DIM + 1 {
        s.push_str("   ");
        for x in (0 .. Board::DIM) {
          let x = String::from_utf8(dump_xcoord(x as u8)).unwrap();
          s.push_str(&x);
          s.push(' ');
        }
        s.push_str("   ");
      } else if col >= 1 && col <= Board::DIM {
        let y = (Board::DIM - 1) - (col - 1);
        if (y + 1) < 10 {
          s.push(' ');
        }
        s.push_str(&format!("{} ", y + 1));
        for x in (0 .. Board::DIM) {
          let point = Point::from_coord(Coord::new(x as u8, y as u8));
          let stone = self.position.stones[point.idx()];
          match stone {
            Stone::Black => s.push_str("X "),
            Stone::White => s.push_str("O "),
            Stone::Empty => s.push_str(". "),
          }
        }
        s.push_str(&format!("{} ", y + 1));
        if (y + 1) < 10 {
          s.push(' ');
        }
      }
      strs.push(s);
    }
    strs
  }
}
