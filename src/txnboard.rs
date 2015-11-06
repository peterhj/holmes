use std::iter::{repeat};

pub const TOMBSTONE:    Point = Point(-1);

pub struct Board;

impl Board {
  pub const CENTER:     Point = Point(9);
  pub const DIM:        Point = Point(19);
  pub const MAX_POINT:  Point = Point(361);
  pub const SIDE_LEN:   usize = 19;
  pub const SIZE:       usize = 361;
}

pub fn for_each_adjacent<F>(point: Point, mut f: F) where F: FnMut(Point) {
  let (x, y) = (point.0 % Board::DIM.0, point.0 / Board::DIM.0);
  let upper = Board::DIM.0 - 1;
  if x >= 1 {
    f(Point(point.0 - 1));
  }
  if y >= 1 {
    f(Point(point.0 - Board::DIM.0));
  }
  if x < upper {
    f(Point(point.0 + 1));
  }
  if y < upper {
    f(Point(point.0 + Board::DIM.0));
  }
}

pub fn for_each_diagonal<F>(point: Point, mut f: F) where F: FnMut(Point) {
  let (x, y) = (point.0 % Board::DIM.0, point.0 / Board::DIM.0);
  let upper = Board::DIM.0 - 1;
  if x >= 1 && y >= 1 {
    f(Point(point.0 - 1 - Board::DIM.0));
  }
  if x >= 1 && y < upper {
    f(Point(point.0 - 1 + Board::DIM.0));
  }
  if x < upper && y >= 1 {
    f(Point(point.0 + 1 - Board::DIM.0));
  }
  if x < upper && y < upper {
    f(Point(point.0 + 1 + Board::DIM.0));
  }
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Stone {
  Black = 0,
  White = 1,
  Empty = 2,
}

impl Stone {
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
  #[inline]
  pub fn idx(self) -> usize {
    self.0 as usize
  }

  #[inline]
  fn is_edge(self) -> bool {
    let (x, y) = (self.0 % Board::DIM.0, self.0 / Board::DIM.0);
    (x == 0 || x == (Board::DIM.0 - 1) ||
     y == 0 || y == (Board::DIM.0 - 1))
  }
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Action {
  Place{point: Point},
  Pass,
  Resign,
}

/*pub struct FlatState {
  //turn:     Stone,
  stones:   Vec<Stone>,
}

impl FlatState {
  pub fn new() -> FlatState {
    let mut stones: Vec<Stone> = repeat(Stone::Empty).take(Board::SIZE).collect();
    FlatState{
      //turn: Stone::Black,
      stones: stones,
    }
  }

  pub fn place(&mut self, stone: Stone, point: Point) {
    self.stones[point.idx()] = stone;
  }
}*/

#[derive(Clone, Copy, Debug)]
pub enum TxnResult {
  Success,
  FailureIllegal,
}

pub trait TxnStateData: Default {
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

#[derive(Clone, Default)]
pub struct TxnStateHeavyData;

impl TxnStateData for TxnStateHeavyData {
  fn undo(&mut self) {
    panic!("FATAL: TxnStateHeavyData does not support txn undo!");
  }
}

#[derive(Clone, Debug)]
pub struct TxnPosition {
  turn:     Stone,
  ko:       Option<Point>,
  stones:   Vec<Stone>,
}

#[derive(Clone, Debug)]
pub struct TxnPositionProposal {
  // Fields to directly overwrite upon commit.
  next_turn:    Stone,
  prev_turn:    Stone,
  next_ko:      Option<Point>,
  prev_ko:      Option<Point>,
  ko_candidate: Option<Point>,

  // For reverting stateful data structures.
  place:        Option<(Stone, Point)>,
  prev_place:   (Stone, Point),
}

impl TxnPositionProposal {
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
    // TODO(20151105)
    unimplemented!();
  }

  /// Returns 0, 1, 2, or 3.
  pub fn approx_count_libs_3(&self) -> usize {
    // TODO(20151105)
    unimplemented!();
  }

  pub fn insert_pseudolib(&mut self, point: Point) {
    self.diff_insert_ps_libs.push(point);
    self.ps_libs.push(point);
  }

  pub fn remove_pseudolib(&mut self, point: Point) {
    self.diff_remove_ps_libs.push(point);
    self.ps_libs.push(point);
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
    let mut i = 0;
    let mut diff_j = 0;
    while i < ps_libs.len() {
      while diff_j < diff_insert_ps_libs.len() && ps_libs[i] > diff_insert_ps_libs[diff_j] {
        diff_j += 1;
      }
      if ps_libs[i] != diff_insert_ps_libs[diff_j] {
        ps_libs[i] = TOMBSTONE;
      }
      i += 1;
    }
    i = 0;
    while i < ps_libs.len() {
      if ps_libs[i] == TOMBSTONE {
        ps_libs.swap_remove(i);
      } else {
        i += 1;
      }
    }
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
  Kill{head: Point, cap_chain: Box<Chain>},
}

#[derive(Clone, Debug)]
pub struct TxnChainsList {
  chains: Vec<Option<Box<Chain>>>,
  roots:  Vec<Point>,
  links:  Vec<Point>,

  mut_heads:  Vec<Point>,
  ops:        Vec<ChainOp>,
}

impl TxnChainsList {
  pub fn new() -> TxnChainsList {
    let mut nils: Vec<Point> = repeat(TOMBSTONE).take(Board::SIZE).collect();
    TxnChainsList{
      chains: Vec::with_capacity(4),
      roots:  nils.clone(),
      links:  nils,
      mut_heads:  Vec::with_capacity(4),
      ops:        Vec::with_capacity(4),
    }
  }

  pub fn reset(&mut self) {
    self.chains.clear();
    for p in (0 .. Board::SIZE) {
      self.roots[p] = TOMBSTONE;
      self.links[p] = TOMBSTONE;
    }
    self.mut_heads.clear();
    self.ops.clear();
  }

  pub fn find_chain(&self, point: Point) -> Point {
    // TODO(20151105)
    //unimplemented!();
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
    self.mut_heads.push(head);
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

  pub fn iter_chain<F>(&self, head: Point, mut f: F) where F: FnMut(Point) {
    // TODO(20151105)
    unimplemented!();
  }

  pub fn iter_chain_mut<F>(&mut self, head: Point, mut f: F) where F: FnMut(&mut TxnChainsList, Point) {
    // TODO(20151105)
    //unimplemented!();
    let mut next_link = head;
    loop {
      next_link = self.links[next_link.idx()];
      f(self, next_link);
      if next_link == head {
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
    let size1 = self.get_chain(head1).unwrap().size;
    let size2 = self.get_chain(head2).unwrap().size;
    let (new_head, old_head, old_size) = if size1 > size2 {
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
    self.get_chain_mut(new_head).unwrap().size += old_size;
    let old_chain = self.take_chain(old_head).unwrap();
    self.ops.push(ChainOp::ExtendUnion{old_head: old_head, new_head: new_head, old_chain: old_chain});
  }

  pub fn extend_chain(&mut self, head: Point, point: Point, position: &TxnPosition) {
    let other_head = self.find_chain(point);
    if other_head == TOMBSTONE {
      self.link_chain(head, point, position);
    } else if head != other_head {
      self.union_chains(head, other_head);
    }
  }

  pub fn kill_chain(&mut self, head: Point) -> usize {
    // TODO(20151105)
    let cap_chain = self.take_chain(head).unwrap();
    let cap_size = cap_chain.size;
    self.ops.push(ChainOp::Kill{head: head, cap_chain: cap_chain});
    cap_size as usize
  }

  pub fn commit(&mut self) {
    {
      let &mut TxnChainsList{
        ref mut_heads, ref mut chains, .. } = self;
      for &head in mut_heads.iter() {
        // XXX: Some modified chains were killed during txn, necessitating the
        // option check.
        if let Some(mut chain) = chains[head.idx()].as_mut().map(|ch| &mut **ch) {
          chain.checkpoint();
        }
      }
    }
    self.mut_heads.clear();
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
        ChainOp::Kill{head, cap_chain} => {
          self.insert_chain(head, cap_chain);
        }
      }
    }
    // XXX: Roll back the chain metadata (pseudolibs, size).
    {
      let &mut TxnChainsList{
        ref mut_heads, ref mut chains, .. } = self;
      for &head in mut_heads.iter() {
        // XXX: Some modified chains were created during txn and have since been
        // rolled back, necessitating the option check.
        if let Some(mut chain) = chains[head.idx()].as_mut().map(|ch| &mut **ch) {
          chain.rollback();
        }
      }
    }
    self.mut_heads.clear();
    self.ops.clear();
  }
}

/*#[derive(Clone, Debug)]
pub struct TxnStateScratch {
  nothing: usize,
}*/

#[derive(Clone)]
pub struct TxnState<Data=()> where Data: Clone + TxnStateData {
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

impl<Data> TxnState<Data> where Data: Clone + TxnStateData {
  pub fn new() -> TxnState<Data> {
    let mut stones: Vec<Stone> = repeat(Stone::Empty).take(Board::SIZE).collect();
    let mut state = TxnState{
      in_txn: false,
      txn_mut: false,
      resigned: None,
      passed: [false, false],
      position: TxnPosition{
        turn: Stone::Black,
        ko: None,
        stones: stones,
      },
      proposal: TxnPositionProposal{
        next_turn: Stone::Black,
        prev_turn: Stone::Black,
        next_ko: None,
        prev_ko: None,
        ko_candidate: None,
        place: None,
        prev_place: (Stone::Empty, TOMBSTONE),
      },
      chains: TxnChainsList::new(),
      data: Default::default(),
    };
    state.reset();
    state
  }

  pub fn reset(&mut self) {
    // TODO(20151105)
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

  pub fn iter_legal_moves<F>(&mut self, turn: Stone, /*scratch: &mut TxnStateScratch,*/ mut f: F) where F: FnMut(Point) {
    for p in (0 .. Board::SIZE as i16) {
      let point = Point(p);
      if self.check_illegal_move_simple(turn, point) {
        continue;
      }
      match self.try_place(turn, point) {
        TxnResult::Success => {
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

  pub fn check_illegal_move_simple(&self, turn: Stone, point: Point) -> bool {
    let p = point.idx();

    // Illegal to place a stone on top of an existing stone.
    if self.position.stones[p] != Stone::Empty {
      return true;
    }

    // Illegal to suicide (place in opponent's eye).
    if self.is_eyelike(turn.opponent(), point) {
      return true;
    }

    // Illegal to place at ko point.
    if let Some(ko) = self.position.ko {
      if ko == point {
        return false;
      }
    }

    false
  }

  pub fn check_ko_candidate(&self, turn: Stone, place_point: Point, ko_point: Point) -> bool {
    let opp_turn = turn.opponent();
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
        self.position.turn = turn.opponent();
        TxnResult::Success
      }
      Action::Resign => {
        self.resigned = Some(turn);
        self.position.turn = turn.opponent();
        TxnResult::Success
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
        chains.extend_chain(adj_head, adj_point, position);
        extended = true;
      }
    });
    if !extended {
      // If necessary, create a new chain.
      let &mut TxnState{
        ref position, ref mut chains, .. } = self;
      chains.create_chain(place_point, position);
    }
  }

  fn capture_chains(&mut self, captor: Stone, opponent: Stone, place_point: Point, set_ko: bool) -> usize {
    let mut num_captured_stones = 0;
    for_each_adjacent(place_point, |adj_point| {
      let adj_stone = self.position.stones[adj_point.idx()];
      if adj_stone == opponent {
        let adj_head = self.chains.find_chain(adj_point);
        assert!(TOMBSTONE != adj_head);
        // Update adjacent opponent chain pseudoliberties.
        {
          let mut adj_chain = self.chains.get_chain_mut(adj_head).unwrap();
          adj_chain.remove_pseudolib(place_point);
        }
        let adj_chain_libs = self.chains.get_chain(adj_head).unwrap().count_pseudolibs();
        if adj_chain_libs == 0 {
          // Add captured opponent chain to own chains' liberties.
          let &mut TxnState{
            ref position, ref mut chains, .. } = self;
          chains.iter_chain_mut(adj_head, |chains, ch_point| {
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
          let num_cap = chains.kill_chain(adj_head);
          if set_ko {
            if num_cap == 1 {
              self.proposal.ko_candidate = Some(adj_head);
            }
          }
          num_captured_stones += num_cap;
        }
      }
    });
    if set_ko {
      if num_captured_stones > 1 {
        self.proposal.ko_candidate = None;
      }
    }
    num_captured_stones
  }

  pub fn try_place(&mut self, turn: Stone, place_point: Point/*, scratch: &mut TxnStateScratch*/) -> TxnResult {
    self.in_txn = true;

    // First, try to terminate txn w/out mutating state.
    self.txn_mut = false;

    // Do not allow an empty turn.
    if Stone::Empty == turn {
      return TxnResult::FailureIllegal;
    }
    // TODO(20151105): Allow placements out of turn, but somehow warn about it?
    let place_p = place_point.idx();
    // Do not allow simple illegal moves.
    if self.check_illegal_move_simple(turn, place_point) {
      return TxnResult::FailureIllegal;
    }

    // Then, enable mutating state (and full undos).
    self.txn_mut = true;

    // Following Tromp-Taylor evaluation order:

    // 1. Place the stone.
    self.proposal.next_turn = turn.opponent();
    self.proposal.prev_turn = self.position.turn;
    self.proposal.next_ko = None;
    self.proposal.prev_ko = self.position.ko;
    self.proposal.place = Some((turn, place_point));
    self.proposal.prev_place = (self.position.stones[place_p], place_point);
    self.position.turn = turn;
    self.position.stones[place_p] = turn;
    self.merge_chains(turn, place_point);

    // 2. Capture opponent chains.
    let opp_turn = turn.opponent();
    let num_captured_stones = self.capture_chains(turn, opp_turn, place_point, true);

    // 3. Suicide own chains.
    let num_suicided_stones = self.capture_chains(opp_turn, turn, place_point, false);

    if let Some(ko_candidate) = self.proposal.ko_candidate {
      if self.check_ko_candidate(turn, place_point, ko_candidate) {
        self.proposal.next_ko = Some(ko_candidate);
      }
    }

    self.data.update(&self.position, &self.chains);

    if num_suicided_stones > 0 {
      TxnResult::FailureIllegal
    } else {
      TxnResult::Success
    }
  }

  pub fn commit(&mut self) {
    assert!(self.in_txn);

    self.position.turn = self.proposal.next_turn;
    self.position.ko = self.proposal.next_ko;

    // TODO(20151029)
    //unimplemented!();

    self.chains.commit();
    self.data.commit();

    self.in_txn = false;
    self.txn_mut = false;
  }

  pub fn undo(&mut self) {
    assert!(self.in_txn);
    if self.txn_mut {
      self.data.undo();
      self.chains.undo();

      self.position.turn = self.proposal.prev_turn;
      self.position.ko = self.proposal.prev_ko;
      let (prev_place_stone, prev_place_point) = self.proposal.prev_place;
      self.position.stones[prev_place_point.idx()] = prev_place_stone;

      // TODO(20151029)
      //unimplemented!();
    }
    self.in_txn = false;
    self.txn_mut = false;
  }
}
