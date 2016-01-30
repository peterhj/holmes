use board::{Board, Rules, RuleSet, Coord, PlayerRank, Stone, Point, Action};
use gtp_board::{dump_xcoord};
use pattern::{Pattern3x3};

use bit_set::{BitSet};
use bit_vec::{BitVec};
use std::cmp::{max};
use std::collections::{BTreeSet};
use std::iter::{repeat};

pub mod extras;
pub mod features;

pub const TOMBSTONE:  Point = Point(-1);

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
  if x < upper && y >= 1 {
    f(Point(point.0 + 1 - Board::DIM_PT.0));
  }
  if x < upper && y < upper {
    f(Point(point.0 + 1 + Board::DIM_PT.0));
  }
  if x >= 1 && y < upper {
    f(Point(point.0 - 1 + Board::DIM_PT.0));
  }
}

pub fn for_each_x8<F>(point: Point, mut f: F) where F: FnMut(u8, Point) {
  let (x, y) = (point.0 % Board::DIM_PT.0, point.0 / Board::DIM_PT.0);
  let upper = Board::DIM_PT.0 - 1;
  if x >= 1 && y >= 1 {
    f(0, Point(point.0 - 1 - Board::DIM_PT.0));
  }
  if y >= 1 {
    f(1, Point(point.0 - Board::DIM_PT.0));
  }
  if x < upper && y >= 1 {
    f(2, Point(point.0 + 1 - Board::DIM_PT.0));
  }
  if x >= 1 {
    f(3, Point(point.0 - 1));
  }
  if x < upper {
    f(4, Point(point.0 + 1));
  }
  if x >= 1 && y < upper {
    f(5, Point(point.0 - 1 + Board::DIM_PT.0));
  }
  if y < upper {
    f(6, Point(point.0 + Board::DIM_PT.0));
  }
  if x < upper && y < upper {
    f(7, Point(point.0 + 1 + Board::DIM_PT.0));
  }
}

/// A point is "eyeish" for a stone if:
/// - the point is surrounded on all sides by the stone, and
/// - the surrounding stones have at least 2 liberties.
pub fn is_eyeish(position: &TxnPosition, chains: &TxnChainsList, stone: Stone, point: Point) -> bool {
  let mut eyeish = true;
  for_each_adjacent(point, |adj_point| {
    if position.stones[adj_point.idx()] != stone {
      eyeish = false;
    } else {
      let adj_head = chains.find_chain(adj_point);
      assert!(adj_head != TOMBSTONE);
      let adj_chain = chains.get_chain(adj_head).unwrap();
      if adj_chain.count_libs_up_to_2() <= 1 {
        eyeish = false;
      }
    }
  });
  eyeish
}

/// The "2/4" rule, a conservative estimate of whether a point is an eye.
/// Some true eyes (c.f., "two headed dragon") will not be detected by this
/// rule.
pub fn is_eyelike(position: &TxnPosition, chains: &TxnChainsList, stone: Stone, point: Point) -> bool {
  if !is_eyeish(position, chains, stone, point) {
    return false;
  }

  // XXX: a.k.a. the "2/4 rule".
  let opp_stone = stone.opponent();
  let mut false_count = if point.is_edge() { 1 } else { 0 };
  for_each_diagonal(point, |diag_point| {
    if position.stones[diag_point.idx()] == opp_stone {
      false_count += 1;
    }
  });
  false_count < 2
}

pub fn check_illegal_move_simple(position: &TxnPosition, chains: &TxnChainsList, turn: Stone, point: Point) -> Option<IllegalReason> {
  let p = point.idx();

  // Illegal to place a stone on top of an existing stone.
  if position.stones[p] != Stone::Empty {
    return Some(IllegalReason::NotEmpty);
  }

  // Illegal to place at ko point.
  if let Some((ko_turn, ko_point)) = position.ko {
    if ko_turn == turn && ko_point == point {
      return Some(IllegalReason::Ko);
    }
  }

  // Illegal to suicide (place in opponent's eye).
  if is_eyeish(position, chains, turn.opponent(), point) {
    return Some(IllegalReason::Suicide);
  }

  None
}

pub fn check_legal_move_simple(position: &TxnPosition, chains: &TxnChainsList, turn: Stone, point: Point) -> Option<TxnResult> {
  let p = point.idx();

  // Illegal to place a stone on top of an existing stone.
  if position.stones[p] != Stone::Empty {
    return Some(Err(TxnStatus::Illegal(IllegalReason::NotEmpty)));
  }

  // Illegal to place at ko point.
  if let Some((ko_turn, ko_point)) = position.ko {
    if ko_turn == turn && ko_point == point {
      return Some(Err(TxnStatus::Illegal(IllegalReason::Ko)));
    }
  }

  // Illegal to suicide (place in opponent's eye).
  if is_eyeish(position, chains, turn.opponent(), point) {
    return Some(Err(TxnStatus::Illegal(IllegalReason::Suicide)));
  }

  // If the move is not next to our own atari chain, it is legal (barring superko).
  let mut no_adj_atari = true;
  for_each_adjacent(point, |adj_point| {
    let adj_stone = position.stones[adj_point.idx()];
    if adj_stone == turn {
      let adj_head = chains.find_chain(adj_point);
      let adj_chain = chains.get_chain(adj_head).unwrap();
      if adj_chain.count_libs_up_to_2() == 1 {
        no_adj_atari = false;
      }
    }
  });
  if no_adj_atari {
    return Some(Ok(()));
  }

  None
}

pub fn check_good_move_fast(position: &TxnPosition, chains: &TxnChainsList, turn: Stone, point: Point) -> bool {
  // Don't place in our own true eye.
  if is_eyelike(position, chains, turn, point) {
    return false;
  }

  true
}

pub type TxnResult = Result<(), TxnStatus>;

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum TxnStatus {
  Illegal(IllegalReason),
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum IllegalReason {
  Noop,
  NotEmpty,
  Suicide,
  Ko,
}

pub trait TxnStateData {
  fn reset(&mut self) {
    // Do nothing.
  }

  fn update(&mut self, position: &TxnPosition, chains: &TxnChainsList, update_turn: Stone, update_action: Action) {
    // Do nothing.
  }
}

impl TxnStateData for () {
}

#[derive(Clone, Debug)]
pub struct TxnPosition {
  pub ranks:    [PlayerRank; 2],

  turn:         Stone,
  epoch:        i16,
  num_stones:   [usize; 2],
  stones:       Vec<Stone>,
  stone_epochs: Vec<i16>,

  // State that depends on the previously placed point.
  // FIXME(20151120)
  prev_play:    Option<(Stone, Point)>, // the previously played point.
  ko:           Option<(Stone, Point)>, // ko point.
  prev_ko:      Option<(Stone, Point)>, // previous ko point.
  prev_self_atari:  bool, // previous play put own chain in atari (2-pt semeai opportunity).
  prev_oppo_atari:  bool, // previous play put opponent chain in atari (save opportunity).

  // FIXME(20151107): Semantics of when `last_placed` and `last_killed` are
  // valid is unclear; currently used as tmp variables that last until the end
  // of .commit(). These are also used by TxnStateData.update().
  pub last_move:    Option<(Stone, Action)>,
  pub last_placed:  Option<(Stone, Point)>,
  pub last_ko:      Option<(Stone, Point)>,
  pub last_killed:  Vec<Vec<Point>>,
  pub last_atari:   Vec<Vec<Point>>,
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

  pub fn count_length(&self) -> usize {
    self.size as usize
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
  pub fn count_libs_up_to_2(&self) -> usize {
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

  /// Returns 0, 1, 2, or 3.
  pub fn count_libs_up_to_3(&self) -> usize {
    let mut prev_lib = None;
    let mut prev2_lib = None;
    for &lib in self.ps_libs.iter() {
      match prev_lib {
        None => prev_lib = Some(lib),
        Some(prev_lib) => if prev_lib != lib {
          match prev2_lib {
            None => prev2_lib = Some(lib),
            Some(prev2_lib) => if prev2_lib != lib {
              return 3;
            }
          }
        },
      }
    }
    match (prev_lib, prev2_lib) {
      (None, None) => 0,
      (Some(_), None) => 1,
      (Some(_), Some(_)) => 2,
      _ => unreachable!(),
    }
  }

  /// Returns 0, 1, 2, 3, or 4.
  pub fn count_libs_up_to_4(&self) -> usize {
    let mut prev_lib = None;
    let mut prev2_lib = None;
    let mut prev3_lib = None;
    for &lib in self.ps_libs.iter() {
      match prev_lib {
        None => prev_lib = Some(lib),
        Some(prev_lib) => if prev_lib != lib {
          match prev2_lib {
            None => prev2_lib = Some(lib),
            Some(prev2_lib) => if prev2_lib != lib {
              match prev3_lib {
                None => prev3_lib = Some(lib),
                Some(prev3_lib) => if prev3_lib != lib {
                  return 4;
                }
              }
            }
          }
        }
      }
    }
    match (prev_lib, prev2_lib, prev3_lib) {
      (None, None, None) => 0,
      (Some(_), None, None) => 1,
      (Some(_), Some(_), None) => 2,
      (Some(_), Some(_), Some(_)) => 3,
      _ => unreachable!(),
    }
  }

  /// Returns 0 through 8.
  pub fn count_libs_up_to_8(&self) -> usize {
    let mut prev_libs = [
      None, None, None, None,
      None, None, None, None,
    ];
    let mut num_uniq_libs = 0 ;
    for &lib in self.ps_libs.iter() {
      for t in 0 .. (num_uniq_libs + 1) {
        match prev_libs[t] {
          None => {
            prev_libs[t] = Some(lib);
            num_uniq_libs += 1;
          }
          Some(prev_lib) => {
            if prev_lib == lib {
              break;
            } else if t == 7 {
              return 8;
            }
          }
        }
      }
    }
    num_uniq_libs
  }

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

  pub fn rollback(&mut self, maybe_head: Option<Point>) {
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
          // FIXME(20151107): There is a very tricky thing here.
          // Suppose we are inserting a new liberty. If the liberty already
          // exists, then when undoing, we definitely DO NOT want to delete
          // all instances of that liberty.
          ps_libs[i] = TOMBSTONE;
          diff_j += 1;
        }
        i += 1;
      }
    }
    // Remove "dead" libs in reverse order. This should result in a partially
    // ordered list.
    for i in (0 .. ps_libs.len()).rev() {
      if ps_libs[i] == TOMBSTONE {
        let deleted_lib = ps_libs.swap_remove(i);
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
  Kill{head: Point, cap_ptrs: Vec<(Point, Point, Point)>, cap_chain: Box<Chain>},
  //Kill{head: Point, cap_ptrs: Vec<(Point, Point)>, cap_chain: Box<Chain>},
}

#[derive(Clone, Debug)]
pub struct TxnChainsList {
  num_chains: usize,
  chains:     Vec<Option<Box<Chain>>>,
  roots:      Vec<Point>,
  links:      Vec<Point>,

  mut_chains: Vec<Point>,     // chain heads which were modified this round.
  ops:        Vec<ChainOp>,   // chain operations this round.

  last_mut_heads: Vec<Point>, // a duplicate of `mut_chains` (for Data.update()).
}

impl TxnChainsList {
  pub fn new() -> TxnChainsList {
    let nils: Vec<Point> = repeat(TOMBSTONE).take(Board::SIZE).collect();
    TxnChainsList{
      num_chains: 0,
      chains:     repeat(None).take(Board::SIZE).collect(),
      roots:      nils.clone(),
      links:      nils,
      mut_chains: Vec::with_capacity(4),
      ops:        Vec::with_capacity(4),
      last_mut_heads: Vec::with_capacity(4),
    }
  }

  pub fn reset(&mut self) {
    self.num_chains = 0;
    for p in 0 .. Board::SIZE {
      self.chains[p] = None;
      self.roots[p] = TOMBSTONE;
      self.links[p] = TOMBSTONE;
    }
    self.mut_chains.clear();
    self.ops.clear();
    self.last_mut_heads.clear();
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
    self.num_chains += 1;
    self.chains[h] = Some(chain);
  }

  fn take_chain(&mut self, head: Point) -> Option<Box<Chain>> {
    self.num_chains -= 1;
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
    assert_eq!(self.roots[h], TOMBSTONE);
    assert_eq!(self.links[h], TOMBSTONE);
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

  fn link_chain(&mut self, head: Point, point: Point, position: &TxnPosition) -> Result<(), ()> {
    //println!("DEBUG: link_chain: link {:?} <- {:?}", point, head);
    let h = head.idx();
    let pt = point.idx();
    assert_eq!(self.roots[pt], TOMBSTONE);
    assert_eq!(self.links[pt], TOMBSTONE);
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
    Ok(())
  }

  fn union_chains(&mut self, head1: Point, head2: Point) -> Result<(), ()> {
    match self.get_chain(head1) {
      Some(_) => {}
      None => panic!("union_chains: failed to get head1: {:?}", head1),
    };
    match self.get_chain(head2) {
      Some(_) => {}
      None => {
        panic!("union_chains: failed to get head2: {:?}", head2);
        //println!("WARNING: union_chains: failed to get head2: {:?}", head2);
        //return Err(());
      }
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
    Ok(())
  }

  pub fn extend_chain(&mut self, head: Point, point: Point, position: &TxnPosition) -> Result<(), ()> {
    //println!("DEBUG: extend_chain: extend {:?} with {:?}", head, point);
    let other_head = self.find_chain(point);
    if other_head == TOMBSTONE {
      //println!("DEBUG:   extend_chain: extend point");
      self.link_chain(head, point, position)
    } else if head != other_head {
      //println!("DEBUG:   extend_chain: extend union {:?} <- {:?}", other_head, head);
      self.union_chains(head, other_head)
    } else {
      Ok(())
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
      cap_ptrs.push((next_point, self.roots[ch_p], self.links[ch_p]));
      //cap_ptrs.push((next_point, self.links[ch_p]));
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
    self.last_mut_heads.extend(self.mut_chains.iter());
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
          self.roots[head.idx()] = TOMBSTONE;
          self.links[head.idx()] = TOMBSTONE;
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
          for &(ch_point, ch_root, ch_link) in cap_ptrs.iter() {
            let ch_p = ch_point.idx();
            // XXX: Quick-find optimization.
            //self.roots[ch_p] = head; // instead of `ch_root`.
            self.roots[ch_p] = ch_root;
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
          chain.rollback(Some(head));
        }
      }
    }
    self.mut_chains.clear();
    self.ops.clear();
    self.last_mut_heads.clear();
  }
}

#[derive(Clone, Debug)]
pub struct GnugoPrintSgf {
  pub output:   String,
  pub illegal:  BTreeSet<Point>,
}

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
  rules:        Rules,

  // Transaction markers.
  in_soft_txn:  bool,
  in_txn:       bool,
  txn_mut:      bool,

  // Game state data.
  resigned:     Option<Stone>,
  passed:       [bool; 2],
  num_captures: [usize; 2],

  // Position data. This is managed by the TxnState.
  pub position: TxnPosition,
  proposal:     TxnPositionProposal,

  // Chains data. This is managed separately.
  pub chains:   TxnChainsList,

  // Auxiliary data. This does not have txn support and is updated only during
  // commit.
  pub data:     Data,
}

impl TxnState<()> {
  pub fn shrink_clone_from<Data>(&mut self, other: &TxnState<Data>) where Data: TxnStateData + Clone {
    assert!(!self.in_txn);
    assert!(!other.in_txn);
    self.rules        = other.rules;
    self.in_soft_txn  = false;
    self.in_txn       = false;
    self.txn_mut      = false;
    self.resigned     = other.resigned.clone();
    self.passed       = other.passed.clone();
    self.num_captures = other.num_captures.clone();
    self.position     = other.position.clone();
    self.proposal     = other.proposal.clone();
    self.chains       = other.chains.clone();
  }
}

impl<Data> TxnState<Data> where Data: TxnStateData + Clone {
  pub fn new(ranks: [PlayerRank; 2], rules: Rules, data: Data) -> TxnState<Data> {
    let stones: Vec<Stone> = repeat(Stone::Empty).take(Board::SIZE).collect();
    let stone_epochs: Vec<i16> = repeat(0).take(Board::SIZE).collect();
    TxnState{
      rules: rules,
      in_soft_txn: false,
      in_txn: false,
      txn_mut: false,
      resigned: None,
      passed: [false, false],
      num_captures: [0, 0],
      position: TxnPosition{
        ranks:        ranks,
        turn:         Stone::Black,
        epoch:        0,
        num_stones:   [0, 0],
        stones:       stones,
        stone_epochs: stone_epochs,
        prev_play:    None,
        ko:           None,
        prev_ko:      None,
        prev_self_atari:  false,
        prev_oppo_atari:  false,
        last_move:    None,
        last_placed:  None,
        last_ko:      None,
        last_killed:  vec![
          vec![],
          vec![],
        ],
        last_atari:   vec![
          vec![],
          vec![],
        ],
      },
      proposal: TxnPositionProposal{
        resigned:     None,
        next_turn:    Stone::Black,
        prev_turn:    Stone::Black,
        next_ko:      None,
        prev_ko:      None,
        ko_candidate: None,
        place:        None,
        prev_place:   (Stone::Empty, TOMBSTONE),
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
    self.num_captures[0] = 0;
    self.num_captures[1] = 0;
    self.position.turn = Stone::Black;
    self.position.epoch = 0;
    self.position.num_stones[0] = 0;
    self.position.num_stones[1] = 0;
    for p in 0 .. Board::SIZE {
      self.position.stones[p] = Stone::Empty;
      self.position.stone_epochs[p] = 0;
    }
    self.position.prev_play = None;
    self.position.ko = None;
    self.position.prev_ko = None;
    self.position.prev_self_atari = false;
    self.position.prev_oppo_atari = false;
    self.chains.reset();
    self.data.reset();
  }

  pub fn replace_clone_from<OtherData>(&mut self, other: &TxnState<OtherData>, data: Data) where OtherData: TxnStateData + Clone {
    assert!(!self.in_txn);
    assert!(!other.in_txn);
    self.rules        = other.rules;
    self.in_soft_txn  = false;
    self.in_txn       = false;
    self.txn_mut      = false;
    self.resigned     = other.resigned.clone();
    self.passed       = other.passed.clone();
    self.num_captures = other.num_captures.clone();
    self.position     = other.position.clone();
    self.proposal     = other.proposal.clone();
    self.chains       = other.chains.clone();
    self.data         = data;
  }

  pub fn shrink_clone(&self) -> TxnState<()> {
    assert!(!self.in_txn);
    TxnState{
      rules:        self.rules,
      in_soft_txn:  false,
      in_txn:       false,
      txn_mut:      false,
      resigned:     self.resigned.clone(),
      passed:       self.passed.clone(),
      num_captures: self.num_captures.clone(),
      position:     self.position.clone(),
      proposal:     self.proposal.clone(),
      chains:       self.chains.clone(),
      data:         (),
    }
  }

  pub fn unsafe_set_current_turn(&mut self, turn: Stone) {
    // FIXME(20151115): this can be highly unsafe; make sure no ko points are
    // set.
    self.position.turn = turn;
  }

  pub fn get_data(&self) -> &Data {
    &self.data
  }

  pub fn is_terminal(&self) -> bool {
    // FIXME(20151106): some rulesets require white to pass first?
    self.resigned.is_some() || (self.passed[0] && self.passed[1])
  }

  pub fn current_turn(&self) -> Stone {
    self.position.turn
  }

  pub fn current_stone(&self, point: Point) -> Stone {
    self.position.stones[point.idx()]
  }

  pub fn current_pat3x3(&self, point: Point) -> Pattern3x3 {
    let mut mask8: u16 = 0xffff;
    for_each_x8(point, |i, adj_pt| {
      let adj_st = self.position.stones[adj_pt.idx()];
      match adj_st {
        Stone::Empty => mask8 &= !(0x3_u16 << (2*i)),
        Stone::Black => mask8 &= !(0x2_u16 << (2*i)),
        Stone::White => mask8 &= !(0x1_u16 << (2*i)),
      }
    });
    Pattern3x3(mask8)
  }

  pub fn current_libs_up_to_3(&self, point: Point) -> usize {
    let head = self.chains.find_chain(point);
    if head == TOMBSTONE {
      0
    } else {
      let chain = self.chains.get_chain(head).unwrap();
      chain.count_libs_up_to_3()
    }
  }

  pub fn current_ko(&self) -> Option<(Stone, Point)> {
    self.position.ko
  }

  pub fn current_score_exact(&self, komi: f32) -> f32 {
    let mut b_score = 0.0;
    let mut w_score = 0.0;
    // FIXME(20151125): rule for removing dead stones.
    if self.rules.score_captures {
      b_score += self.num_captures[0] as f32;
      w_score += self.num_captures[1] as f32;
    }
    if self.rules.score_stones {
      b_score += self.position.num_stones[0] as f32;
      w_score += self.position.num_stones[1] as f32;
    }
    if self.rules.score_territory {
      // FIXME(20151107)
    }
    // TODO(20151107): handicap komi (for non-Japanese rulesets).
    //w_score - b_score + komi
    unimplemented!();
  }

  pub fn current_score_tromp_taylor(&self, komi: f32) -> f32 {
    let mut score: [f32; 2] = [0.0, 0.0];

    // Add own stones (area counting).
    score[0] += self.position.num_stones[0] as f32;
    score[1] += self.position.num_stones[1] as f32;

    // TODO(20151126): Remove dead stones (mark dead w/ uniform rollouts).

    // Add territory using a 3-pass flood fill algorithm.
    let mut territory: Vec<u8> = repeat(0).take(Board::SIZE).collect();
    let mut territory_count: [i32; 2] = [0, 0];
    fn flood_fill(p: usize, mask: u8, position: &TxnPosition, territory: &mut [u8]) {
      match position.stones[p] {
        Stone::Empty => {
          if territory[p] & mask != 0 {
            return;
          }
          territory[p] |= mask;
          for_each_adjacent(Point::from_idx(p), |adj_pt| {
            flood_fill(adj_pt.idx(), mask, position, territory);
          });
        }
        _ => {}
      }
    }
    for p in 0 .. Board::SIZE {
      match self.position.stones[p] {
        Stone::Empty => {
          for_each_adjacent(Point::from_idx(p), |adj_pt| {
            match self.position.stones[adj_pt.idx()] {
              Stone::Black => {
                flood_fill(p, 1, &self.position, &mut territory);
              }
              Stone::White => {
                flood_fill(p, 2, &self.position, &mut territory);
              }
              _ => {}
            }
          });
        }
        _ => {}
      }
    }
    for p in 0 .. Board::SIZE {
      match self.position.stones[p] {
        Stone::Empty => {
          match territory[p] {
            1 => territory_count[0] += 1,
            2 => territory_count[1] += 1,
            3 => {}
            _ => unreachable!(),
          }
        }
        _ => {}
      }
    }
    score[0] += territory_count[0] as f32;
    score[1] += territory_count[1] as f32;

    score[1] - score[0] + komi
  }

  pub fn current_score_rollout(&self, komi: f32) -> f32 {
    let mut b_score = 0.0;
    let mut w_score = 0.0;
    b_score += self.position.num_stones[0] as f32;
    w_score += self.position.num_stones[1] as f32;
    for p in 0 .. Board::SIZE {
      match self.position.stones[p] {
        Stone::Empty => {
          if is_eyelike(&self.position, &self.chains, Stone::Black, Point(p as i16)) {
            b_score += 1.0;
          } else if is_eyelike(&self.position, &self.chains, Stone::White, Point(p as i16)) {
            w_score += 1.0;
          }
        }
        _ => {}
      }
    }
    w_score - b_score + komi
  }

  pub fn iter_legal_moves_accurate<F>(&mut self, turn: Stone, /*scratch: &mut TxnStateScratch,*/ mut f: F) where F: FnMut(Point) {
    for p in 0 .. Board::SIZE as i16 {
      let point = Point(p);
      if check_illegal_move_simple(&self.position, &self.chains, turn, point).is_some() {
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

  pub fn is_capture(&self, stone: Stone, point: Point) -> bool {
    let mut capture = false;
    for_each_adjacent(point, |adj_point| {
      if self.position.stones[adj_point.idx()] == stone {
        let adj_head = self.chains.find_chain(adj_point);
        assert!(adj_head != TOMBSTONE);
        let adj_chain = self.chains.get_chain(adj_head).unwrap();
        if adj_chain.count_libs_up_to_2() <= 1 {
          capture = true;
        }
      }
    });
    capture
  }

  pub fn check_ko_candidate(&self, place_turn: Stone, place_point: Point, ko_point: Point) -> bool {
    let opp_turn = place_turn.opponent();
    let mut is_ko = true;
    for_each_adjacent(ko_point, |adj_ko_point| {
      let adj_ko_stone = self.position.stones[adj_ko_point.idx()];
      if adj_ko_stone != Stone::Empty {
        // This checks two cases:
        // - an adjacent stone that is also the placed stone must have only
        //   one liberty and have chain size of one
        // - otherwise, adjacent stones must have at least two liberties
        let adj_ko_head = self.chains.find_chain(adj_ko_point);
        let adj_ko_chain = self.chains.get_chain(adj_ko_head).unwrap();
        let adj_ko_libs = adj_ko_chain.count_libs_up_to_2();
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
      } else {
        unreachable!();
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
        self.position.last_move = Some((turn, action));
        Ok(())
      }
      Action::Resign => {
        self.in_soft_txn = true;
        self.proposal.resigned = Some(turn);
        self.proposal.next_turn = turn.opponent();
        self.proposal.next_ko = None;
        self.position.last_move = Some((turn, action));
        Ok(())
      }
    }
  }

  fn merge_adjacent_chains(&mut self, turn: Stone, place_point: Point) {
    let mut failed = false;
    let mut extended = false;
    for_each_adjacent(place_point, |adj_point| {
      if failed {
        return;
      }
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
        if chains.extend_chain(adj_head, place_point, position).is_err() {
          failed = true;
          return;
        }
        //println!("DEBUG: merge_adjacent_chains: extend {:?} {:?} <- {:?}", turn, place_point, adj_head);
        extended = true;
      }
    });
    if failed {
      for s in self.to_debug_strings().iter() {
        println!("{}", s);
      }
      panic!();
    }
    if !extended {
      // If necessary, create a new chain.
      let &mut TxnState{
        ref position, ref mut chains, .. } = self;
      chains.create_chain(place_point, position);
      //println!("DEBUG: merge_adjacent_chains: create {:?} {:?}", turn, place_point);
    }
  }

  fn capture_chain(&mut self, captor: Stone, opponent: Stone, adj_head: Point, set_ko: bool) -> usize {
    let opp_off = opponent.offset();
    let adj_chain_libs = self.chains.get_chain(adj_head).unwrap().count_pseudolibs();
    if adj_chain_libs == 0 {
      //println!("DEBUG: capture_chains: capture {:?} by {:?}", adj_head, place_point);
      // Clear captured opponent chain stones from position, and add
      // captured opponent chain to own chains' liberties.
      let &mut TxnState{
        ref mut position, ref mut chains, .. } = self;
      let mut num_cap_stones = 0;
      chains.iter_chain_mut(adj_head, |chains, ch_point| {
        //let cap_stone = position.stones[ch_point.idx()];
        position.stones[ch_point.idx()] = Stone::Empty;
        position.last_killed[opp_off].push(ch_point);
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
      num_cap_stones
    } else {
      0
    }
  }

  fn capture_adjacent_chains(&mut self, captor: Stone, opponent: Stone, place_point: Point, set_ko: bool) -> usize {
    let opp_off = opponent.offset();
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
          //println!("DEBUG: capture_chains: remove lib {:?} from {:?}", place_point, adj_head);
        }
        let num_cap_stones = self.capture_chain(captor, opponent, adj_head, set_ko);
        total_num_captured_stones += num_cap_stones;
      }
    });
    if set_ko {
      if total_num_captured_stones > 1 {
        self.proposal.ko_candidate = None;
      }
    }
    self.position.num_stones[opp_off] -= total_num_captured_stones;
    total_num_captured_stones
  }

  pub fn try_place(&mut self, turn: Stone, place_point: Point/*, scratch: &mut TxnStateScratch*/) -> TxnResult {
    self.in_txn = true;

    // First, try to terminate txn w/out mutating state.
    self.txn_mut = false;

    // Do not allow an empty turn.
    if Stone::Empty == turn {
      return Err(TxnStatus::Illegal(IllegalReason::Noop));
    }
    // TODO(20151105): Allow placements out of turn, but somehow warn about it?
    let place_p = place_point.idx();
    // Do not allow simple illegal moves.
    if let Some(reason) = check_illegal_move_simple(&self.position, &self.chains, turn, place_point) {
      return Err(TxnStatus::Illegal(reason));
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
    self.position.num_stones[turn.offset()] += 1;
    self.position.stones[place_p] = turn;
    self.position.last_move = Some((turn, Action::Place{point: place_point}));
    self.position.last_placed = Some((turn, place_point));
    self.position.last_ko = self.position.ko;
    self.position.last_killed[0].clear();
    self.position.last_killed[1].clear();
    self.position.last_atari[0].clear();
    self.position.last_atari[1].clear();
    self.merge_adjacent_chains(turn, place_point);

    // 2. Capture opponent chains.
    let opp_turn = turn.opponent();
    self.capture_adjacent_chains(turn, opp_turn, place_point, true);

    // 3. Suicide own chains.
    let num_suicided_stones = self.capture_adjacent_chains(opp_turn, turn, place_point, false);

    // Determine local atari flags (used by local features).
    {
      let place_head = self.chains.find_chain(place_point);
      if place_head != TOMBSTONE {
        let place_chain = self.chains.get_chain(place_head).unwrap();
        if place_chain.count_libs_up_to_2() == 1 {
          self.position.last_atari[turn.offset()].push(place_head);
        }
      }
      for_each_adjacent(place_point, |adj_point| {
        let adj_stone = self.position.stones[adj_point.idx()];
        let adj_head = self.chains.find_chain(adj_point);
        if adj_head != TOMBSTONE {
          let adj_chain = self.chains.get_chain(adj_head).unwrap();
          if adj_chain.count_libs_up_to_2() == 1 {
            self.position.last_atari[adj_stone.offset()].push(adj_head);
          }
        }
      });
    }

    // Finalize the ko point.
    if let Some(ko_candidate) = self.proposal.ko_candidate {
      if self.check_ko_candidate(turn, place_point, ko_candidate) {
        self.proposal.next_ko = Some(ko_candidate);
      }
    }

    // Currently, always disallow suicide.
    if num_suicided_stones > 0 {
      Err(TxnStatus::Illegal(IllegalReason::Suicide))
    } else {
      Ok(())
    }
  }

  pub fn commit(&mut self) {
    // Roll forward the current turn.
    self.position.turn = self.proposal.next_turn;
    self.position.epoch += 1;

    // Roll forward information about this turn (now the previous turn).
    self.position.prev_play = match self.position.last_move {
      Some((update_turn, Action::Place{point})) => Some((update_turn, point)),
      _ => None,
    };
    self.position.ko = self.proposal.next_ko.map(|ko_point| (self.proposal.next_turn, ko_point));
    self.position.prev_ko = self.proposal.prev_ko;

    let (update_turn, update_action) = self.position.last_move.unwrap();

    if self.in_soft_txn {
      self.resigned = self.proposal.resigned;
      self.position.prev_self_atari = false;
      self.position.prev_oppo_atari = false;
    } else {
      assert!(self.in_txn);

      if let Some((_, update_point)) = self.position.prev_play {
        self.position.stone_epochs[update_point.idx()] = self.position.epoch - 1;
      }

      // Commit more stateful changes to the position.
      if self.position.last_atari[update_turn.offset()].len() > 0 {
        self.position.prev_self_atari = true;
      } else {
        self.position.prev_self_atari = false;
      }
      if self.position.last_atari[update_turn.opponent().offset()].len() > 0 {
        self.position.prev_oppo_atari = true;
      } else {
        self.position.prev_oppo_atari = false;
      }

      self.num_captures[0] += self.position.last_killed[1].len();
      self.num_captures[1] += self.position.last_killed[0].len();

      // Commit changes to the chains list. This performs checkpointing per chain.
      self.chains.commit();
    }

    // Run changes on the extra data.
    self.data.update(&self.position, &self.chains, update_turn, update_action);

    /*self.position.last_move = None;
    self.position.last_placed = None;
    self.position.last_ko = None;
    self.position.last_killed[0].clear();
    self.position.last_killed[1].clear();
    self.position.last_atari[0].clear();
    self.position.last_atari[1].clear();*/

    // FIXME(20151124): currently, no good way to reset ChainsList `last_*`
    // fields, so doing it here instead.
    self.chains.last_mut_heads.clear();

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
        // Undo changes in the chains list.
        self.chains.undo();

        // Roll back the position using saved values in the proposal.
        let last_turn = self.position.turn;

        for &cap_point in self.position.last_killed[0].iter() {
          self.position.stones[cap_point.idx()] = Stone::Black;
        }
        for &cap_point in self.position.last_killed[1].iter() {
          self.position.stones[cap_point.idx()] = Stone::White;
        }
        self.position.num_stones[0] += self.position.last_killed[0].len();
        self.position.num_stones[1] += self.position.last_killed[1].len();

        let (prev_place_stone, prev_place_point) = self.proposal.prev_place;
        self.position.num_stones[last_turn.offset()] -= 1;
        self.position.stones[prev_place_point.idx()] = prev_place_stone;

        self.position.turn = self.proposal.prev_turn;
        //self.position.ko = self.proposal.prev_ko;

        self.position.last_move = None;
        self.position.last_placed = None;
        self.position.last_ko = None;
        self.position.last_killed[0].clear();
        self.position.last_killed[1].clear();
        self.position.last_atari[0].clear();
        self.position.last_atari[1].clear();

        self.chains.last_mut_heads.clear();
      }
    }

    /*let test_head = self.chains.find_chain(Point(283));
    if test_head != TOMBSTONE {
      let test_chain = self.chains.get_chain(test_head).unwrap();
      println!("DEBUG: try_place: UNDO 283: ps_libs: {:?}",
          test_chain.ps_libs);
    }*/

    self.txn_mut = false;
    self.in_txn = false;
    self.in_soft_txn = false;
  }

  pub fn to_debug_strings(&self) -> Vec<String> {
    // TODO(20151107)
    let empty_indent = repeat(' ').take(48).collect::<String>();
    let mut strs = vec![];
    let mut chain_h = 0;
    for col in 0 .. max(Board::DIM + 2, self.chains.num_chains + 1) {
      let mut s = String::new();
      if col == 0 || col == Board::DIM + 1 {
        s.push_str("   ");
        for x in 0 .. Board::DIM {
          let x = String::from_utf8(dump_xcoord(x as u8)).unwrap();
          s.push_str(&x);
          s.push(' ');
        }
        s.push_str("   ");
        s.push_str("    ");
      } else if col >= 1 && col <= Board::DIM {
        let y = (Board::DIM - 1) - (col - 1);
        if (y + 1) < 10 {
          s.push(' ');
        }
        s.push_str(&format!("{} ", y + 1));
        for x in 0 .. Board::DIM {
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
        s.push_str("    ");
      } else {
        s.push_str(&empty_indent);
      }
      if col == 0 {
        s.push_str(&format!("turn: {:?}   num chains: {}   ko: {:?}   prev ko: {:?}",
            self.current_turn(), self.chains.num_chains,
            self.position.ko.map(|k| (k.0, k.1.to_coord().to_string())),
            self.position.prev_ko.map(|k| (k.0, k.1.to_coord().to_string())),
        ));
      } else if col >= 1 && col <= self.chains.num_chains {
        while self.chains.chains[chain_h].is_none() {
          chain_h += 1;
        }
        let chain_head = Point(chain_h as i16);
        let chain_stone = self.position.stones[chain_h];
        s.push_str(&format!("chain {:?} {}:",
            chain_stone,
            chain_head.to_coord().to_string(),
        ));
        if chain_head.to_coord().y < 10 {
          s.push(' ');
        }
        let chain = self.chains.get_chain(chain_head).unwrap();
        s.push_str(&format!(" ({}, {})",
            chain.size,
            chain.count_libs_up_to_2(),
        ));
        self.chains.iter_chain(chain_head, |ch_point| {
          s.push_str(&format!(" {}", ch_point.to_coord().to_string()));
        });
        chain_h += 1;
      }
      strs.push(s);
    }
    strs
  }

  pub fn to_gnugo_printsgf(&mut self, date_str: &str, komi: f32, ruleset: RuleSet) -> GnugoPrintSgf {
    let mut s = String::new();
    let mut illegal_points = BTreeSet::new();

    s.push_str("(;GM[1]FF[4]\n");
    s.push_str(&format!("SZ[{}]\n", Board::DIM));
    s.push_str("GN[GNU Go 3.8 load and print]\n");
    s.push_str(&format!("DT[{}]\n", date_str));

    let relative_komi = komi + self.num_captures[1] as f32 - self.num_captures[0] as f32;
    let mut line = String::new();
    line.push_str(&format!("KM[{:.1}]HA[{}]RU[{}]AP[GNU Go:3.8]",
        relative_komi,
        0,
        match ruleset {
          RuleSet::KgsJapanese => "Japanese",
          _ => unimplemented!(),
        },
    ));

    enum PrintState {
      AB{idx: usize},
      AW{idx: usize},
      PL,
      IL{ptr: usize},
      Terminal,
    }

    let parity = Stone::Black;
    let mut state = match parity {
      Stone::Black => PrintState::AW{idx: 0},
      Stone::White => PrintState::AB{idx: 0},
      _ => unimplemented!(),
    };
    let mut w_ptr = 0;
    let mut b_ptr = 0;
    let mut num_illegal = 0;
    loop {
      let mut flushed = false;
      if line.len() + 4 > 64 {
        flushed = true;
        s.push_str(&line);
        s.push('\n');
        line.clear();
      }
      match state {
        PrintState::AB{mut idx} => {
          if idx < self.position.num_stones[0] {
            if idx == 0 {
              line.push_str("AB");
            }
            while b_ptr < Board::SIZE && self.position.stones[b_ptr] != Stone::Black {
              b_ptr += 1;
            }
            line.push_str(&format!("[{}]", Point(b_ptr as i16).to_coord().to_sgf()));
            flushed = false;
            idx += 1;
            b_ptr += 1;
          }
          if idx == self.position.num_stones[0] {
            if idx > 0 && !flushed {
              flushed = true;
              s.push_str(&line);
              s.push('\n');
              line.clear();
            }
            match parity {
              Stone::Black => state = PrintState::PL,
              Stone::White => state = PrintState::AW{idx: 0},
              _ => unimplemented!(),
            };
          } else {
            state = PrintState::AB{idx: idx};
          }
        }
        PrintState::AW{mut idx} => {
          if idx < self.position.num_stones[1] {
            if idx == 0 {
              line.push_str("AW");
            }
            while w_ptr < Board::SIZE && self.position.stones[w_ptr] != Stone::White {
              w_ptr += 1;
            }
            line.push_str(&format!("[{}]", Point(w_ptr as i16).to_coord().to_sgf()));
            flushed = false;
            idx += 1;
            w_ptr += 1;
          }
          if idx == self.position.num_stones[1] {
            if idx > 0 && !flushed {
              flushed = true;
              s.push_str(&line);
              s.push('\n');
              line.clear();
            }
            match parity {
              Stone::Black => state = PrintState::AB{idx: 0},
              Stone::White => state = PrintState::PL,
              _ => unimplemented!(),
            };
          } else {
            state = PrintState::AW{idx: idx};
          }
        }
        PrintState::PL => {
          line.push_str(&format!("PL[{}]", match self.current_turn() {
            Stone::Black => "B",
            Stone::White => "W",
            _ => unimplemented!(),
          }));
          flushed = false;
          state = PrintState::IL{ptr: 0};
        }
        PrintState::IL{mut ptr} => {
          if ptr < Board::SIZE {
            let mut illegal = false;
            if self.position.stones[ptr] == Stone::Empty {
              let turn = self.current_turn();
              let point = Point(ptr as i16);
              if check_illegal_move_simple(&self.position, &self.chains, turn, point).is_some() {
                illegal = true;
              } else {
                if self.try_place(turn, point).is_err() {
                  illegal = true;
                }
                self.undo();
              }
              if illegal {
                if num_illegal == 0 {
                  line.push_str("IL");
                }
                illegal_points.insert(point);
                line.push_str(&format!("[{}]", point.to_coord().to_sgf()));
                flushed = false;
                num_illegal += 1;
              }
            }
            /*if ptr == 264 {
              println!("DEBUG: IL: point {:?}", Point(264).to_coord().to_string());
              println!("DEBUG: IL: illegal? {:?}", illegal);
              println!("DEBUG: IL: empty? {:?}", self.position.stones[ptr] == Stone::Empty);
            }*/
            ptr += 1;
          }
          if ptr == Board::SIZE {
            state = PrintState::Terminal;
          } else {
            state = PrintState::IL{ptr: ptr};
          }
        }
        PrintState::Terminal => {
          if !flushed && line.len() > 0 {
            flushed = true;
            s.push_str(&line);
            s.push('\n');
            line.clear();
          }
          break;
        }
      }
    }
    s.push_str(")\n");

    GnugoPrintSgf{
      output:   s,
      illegal:  illegal_points,
    }
  }
}
