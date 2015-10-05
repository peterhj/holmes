use bit_set::{BitSet};
use bit_vec::{BitVec};
use std::iter::{repeat};

const TOMBSTONE: Pos = -1;

pub trait IntoIndex {
  fn idx(&self) -> usize;
}

pub type Pos = i16;

impl IntoIndex for Pos {
  #[inline]
  fn idx(&self) -> usize {
    *self as usize
  }
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Stone {
  Black = 0,
  White = 1,
  Empty = 2,
}

impl Stone {
  #[inline]
  pub fn opponent(&self) -> Stone {
    match *self {
      Stone::Black => Stone::White,
      Stone::White => Stone::Black,
      Stone::Empty => unreachable!(),
    }
  }

  #[inline]
  pub fn offset(&self) -> usize {
    match *self {
      Stone::Empty => unreachable!(),
      x => x as usize,
    }
  }
}

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
pub enum Action {
  Resign,
  Pass,
  Place{pos: Pos},
}

#[derive(Clone, Copy, Debug)]
pub enum ActionStatus {
  Legal           = 0,
  IllegalUnknown  = 1,
  IllegalOccupied = 2,
  IllegalSuicide  = 3,
  IllegalKo       = 4,
  IllegalSuperko  = 5,
}

/*#[derive(Clone, Copy)]
pub struct ActionStatusSet {
  mask: u8,
}

impl ActionStatusSet {
  pub fn new() -> ActionStatusSet {
    ActionStatusSet{mask: 0}
  }

  pub fn add(mut self, status: ActionStatus) -> ActionStatusSet {
    self.mask |= status as u8;
    self
  }

  pub fn contains(&self, status: ActionStatus) -> bool {
    (self.mask | status as u8) != 0
  }
}*/

pub enum KoRule {
  None,
  Ko,
  Superko,
}

pub enum SuicideRule {
  NotAllowed,
  Allowed,
}

pub struct RuleSet {
  ko:       KoRule,
  suicide:  SuicideRule,
}

impl RuleSet {
  pub fn allow_any_ko(&self) -> bool {
    match self.ko {
      KoRule::None => true,
      _ => false,
    }
  }

  pub fn allow_suicide(&self) -> bool {
    match self.suicide {
      SuicideRule::NotAllowed => false,
      SuicideRule::Allowed => true,
    }
  }
}

#[derive(Clone, Debug)]
pub struct Chain {
  ps_libs: Vec<Pos>,
}

impl Chain {
  pub fn new() -> Chain {
    Chain{ps_libs: Vec::with_capacity(4) }
  }

  pub fn add_pseudoliberty(&mut self, lib: Pos) {
    self.ps_libs.push(lib);
  }

  pub fn remove_pseudoliberty(&mut self, lib: Pos) {
    let mut i = 0;
    while i < self.ps_libs.len() {
      if self.ps_libs[i] == lib {
        self.ps_libs.swap_remove(i);
      } else {
        i += 1;
      }
    }
  }

  pub fn count_pseudoliberties(&self) -> usize {
    self.ps_libs.len()
  }

  pub fn approx_count_liberties(&self) -> usize {
    let mut prev_lib = None;
    for &lib in self.ps_libs.iter() {
      match prev_lib {
        None    => prev_lib = Some(lib),
        Some(p) => if p != lib {
          return 2;
        },
      }
    }
    match prev_lib {
      None    => return 0,
      Some(_) => return 1,
    }
  }
}

pub struct FastBoardWork {
  // Temporary buffers used during play() and stored here to avoid extra heap
  // allocations.
  merge_adj_heads: Vec<Pos>,

  // TODO(20151004): Support for `undo`.
  //
  // What is modified during play():
  // - stone state (pos flags, adj flags)
  // - chain state
  //
  // Stone state is relatively easy to revert and consists of 2 operations:
  // - place stone:
  //   Most flags can be reverted easily, while a few need to be saved:
  //   last pos, first play flag.
  // - capture stone
  //   Same as above.
  //
  // Chain state is significantly more tricky and comprises the following
  // operations:
  // - create chain
  // - append stone to chain
  // - merge chains (2 to 4 chain unions)
  // - kill chain
  //
  // The basic idea for `undo`-safe chain merge is to mutate as little as
  // necessary:
  // - In non-`undo`-safe merge, weighted union is used for balancing; this
  //   leads to some chains acting as both merger and mergee.
  //   In `undo`-safe merge, do not use weighted union, so that only a single
  //   chain per play acts as the "merger" chain.
  // - In non-`undo`-safe merge, quick-union find w/ path compression is used
  //   during unions; this mutates chain roots other than the chain heads.
  //   In `undo`-safe merge, use slow but immutable find operation during
  //   unions, so that only the chain heads are mutated in .ch_roots.
  //   (Note: This has the side effect of a small performance gain.)
  // - In non-`undo`-safe merge, both chains' links are "glued" at both ends.
  //   Tracking the glue points can revert .ch_links.
  // - In non-`undo`-safe merge, chains' sizes are both accumulated and zeroed.
  //   In `undo`-safe merge, only mutating the merger chain preserves .ch_sizes.
  //   (Note: .ch_sizes is still necessary despite unweighted union if `undo`
  //   ops are to coexist with non-`undo` ops.)
  // - In non-`undo`-safe merge, a merged chain's liberties are dropped after
  //   being used to linearly extend the merging chain's liberties.
  //   In `undo`-safe merge, keep the merged chains' liberties, and to revert
  //   the merger chain's liberties simply truncate them at the extend point.
  //
  // The same idea applies to `undo`-safe chain kill:
  // - Chain kill calls stone capture, which can be safely reverted.
  // - In non-`undo`-safe kill, roots, links, and sizes are zeroed out.
  //   In `undo`-safe kill, simply do not touch .ch_roots, .ch_links, and
  //   .ch_sizes and leave them in an invalidated state.
  // - In non-`undo`-safe kill, the chain liberties are dropped.
  //   In `undo`-safe kill, keep the chain liberties.
  undo_place_pos:       Pos,
  undo_place_last_pos:  Option<Pos>,
  undo_place_st_first:  bool,
  undo_cap_stones:      Vec<(Pos, Stone)>,
  undo_merge_glue:      Vec<Pos>,
  undo_merge_size:      u16,
  undo_merge_chains:    Vec<Option<Box<Chain>>>,
  undo_merge_extend:    usize,
  undo_kill_pslib_lims: Vec<(Pos, usize)>,
  undo_kill_chains:     Vec<Option<Box<Chain>>>,
  undo_kill_ko_pos:     Option<Pos>,
}

impl FastBoardWork {
  pub fn new() -> FastBoardWork {
    FastBoardWork{
      merge_adj_heads:      Vec::with_capacity(4),
      undo_place_pos:       TOMBSTONE,
      undo_place_last_pos:  None,
      undo_place_st_first:  false,
      undo_cap_stones:      Vec::with_capacity(4),
      undo_merge_glue:      Vec::with_capacity(4),
      undo_merge_size:      0,
      undo_merge_chains:    Vec::with_capacity(4),
      undo_merge_extend:    0,
      undo_kill_pslib_lims: Vec::with_capacity(4),
      undo_kill_chains:     Vec::with_capacity(4),
      undo_kill_ko_pos:     None,
    }
  }

  #[inline]
  pub fn reset(&mut self) {
    self.merge_adj_heads.clear();
  }

  pub fn txn_reset(&mut self) {
    self.reset();

    // TODO
    unimplemented!();
  }
}

#[derive(Clone)]
pub struct FastBoardAux {
  // Extra data structures (some may be moved back to FastBoard).
  st_first: BitVec,         // first play at position.
  st_empty: BitVec,         // position is empty.
  st_adj:   Vec<BitVec>,    // adjacent stones of a certain color (4 bits/pos).
  st_legal: BitSet,         // legal positions.
  zob_hash: u64,            // Zobrist hash.
}

impl FastBoardAux {
  pub fn new() -> FastBoardAux {
    FastBoardAux{
      st_first: BitVec::from_elem(FastBoard::BOARD_SIZE, true),
      st_empty: BitVec::from_elem(FastBoard::BOARD_SIZE, true),
      st_adj:   vec![
        BitVec::from_elem(FastBoard::BOARD_SIZE * 4, false),
        BitVec::from_elem(FastBoard::BOARD_SIZE * 4, false),
      ],
      st_legal: BitSet::with_capacity(FastBoard::BOARD_SIZE),
      zob_hash: 0,
    }
  }

  pub fn reset(&mut self) {
    self.st_first.set_all();
    self.st_empty.set_all();
    self.st_adj[0].clear();
    self.st_adj[1].clear();
    self.st_legal.extend((0 .. FastBoard::BOARD_SIZE));
    self.zob_hash = 0;
  }

  fn check_move_simple(&self, stone: Stone, action: Action, board: &FastBoard) -> Option<ActionStatus> {
    match action {
      Action::Resign  => { return Some(ActionStatus::Legal); }
      Action::Pass    => { return Some(ActionStatus::Legal); }
      Action::Place{pos} => {
        let i = pos.idx();
        // Position is occupied.
        if !self.st_empty[i] {
          return Some(ActionStatus::IllegalOccupied);
        }
        // No suicide.
        let opp_stone = stone.opponent();
        let mut is_eye_suicide = true;
        FastBoard::for_each_adjacent_labeled(pos, |_, adj_direction| {
          if !self.st_adj[opp_stone.offset()].get(i * 4 + adj_direction).unwrap() {
            is_eye_suicide = false;
          }
        });
        if is_eye_suicide {
          return Some(ActionStatus::IllegalSuicide);
        }
        // Ko rule.
        let first_play = self.st_first[i];
        if first_play {
          return Some(ActionStatus::Legal);
        }
        if let Some(ko_pos) = board.ko_pos {
          if ko_pos == pos {
            return Some(ActionStatus::IllegalKo);
          }
        }
        // Simple case: first play on the position, move does not reduce any
        // chain's liberties to zero. Doesn't require simulating the move.
        if first_play {
          let mut is_chain_suicide = false;
          let mut is_captor = false;
          FastBoard::for_each_adjacent(pos, |adj_pos| {
            // XXX: Check both player's and opponent's adjacent chains.
            let adj_head = board.traverse_chain_slow(adj_pos);
            if board.get_chain(adj_head).approx_count_liberties() <= 1 {
              if board.stones[adj_pos.idx()] == stone {
                is_chain_suicide = true;
              } else if board.stones[adj_pos.idx()] == opp_stone {
                is_captor = true;
              }
            }
          });
          if !is_captor {
            if is_chain_suicide {
              return Some(ActionStatus::IllegalSuicide);
            }
            return Some(ActionStatus::Legal);
          }
        }
        // Otherwise, we have to simulate the move to check legality.
        None
      }
    }
  }

  pub fn check_move(&self, stone: Stone, action: Action, board: &FastBoard) -> ActionStatus {
    // FIXME
    match self.check_move_simple(stone, action, board) {
      Some(status)  => status,
      None          => ActionStatus::IllegalUnknown,
    }
  }

  pub fn mark_legal_positions(&self, stone: Stone, board: &FastBoard) -> BitSet {
    let mut mask = BitSet::with_capacity(FastBoard::BOARD_SIZE);
    for p in (0 .. FastBoard::BOARD_SIZE) {
      match self.check_move(stone, Action::Place{pos: p as Pos}, board) {
        ActionStatus::Legal => { mask.insert(p); }
        _ => {}
      }
    }
    mask
  }

  pub fn get_legal_positions(&self, stone: Stone, board: &FastBoard) -> &BitSet {
    // TODO
    unimplemented!();
  }
}

#[derive(Clone, Debug)]
pub struct FastBoard {
  in_txn:   bool,

  // Basic board structure.
  turn:     Stone,
  stones:   Vec<Stone>,
  ko_pos:   Option<Pos>,

  // Previous board structure.
  last_pos: Option<Pos>,

  // Chain data structures. Namely, this implements a linked quick-union for
  // finding and iterating over connected components, as well as a list of
  // chain liberties.
  chains:   Vec<Option<Box<Chain>>>,
  ch_roots: Vec<Pos>,
  ch_links: Vec<Pos>,
  ch_sizes: Vec<u16>,
}

impl FastBoard {
  pub const BOARD_DIM:  i16   = 19;
  pub const BOARD_SIZE: usize = 361;

  pub fn for_each_adjacent<F>(pos: Pos, mut f: F) where F: FnMut(Pos) {
    let (x, y) = (pos % Self::BOARD_DIM, pos / Self::BOARD_DIM);
    if x >= 1 {
      f(pos - 1);
    }
    if y >= 1 {
      f(pos - Self::BOARD_DIM);
    }
    if x < Self::BOARD_DIM - 1 {
      f(pos + 1);
    }
    if y < Self::BOARD_DIM - 1 {
      f(pos + Self::BOARD_DIM);
    }
  }

  pub fn for_each_adjacent_labeled<F>(pos: Pos, mut f: F) where F: FnMut(Pos, usize) {
    let (x, y) = (pos % Self::BOARD_DIM, pos / Self::BOARD_DIM);
    if x >= 1 {
      f(pos - 1,                0);
    }
    if y >= 1 {
      f(pos - Self::BOARD_DIM,  1);
    }
    if x < Self::BOARD_DIM - 1 {
      f(pos + 1,                2);
    }
    if y < Self::BOARD_DIM - 1 {
      f(pos + Self::BOARD_DIM,  3);
    }
  }

  pub fn new() -> FastBoard {
    FastBoard{
      in_txn:   false,
      turn:     Stone::Black,
      stones:   Vec::with_capacity(Self::BOARD_SIZE),
      ko_pos:   None,
      last_pos: None,
      chains:   Vec::with_capacity(Self::BOARD_SIZE),
      ch_roots: Vec::with_capacity(Self::BOARD_SIZE),
      ch_links: Vec::with_capacity(Self::BOARD_SIZE),
      ch_sizes: Vec::with_capacity(Self::BOARD_SIZE),
    }
  }

  pub fn current_turn(&self) -> Stone {
    self.turn
  }

  pub fn reset(&mut self) {
    self.in_txn = false;

    // FIXME(20151004): some of this should depend on a static configuration.
    self.turn = Stone::Black;
    self.stones.clear();
    self.stones.extend(repeat(Stone::Empty).take(Self::BOARD_SIZE));
    self.ko_pos = None;

    self.last_pos = None;

    self.chains.clear();
    self.chains.extend(repeat(None).take(Self::BOARD_SIZE));
    self.ch_roots.clear();
    self.ch_roots.extend(repeat(TOMBSTONE).take(Self::BOARD_SIZE));
    self.ch_links.clear();
    self.ch_links.extend(repeat(TOMBSTONE).take(Self::BOARD_SIZE));
    self.ch_sizes.clear();
    self.ch_sizes.extend(repeat(0).take(Self::BOARD_SIZE));
  }

  pub fn play_turn(&mut self, action: Action, work: &mut FastBoardWork, aux: &mut Option<FastBoardAux>) {
    let turn = self.turn;
    self.play(turn, action, work, aux);
  }

  pub fn play(&mut self, stone: Stone, action: Action, work: &mut FastBoardWork, aux: &mut Option<FastBoardAux>) {
    //println!("DEBUG: play: {:?} {:?}", stone, action);
    assert!(!self.in_txn);
    work.reset();
    match action {
      Action::Resign => {}
      Action::Pass => {}
      Action::Place{pos} => {
        // TODO(20151003)
        //println!("DEBUG: place stone");
        self.place_stone(stone, pos, aux);
        //println!("DEBUG: update chains");
        self.update_chains(stone, pos, work, aux);
      }
    }
    // TODO
    self.turn = self.turn.opponent();
  }

  pub fn begin_txn_play(&mut self, stone: Stone, action: Action, work: &mut FastBoardWork, aux: &mut Option<FastBoardAux>) {
    assert!(!self.in_txn);
    work.txn_reset();
    match action {
      Action::Resign => {}
      Action::Pass => {}
      Action::Place{pos} => {
        // TODO(20151003)
        self.txn_place_stone(stone, pos, work, aux);
        self.txn_update_chains(stone, pos, work, aux);
      }
    }
    // TODO
    self.in_txn = true;
  }

  pub fn commit_txn_play(&mut self) {
    self.in_txn = false;
    self.turn = self.turn.opponent();
  }

  pub fn undo_txn_play(&mut self) {
    // TODO
    self.in_txn = false;
  }

  fn place_stone(&mut self, stone: Stone, pos: Pos, aux: &mut Option<FastBoardAux>) {
    let i = pos.idx();
    //println!("DEBUG: place stone: {} {}", i, self.stones.len());
    self.stones[i] = stone;
    self.last_pos = Some(pos);
    if let &mut Some(ref mut aux) = aux {
      //println!("DEBUG: aux: set flags {} {}", stone.offset(), aux.st_adj.len());
      aux.st_first.set(i, false);
      aux.st_empty.set(i, false);
      Self::for_each_adjacent_labeled(pos, |adj_pos, adj_direction| {
        aux.st_adj[stone.offset()].set(adj_pos.idx() * 4 + adj_direction ^ 0x02, true);
      });
    }
  }

  fn capture_stone(&mut self, pos: Pos, aux: &mut Option<FastBoardAux>) {
    let i = pos.idx();
    let stone = self.stones[i];
    self.stones[i] = Stone::Empty;
    if let &mut Some(ref mut aux) = aux {
      aux.st_empty.set(i, true);
      Self::for_each_adjacent_labeled(pos, |adj_pos, adj_direction| {
        aux.st_adj[stone.offset()].set(adj_pos.idx() * 4 + adj_direction ^ 0x02, false);
      });
    }
  }

  fn txn_place_stone(&mut self, stone: Stone, pos: Pos, work: &mut FastBoardWork, aux: &mut Option<FastBoardAux>) {
    let i = pos.idx();
    work.undo_place_pos = pos;
    self.stones[i] = stone;
    work.undo_place_last_pos = self.last_pos;
    self.last_pos = Some(pos);
    if let &mut Some(ref mut aux) = aux {
      work.undo_place_st_first = aux.st_first.get(i).unwrap();
      aux.st_first.set(i, false);
      aux.st_empty.set(i, false);
      Self::for_each_adjacent_labeled(pos, |adj_pos, adj_direction| {
        aux.st_adj[stone.offset()].set(adj_pos.idx() * 4 + adj_direction ^ 0x02, true);
      });
    }
  }

  fn txn_capture_stone(&mut self, pos: Pos, work: &mut FastBoardWork, aux: &mut Option<FastBoardAux>) {
    let i = pos.idx();
    let stone = self.stones[i];
    work.undo_cap_stones.push((pos, stone));
    self.stones[i] = Stone::Empty;
    if let &mut Some(ref mut aux) = aux {
      aux.st_empty.set(i, true);
      Self::for_each_adjacent_labeled(pos, |adj_pos, adj_direction| {
        aux.st_adj[stone.offset()].set(adj_pos.idx() * 4 + adj_direction ^ 0x02, false);
      });
    }
  }

  fn get_chain(&self, head: Pos) -> &Chain {
    &**self.chains[head.idx()].as_ref().unwrap()
      //.expect(&format!("get_chain expected a chain: {}", head))
  }

  fn get_chain_mut(&mut self, head: Pos) -> &mut Chain {
    &mut **self.chains[head.idx()].as_mut().unwrap()
      //.expect(&format!("get_chain_mut expected a chain: {}", head))
  }

  fn take_chain(&mut self, head: Pos) -> Box<Chain> {
    self.chains[head.idx()].take().unwrap()
      //.expect(&format!("take_chain expected a chain: {}", head))
  }

  fn traverse_chain(&mut self, mut pos: Pos) -> Pos {
    if self.ch_roots[pos.idx()] == TOMBSTONE {
      return TOMBSTONE;
    }
    while pos != self.ch_roots[pos.idx()] {
      self.ch_roots[pos.idx()] = self.ch_roots[self.ch_roots[pos.idx()].idx()];
      pos = self.ch_roots[pos.idx()];
    }
    pos
  }

  fn traverse_chain_slow(&self, mut pos: Pos) -> Pos {
    if self.ch_roots[pos.idx()] == TOMBSTONE {
      return TOMBSTONE;
    }
    while pos != self.ch_roots[pos.idx()] {
      pos = self.ch_roots[pos.idx()];
    }
    pos
  }

  fn union_chains(&mut self, pos1: Pos, pos2: Pos) -> Pos {
    let head1 = self.traverse_chain_slow(pos1);
    let head2 = self.traverse_chain_slow(pos2);
    if head1 == head2 {
      head1
    } else if self.ch_sizes[head1.idx()] < self.ch_sizes[head2.idx()] {
      self.ch_roots[head1.idx()] = head2;
      let prev_tail = self.ch_links[head2.idx()];
      let new_tail = self.ch_links[head1.idx()];
      self.ch_links[head1.idx()] = prev_tail;
      self.ch_links[head2.idx()] = new_tail;
      self.ch_sizes[head2.idx()] += self.ch_sizes[head1.idx()];
      self.ch_sizes[head1.idx()] = 0;
      let old_chain = self.take_chain(head1);
      self.get_chain_mut(head2).ps_libs.extend(&old_chain.ps_libs);
      head2
    } else {
      self.ch_roots[head2.idx()] = head1;
      let prev_tail = self.ch_links[head1.idx()];
      let new_tail = self.ch_links[head2.idx()];
      self.ch_links[head2.idx()] = prev_tail;
      self.ch_links[head1.idx()] = new_tail;
      self.ch_sizes[head1.idx()] += self.ch_sizes[head2.idx()];
      self.ch_sizes[head2.idx()] = 0;
      let old_chain = self.take_chain(head2);
      self.get_chain_mut(head1).ps_libs.extend(&old_chain.ps_libs);
      head1
    }
  }

  fn txn_union_chains(&mut self, pos1: Pos, pos2: Pos, work: &mut FastBoardWork) -> Pos {
    let head1 = self.traverse_chain_slow(pos1);
    let head2 = self.traverse_chain_slow(pos2);
    if head1 == head2 {
      head1
    } else {
      // FIXME(20151004)
      self.ch_roots[head2.idx()] = head1;
      let prev_tail = self.ch_links[head1.idx()];
      let new_tail = self.ch_links[head2.idx()];
      self.ch_links[head2.idx()] = prev_tail;
      self.ch_links[head1.idx()] = new_tail;
      self.ch_sizes[head1.idx()] += self.ch_sizes[head2.idx()];
      //self.ch_sizes[head2.idx()] = 0;
      let old_chain = self.take_chain(head2);
      self.get_chain_mut(head1).ps_libs.extend(&old_chain.ps_libs);
      head1
    }
  }

  fn update_chains(&mut self, stone: Stone, pos: Pos, work: &mut FastBoardWork, aux: &mut Option<FastBoardAux>) {
    let opp_stone = stone.opponent();
    Self::for_each_adjacent(pos, |adj_pos| {
      if self.stones[adj_pos.idx()] != Stone::Empty {
        let adj_head = self.traverse_chain(adj_pos);
        self.get_chain_mut(adj_head).remove_pseudoliberty(pos);
        if self.stones[adj_head.idx()] == opp_stone {
          if self.get_chain(adj_head).count_pseudoliberties() == 0 {
            self.kill_chain(adj_head, aux);
          }
        } else {
          work.merge_adj_heads.push(adj_head);
        }
      }
    });
    self.merge_chains_and_append(&work.merge_adj_heads, pos);
  }

  fn create_chain(&mut self, head: Pos) {
    self.ch_roots[head.idx()] = head;
    self.ch_links[head.idx()] = head;
    self.ch_sizes[head.idx()] = 1;
    self.chains[head.idx()] = Some(Box::new(Chain::new()));
    Self::for_each_adjacent(head, |adj_pos| {
      if let Stone::Empty = self.stones[adj_pos.idx()] {
        self.get_chain_mut(head).add_pseudoliberty(adj_pos);
      }
    });
  }

  fn add_to_chain(&mut self, head: Pos, pos: Pos) {
    self.ch_roots[pos.idx()] = head;
    //assert_eq!(root, self.traverse_chain(pos));
    let prev_tail = self.ch_links[head.idx()];
    self.ch_links[pos.idx()] = prev_tail;
    self.ch_links[head.idx()] = pos;
    self.ch_sizes[head.idx()] += 1;
    Self::for_each_adjacent(pos, |adj_pos| {
      if let Stone::Empty = self.stones[adj_pos.idx()] {
        self.get_chain_mut(head).add_pseudoliberty(adj_pos);
      }
    });
  }

  fn merge_chains_and_append(&mut self, heads: &[Pos], pos: Pos) {
    let mut prev_root = None;
    for i in (1 .. heads.len()) {
      match prev_root {
        None    => prev_root = Some(self.union_chains(heads[i-1], heads[i])),
        Some(r) => prev_root = Some(self.union_chains(r, heads[i])),
      }
    }
    match prev_root {
      None    => self.create_chain(pos),
      Some(r) => self.add_to_chain(r, pos),
    }
  }

  fn kill_chain(&mut self, head: Pos, aux: &mut Option<FastBoardAux>) {
    let mut num_captured = 0;
    let mut prev;
    let mut next = head;
    let opp_stone = self.stones[head.idx()].opponent();
    loop {
      prev = next;
      next = self.ch_links[prev.idx()];
      self.capture_stone(next, aux);
      Self::for_each_adjacent(next, |adj_pos| {
        if self.stones[adj_pos.idx()] == opp_stone {
          self.get_chain_mut(adj_pos).add_pseudoliberty(next);
        }
      });
      self.ch_roots[next.idx()] = TOMBSTONE;
      self.ch_links[prev.idx()] = TOMBSTONE;
      num_captured += 1;
      if next == head {
        break;
      }
    }
    self.ch_sizes[head.idx()] = 0;
    self.take_chain(head);
    if num_captured == 1 {
      // FIXME(20151004): there can be multiple ko points.
      self.ko_pos = Some(prev);
    }
  }

  fn txn_update_chains(&mut self, stone: Stone, pos: Pos, work: &mut FastBoardWork, aux: &mut Option<FastBoardAux>) {
    let opp_stone = stone.opponent();
    Self::for_each_adjacent(pos, |adj_pos| {
      if self.stones[adj_pos.idx()] != Stone::Empty {
        let adj_head = self.traverse_chain(adj_pos);
        self.get_chain_mut(adj_head).remove_pseudoliberty(pos);
        if self.stones[adj_head.idx()] == opp_stone {
          if self.get_chain(adj_head).count_pseudoliberties() == 0 {
            self.txn_kill_chain(adj_head, work, aux);
          }
        } else {
          work.merge_adj_heads.push(adj_head);
        }
      }
    });
    self.txn_merge_chains_and_append(pos, work);
  }

  fn txn_create_chain(&mut self, head: Pos, work: &mut FastBoardWork) {
    // FIXME(20151004)
    self.ch_roots[head.idx()] = head;
    self.ch_links[head.idx()] = head;
    self.ch_sizes[head.idx()] = 1;
    self.chains[head.idx()] = Some(Box::new(Chain::new()));
    Self::for_each_adjacent(head, |adj_pos| {
      if let Stone::Empty = self.stones[adj_pos.idx()] {
        self.get_chain_mut(head).add_pseudoliberty(adj_pos);
      }
    });
  }

  fn txn_add_to_chain(&mut self, head: Pos, pos: Pos, work: &mut FastBoardWork) {
    // FIXME(20151004)
    self.ch_roots[pos.idx()] = head;
    let prev_tail = self.ch_links[head.idx()];
    self.ch_links[pos.idx()] = prev_tail;
    self.ch_links[head.idx()] = pos;
    self.ch_sizes[head.idx()] += 1;
    Self::for_each_adjacent(pos, |adj_pos| {
      if let Stone::Empty = self.stones[adj_pos.idx()] {
        self.get_chain_mut(head).add_pseudoliberty(adj_pos);
      }
    });
  }

  fn txn_merge_chains_and_append(&mut self, pos: Pos, work: &mut FastBoardWork) {
    let mut prev_root = None;
    for i in (1 .. work.merge_adj_heads.len()) {
      match prev_root {
        None    => prev_root = Some(self.txn_union_chains(work.merge_adj_heads[i-1], work.merge_adj_heads[i], work)),
        Some(r) => prev_root = Some(self.txn_union_chains(r, work.merge_adj_heads[i], work)),
      }
    }
    match prev_root {
      None    => self.txn_create_chain(pos, work),
      Some(r) => self.txn_add_to_chain(r, pos, work),
    }
  }

  fn txn_kill_chain(&mut self, head: Pos, work: &mut FastBoardWork, aux: &mut Option<FastBoardAux>) {
    let mut num_captured = 0;
    let mut prev;
    let mut next = head;
    let opp_stone = self.stones[head.idx()].opponent();
    loop {
      prev = next;
      next = self.ch_links[prev.idx()];
      self.txn_capture_stone(next, work, aux);
      Self::for_each_adjacent(next, |adj_pos| {
        if self.stones[adj_pos.idx()] == opp_stone {
          let adj_head = self.traverse_chain_slow(adj_pos);
          work.undo_kill_pslib_lims.push((adj_head, self.get_chain(adj_head).count_pseudoliberties()));
          self.get_chain_mut(adj_pos).add_pseudoliberty(next);
        }
      });
      //self.ch_roots[next.idx()] = TOMBSTONE;
      //self.ch_links[prev.idx()] = TOMBSTONE;
      num_captured += 1;
      if next == head {
        break;
      }
    }
    //self.ch_sizes[head.idx()] = 0;
    let captured_chain = self.take_chain(head);
    work.undo_kill_chains.push(Some(captured_chain));
    if num_captured == 1 {
      // FIXME(20151004): there can be multiple ko points.
      work.undo_kill_ko_pos = self.ko_pos;
      self.ko_pos = Some(prev);
    }
  }
}
