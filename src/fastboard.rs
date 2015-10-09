use gtp_board::{Player, Coord, Vertex};

use bit_set::{BitSet};
use bit_vec::{BitVec};
use std::cmp::{min};
use std::collections::{HashSet};
use std::iter::{repeat};
use std::str::{from_utf8};

const TOMBSTONE: Pos = -1;

pub trait PosExt {
  fn from_coord(coord: Coord) -> Self where Self: Sized;
  fn to_coord(self) -> Coord;
  fn idx(self) -> usize;
  fn rot(self, r: u8) -> Pos;
  fn flip(self, f: u8) -> Pos;
}

pub type Pos = i16;

impl PosExt for Pos {
  fn from_coord(coord: Coord) -> Pos {
    assert!(coord.x < 19);
    assert!(coord.y < 19);
    coord.x as Pos + (coord.y as Pos) * 19
  }

  fn to_coord(self) -> Coord {
    let coord = Coord{x: (self % 19) as u8, y: (self / 19) as u8};
    assert!(coord.x < 19);
    assert!(coord.y < 19);
    coord
  }

  #[inline]
  fn idx(self) -> usize {
    self as usize
  }

  #[inline]
  fn rot(self, r: u8) -> Pos {
    let (x, y) = (self % 19, self / 19);
    match r {
      0 => self,
      1 => y        + (19-1-x) * 19,
      2 => (19-1-x) + (19-1-y) * 19,
      3 => (19-1-y) + x * 19,
      _ => unreachable!(),
    }
  }

  #[inline]
  fn flip(self, f: u8) -> Pos {
    let (x, y) = (self % 19, self / 19);
    match f {
      0 => self,
      1 => (19-1-x) + y * 19,
      2 => x        + (19-1-y) * 19,
      3 => (19-1-x) + (19-1-y) * 19,
      _ => unreachable!(),
    }
  }
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Stone {
  Black = 0,
  White = 1,
  Empty = 2,
}

impl Stone {
  pub fn from_player(player: Player) -> Stone {
    match player {
      Player::Black => Stone::Black,
      Player::White => Stone::White,
    }
  }

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

impl Action {
  pub fn from_vertex(vertex: Vertex) -> Action {
    match vertex {
      Vertex::Resign  => Action::Resign,
      Vertex::Pass    => Action::Pass,
      Vertex::Play(c) => Action::Place{pos: Pos::from_coord(c)},
    }
  }

  pub fn to_vertex(&self) -> Vertex {
    match *self {
      Action::Resign      => Vertex::Resign,
      Action::Pass        => Vertex::Pass,
      Action::Place{pos}  => Vertex::Play(pos.to_coord()),
    }
  }
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
pub struct ChainLibs {
  ps_libs: Vec<Pos>,
}

impl ChainLibs {
  pub fn new() -> ChainLibs {
    ChainLibs{ps_libs: Vec::with_capacity(4) }
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

#[derive(Clone)]
pub struct FastBoardWork {
  // Temporary buffers used during play() and stored here to avoid extra heap
  // allocations.
  merge_adj_heads:  Vec<Pos>,

  check_pos:        BitVec,

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
  /*undo_place_pos:       Pos,
  undo_place_last_pos:  Option<Pos>,
  undo_place_st_first:  bool,
  undo_cap_stones:      Vec<(Pos, Stone)>,
  undo_merge_glue:      Vec<Pos>,
  undo_merge_size:      u16,
  undo_merge_chains:    Vec<Option<Box<ChainLibs>>>,
  undo_merge_extend:    usize,
  undo_kill_pslib_lims: Vec<(Pos, usize)>,
  undo_kill_chains:     Vec<Option<Box<ChainLibs>>>,
  undo_kill_ko_pos:     Option<Pos>,*/
}

impl FastBoardWork {
  pub fn new() -> FastBoardWork {
    FastBoardWork{
      merge_adj_heads:      Vec::with_capacity(4),
      check_pos:            BitVec::from_elem(FastBoard::BOARD_SIZE, false),
      /*undo_place_pos:       TOMBSTONE,
      undo_place_last_pos:  None,
      undo_place_st_first:  false,
      undo_cap_stones:      Vec::with_capacity(4),
      undo_merge_glue:      Vec::with_capacity(4),
      undo_merge_size:      0,
      undo_merge_chains:    Vec::with_capacity(4),
      undo_merge_extend:    0,
      undo_kill_pslib_lims: Vec::with_capacity(4),
      undo_kill_chains:     Vec::with_capacity(4),
      undo_kill_ko_pos:     None,*/
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
  // Stone state data structures (some may be moved back to FastBoard).
  st_first:     BitVec,         // first play at position.
  st_empty:     BitVec,         // position is empty.
  st_legal:     Vec<BitSet>,    // legal positions.
  st_reach:     Vec<BitVec>,    // whether stones of a certain color reach empty.
  //st_adj:       Vec<BitVec>,    // adjacent stones of a certain color (4 bits/pos).

  // Chain state data structures.
  ch_heads:     HashSet<Pos>,

  // Data about the last play. The intended sequence of FastBoard and
  // FastBoardAux operation is:
  // 1. state.play_turn(..., aux_state)
  // 2. aux_state.update_turn(state, ...)
  last_adj_chs: Vec<Pos>,       // last move adjacent chains.
  last_cap_chs: Vec<Pos>,       // last move captured chains.

  //hash: u64,            // Zobrist hash.
}

impl FastBoardAux {
  pub fn new() -> FastBoardAux {
    FastBoardAux{
      st_first:     BitVec::from_elem(FastBoard::BOARD_SIZE, true),
      st_empty:     BitVec::from_elem(FastBoard::BOARD_SIZE, true),
      st_legal:     vec![
        BitSet::with_capacity(FastBoard::BOARD_SIZE),
        BitSet::with_capacity(FastBoard::BOARD_SIZE),
      ],
      st_reach:     vec![
        BitVec::from_elem(FastBoard::BOARD_SIZE, false),
        BitVec::from_elem(FastBoard::BOARD_SIZE, false),
      ],
      /*st_adj:       vec![
        BitVec::from_elem(FastBoard::BOARD_SIZE * 4, false),
        BitVec::from_elem(FastBoard::BOARD_SIZE * 4, false),
      ],*/
      ch_heads:     HashSet::new(),
      last_adj_chs: Vec::with_capacity(4),
      last_cap_chs:   Vec::with_capacity(4),
      //hash: 0,
    }
  }

  pub fn reset(&mut self) {
    self.st_first.set_all();
    self.st_empty.set_all();
    self.st_legal[0].extend((0 .. FastBoard::BOARD_SIZE));
    self.st_legal[1].extend((0 .. FastBoard::BOARD_SIZE));
    self.st_reach[0].clear();
    self.st_reach[1].clear();
    //self.st_adj[0].clear();
    //self.st_adj[1].clear();
    self.ch_heads.clear();
    //self.hash = 0;
  }

  /*fn update_turn(&mut self, board: &FastBoard, work: &mut FastBoardWork) {
    let turn = board.current_turn();
    self.update(turn, board, work);
  }*/

  fn update(&mut self, turn: Stone, board: &FastBoard, work: &mut FastBoardWork) {
    self.update_legal_positions(board, work);
  }

  fn check_move_simple(&self, stone: Stone, action: Action, board: &FastBoard, verbose: bool) -> Option<ActionStatus> {
    match action {
      Action::Resign  => { return Some(ActionStatus::Legal); }
      Action::Pass    => { return Some(ActionStatus::Legal); }
      Action::Place{pos} => {
        // Local checks: these touch only the candidate position and its
        // 4-adjacent positions.
        let i = pos.idx();
        // Position is occupied.
        if !self.st_empty[i] {
          return Some(ActionStatus::IllegalOccupied);
        }
        // Position is not an eye.
        // Approximate definition of an eye:
        // 1. surrounded on all 4 sides by the opponent's chain;
        // 2. the surrounding chain has at least 2 liberties.
        let opp_stone = stone.opponent();
        let mut n_adj_opp = 0;
        let mut n_adj = 0;
        FastBoard::for_each_adjacent(pos, |adj_pos| {
          n_adj += 1;
          if board.stones[adj_pos.idx()] == opp_stone {
            n_adj_opp += 1;
          }
        });
        if n_adj_opp == n_adj {
          let mut eye_adj_libs = 2;
          FastBoard::for_each_adjacent(pos, |adj_pos| {
            let adj_head = board.traverse_chain_slow(adj_pos);
            assert!(adj_head != TOMBSTONE);
            let adj_chain = board.get_chain(adj_head);
            eye_adj_libs = min(eye_adj_libs, adj_chain.approx_count_liberties());
          });
          if eye_adj_libs >= 2 {
            return Some(ActionStatus::IllegalSuicide);
          }
        }
        // Ko rule.
        let first_play = self.st_first[i];
        if first_play {
          return Some(ActionStatus::Legal);
        }
        // FIXME(20151009): fix ko point stuff.
        /*if let Some(ko_pos) = board.ko_pos {
          if ko_pos == pos {
            return Some(ActionStatus::IllegalKo);
          }
        }*/

        // TODO(20151009): leaving this as the terminal legal move check.
        // What else needs to be checked?

        // Condition for quick global check: first play on the position, move
        // does not reduce any chain's liberties to zero. No simulation needed.
        //if first_play {
          let mut is_chain_suicide = false;
          let mut is_captor = false;
          FastBoard::for_each_adjacent(pos, |adj_pos| {
            // XXX: Check both player's and opponent's adjacent chains.
            if board.stones[adj_pos.idx()] != Stone::Empty {
              let adj_head = board.traverse_chain_slow(adj_pos);
              assert!(adj_head != TOMBSTONE);
              if board.get_chain(adj_head).approx_count_liberties() <= 1 {
                if board.stones[adj_pos.idx()] == stone {
                  is_chain_suicide = true;
                } else if board.stones[adj_pos.idx()] == opp_stone {
                  is_captor = true;
                }
              }
            }
          });
          /*if !is_captor {
            if is_chain_suicide {
              return Some(ActionStatus::IllegalSuicide);
            }
            return Some(ActionStatus::Legal);
          }*/
          if is_captor {
            return Some(ActionStatus::Legal);
          } else if is_chain_suicide {
            return Some(ActionStatus::IllegalSuicide);
          } else {
            return Some(ActionStatus::Legal);
          }
        //}

        // Otherwise, we have to simulate the move to check legality.
        None
      }
    }
  }

  pub fn check_move(&self, turn: Stone, action: Action, board: &FastBoard, work: &mut FastBoardWork, verbose: bool) -> ActionStatus {
    match self.check_move_simple(turn, action, board, verbose) {
      Some(status)  => status,
      // FIXME(20151004): simulate move to check legality.
      None          => ActionStatus::IllegalUnknown,
    }
  }

  pub fn is_legal_move_cached(&self, turn: Stone, action: Action) -> bool {
    match action {
      Action::Resign  => { true }
      Action::Pass    => { true }
      Action::Place{pos} => {
        self.st_legal[turn.offset()].contains(&pos.idx())
      }
    }
  }

  pub fn get_legal_positions(&self, turn: Stone) -> &BitSet {
    &self.st_legal[turn.offset()]
  }

  pub fn update_legal_positions(&mut self, board: &FastBoard, work: &mut FastBoardWork) {
    // TODO(20151004): Shortcuts are possible depending on the last play:
    // - last pos touched no prior chains:
    //   only update adjacent moves
    // - last pos touched chains but did not kill any:
    //   update adjacent moves and liberties of touched chains
    // - last pos killed a chain:
    //   easy way out is to redo the whole board; otherwise need to propagate
    //   affected chains and update affected liberties
    if let Some(last_pos) = board.last_pos {
      for &turn in &[Stone::Black, Stone::White] {
        let turn_off = turn.offset();
        if self.last_cap_chs.len() > 0 {
          for p in (0 .. FastBoard::BOARD_SIZE) {
            match self.check_move(turn, Action::Place{pos: p as Pos}, board, work, false) {
              ActionStatus::Legal => { self.st_legal[turn_off].insert(p); }
              _                   => { self.st_legal[turn_off].remove(&p); }
            }
          }
        } else {
          work.check_pos.clear();
          for &h in self.last_adj_chs.iter() {
            let head = board.traverse_chain_slow(h);
            assert!(head != TOMBSTONE);
            let chain = board.get_chain(head);
            for &lib in chain.ps_libs.iter() {
              if !work.check_pos.get(lib.idx()).unwrap() {
                work.check_pos.set(lib.idx(), true);
                match self.check_move(turn, Action::Place{pos: lib}, board, work, false) {
                  ActionStatus::Legal => { self.st_legal[turn_off].insert(lib.idx()); }
                  _                   => { self.st_legal[turn_off].remove(&lib.idx()); }
                }
              }
            }
          }
          FastBoard::for_each_adjacent(last_pos, |adj_pos| {
            if !work.check_pos.get(adj_pos.idx()).unwrap() {
              match self.check_move(turn, Action::Place{pos: adj_pos}, board, work, false) {
                ActionStatus::Legal => { self.st_legal[turn_off].insert(adj_pos.idx()); }
                _                   => { self.st_legal[turn_off].remove(&adj_pos.idx()); }
              }
            }
          });
          match self.check_move(turn, Action::Place{pos: last_pos}, board, work, false) {
            ActionStatus::Legal => { self.st_legal[turn_off].insert(last_pos.idx()); }
            _                   => { self.st_legal[turn_off].remove(&last_pos.idx()); }
          }
        }
      }
    }
  }
}

#[derive(Clone, Debug)]
pub struct FastBoard {
  // FIXME(20151006): for clarity, should group fields into ones modified during
  // .play() and ones modified during .update().

  // Undo-safe txns.
  //in_txn:     bool,           // is the board in a txn?

  // Basic board structure.
  turn:       Stone,          // whose turn it is to move.
  stones:     Vec<Stone>,     // the stone configuration on the board.
  ko_pos:     Option<Pos>,    // the ko point, if any.
  // TODO(20151009): rework ko point code.
  //ko_point:   Option<(Stone, Pos)>,    // the ko point, if any.

  // Previous board structures.
  last_turn:  Option<Stone>,  // the previously played turn.
  last_pos:   Option<Pos>,    // the previously played position.
  last_caps:  Vec<Pos>,       // previously captured positions.

  // Chain data structures. Namely, this implements a linked quick-union for
  // finding and iterating over connected components, as well as a list of
  // chain liberties.
  ch_roots:   Vec<Pos>,
  ch_links:   Vec<Pos>,
  ch_sizes:   Vec<u16>,
  ch_libs:    Vec<Option<Box<ChainLibs>>>,
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
      //in_txn:     false,
      turn:       Stone::Black,
      stones:     Vec::with_capacity(Self::BOARD_SIZE),
      ko_pos:     None,
      last_turn:  None,
      last_pos:   None,
      last_caps:  Vec::with_capacity(4),
      ch_roots:   Vec::with_capacity(Self::BOARD_SIZE),
      ch_links:   Vec::with_capacity(Self::BOARD_SIZE),
      ch_sizes:   Vec::with_capacity(Self::BOARD_SIZE),
      ch_libs:    Vec::with_capacity(Self::BOARD_SIZE),
    }
  }

  pub fn current_turn(&self) -> Stone {
    self.turn
  }

  pub fn last_captures(&self) -> &[Pos] {
    &self.last_caps
  }

  pub fn get_stone(&self, pos: Pos) -> Stone {
    self.stones[pos.idx()]
  }

  pub fn get_digest(&self) -> BitSet {
    let mut digest = BitSet::with_capacity(Self::BOARD_SIZE * 2);
    for (p, &stone) in self.stones.iter().enumerate() {
      match stone {
        Stone::Black => { digest.insert(p * 2); }
        Stone::White => { digest.insert(p * 2 + 1); }
        _ => {}
      }
    }
    digest
  }

  pub fn reset(&mut self) {
    //self.in_txn = false;

    // FIXME(20151004): some of this should depend on a static configuration.
    self.turn = Stone::Black;
    self.stones.clear();
    self.stones.extend(repeat(Stone::Empty).take(Self::BOARD_SIZE));
    self.ko_pos = None;

    self.last_pos = None;
    self.last_turn = None;
    self.last_caps.clear();

    self.ch_roots.clear();
    self.ch_roots.extend(repeat(TOMBSTONE).take(Self::BOARD_SIZE));
    self.ch_links.clear();
    self.ch_links.extend(repeat(TOMBSTONE).take(Self::BOARD_SIZE));
    self.ch_sizes.clear();
    self.ch_sizes.extend(repeat(0).take(Self::BOARD_SIZE));
    self.ch_libs.clear();
    self.ch_libs.extend(repeat(None).take(Self::BOARD_SIZE));
  }

  pub fn is_legal_move_fast(&self, stone: Stone, pos: Pos) -> bool {
    // TODO(20151008): check suicide, eye, ko.
    // Local checks: these touch only the candidate position and its
    // 4-adjacent positions.
    let i = pos.idx();
    // Position is not occupied.
    if self.stones[i] != Stone::Empty {
      return false;
    }
    // FIXME(20151009): fix this psuedo-eye logic.
    // Position is not an eye.
    // Approximate definition of an eye:
    // 1. surrounded on all 4 sides by the opponent's chain;
    // 2. the surrounding chain has at least 2 liberties.
    let opp_stone = stone.opponent();
    let mut is_eye_1 = true;
    FastBoard::for_each_adjacent(pos, |adj_pos| {
      if self.stones[adj_pos.idx()] != opp_stone {
        is_eye_1 = false;
      }
    });
    if is_eye_1 {
      let mut eye_adj_libs = None;
      FastBoard::for_each_adjacent(pos, |adj_pos| {
        if eye_adj_libs.is_none() {
          let adj_head = self.traverse_chain_slow(adj_pos);
          let adj_chain = self.get_chain(adj_head);
          eye_adj_libs = Some(adj_chain.approx_count_liberties());
        }
      });
      if let Some(2) = eye_adj_libs {
        return false;
      }
    }
    // FIXME(20151009): ko rule.
    // Ko rule.
    true
  }

  pub fn score_fast(&self, komi: f32) -> f32 {
    // TODO(20151008): just count stones and eyes, then add komi;
    // negative scores go to black, positive scores go to white.
    //unimplemented!();
    let mut b_score = 0.0;
    let mut w_score = 0.0;
    for (p, &stone) in self.stones.iter().enumerate() {
      match stone {
        Stone::Black => b_score += 1.0,
        Stone::White => w_score += 1.0,
        Stone::Empty => {
          let mut n_b = 0;
          let mut n_w = 0;
          FastBoard::for_each_adjacent(p as Pos, |adj_pos| {
            match self.stones[adj_pos.idx()] {
              Stone::Black => n_b += 1,
              Stone::White => n_w += 1,
              _ => {}
            }
          });
          if n_b == 4 {
            b_score += 1.0;
          } else if n_w == 4 {
            w_score += 1.0;
          }
        }
      }
    }
    w_score - b_score + komi
  }

  pub fn play(&mut self, stone: Stone, action: Action, work: &mut FastBoardWork, aux: &mut Option<FastBoardAux>, verbose: bool) {
    // FIXME(20151008): ko point logic is seriously messed up.
    //println!("DEBUG: play: {:?} {:?}", stone, action);
    //assert!(!self.in_txn);
    work.reset();
    self.last_caps.clear();
    if let &mut Some(ref mut aux) = aux {
      aux.last_adj_chs.clear();
      aux.last_cap_chs.clear();
    }
    match action {
      Action::Resign | Action::Pass => {
        self.last_pos = None;
      }
      Action::Place{pos} => {
        self.place_stone(stone, pos, aux);
        let opp_stone = stone.opponent();
        let mut num_captured = 0;
        Self::for_each_adjacent(pos, |adj_pos| {
          if self.stones[adj_pos.idx()] != Stone::Empty {
            if verbose {
              println!("DEBUG: adjacent to {}: {}", pos.to_coord().to_string(), adj_pos.to_coord().to_string());
            }
            let adj_head = self.traverse_chain(adj_pos);
            assert!(adj_head != TOMBSTONE);
            if let &mut Some(ref mut aux) = aux {
              aux.last_adj_chs.push(adj_head);
            }
            self.get_chain_mut(adj_head).remove_pseudoliberty(pos);
            if self.stones[adj_head.idx()] == opp_stone {
              if self.get_chain(adj_head).count_pseudoliberties() == 0 {
                if verbose {
                  println!("DEBUG: about to kill chain: {} size: {}",
                      from_utf8(&adj_head.to_coord().to_bytestring()).unwrap(),
                      self.ch_sizes[adj_head.idx()]);
                }
                num_captured += self.kill_chain(adj_head, aux);
              }
            } else {
              if verbose {
                println!("DEBUG: can merge {}: {} -> {}", pos.to_coord().to_string(), adj_pos.to_coord().to_string(), adj_head.to_coord().to_string());
              }
              work.merge_adj_heads.push(adj_head);
            }
          }
        });
        if num_captured > 1 {
          self.ko_pos = None;
        }
        self.merge_chains_and_append(&work.merge_adj_heads, pos, aux, verbose);
        if let Some(ko_pos) = self.ko_pos {
          // TODO(20151005): check if not actually a kos pos.
        }
        self.last_pos = Some(pos);
      }
    }
    self.last_turn = Some(stone);
    self.turn = stone.opponent();
  }

  pub fn update(&mut self, turn: Stone, action: Action, work: &mut FastBoardWork, aux: &mut Option<FastBoardAux>, verbose: bool) {
    if let &mut Some(ref mut aux) = aux {
      aux.update(turn, self, work);
    }
  }

  fn place_stone(&mut self, stone: Stone, pos: Pos, aux: &mut Option<FastBoardAux>) {
    let i = pos.idx();
    self.stones[i] = stone;
    self.ch_roots[i] = TOMBSTONE;
    if let &mut Some(ref mut aux) = aux {
      aux.st_first.set(i, false);
      aux.st_empty.set(i, false);
      /*Self::for_each_adjacent_labeled(pos, |adj_pos, adj_direction| {
        aux.st_adj[stone.offset()].set(adj_pos.idx() * 4 + adj_direction ^ 0x02, true);
      });*/
    }
  }

  fn capture_stone(&mut self, pos: Pos, aux: &mut Option<FastBoardAux>) {
    let i = pos.idx();
    let stone = self.stones[i];
    self.stones[i] = Stone::Empty;
    self.ch_roots[i] = TOMBSTONE;
    self.last_caps.push(pos);
    if let &mut Some(ref mut aux) = aux {
      aux.st_empty.set(i, true);
      /*Self::for_each_adjacent_labeled(pos, |adj_pos, adj_direction| {
        aux.st_adj[stone.offset()].set(adj_pos.idx() * 4 + adj_direction ^ 0x02, false);
      });*/
    }
  }

  /*fn update_chains(&mut self, stone: Stone, pos: Pos, work: &mut FastBoardWork, aux: &mut Option<FastBoardAux>) {
  }*/

  fn get_chain(&self, head: Pos) -> &ChainLibs {
    &**self.ch_libs[head.idx()].as_ref()
      .unwrap()
      //.expect(&format!("get_chain expected a chain: {}", head))
  }

  fn get_chain_mut(&mut self, head: Pos) -> &mut ChainLibs {
    &mut **self.ch_libs[head.idx()].as_mut()
      .unwrap()
      //.expect(&format!("get_chain_mut expected a chain: {}", head))
  }

  fn take_chain(&mut self, head: Pos) -> Box<ChainLibs> {
    self.ch_libs[head.idx()].take()
      .unwrap()
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

  fn union_chains(&mut self, pos1: Pos, pos2: Pos, aux: &mut Option<FastBoardAux>) -> Pos {
    let head1 = self.traverse_chain_slow(pos1);
    let head2 = self.traverse_chain_slow(pos2);
    if head1 == head2 {
      head1
    } else {
      let (new_head, old_head) = if self.ch_sizes[head1.idx()] < self.ch_sizes[head2.idx()] {
        (head2, head1)
      } else {
        (head1, head2)
      };
      self.ch_roots[old_head.idx()] = new_head;
      let prev_tail = self.ch_links[new_head.idx()];
      let new_tail = self.ch_links[old_head.idx()];
      self.ch_links[old_head.idx()] = prev_tail;
      self.ch_links[new_head.idx()] = new_tail;
      self.ch_sizes[new_head.idx()] += self.ch_sizes[old_head.idx()];
      self.ch_sizes[old_head.idx()] = 0;
      let old_chain = self.take_chain(old_head);
      if let &mut Some(ref mut aux) = aux {
        aux.ch_heads.remove(&old_head);
      }
      self.get_chain_mut(new_head).ps_libs.extend(&old_chain.ps_libs);
      new_head
    }
  }

  fn create_chain(&mut self, head: Pos, aux: &mut Option<FastBoardAux>, verbose: bool) {
    if verbose {
      println!("DEBUG: creating a chain: {}", head.to_coord().to_string());
    }
    self.ch_roots[head.idx()] = head;
    self.ch_links[head.idx()] = head;
    self.ch_sizes[head.idx()] = 1;
    self.ch_libs[head.idx()] = Some(Box::new(ChainLibs::new()));
    if let &mut Some(ref mut aux) = aux {
      aux.ch_heads.insert(head);
    }
    Self::for_each_adjacent(head, |adj_pos| {
      if let Stone::Empty = self.stones[adj_pos.idx()] {
        self.get_chain_mut(head).add_pseudoliberty(adj_pos);
      }
    });
  }

  fn add_to_chain(&mut self, head: Pos, pos: Pos, verbose: bool) {
    if verbose {
      println!("DEBUG: adding stone {} to chain: {}", pos.to_coord().to_string(), head.to_coord().to_string());
    }
    assert!(head != TOMBSTONE);
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

  fn merge_chains_and_append(&mut self, heads: &[Pos], pos: Pos, aux: &mut Option<FastBoardAux>, verbose: bool) {
    let mut prev_root = None;
    for i in (0 .. heads.len()) {
      match prev_root {
        None    => prev_root = Some(heads[i]),
        Some(r) => prev_root = Some(self.union_chains(r, heads[i], aux)),
      }
    }
    match prev_root {
      None    => self.create_chain(pos, aux, verbose),
      Some(r) => self.add_to_chain(r, pos, verbose),
    }
  }

  fn kill_chain(&mut self, head: Pos, aux: &mut Option<FastBoardAux>) -> usize {
    let mut num_captured = 0;
    let mut prev;
    let mut next = head;
    let stone = self.stones[head.idx()].opponent();
    loop {
      prev = next;
      next = self.ch_links[prev.idx()];
      self.capture_stone(next, aux);
      Self::for_each_adjacent(next, |adj_pos| {
        if self.stones[adj_pos.idx()] == stone {
          let adj_head = self.traverse_chain(adj_pos);
          // XXX: Need to check for tombstone here. The killed chain is adjacent
          // to the freshly placed stone, which is currently not in a chain.
          if adj_head != TOMBSTONE {
            self.get_chain_mut(adj_head).add_pseudoliberty(next);
          }
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
    if let &mut Some(ref mut aux) = aux {
      aux.ch_heads.remove(&head);
      aux.last_cap_chs.push(head);
    }
    if num_captured == 1 {
      self.ko_pos = Some(prev);
    }
    num_captured
  }

  /*pub fn begin_txn_play(&mut self, stone: Stone, action: Action, work: &mut FastBoardWork, aux: &mut Option<FastBoardAux>) {
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

  fn txn_place_stone(&mut self, stone: Stone, pos: Pos, work: &mut FastBoardWork, aux: &mut Option<FastBoardAux>) {
    let i = pos.idx();
    work.undo_place_pos = pos;
    self.stones[i] = stone;
    work.undo_place_last_pos = self.last_pos;
    self.last_pos = Some(pos);
    // FIXME
    self.ch_roots[i] = TOMBSTONE;
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
    self.ch_libs[head.idx()] = Some(Box::new(ChainLibs::new()));
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
          // XXX: Need to check for tombstone here. The killed chain is adjacent
          // to the freshly placed stone, which is currently not in a chain.
          if adj_head != TOMBSTONE {
            work.undo_kill_pslib_lims.push((adj_head, self.get_chain(adj_head).count_pseudoliberties()));
            self.get_chain_mut(adj_pos).add_pseudoliberty(next);
          }
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
  }*/
}
