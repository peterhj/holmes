use gtp_board::{Player, Coord, Vertex};

use array::{Array2d, Array3d};

use bit_set::{BitSet};
use bit_vec::{BitVec};
use std::cmp::{min};
use std::collections::{HashSet};
use std::iter::{repeat};
use std::ops::{Range};
use std::str::{from_utf8};

const TOMBSTONE: Pos = -1;

// XXX: Absolute feature representation.
// Current board position.
const BLACK_PLANE:        usize = 0;
const WHITE_PLANE:        usize = FastBoard::BOARD_SIZE;
// Previous board position.
const PREV_BLACK_PLANE:   usize = FastBoard::BOARD_SIZE * 2;
const PREV_WHITE_PLANE:   usize = FastBoard::BOARD_SIZE * 3;
// Atari/liveness status.
/*const BLACK_ATARI_PLANE:  usize = FastBoard::BOARD_SIZE * 4;
const BLACK_LIVE_PLANE:   usize = FastBoard::BOARD_SIZE * 5;
const WHITE_ATARI_PLANE:  usize = FastBoard::BOARD_SIZE * 6;
const WHITE_LIVE_PLANE:   usize = FastBoard::BOARD_SIZE * 7;*/
const NUM_PLANES:         usize = 4;
//const NUM_PLANES:         usize = 8;

pub trait PosExt {
  fn iter_trans() -> Range<u8>;
  fn from_coord(coord: Coord) -> Self where Self: Sized;
  fn to_coord(self) -> Coord;
  fn idx(self) -> usize;
  fn trans(self, t: u8) -> Pos;
  fn inv_trans(self, t: u8) -> Pos;
  fn is_edge(self) -> bool;
}

pub type Pos = i16;

impl PosExt for Pos {
  fn iter_trans() -> Range<u8> {
    (0 .. 8)
  }

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
  fn trans(self, t: u8) -> Pos {
    let (x, y) = (self % 19, self / 19);
    let (u, v) = (x - 9, y - 9);
    match t {
      0 => self,
      1 => (9 - u) + (9 + v) * 19,
      2 => (9 + u) + (9 - v) * 19,
      3 => (9 - u) + (9 - v) * 19,
      4 => (9 + v) + (9 + u) * 19,
      5 => (9 - v) + (9 + u) * 19,
      6 => (9 + v) + (9 - u) * 19,
      7 => (9 - v) + (9 - u) * 19,
      _ => unreachable!(),
    }
  }

  #[inline]
  fn inv_trans(self, t: u8) -> Pos {
    let (x, y) = (self % 19, self / 19);
    let (u, v) = (x - 9, y - 9);
    // XXX: Most inverse transforms are the same as the forward transforms,
    // except for two (labeled below).
    match t {
      0 => self,
      1 => (9 - u) + (9 + v) * 19,
      2 => (9 + u) + (9 - v) * 19,
      3 => (9 - u) + (9 - v) * 19,
      4 => (9 + v) + (9 + u) * 19,
      5 => (9 + v) + (9 - u) * 19,  // XXX: this one is inverted.
      6 => (9 - v) + (9 + u) * 19,  // XXX: this one is inverted.
      7 => (9 - v) + (9 - u) * 19,
      _ => unreachable!(),
    }
  }

  #[inline]
  fn is_edge(self) -> bool {
    let (x, y) = (self % 19, self / 19);
    (x == 0 || x == (19 - 1) ||
     y == 0 || y == (19 - 1))
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
  Place{pos: Pos},
  Pass,
  Resign,
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
pub struct Chain {
  ps_libs:  Vec<Pos>,
  //ps_opps:  Vec<Pos>,
}

impl Chain {
  pub fn new() -> Chain {
    Chain{
      ps_libs: Vec::with_capacity(4),
      //ps_opps: Vec::with_capacity(4),
    }
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

  /*pub fn add_pseudoenemy(&mut self, opp: Pos) {
    self.ps_opps.push(opp);
  }

  pub fn remove_pseudoenemy(&mut self, opp: Pos) {
    let mut i = 0;
    while i < self.ps_opps.len() {
      if self.ps_opps[i] == opp {
        self.ps_opps.swap_remove(i);
      } else {
        i += 1;
      }
    }
  }*/

  pub fn count_pseudoliberties(&self) -> usize {
    self.ps_libs.len()
  }

  // FIXME(20151009): these approximate liberties are almost certainly wrong;
  // only a liberty if fully surrounded by same color.
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
  undo_merge_chains:    Vec<Option<Box<Chain>>>,
  undo_merge_extend:    usize,
  undo_kill_pslib_lims: Vec<(Pos, usize)>,
  undo_kill_chains:     Vec<Option<Box<Chain>>>,
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

/*struct EdgeOrderTree;

impl EdgeOrderTree {
  pub fn link(&mut self, u: Pos, v: Pos) {
  }

  pub fn cut(&mut self, u: Pos, e: usize) {
  }

  pub fn split(&mut self, v: Pos, e: usize) {
  }

  pub fn merge(&mut self, u: Pos, w: Pos) {
  }
}*/

#[derive(Clone)]
pub struct FastBoardAux {
  // Stone state data structures (some may be moved back to FastBoard).
  st_first:     BitVec,         // first play at position.
  st_empty:     BitVec,         // position is empty.
  st_legal:     Vec<BitSet>,    // legal positions.
  st_reach:     Vec<BitVec>,    // whether stones of a certain color reach empty.

  // Empty regions are tracked using a dynamic connectivity data structure.
  // Possible data structures:
  // - Eulerian tour tree: see lecture notes Williams 2015.
  // - Edge-ordered dynamic tree: see Eppstein et al. 1990; based on ST-trees
  //   from Sleator and Tarjan 1982.
  //regions:      EdgeOrderTree,

  // Chain state data structures.
  // TODO(20151009): currently unused.
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
      ch_heads:     HashSet::new(),
      last_adj_chs: Vec::with_capacity(4),
      last_cap_chs: Vec::with_capacity(4),
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
    self.ch_heads.clear();
    //self.hash = 0;
  }

  pub fn update(&mut self, _turn: Stone, board: &FastBoard, work: &mut FastBoardWork, tmp_board: &mut FastBoard, tmp_aux: &mut FastBoardAux) {
    self.update_legal_positions(board, work, tmp_board, tmp_aux);
  }

  fn check_move_simple(&self, stone: Stone, action: Action, board: &FastBoard, work: &mut FastBoardWork, tmp_board: &mut FastBoard, tmp_aux: &mut FastBoardAux, verbose: bool) -> Option<ActionStatus> {
    match action {
      Action::Resign  => { return Some(ActionStatus::Legal); }
      Action::Pass    => { return Some(ActionStatus::Legal); }
      Action::Place{pos} => {
        if verbose {
          println!("DEBUG: check move: {:?} {}", stone, pos.to_coord().to_string());
        }

        // Local checks: these touch only the candidate position and its
        // 4-adjacent positions.
        let i = pos.idx();

        // Position is occupied.
        if !self.st_empty[i] {
          if verbose {
            println!("DEBUG: illegal move: occupied");
          }
          return Some(ActionStatus::IllegalOccupied);
        }

        // Position is an opponent's eye.
        if board.is_eye1(stone, pos) {
          if verbose {
            println!("DEBUG: illegal move: playing in opponent's 1-eye");
          }
          return Some(ActionStatus::IllegalSuicide);
        }

        // Play at the ko point.
        // FIXME(20151009): fix ko point stuff.
        /*let first_play = self.st_first[i];
        if first_play {
          return Some(ActionStatus::Legal);
        }
        if let Some(ko_pos) = board.ko_pos {
          if ko_pos == pos {
            return Some(ActionStatus::IllegalKo);
          }
        }*/

        // TODO(20151009): leaving this as the terminal legal move check.
        // What else needs to be checked?
        tmp_board.clone_from(board);
        tmp_aux.clone_from(self);
        //tmp_board.play(stone, action, work, &mut Some(&mut self.clone()), false);
        tmp_board.play(stone, action, work, &mut Some(tmp_aux), false);
        //if tmp_board.get_last_suicide_captures().len() == 0 {
        if tmp_board.last_move_was_legal() {
          Some(ActionStatus::Legal)
        } else {
          Some(ActionStatus::IllegalSuicide)
        }

        /*// Condition for quick global check: first play on the position, move
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
        println!("DEBUG: move is chain suicide? {} is captor? {}",
            is_chain_suicide, is_captor);
          /*if !is_captor {
            if is_chain_suicide {
              return Some(ActionStatus::IllegalSuicide);
            }
            return Some(ActionStatus::Legal);
          }*/
        if is_captor {
          return Some(ActionStatus::Legal);
        } else if is_chain_suicide {
          if verbose {
            println!("DEBUG: illegal move: chain suicide");
          }
          return Some(ActionStatus::IllegalSuicide);
        } else {
          return Some(ActionStatus::Legal);
        }
        //}

        // Otherwise, we have to simulate the move to check legality.
        None*/
      }
    }
  }

  pub fn check_move(&self, turn: Stone, action: Action, board: &FastBoard, work: &mut FastBoardWork, tmp_board: &mut FastBoard, tmp_aux: &mut FastBoardAux, verbose: bool) -> ActionStatus {
    match self.check_move_simple(turn, action, board, work, tmp_board, tmp_aux, verbose) {
      Some(status)  => status,
      // FIXME(20151004): simulate move to check legality.
      None          => {
        if verbose {
          println!("DEBUG: illegal move: unknown, must simulate");
        }
        ActionStatus::IllegalUnknown
      }
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

  pub fn update_legal_positions(&mut self, board: &FastBoard, work: &mut FastBoardWork, tmp_board: &mut FastBoard, tmp_aux: &mut FastBoardAux) {
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
            match self.check_move(turn, Action::Place{pos: p as Pos}, board, work, tmp_board, tmp_aux, false) {
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
                match self.check_move(turn, Action::Place{pos: lib}, board, work, tmp_board, tmp_aux, false) {
                  ActionStatus::Legal => { self.st_legal[turn_off].insert(lib.idx()); }
                  _                   => { self.st_legal[turn_off].remove(&lib.idx()); }
                }
              }
            }
          }
          FastBoard::for_each_adjacent(last_pos, |adj_pos| {
            if !work.check_pos.get(adj_pos.idx()).unwrap() {
              match self.check_move(turn, Action::Place{pos: adj_pos}, board, work, tmp_board, tmp_aux, false) {
                ActionStatus::Legal => { self.st_legal[turn_off].insert(adj_pos.idx()); }
                _                   => { self.st_legal[turn_off].remove(&adj_pos.idx()); }
              }
            }
          });
          match self.check_move(turn, Action::Place{pos: last_pos}, board, work, tmp_board, tmp_aux, false) {
            ActionStatus::Legal => { self.st_legal[turn_off].insert(last_pos.idx()); }
            _                   => { self.st_legal[turn_off].remove(&last_pos.idx()); }
          }
        }
      }
    }
  }
}

pub enum PlayResult {
  Okay,
  FatalError,
}

#[derive(Clone, Debug)]
pub struct FastBoard {
  // FIXME(20151006): for clarity, should group fields into ones modified during
  // .play() and ones modified during .update().

  // Undo-safe txns.
  //in_txn:     bool,           // is the board in a txn?

  // Feature extraction.
  features:   Array3d<u8>,    // board features.
  mask:       Array2d<f32>,   // legal moves mask.

  // Current board structure.
  turn:       Stone,          // whose turn it is to move.
  stones:     Vec<Stone>,     // the stone configuration on the board.
  ko_pos:     Option<Pos>,    // the ko point, if any.
  // TODO(20151009): rework ko point code.
  //ko_point:   Option<(Stone, Pos)>,    // the ko point, if any.

  // Previous board structures.
  last_turn:  Option<Stone>,  // the previously played turn.
  last_pos:   Option<Pos>,    // the previously played position.
  last_clob:  bool,           // previous move clobbered a stone.
  last_caps:  Vec<Pos>,       // previously captured positions.
  last_suics: Vec<Pos>,       // previously suicided chains.
  //last_toucs: Vec<Pos>,       // previously "touched" positions.

  // Chain data structures. Namely, this implements a linked quick-union for
  // finding and iterating over connected components, as well as a list of
  // chain liberties.
  chains:     Vec<Option<Box<Chain>>>,
  ch_roots:   Vec<Pos>,
  ch_links:   Vec<Pos>,
  ch_sizes:   Vec<u16>,

  // History data structures.
  // TODO(20151011)
  //cap_counts: Vec<u32>,
}

impl FastBoard {
  pub const BOARD_DIM:      i16   = 19;
  pub const BOARD_SIZE:     usize = 361;
  pub const BOARD_MAX_POS:  i16   = 361;

  pub fn for_each_adjacent<F>(pos: Pos, mut f: F) where F: FnMut(Pos) {
    let (x, y) = (pos % Self::BOARD_DIM, pos / Self::BOARD_DIM);
    let upper = Self::BOARD_DIM - 1;
    if x >= 1 {
      f(pos - 1);
    }
    if y >= 1 {
      f(pos - Self::BOARD_DIM);
    }
    if x < upper {
      f(pos + 1);
    }
    if y < upper {
      f(pos + Self::BOARD_DIM);
    }
  }

  pub fn for_each_diagonal<F>(pos: Pos, mut f: F) where F: FnMut(Pos) {
    let (x, y) = (pos % Self::BOARD_DIM, pos / Self::BOARD_DIM);
    let upper = Self::BOARD_DIM - 1;
    if x >= 1 && y >= 1 {
      f(pos - 1 - Self::BOARD_DIM);
    }
    if x >= 1 && y < upper {
      f(pos - 1 + Self::BOARD_DIM);
    }
    if x < upper && y >= 1 {
      f(pos + 1 - Self::BOARD_DIM);
    }
    if x < upper && y < upper {
      f(pos + 1 + Self::BOARD_DIM);
    }
  }

  /*pub fn for_each_adjacent_labeled<F>(pos: Pos, mut f: F) where F: FnMut(Pos, usize) {
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
  }*/

  pub fn new() -> FastBoard {
    FastBoard{
      //in_txn:     false,
      features:   Array3d::with_zeros((Self::BOARD_DIM as usize, Self::BOARD_DIM as usize, NUM_PLANES)),
      mask:       Array2d::with_zeros((Self::BOARD_SIZE, 1)),
      turn:       Stone::Black,
      stones:     Vec::with_capacity(Self::BOARD_SIZE),
      ko_pos:     None,
      last_turn:  None,
      last_pos:   None,
      last_clob:  false,
      last_caps:  Vec::with_capacity(4),
      last_suics: Vec::with_capacity(4),
      //last_toucs: Vec::with_capacity(4),
      chains:     Vec::with_capacity(Self::BOARD_SIZE),
      ch_roots:   Vec::with_capacity(Self::BOARD_SIZE),
      ch_links:   Vec::with_capacity(Self::BOARD_SIZE),
      ch_sizes:   Vec::with_capacity(Self::BOARD_SIZE),
    }
  }

  pub fn current_turn(&self) -> Stone {
    self.turn
  }

  pub fn last_position(&self) -> Option<Pos> {
    self.last_pos
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

  pub fn get_bytedigest(&self) -> Vec<u32> {
    let mut digest = Vec::with_capacity((Self::BOARD_SIZE + 16 - 1) / 16);
    let mut offset = 0;
    let mut mask: u32 = 0;
    for (p, &stone) in self.stones.iter().enumerate() {
      assert_eq!(p % 16, offset);
      match stone {
        Stone::Black => {
          mask |= 1 << (2 * offset);
        }
        Stone::White => {
          mask |= 1 << (2 * offset + 1);
        }
        Stone::Empty => {
        }
      }
      offset += 1;
      if offset == 16 {
        digest.push(mask);
        offset = 0;
      }
    }
    if offset != 0 {
      digest.push(mask);
    }
    digest
  }

  pub fn extract_features(&self) -> &Array3d<u8> {
    &self.features
  }

  pub fn extract_mask(&self, turn: Stone) -> &Array2d<f32> {
    // FIXME(20151023): need to maintain 2 masks.
    &self.mask
  }

  pub fn reset(&mut self) {
    //self.in_txn = false;

    self.features.zero_out();
    self.mask.zero_out();

    // FIXME(20151004): some of this should depend on a static configuration.
    self.turn = Stone::Black;
    self.stones.clear();
    self.stones.extend(repeat(Stone::Empty).take(Self::BOARD_SIZE));
    self.ko_pos = None;

    self.last_pos = None;
    self.last_turn = None;
    self.last_clob = false;
    self.last_caps.clear();
    self.last_suics.clear();
    //self.last_toucs.clear();

    self.chains.clear();
    self.chains.extend(repeat(None).take(Self::BOARD_SIZE));
    self.ch_roots.clear();
    self.ch_roots.extend(repeat(TOMBSTONE).take(Self::BOARD_SIZE));
    self.ch_links.clear();
    self.ch_links.extend(repeat(TOMBSTONE).take(Self::BOARD_SIZE));
    self.ch_sizes.clear();
    self.ch_sizes.extend(repeat(0).take(Self::BOARD_SIZE));
  }

  pub fn last_move_was_legal(&self) -> bool {
    !self.last_clob &&
    self.last_suics.len() == 0
  }

  /*pub fn get_last_suicide_captures(&self) -> &[Pos] {
    &self.last_suics
  }*/

  #[inline]
  pub fn is_occupied(&self, pos: Pos) -> bool {
    // Position is not occupied.
    if self.stones[pos.idx()] != Stone::Empty {
      false
    } else {
      true
    }
  }

  #[inline]
  pub fn is_eyeish(&self, stone: Stone, pos: Pos) -> bool {
    let opp_stone = stone.opponent();
    let mut eyeish = true;
    FastBoard::for_each_adjacent(pos, |adj_pos| {
      if self.stones[adj_pos.idx()] != opp_stone {
        eyeish = false;
      }
    });
    eyeish
  }

  #[inline]
  pub fn is_eyelike(&self, stone: Stone, pos: Pos) -> bool {
    let opp_stone = stone.opponent();
    if !self.is_eyeish(stone, pos) {
      return false;
    }
    let mut false_count = if pos.is_edge() { 1 } else { 0 };
    FastBoard::for_each_diagonal(pos, |diag_pos| {
      if self.stones[diag_pos.idx()] == opp_stone {
        false_count += 1;
      }
    });
    false_count < 2
  }

  #[inline]
  pub fn is_eye1(&self, stone: Stone, pos: Pos) -> bool {
    // Approximate definition of an eye:
    // 1. surrounded on all sides by the opponent's chains;
    // 2. the surrounding chains each have at least 2 liberties.
    let opp_stone = stone.opponent();
    let mut n_adj_opp = 0;
    let mut n_adj = 0;
    FastBoard::for_each_adjacent(pos, |adj_pos| {
      n_adj += 1;
      if self.stones[adj_pos.idx()] == opp_stone {
        n_adj_opp += 1;
      }
    });
    if n_adj_opp == n_adj {
      // XXX: This part checks that none of the surrounding opponent chains are
      // susceptible to capture by the placed stone.
      let mut eye_adj_libs = 2;
      FastBoard::for_each_adjacent(pos, |adj_pos| {
        let adj_head = self.traverse_chain_slow(adj_pos);
        assert!(adj_head != TOMBSTONE);
        let adj_chain = self.get_chain(adj_head);
        eye_adj_libs = min(eye_adj_libs, adj_chain.approx_count_liberties());
      });
      if eye_adj_libs >= 2 {
        true
      } else {
        false
      }
    } else {
      false
    }
  }

  pub fn is_legal_move_fast(&self, stone: Stone, pos: Pos) -> bool {
    // TODO(20151008): check suicide, eye, ko.
    //let i = pos.idx();
    if self.is_occupied(pos) {
      return false;
    }
    //if self.is_single_eye(stone, pos) {
    if self.is_eyelike(stone, pos) {
      return false;
    }
    // FIXME(20151009): ko rule.
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

  pub fn play(&mut self, stone: Stone, action: Action, work: &mut FastBoardWork, aux: &mut Option<&mut FastBoardAux>, verbose: bool) -> PlayResult {
    // FIXME(20151008): ko point logic is seriously messed up.
    //println!("DEBUG: play: {:?} {:?}", stone, action);
    //assert!(!self.in_txn);
    let mut result = PlayResult::Okay;
    work.reset();
    self.turn = stone;
    self.last_clob = false;
    self.last_caps.clear();
    self.last_suics.clear();
    //self.last_toucs.clear();
    aux.as_mut().map(|aux| {
      aux.last_adj_chs.clear();
      aux.last_cap_chs.clear();
    });

    match action {
      Action::Resign | Action::Pass => {
        self.last_pos = None;
      }
      Action::Place{pos} => {
        self.place_stone(stone, pos, aux);
        let opp_stone = stone.opponent();
        let mut num_captured = 0;
        Self::for_each_adjacent(pos, |adj_pos| {
          if let PlayResult::FatalError = result {
            return;
          }
          if self.stones[adj_pos.idx()] != Stone::Empty {
            if verbose {
              println!("DEBUG: adjacent to {}: {}", pos.to_coord().to_string(), adj_pos.to_coord().to_string());
            }
            let adj_head = self.traverse_chain(adj_pos);
            //assert!(adj_head != TOMBSTONE);
            if adj_head == TOMBSTONE {
              println!("WARNING: play: adj head is TOMBSTONE!");
              result = PlayResult::FatalError;
              return;
            }
            //self.last_toucs.extend(self.get_chain(adj_head).ps_libs.iter());
            aux.as_mut().map(|aux| {
              aux.last_adj_chs.push(adj_head);
            });
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
        if let PlayResult::FatalError = result {
          println!("WARNING: failed to play due to error!");
          return result;
        }
        if num_captured > 1 {
          self.ko_pos = None;
        }
        self.merge_chains_and_append(&work.merge_adj_heads, pos, aux, verbose);
        aux.as_mut().map(|aux| {
          for &head in aux.ch_heads.iter() {
            if self.stones[head.idx()] == stone &&
                self.get_chain(head).count_pseudoliberties() == 0
            {
              self.last_suics.push(head);
            }
          }
        });
        if let Some(ko_pos) = self.ko_pos {
          // TODO(20151005): check if not actually a kos pos.
        }
        self.last_pos = Some(pos);
        // XXX(20151019): Update mask for network evaluation.
        self.mask.as_mut_slice()[pos.idx()] = 1.0;
      }
    }
    self.last_turn = Some(stone);
    {
      // XXX(20151019): Update mask for network evaluation.
      let mut mask = self.mask.as_mut_slice();
      for &cap_pos in self.last_caps.iter() {
        mask[cap_pos.idx()] = 0.0;
      }
    }
    self.turn = stone.opponent();

    result
  }

  fn place_stone(&mut self, stone: Stone, pos: Pos, aux: &mut Option<&mut FastBoardAux>) {
    assert!(stone != Stone::Empty);
    let i = pos.idx();
    let prev_stone = self.stones[i];
    if prev_stone != Stone::Empty {
      self.last_clob = true;
    }
    self.stones[i] = stone;
    {
      let mut features = self.features.as_mut_slice();
      match stone {
        Stone::Black => {
          features[BLACK_PLANE + i] = 1;
          features[WHITE_PLANE + i] = 0;
        }
        Stone::White => {
          features[BLACK_PLANE + i] = 0;
          features[WHITE_PLANE + i] = 1;
        }
        _ => unreachable!(),
      }
      match prev_stone {
        Stone::Black => {
          features[PREV_BLACK_PLANE + i] = 1;
          features[PREV_WHITE_PLANE + i] = 0;
        }
        Stone::White => {
          features[PREV_BLACK_PLANE + i] = 0;
          features[PREV_WHITE_PLANE + i] = 1;
        }
        Stone::Empty => {
          features[PREV_BLACK_PLANE + i] = 0;
          features[PREV_WHITE_PLANE + i] = 0;
        }
      }
    }
    aux.as_mut().map(|aux| {
      aux.st_first.set(i, false);
      aux.st_empty.set(i, false);
    });
  }

  fn capture_stone(&mut self, pos: Pos, aux: &mut Option<&mut FastBoardAux>) {
    let i = pos.idx();
    let prev_stone = self.stones[i];
    self.stones[i] = Stone::Empty;
    self.last_caps.push(pos);
    {
      let mut features = self.features.as_mut_slice();
      features[BLACK_PLANE + i] = 0;
      features[WHITE_PLANE + i] = 0;
      match prev_stone {
        Stone::Black => {
          features[PREV_BLACK_PLANE + i] = 1;
          features[PREV_WHITE_PLANE + i] = 0;
        }
        Stone::White => {
          features[PREV_BLACK_PLANE + i] = 0;
          features[PREV_WHITE_PLANE + i] = 1;
        }
        Stone::Empty => {
          features[PREV_BLACK_PLANE + i] = 0;
          features[PREV_WHITE_PLANE + i] = 0;
        }
      }
    }
    aux.as_mut().map(|aux| {
      aux.st_empty.set(i, true);
    });
  }

  /*fn update_chains(&mut self, stone: Stone, pos: Pos, work: &mut FastBoardWork, aux: &mut Option<FastBoardAux>) {
  }*/

  fn get_chain(&self, head: Pos) -> &Chain {
    &**self.chains[head.idx()].as_ref()
      .unwrap()
      //.expect(&format!("get_chain expected a chain: {}", head))
  }

  fn get_chain_mut(&mut self, head: Pos) -> &mut Chain {
    &mut **self.chains[head.idx()].as_mut()
      .unwrap()
      //.expect(&format!("get_chain_mut expected a chain: {}", head))
  }

  fn take_chain(&mut self, head: Pos) -> Box<Chain> {
    self.chains[head.idx()].take()
      .unwrap()
      //.expect(&format!("take_chain expected a chain: {}", head))
  }

  fn traverse_chain(&mut self, mut pos: Pos) -> Pos {
    if self.ch_roots[pos.idx()] == TOMBSTONE {
      return TOMBSTONE;
    }
    while pos != self.ch_roots[pos.idx()] {
      match self.ch_roots[self.ch_roots[pos.idx()].idx()] {
        TOMBSTONE => {
          return TOMBSTONE;
        }
        root => {
          self.ch_roots[pos.idx()] = root;
          pos = root;
        }
      }
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

  fn union_chains(&mut self, pos1: Pos, pos2: Pos, aux: &mut Option<&mut FastBoardAux>) -> Pos {
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
      aux.as_mut().map(|aux| {
        aux.ch_heads.remove(&old_head);
      });
      self.get_chain_mut(new_head).ps_libs.extend(&old_chain.ps_libs);
      new_head
    }
  }

  fn create_chain(&mut self, head: Pos, aux: &mut Option<&mut FastBoardAux>, verbose: bool) {
    if verbose {
      println!("DEBUG: creating a chain: {}", head.to_coord().to_string());
    }
    self.ch_roots[head.idx()] = head;
    self.ch_links[head.idx()] = head;
    self.ch_sizes[head.idx()] = 1;
    self.chains[head.idx()] = Some(Box::new(Chain::new()));
    aux.as_mut().map(|aux| {
      aux.ch_heads.insert(head);
    });
    Self::for_each_adjacent(head, |adj_pos| {
      let adj_stone = self.stones[adj_pos.idx()];
      if let Stone::Empty = adj_stone {
        self.get_chain_mut(head).add_pseudoliberty(adj_pos);
      /*} else if self.stones[head.idx()].opponent() == adj_stone {
        self.get_chain_mut(head).add_pseudoenemy(adj_pos);*/
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
      let adj_stone = self.stones[adj_pos.idx()];
      if let Stone::Empty = adj_stone {
        self.get_chain_mut(head).add_pseudoliberty(adj_pos);
      /*} else if self.stones[pos.idx()].opponent() == adj_stone {
        self.get_chain_mut(head).add_pseudoenemy(adj_pos);*/
      }
    });
    if verbose {
      let ps_lib_coords: Vec<_> =
          self.get_chain(head).ps_libs.iter()
            .map(|lib| lib.to_coord().to_string())
            .collect();
      println!("DEBUG: chain psuedoliberties: {:?}",
          //pos.to_coord().to_string(), head.to_coord().to_string()
          ps_lib_coords
      );
    }
  }

  fn merge_chains_and_append(&mut self, heads: &[Pos], pos: Pos, aux: &mut Option<&mut FastBoardAux>, verbose: bool) {
    let mut prev_head = None;
    for i in (0 .. heads.len()) {
      match prev_head {
        None    => prev_head = Some(heads[i]),
        Some(r) => prev_head = Some(self.union_chains(r, heads[i], aux)),
      }
    }
    // TODO(20151022): update features with atari.
    match prev_head {
      None    => {
        self.create_chain(pos, aux, verbose);
        // XXX: Update atari features for newly created chain (consists of only
        // a single stone).
        /*{
          let num_libs = self.get_chain(pos).approx_count_liberties();
          let (atari_feat, live_feat) = if num_libs == 1 {
            (1.0, 0.0)
          } else if num_libs >= 2 {
            (0.0, 1.0)
          } else {
            unreachable!();
          };
          let mut features = self.features.as_mut_slice();
          let i = pos.idx();
          match stone {
            Stone::Black => {
              features[BLACK_ATARI_PLANE + i] = atari_feat;
              features[BLACK_LIVE_PLANE + i] = live_feat;
            }
            Stone::White => {
              features[WHITE_ATARI_PLANE + i] = atari_feat;
              features[WHITE_LIVE_PLANE + i] = live_feat;
            }
            Stone::Empty => unreachable!(),
          }
        }*/
        // XXX: Update atari features for adjacent enemies.
        // TODO(20151022)
      }
      Some(r) => {
        // TODO: update chain at prev_head.
        self.add_to_chain(r, pos, verbose);
        // TODO: update chain at r.
      }
    }
  }

  fn kill_chain(&mut self, head: Pos, aux: &mut Option<&mut FastBoardAux>) -> usize {
    let mut num_captured = 0;
    let mut prev;
    let mut next = head;
    let stone = self.stones[head.idx()];
    let opp_stone = stone.opponent();
    loop {
      prev = next;
      next = self.ch_links[prev.idx()];
      self.capture_stone(next, aux);
      Self::for_each_adjacent(next, |adj_pos| {
        if self.stones[adj_pos.idx()] == opp_stone {
          let adj_head = self.traverse_chain(adj_pos);
          // XXX: Need to check for tombstone here. The killed chain is adjacent
          // to the freshly placed stone, which is currently not in a chain.
          if adj_head != TOMBSTONE {
            let mut adj_chain = self.get_chain_mut(adj_head);
            adj_chain.add_pseudoliberty(next);
            /*adj_chain.remove_pseudoenemy(next);*/
          }
        }
      });
      self.ch_roots[next.idx()] = TOMBSTONE;
      self.ch_links[prev.idx()] = TOMBSTONE;
      // XXX: Update atari features for killed chain's own stones.
      /*{
        let mut features = self.features.as_mut_slice();
        let next_i = next.idx();
        // FIXME(20151022): convert to absolute features.
        if stone == self.turn {
          features[GREEN_ATARI_PLANE + next_i] = 0.0;
          features[GREEN_NON_ATARI_PLANE + next_i] = 0.0;
        } else if stone == self.turn.opponent() {
          features[RED_ATARI_PLANE + next_i] = 0.0;
          features[RED_NON_ATARI_PLANE + next_i] = 0.0;
        } else {
          unreachable!();
        }
      }*/
      num_captured += 1;
      if next == head {
        break;
      }
    }
    self.ch_sizes[head.idx()] = 0;
    let killed_chain = self.take_chain(head);
    // XXX: Update atari features for killed chain's adjacent enemy stones.
    /*{
      let mut features = self.features.as_mut_slice();
      for &opp_pos in killed_chain.ps_opps.iter() {
        let opp_i = opp_pos.idx();
        let stone = self.stones[opp_i];
        // FIXME(20151022): convert to absolute features.
        if stone == self.turn {
          features[GREEN_ATARI_PLANE + opp_i] = 0.0;
          features[GREEN_NON_ATARI_PLANE + opp_i] = 0.0;
        } else if stone == self.turn.opponent() {
          features[RED_ATARI_PLANE + opp_i] = 0.0;
          features[RED_NON_ATARI_PLANE + opp_i] = 0.0;
        } else {
          unreachable!();
        }
      }
    }*/
    aux.as_mut().map(|aux| {
      aux.ch_heads.remove(&head);
      aux.last_cap_chs.push(head);
    });
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
