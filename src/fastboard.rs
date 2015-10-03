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
  Empty = 0,
  Black = 1,
  White = 2,
}

impl Stone {
  #[inline]
  pub fn opponent(&self) -> Stone {
    match *self {
      Stone::Empty => unreachable!(),
      Stone::Black => Stone::White,
      Stone::White => Stone::Black,
    }
  }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum Action {
  Resign,
  Pass,
  Place{pos: Pos},
}

#[derive(Clone, Copy)]
pub enum ActionStatus {
  Legal           = 0,
  IllegalOccupied = 0x01,
  IllegalSuicide  = 0x02,
  IllegalKo       = 0x04,
}

#[derive(Clone, Copy)]
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
}

pub struct FastBoardWork {
  merge_adj_roots: Vec<Pos>,
}

impl FastBoardWork {
  pub fn new() -> FastBoardWork {
    FastBoardWork{
      merge_adj_roots: Vec::with_capacity(4),
    }
  }

  pub fn reset(&mut self) {
    self.merge_adj_roots.clear();
  }
}

#[derive(Clone, Debug)]
pub struct FastBoard {
  stones:   Vec<Stone>,
  ch_roots: Vec<Pos>,
  ch_links: Vec<Pos>,
  ch_sizes: Vec<u16>,
  chains:   Vec<Option<Box<Chain>>>,
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

  pub fn new() -> FastBoard {
    FastBoard{
      stones:   Vec::with_capacity(Self::BOARD_SIZE),
      ch_roots: Vec::with_capacity(Self::BOARD_SIZE),
      ch_links: Vec::with_capacity(Self::BOARD_SIZE),
      ch_sizes: Vec::with_capacity(Self::BOARD_SIZE),
      chains:   Vec::with_capacity(Self::BOARD_SIZE),
    }
  }

  pub fn reset(&mut self) {
    self.stones.clear();
    self.ch_roots.clear();
    self.ch_links.clear();
    self.ch_sizes.clear();
    self.chains.clear();
    self.stones.extend(repeat(Stone::Empty).take(Self::BOARD_SIZE));
    self.ch_roots.extend(repeat(TOMBSTONE).take(Self::BOARD_SIZE));
    self.ch_links.extend(repeat(TOMBSTONE).take(Self::BOARD_SIZE));
    self.ch_sizes.extend(repeat(0).take(Self::BOARD_SIZE));
    self.chains.extend(repeat(None).take(Self::BOARD_SIZE));
  }

  pub fn play(&mut self, stone: Stone, action: Action, work: &mut FastBoardWork) {
    work.reset();
    match action {
      Action::Resign => {}
      Action::Pass => {}
      Action::Place{pos} => {
        // TODO(20151003)
        self.place_stone(stone, pos);
        self.update_chains(stone, pos, work);
      }
    }
    // TODO
  }

  fn place_stone(&mut self, stone: Stone, pos: Pos) {
    self.stones[pos.idx()] = stone;
  }

  fn get_chain(&self, root: Pos) -> &Chain {
    &**self.chains[root.idx()].as_ref().unwrap()
      //.expect(&format!("get_chain expected a chain: {}", root))
  }

  fn get_chain_mut(&mut self, root: Pos) -> &mut Chain {
    &mut **self.chains[root.idx()].as_mut().unwrap()
      //.expect(&format!("get_chain_mut expected a chain: {}", root))
  }

  fn take_chain(&mut self, root: Pos) -> Box<Chain> {
    self.chains[root.idx()].take().unwrap()
      //.expect(&format!("take_chain expected a chain: {}", root))
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

  fn union_chains(&mut self, root1: Pos, root2: Pos) -> Pos {
    let root1 = self.traverse_chain(root1);
    let root2 = self.traverse_chain(root2);
    if root1 == root2 {
      root1
    } else if self.ch_sizes[root1.idx()] < self.ch_sizes[root2.idx()] {
      self.ch_roots[root1.idx()] = root2;
      let prev_tail = self.ch_links[root2.idx()];
      let new_tail = self.ch_links[root1.idx()];
      self.ch_links[root1.idx()] = prev_tail;
      self.ch_links[root2.idx()] = new_tail;
      self.ch_sizes[root2.idx()] += self.ch_sizes[root1.idx()];
      self.ch_sizes[root1.idx()] = 0;
      let old_chain = self.take_chain(root1);
      self.get_chain_mut(root2).ps_libs.extend(&old_chain.ps_libs);
      root2
    } else {
      self.ch_roots[root2.idx()] = root1;
      let prev_tail = self.ch_links[root1.idx()];
      let new_tail = self.ch_links[root2.idx()];
      self.ch_links[root2.idx()] = prev_tail;
      self.ch_links[root1.idx()] = new_tail;
      self.ch_sizes[root1.idx()] += self.ch_sizes[root2.idx()];
      self.ch_sizes[root2.idx()] = 0;
      let old_chain = self.take_chain(root2);
      self.get_chain_mut(root1).ps_libs.extend(&old_chain.ps_libs);
      root1
    }
  }

  fn update_chains(&mut self, stone: Stone, pos: Pos, work: &mut FastBoardWork) {
    let opp_stone = stone.opponent();
    Self::for_each_adjacent(pos, |adj_pos| {
      let adj_root = self.traverse_chain(adj_pos);
      if adj_root != TOMBSTONE {
        self.get_chain_mut(adj_root).remove_pseudoliberty(pos);
        if self.stones[adj_root.idx()] == opp_stone {
          if self.get_chain(adj_root).count_pseudoliberties() == 0 {
            self.kill_chain(adj_root);
          }
        } else {
          work.merge_adj_roots.push(adj_root);
        }
      }
    });
    self.merge_chains_and_append(&work.merge_adj_roots, pos);
  }

  fn create_chain(&mut self, root: Pos) {
    self.ch_roots[root.idx()] = root;
    self.ch_links[root.idx()] = root;
    self.ch_sizes[root.idx()] = 1;
    self.chains[root.idx()] = Some(Box::new(Chain::new()));
    Self::for_each_adjacent(root, |adj_pos| {
      if let Stone::Empty = self.stones[adj_pos.idx()] {
        self.get_chain_mut(root).add_pseudoliberty(adj_pos);
      }
    });
  }

  fn add_to_chain(&mut self, root: Pos, pos: Pos) {
    self.ch_roots[pos.idx()] = root;
    //assert_eq!(root, self.traverse_chain(pos));
    let prev_tail = self.ch_links[root.idx()];
    self.ch_links[pos.idx()] = prev_tail;
    self.ch_links[root.idx()] = pos;
    self.ch_sizes[root.idx()] += 1;
    Self::for_each_adjacent(pos, |adj_pos| {
      if let Stone::Empty = self.stones[adj_pos.idx()] {
        self.get_chain_mut(root).add_pseudoliberty(adj_pos);
      }
    });
  }

  fn merge_chains_and_append(&mut self, roots: &[Pos], pos: Pos) {
    let mut prev_root = None;
    for i in (1 .. roots.len()) {
      match prev_root {
        None    => prev_root = Some(self.union_chains(roots[i-1], roots[i])),
        Some(r) => prev_root = Some(self.union_chains(r, roots[i])),
      }
    }
    match prev_root {
      None    => self.create_chain(pos),
      Some(r) => self.add_to_chain(r, pos),
    }
  }

  fn kill_chain(&mut self, root: Pos) {
    let mut next = root;
    loop {
      let prev = next;
      next = self.ch_links[prev.idx()];
      self.stones[next.idx()] = Stone::Empty;
      self.ch_roots[next.idx()] = TOMBSTONE;
      self.ch_links[prev.idx()] = TOMBSTONE;
      if next == root {
        break;
      }
    }
    self.ch_sizes[root.idx()] = 0;
    self.take_chain(root);
  }

  fn set_ko(&mut self) {
  }

  fn unset_ko(&mut self) {
  }
}
