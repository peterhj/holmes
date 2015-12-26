use board::{Stone, Point};
use txnstate::{TxnState};
use txnstate::extras::{TxnStateNodeData};

use std::cell::{RefCell};
use std::rc::{Rc};

pub struct MinimaxNode {
  state:        TxnState<TxnStateNodeData>,
  turn:         Stone,
  value:        f32,
  action:       Option<Point>,

  //prune_moves:  Vec<Point>,
  child_nodes:  Vec<(Point, Option<Rc<RefCell<MinimaxNode>>>)>,
}

impl MinimaxNode {
  pub fn new(state: &TxnState<TxnStateNodeData>) -> MinimaxNode {
    // TODO(20151226)
    MinimaxNode{
      state:        state.clone(),
      turn:         state.current_turn(),
      value:        0.5,
      action:       None,
      //prune_moves:  vec![],
      child_nodes:  vec![],
    }
  }
}

#[derive(Clone, Copy)]
pub struct MinimaxSearch {
  pub depth:    usize,
}

pub struct MinimaxSearchServer;

impl MinimaxSearchServer {
  pub fn search(&mut self, search: MinimaxSearch, root_state: &TxnState<TxnStateNodeData>) {
  }

  fn fail_soft_alpha_beta(&mut self, node: Rc<RefCell<MinimaxNode>>, min_bound: f32, max_bound: f32, depth: usize) -> f32 {
    if depth == 0 {
      // TODO(20151226): call parallel search evaluation.
      return 0.5;
    }
    let mut node = node.borrow_mut();
    match node.turn {
      Stone::Black => {
        let mut max_bound = max_bound;
        let mut min_value = 1.0;
        let mut best_action = None;
        for &mut (action, ref mut child_node) in node.child_nodes.iter_mut() {
          if child_node.is_none() {
            // TODO(20151226): instantiate node.
            *child_node = None;
          }
          let x = self.fail_soft_alpha_beta(child_node.as_ref().unwrap().clone(), min_bound, max_bound, depth - 1);
          if x < min_value {
            min_value = x;
            best_action = Some(action);
            if x < max_bound {
              max_bound = x;
            }
            if min_bound >= max_bound {
              break;
            }
          }
        }
        node.value = min_value;
        node.action = best_action;
        min_value
      }
      Stone::White => {
        let mut min_bound = min_bound;
        let mut max_value = 0.0;
        let mut best_action = None;
        for &mut (action, ref mut child_node) in node.child_nodes.iter_mut() {
          if child_node.is_none() {
            // TODO(20151226): instantiate node.
            *child_node = None;
          }
          let x = self.fail_soft_alpha_beta(child_node.as_ref().unwrap().clone(), min_bound, max_bound, depth - 1);
          if x > max_value {
            max_value = x;
            best_action = Some(action);
            if x > min_bound {
              min_bound = x;
            }
            if min_bound >= max_bound {
              break;
            }
          }
        }
        node.value = max_value;
        node.action = best_action;
        max_value
      }
      _ => unreachable!(),
    }
  }
}
