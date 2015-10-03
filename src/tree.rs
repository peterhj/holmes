use contains::hash_map::{OpenCopyHashMap};
use fastboard::{Action, FastBoard};
use policy::{TreePolicy, PlayoutPolicy};

#[derive(Clone, Copy)]
pub struct QValueEdge<E> where E: Clone + Copy + Default {
  pub next_id:  usize,
  pub action:   Action,
  pub q_value:  f32,
  pub n_plays:  i32,
  pub n_wins:   i32,
  pub data:     E,
}

impl<E> QValueEdge<E> where E: Clone + Copy + Default {
  pub fn new(next_id: usize, action: Action, init_q: f32) -> QValueEdge<E> {
    QValueEdge{
      next_id:  next_id,
      action:   action,
      q_value:  init_q,
      n_plays:  0,
      n_wins:   0,
      data:     Default::default(),
    }
  }
}

pub struct QValueNode<D, E> where D: Default, E: Clone + Copy + Default {
  pub id:       usize,
  pub prev_id:  Option<usize>,
  pub state:    FastBoard,
  pub depth:    i32,
  pub n_plays:  i32,
  pub data:     D,
  pub childs:   OpenCopyHashMap<usize, QValueEdge<E>>,
}

impl<D, E> QValueNode<D, E> where D: Default, E: Clone + Copy + Default{
  pub fn new(prev_id: Option<usize>, id: usize, state: FastBoard, depth: i32) -> QValueNode<D, E> {
    QValueNode{
      id:       id,
      prev_id:  prev_id,
      state:    state,
      depth:    depth,
      n_plays:  0,
      data:     Default::default(),
      childs:   OpenCopyHashMap::with_capacity(1),
    }
  }
}

pub struct QValueTree<P> where P: TreePolicy {
  pub nodes:    Vec<QValueNode<P::NodeData, P::EdgeData>>,
}

impl<P> QValueTree<P> where P: TreePolicy {
  pub fn new() -> QValueTree<P> {
    QValueTree{nodes: Vec::new()}
  }

  pub fn reset(&mut self, init_state: &FastBoard, init_q: f32) {
    self.nodes.clear();
    self.expand(None, init_state.clone(), init_q);
  }

  pub fn expand(&mut self, prev: Option<(usize, Action)>, next_state: FastBoard, init_q: f32) {
    let id = self.nodes.len();
    let (prev_id, next_depth) = if let Some((prev_id, action)) = prev {
      let prev_node = &mut self.nodes[prev_id];
      prev_node.childs.insert(id, QValueEdge::new(id, action, init_q));
      (Some(prev_id), prev_node.depth + 1)
    } else {
      (None, 0)
    };
    self.nodes.push(QValueNode::new(prev_id, id, next_state, next_depth));
  }

  pub fn simulate(&mut self, action: Action, id: usize, tree_policy: &P, playout_policy: &mut PlayoutPolicy) {
    let result = {
      let node = &self.nodes[id];
      playout_policy.execute_playout(&node.state, node.depth)
    };
    let mut next_id = None;
    let mut update_id = id;
    loop {
      tree_policy.update_tree_node(self, update_id, action, next_id, result);
      match self.nodes[update_id].prev_id {
        Some(prev_id) => {
          next_id = Some(update_id);
          update_id = prev_id;
        }
        None => break,
      }
    }
  }
}
