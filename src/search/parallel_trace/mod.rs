use array_util::{array_argmax};
use board::{Board, Action};
use search::{translate_score_to_reward};
use search::parallel_policies::{
  SearchPolicyWorker,
  PriorPolicy, DiffPriorPolicy,
};
use search::parallel_tree::{TreePolicyConfig, Node};
use txnstate::{TxnState};
use txnstate::extras::{TxnStateNodeData};
use worker::{WorkerSharedData};

//use std::cell::{RefCell};
//use std::rc::{Rc};
use std::sync::{Arc, Barrier, Mutex, RwLock};
use std::sync::atomic::{AtomicUsize, Ordering, fence};
use vec_map::{VecMap};

pub mod omega;

pub struct SearchTrace {
  pub tree_cfg:     Option<TreePolicyConfig>,
  pub root_state:   Option<TxnState<TxnStateNodeData>>,
  pub root_trace:   Option<TreeExpansionTrace>,
  pub batches:      Vec<SearchTraceBatch>,
}

impl SearchTrace {
  pub fn new() -> SearchTrace {
    SearchTrace{
      tree_cfg:     None, //tree_cfg,
      root_state:   None,
      root_trace:   None,
      batches:      vec![],
    }
  }

  //pub fn reset(&mut self, tree_cfg: TreePolicyConfig, root_state: TxnState<TxnStateNodeData>) {
  pub fn reset(&mut self, tree_cfg: TreePolicyConfig, root_node: &Node) {
    self.tree_cfg = Some(tree_cfg);
    self.root_state = Some(root_node.state.clone());
    self.root_trace = Some(TreeExpansionTrace::new(root_node));
    /*self.root_trace = Some(TreeExpansionTrace{
      valid_actions:  root_node.valid_moves.iter().map(|&pt| Action::Place{point: pt}).collect(),
      prior_values:   root_node.values.prior_values.clone(),
    });*/
    self.batches.clear();
  }

  pub fn start_batch(&mut self, batch_size: usize) {
    let mut traj_traces = Vec::with_capacity(batch_size);
    for _ in 0 .. batch_size {
      traj_traces.push(SearchTrajTrace{
        tree_trace:      TreeTrajTrace{
          decisions:    vec![],
          expansion:    None,
        },
        rollout_trace:   RolloutTrajTrace{
          actions:      vec![],
          score:        None,
        },
      });
    }
    self.batches.push(SearchTraceBatch{
      batch_size:   batch_size,
      traj_traces:  traj_traces,
    });
  }
}

#[derive(Clone)]
pub struct SearchTraceBatch {
  pub batch_size:       usize,
  pub traj_traces:      Vec<SearchTrajTrace>,
}

#[derive(Clone)]
pub struct SearchTrajTrace {
  pub tree_trace:       TreeTrajTrace,
  pub rollout_trace:    RolloutTrajTrace,
}

#[derive(Clone)]
pub struct TreeTrajTrace {
  pub decisions:    Vec<TreeDecisionTrace>,
  pub expansion:    Option<TreeExpansionTrace>,
}

#[derive(Clone)]
pub struct TreeDecisionTrace {
  pub action:       Action,
  pub action_rank:  usize,
  pub horizon:      usize,
}

impl TreeDecisionTrace {
  pub fn new(action: Action, action_rank: usize, horizon: usize) -> TreeDecisionTrace{
    TreeDecisionTrace{
      action:       action,
      action_rank:  action_rank,
      horizon:      horizon,
    }
  }
}

#[derive(Clone)]
pub struct TreeExpansionTrace {
  pub valid_actions:    Vec<Action>,
  pub prior_values:     Vec<f32>,
}

impl TreeExpansionTrace {
  pub fn new(node: &Node) -> TreeExpansionTrace {
    TreeExpansionTrace{
      valid_actions:  node.valid_moves.iter().map(|&pt| Action::Place{point: pt}).collect(),
      prior_values:   node.values.prior_values.clone(),
    }
  }
}

#[derive(Clone)]
pub struct RolloutTrajTrace {
  pub actions:      Vec<Action>,
  pub score:        Option<f32>,
}

#[derive(Clone)]
pub struct ReconSharedTree {
  pub root_node:    Arc<Mutex<Option<Arc<RwLock<ReconNode>>>>>,
}

impl ReconSharedTree {
  pub fn new() -> ReconSharedTree {
    ReconSharedTree{
      root_node:    Arc::new(Mutex::new(None)),
    }
  }

  pub fn try_clear(&self) {
    let mut root_node = self.root_node.lock().unwrap();
    if root_node.is_some() {
      *root_node = None;
    }
  }

  pub fn try_reset(&self, root_state: TxnState<TxnStateNodeData>, /*prior_policy: &mut PriorPolicy*/) {
    let mut root_node = self.root_node.lock().unwrap();
    if root_node.is_none() {
      *root_node = Some(Arc::new(RwLock::new(ReconNode::new(root_state, /*prior_policy*/))));
    }
  }
}

pub struct ReconNode {
  pub state:            TxnState<TxnStateNodeData>,
  initialized:          bool,
  pub valid_actions:    Vec<Action>,
  pub prior_values:     Vec<f32>,
  pub child_nodes:      VecMap<Arc<RwLock<ReconNode>>>,
  pub visit_count:      AtomicUsize,
  pub succ_counts:      Vec<AtomicUsize>,
  pub trial_counts:     Vec<AtomicUsize>,
}

impl ReconNode {
  pub fn new(state: TxnState<TxnStateNodeData>, /*prior_policy: &mut PriorPolicy*/) -> ReconNode {
    ReconNode{
      state:            state,
      initialized:      false,
      valid_actions:    Vec::with_capacity(Board::SIZE),
      prior_values:     Vec::with_capacity(Board::SIZE),
      child_nodes:      VecMap::with_capacity(Board::SIZE),
      visit_count:      AtomicUsize::new(0),
      succ_counts:      Vec::with_capacity(Board::SIZE),
      trial_counts:     Vec::with_capacity(Board::SIZE),
    }
  }

  pub fn get_value(&self) -> f32 {
    let trial_counts: Vec<f32> = self.trial_counts.iter().map(|x| x.load(Ordering::Acquire) as f32).collect();
    if let Some(j) = array_argmax(&trial_counts) {
      let n_j = trial_counts[j];
      if n_j > 0.0 {
        let s_j = self.succ_counts[j].load(Ordering::Acquire) as f32;
        s_j / n_j
      } else {
        0.5
      }
    } else {
      0.5
    }
  }

  pub fn initialize(&mut self, valid_actions: &Vec<Action>, prior_values: &Vec<f32>) {
    if self.initialized {
      return;
    }
    self.initialized = true;
    self.valid_actions.clone_from(valid_actions);
    self.prior_values = prior_values.clone();
    let num_actions = self.valid_actions.len();
    self.succ_counts.reserve(num_actions);
    for _ in 0 .. num_actions {
      self.succ_counts.push(AtomicUsize::new(0));
    }
    self.trial_counts.reserve(num_actions);
    for _ in 0 .. num_actions {
      self.trial_counts.push(AtomicUsize::new(0));
    }
  }

  pub fn backup(&self, action: Action, score: f32) {
    self.visit_count.fetch_add(1, Ordering::AcqRel);
    if let Action::Place{point} = action {
      let p = point.idx();
      self.trial_counts[p].fetch_add(1, Ordering::AcqRel);
      match translate_score_to_reward(self.state.current_turn(), score) {
        0 => {}
        1 => {
          self.succ_counts[p].fetch_add(1, Ordering::AcqRel);
        }
        _ => { unreachable!(); }
      }
    }
  }
}

pub trait TreeBatchWorker {
  //fn policy_worker(&mut self) -> &SearchPolicyWorker;
  //fn prior_policy(&mut self) -> &mut DiffPriorPolicy;

  //fn with_prior_policy<V>(&mut self, mut f: &mut FnMut(&mut PriorPolicy) -> V) -> V;
  fn with_search_worker<F, V>(&mut self, mut f: F) -> V where F: FnOnce(&mut SearchPolicyWorker) -> V;
  fn with_prior_policy<F, V>(&mut self, mut f: F) -> V where F: FnOnce(&mut PriorPolicy) -> V;
  fn start_instance(&mut self, tree_cfg: TreePolicyConfig);
  fn update_batch(&mut self, root_node: Arc<RwLock<ReconNode>>, trace_batch: &SearchTraceBatch);
  fn update_instance(&mut self, root_node: Arc<RwLock<ReconNode>>);
}

pub trait RolloutTrajWorker {
  fn update_traj(&mut self, init_node: Arc<RwLock<ReconNode>>, traj_trace: &SearchTrajTrace);
}

impl RolloutTrajWorker for () {
  fn update_traj(&mut self, _init_node: Arc<RwLock<ReconNode>>, _traj_trace: &SearchTrajTrace) {
  }
}

#[derive(Clone)]
pub struct ParallelSearchReconWorkerBuilder {
  worker_data:  WorkerSharedData,
  shared_tree:  ReconSharedTree,
}

impl ParallelSearchReconWorkerBuilder {
  pub fn new(num_workers: usize) -> ParallelSearchReconWorkerBuilder {
    ParallelSearchReconWorkerBuilder{
      worker_data:  WorkerSharedData::new(num_workers),
      shared_tree:  ReconSharedTree::new(),
    }
  }

  pub fn into_worker<TreeWork, RollWork>(
      self,
      tid: usize,
      tree_batch_worker:    TreeWork,
      rollout_traj_worker:  RollWork,
  ) -> ParallelSearchReconWorker<TreeWork, RollWork>
  where TreeWork: TreeBatchWorker,
        RollWork: RolloutTrajWorker,
  {
    ParallelSearchReconWorker{
      tid:  tid,
      worker_data:  self.worker_data,
      shared_tree:  self.shared_tree,
      tree_batch_worker:    tree_batch_worker,
      rollout_traj_worker:  rollout_traj_worker,
    }
  }
}

pub struct ParallelSearchReconWorker<TreeWork, RollWork>
where TreeWork: TreeBatchWorker,
      RollWork: RolloutTrajWorker,
{
  tid:          usize,
  worker_data:  WorkerSharedData,
  pub shared_tree:  ReconSharedTree,
  tree_batch_worker:    TreeWork,
  rollout_traj_worker:  RollWork,
}

impl<TreeWork, RollWork> ParallelSearchReconWorker<TreeWork, RollWork>
where TreeWork: TreeBatchWorker,
      RollWork: RolloutTrajWorker,
{
  pub fn with_tree_worker<F, V>(&mut self, f: F) -> V
  where F: FnOnce(&mut TreeWork) -> V {
    f(&mut self.tree_batch_worker)
  }

  pub fn reconstruct_trace(&mut self, search_trace: &SearchTrace) {
    assert!(search_trace.root_state.is_some());
    /*tree_batch_worker.with_prior_policy(|prior_policy| {
      self.shared_tree.try_reset(search_trace.root_state.as_ref().unwrap().clone(), prior_policy);
    });*/
    self.shared_tree.try_clear();
    self.worker_data.sync();
    self.shared_tree.try_reset(search_trace.root_state.as_ref().unwrap().clone());
    self.worker_data.sync();

    self.tree_batch_worker.start_instance(search_trace.tree_cfg.as_ref().unwrap().clone());

    let root_node = {
      let root_node = self.shared_tree.root_node.lock().unwrap();
      root_node.as_ref().unwrap().clone()
    };

    let num_batches = search_trace.batches.len();
    for batch in 0 .. num_batches {
      let trace_batch = &search_trace.batches[batch];

      // Populate all the tree nodes for this batch.
      let mut leaf_nodes = Vec::with_capacity(trace_batch.batch_size);
      for batch_idx in 0 .. trace_batch.batch_size {
        let traj_trace = &trace_batch.traj_traces[batch_idx];
        let mut node = root_node.clone();
        let mut terminated = false;
        for decision in traj_trace.tree_trace.decisions.iter() {
          assert!(!terminated);
          let action_idx = decision.action.idx();
          let (turn, visit_count, has_child) = {
            let node = node.read().unwrap();
            (
              node.state.current_turn(),
              node.visit_count.load(Ordering::Acquire),
              node.child_nodes.contains_key(&action_idx),
            )
          };
          if visit_count < search_trace.tree_cfg.unwrap().visit_thresh {
            terminated = true;
          }
          let next_node = if has_child {
            node.read().unwrap().child_nodes[action_idx].clone()
          } else {
            let mut new_state = node.read().unwrap().state.clone();
            if new_state.try_action(turn, decision.action).is_err() {
              panic!("reconstruct_trace: unexpected illegal action: {:?}", decision.action);
            } else {
              new_state.commit();
            }
            /*let new_node = tree_batch_worker.with_prior_policy(move |prior_policy| {
              Arc::new(RwLock::new(ReconNode::new(new_state, /*prior_policy*/)))
            });*/
            let new_node = Arc::new(RwLock::new(ReconNode::new(new_state)));
            let mut node = node.write().unwrap();
            if node.child_nodes.contains_key(&action_idx) {
              node.child_nodes[action_idx].clone()
            } else {
              node.child_nodes.insert(action_idx, new_node.clone());
              new_node
            }
          };
          node = next_node;
        }
        leaf_nodes.push(node);
      }
      for batch_idx in 0 .. trace_batch.batch_size {
        let traj_trace = &trace_batch.traj_traces[batch_idx];
        if let Some(ref expansion) = traj_trace.tree_trace.expansion {
          let mut leaf_node = &mut leaf_nodes[batch_idx];
          let mut leaf_node = leaf_node.write().unwrap();
          leaf_node.initialize(&expansion.valid_actions, &expansion.prior_values);
        }
      }
      self.worker_data.sync();

      // Update the tree batch worker.
      self.tree_batch_worker.update_batch(root_node.clone(), trace_batch);
      self.worker_data.sync();

      // Update the rollout trajectory worker.
      for batch_idx in 0 .. trace_batch.batch_size {
        let init_node = leaf_nodes[batch_idx].clone();
        let traj_trace = &trace_batch.traj_traces[batch_idx];
        self.rollout_traj_worker.update_traj(init_node, traj_trace);
      }

      // Backup the tree nodes.
      for batch_idx in 0 .. trace_batch.batch_size {
        let traj_trace = &trace_batch.traj_traces[batch_idx];
        let tree_trace = &traj_trace.tree_trace;
        let rollout_trace = &traj_trace.rollout_trace;
        assert!(rollout_trace.score.is_some());
        let score = rollout_trace.score.unwrap();

        if rollout_trace.actions.len() >= 1 {
          let first_action = rollout_trace.actions[0];
          leaf_nodes[batch_idx].read().unwrap().backup(first_action, score);
        }

        let mut node = root_node.clone();
        for decision in tree_trace.decisions.iter() {
          let action_idx = decision.action.idx();
          let next_node = {
            let node = node.read().unwrap();
            node.backup(decision.action, score);
            node.child_nodes[action_idx].clone()
          };
          node = next_node;
        }
      }
      self.worker_data.sync();
    }

    self.tree_batch_worker.update_instance(root_node.clone());
    self.worker_data.sync();
  }
}
