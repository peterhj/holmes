use board::{Action};
use search::{translate_score_to_reward};
use search::parallel_policies::{DiffPriorPolicy};
use search::parallel_tree::{TreePolicyConfig};
use txnstate::{TxnState};
use txnstate::extras::{TxnStateNodeData};

use std::sync::{Arc, Barrier, Mutex, RwLock};
use std::sync::atomic::{AtomicUsize, Ordering};
use vec_map::{VecMap};

pub mod magic;

pub struct SearchTrace {
  pub tree_cfg:     TreePolicyConfig,
  pub root_state:   Option<TxnState<TxnStateNodeData>>,
  pub batches:      Vec<SearchTraceBatch>,
}

impl SearchTrace {
  pub fn new(tree_cfg: TreePolicyConfig) -> SearchTrace {
    SearchTrace{
      tree_cfg:     tree_cfg,
      root_state:   None,
      batches:      vec![],
    }
  }

  pub fn reset(&mut self, root_state: TxnState<TxnStateNodeData>) {
    //self.tree_cfg = tree_cfg;
    self.root_state = Some(root_state);
    self.batches.clear();
  }

  pub fn start_batch(&mut self, batch_size: usize) {
    let mut traj_traces = Vec::with_capacity(batch_size);
    for _ in 0 .. batch_size {
      traj_traces.push(SearchTrajTrace{
        explore_traj:   ExploreTrajTrace{
          decisions:    vec![],
          rollout:      false,
        },
        rollout_traj:   RolloutTrajTrace{
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
  pub batch_size:   usize,
  pub traj_traces:  Vec<SearchTrajTrace>,
}

#[derive(Clone)]
pub struct SearchTrajTrace {
  pub explore_traj: ExploreTrajTrace,
  pub rollout_traj: RolloutTrajTrace,
}

#[derive(Clone)]
pub struct ExploreTrajTrace {
  //pub actions:      Vec<Action>,
  pub decisions:    Vec<ExploreDecisionTrace>,
  pub rollout:      bool,
}

#[derive(Clone)]
pub struct ExploreDecisionTrace {
  pub action:       Action,
  pub action_rank:  usize,
  pub horizon:      usize,
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

  pub fn try_reset(&self, root_state: &TxnState<TxnStateNodeData>) {
    // FIXME(20160222)
    unimplemented!();
  }
}

pub struct ReconNode {
  pub state:            TxnState<TxnStateNodeData>,
  pub valid_actions:    Vec<Action>,
  pub prior_values:     Vec<f32>,
  pub child_nodes:      VecMap<Arc<RwLock<ReconNode>>>,
  pub visit_count:      AtomicUsize,
  pub succ_counts:      Vec<AtomicUsize>,
  pub trial_counts:     Vec<AtomicUsize>,
}

impl ReconNode {
  pub fn new(state: TxnState<TxnStateNodeData>, prior_policy: &mut DiffPriorPolicy) -> ReconNode {
    // FIXME(20160222)
    unimplemented!();
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

pub trait ExploreBatchWorker {
  fn prior_policy(&mut self) -> &mut DiffPriorPolicy;
  fn update_batch(&mut self, root_node: Arc<RwLock<ReconNode>>, trace_batch: &SearchTraceBatch);
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
  barrier:      Arc<Barrier>,
  shared_tree:  ReconSharedTree,
}

impl ParallelSearchReconWorkerBuilder {
  pub fn new(num_workers: usize) -> ParallelSearchReconWorkerBuilder {
    // FIXME(20160222)
    unimplemented!();
  }

  pub fn build_worker(self) -> ParallelSearchReconWorker {
    // FIXME(20160222)
    unimplemented!();
  }
}

pub struct ParallelSearchReconWorker {
  tid:          usize,
  num_workers:  usize,
  barrier:      Arc<Barrier>,
  shared_tree:  ReconSharedTree,
}

impl ParallelSearchReconWorker {
  pub fn reconstruct_trace<ExpWork, RolWork>(
      &mut self,
      search_trace: SearchTrace,
      mut explore_batch_worker: ExpWork,
      mut rollout_traj_worker:  RolWork,
      )
  where ExpWork: ExploreBatchWorker,
        RolWork: RolloutTrajWorker,
  {
    assert!(search_trace.root_state.is_some());
    self.shared_tree.try_reset(search_trace.root_state.as_ref().unwrap());
    self.barrier.wait();

    let root_node = {
      let root_node = self.shared_tree.root_node.lock().unwrap();
      root_node.as_ref().unwrap().clone()
    };

    let num_batches = search_trace.batches.len();
    for batch in 0 .. num_batches {
      let trace_batch = &search_trace.batches[batch];

      // Populate all the tree nodes for this batch.
      let mut explore_leaf_nodes = Vec::with_capacity(trace_batch.batch_size);
      for batch_idx in 0 .. trace_batch.batch_size {
        let traj_trace = &trace_batch.traj_traces[batch_idx];
        let mut node = root_node.clone();
        let mut terminated = false;
        for decision in traj_trace.explore_traj.decisions.iter() {
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
          if visit_count < search_trace.tree_cfg.visit_thresh {
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
            let new_node = Arc::new(RwLock::new(ReconNode::new(new_state, explore_batch_worker.prior_policy())));
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
        explore_leaf_nodes.push(node);
      }

      // Update the exploration batch worker.
      explore_batch_worker.update_batch(root_node.clone(), trace_batch);

      // Update the rollout trajectory worker.
      for batch_idx in 0 .. trace_batch.batch_size {
        let init_node = explore_leaf_nodes[batch_idx].clone();
        let traj_trace = &trace_batch.traj_traces[batch_idx];
        rollout_traj_worker.update_traj(init_node, traj_trace);
      }

      // Backup the tree nodes.
      for batch_idx in 0 .. trace_batch.batch_size {
        let traj_trace = &trace_batch.traj_traces[batch_idx];
        let explore_traj = &traj_trace.explore_traj;
        let rollout_traj = &traj_trace.rollout_traj;
        assert!(rollout_traj.score.is_some());
        let score = rollout_traj.score.unwrap();

        if rollout_traj.actions.len() >= 1 {
          let first_action = rollout_traj.actions[0];
          explore_leaf_nodes[batch_idx].read().unwrap().backup(first_action, score);
        }

        let mut node = root_node.clone();
        for decision in explore_traj.decisions.iter() {
          let action_idx = decision.action.idx();
          let next_node = {
            let node = node.read().unwrap();
            node.backup(decision.action, score);
            node.child_nodes[action_idx].clone()
          };
          node = next_node;
        }
      }
    }
  }
}
