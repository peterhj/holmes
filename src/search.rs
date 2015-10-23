use fastboard::{Action};
use fasttree::{FastSearchTree, SearchResult, Trajectory};
use policy::{
  SearchPolicy, RolloutPolicy,
};

use statistics_avx2::random::{StreamRng, XorShift128PlusStreamRng};

use bit_set::{BitSet};

pub struct SearchProblem<SearchP, RolloutP> where SearchP: SearchPolicy, RolloutP: RolloutPolicy {
  //stop_time:      Timespec,
  // FIXME(20151008): for now, this version of search starts over from scratch each time.
  pub rng:            XorShift128PlusStreamRng,
  pub tree:           FastSearchTree,
  pub traj:           Trajectory,
  pub tree_policy:    SearchP,
  pub rollout_policy: RolloutP,
}

impl<SearchP, RolloutP> SearchProblem<SearchP, RolloutP> where SearchP: SearchPolicy, RolloutP: RolloutPolicy {
  pub fn join(&mut self) -> Action {
    let &mut SearchProblem{
      ref mut rng,
      ref mut tree,
      ref mut traj,
      ref mut tree_policy,
      ref mut rollout_policy,
      ..} = self;
    // FIXME(20151011): hard coding number of rollouts for now.
    for _ in (0 .. 10000) {
      match tree.walk(tree_policy, traj) {
        SearchResult::Terminal => {
          // FIXME(20151011): do terminal nodes result in a backup?
        }
        SearchResult::Leaf => {
          // FIXME(20151019): simulate using the new Trajectory.
          //tree.simulate(rollout_policy);
          //tree.backup(tree_policy, traj);

          tree.init_rollout(traj);
          loop {
            rollout_policy.preload_batch_state(0, traj.current_state());
            rollout_policy.execute_batch(1);
            if let Some(_) = traj.step_rollout(0, rollout_policy, rng) {
              // TODO(20151023): Do nothing?
            } else {
              break;
            }
          }
          traj.end_rollout();
          tree.backup(tree_policy, traj);
        }
      }
    }
    tree_policy.execute_best(tree.get_root())
  }
}

pub struct BatchSearchProblem<SearchP, RolloutP> where SearchP: SearchPolicy, RolloutP: RolloutPolicy {
  //stop_time:      Timespec,
  // FIXME(20151008): for now, this version of search starts over from scratch each time.
  pub rng:            XorShift128PlusStreamRng,
  pub tree:           FastSearchTree,
  pub trajs:          Vec<Trajectory>,
  pub tree_policy:    SearchP,
  pub rollout_policy: RolloutP,

  // Temporary work variables.
  pub terminal:       BitSet,
  pub rollout_term:   BitSet,
}

impl<SearchP, RolloutP> BatchSearchProblem<SearchP, RolloutP> where SearchP: SearchPolicy, RolloutP: RolloutPolicy {
  pub fn join(&mut self) -> Action {
    let &mut BatchSearchProblem{
      ref mut rng,
      ref mut tree,
      ref mut trajs,
      ref mut tree_policy,
      ref mut rollout_policy,
      ref mut terminal,
      ref mut rollout_term,
      ..} = self;
    // FIXME(20151018): hard coding number of batches/rollouts for now.
    let batch_size = 256;
    let num_batches = 12;
    for _ in (0 .. num_batches) {
      terminal.clear();
      for batch_idx in (0 .. batch_size) {
        match tree.walk(tree_policy, &mut trajs[batch_idx]) {
          SearchResult::Terminal => {
            terminal.insert(batch_idx);
          }
          SearchResult::Leaf => {
            tree.init_rollout(&mut trajs[batch_idx]);
          }
        }
      }
      loop {
        for batch_idx in (0 .. batch_size) {
          if !terminal.contains(&batch_idx) {
            rollout_policy.preload_batch_state(batch_idx, trajs[batch_idx].current_state());
          }
        }
        // FIXME(20151019): also check for rollout termination;
        // need to return cdf and sample outside of rollout execution.
        rollout_policy.execute_batch(batch_size);
        let mut running = false;
        for batch_idx in (0 .. batch_size) {
          if !terminal.contains(&batch_idx) {
            if let Some(_) = trajs[batch_idx].step_rollout(batch_idx, rollout_policy, rng) {
              running = true;
            } else {
              rollout_term.insert(batch_idx);
            }
            running = true;
          }
        }
        if !running {
          break;
        }
      }
      for batch_idx in (0 .. batch_size) {
        // FIXME(20151011): do terminal nodes result in a backup?
        if !terminal.contains(&batch_idx) {
          trajs[batch_idx].end_rollout();
          tree.backup(tree_policy, &trajs[batch_idx]);
        }
      }
    }
    tree_policy.execute_best(tree.get_root())
  }
}
