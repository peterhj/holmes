use fastboard::{Action};
use fasttree::{SearchResult, Trajectory, FastSearchTree};
use policy::{
  SearchPolicy, RolloutPolicy,
};

use statistics_avx2::random::{StreamRng, XorShift128PlusStreamRng};

use bit_set::{BitSet};
use rand::{thread_rng};

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
          tree_policy.backup(tree, traj);
        }
      }
    }
    tree_policy.execute_best(tree.get_root())
  }
}

pub struct BatchSearchProblem {
  //stop_time:      Timespec,
  // FIXME(20151008): for now, this version of search starts over from scratch each time.
  pub rng:            XorShift128PlusStreamRng,
  pub tree:           FastSearchTree,
  pub trajs:          Vec<Trajectory>,
  //pub tree_policy:    SearchP,
  //pub rollout_policy: RolloutP,

  // Temporary work variables.
  pub terminal:       BitSet,
  pub rollout_term:   BitSet,
}

impl BatchSearchProblem {
  pub fn new(tree: FastSearchTree, batch_size: usize) -> BatchSearchProblem {
    //let batch_size = rollout_policy.batch_size();
    let mut trajs = Vec::new();
    for _ in (0 .. batch_size) {
      trajs.push(Trajectory::new());
    }
    BatchSearchProblem{
      //stop_time:      get_time() + Duration::seconds(10), // FIXME(20151006)
      rng:            XorShift128PlusStreamRng::with_rng_seed(&mut thread_rng()),
      tree:           tree,
      trajs:          trajs,
      //tree_policy:    search_policy,
      //rollout_policy: rollout_policy,

      terminal:       BitSet::with_capacity(batch_size),
      rollout_term:   BitSet::with_capacity(batch_size),
    }
  }

  pub fn join(&mut self, tree_policy: &mut SearchPolicy, rollout_policy: &mut RolloutPolicy) -> Action {
    let &mut BatchSearchProblem{
      ref mut rng,
      ref mut tree,
      ref mut trajs,
      //ref mut tree_policy,
      //ref mut rollout_policy,
      ref mut terminal,
      ref mut rollout_term,
      ..} = self;
    // FIXME(20151018): hard coding number of batches/rollouts for now.
    let batch_size = rollout_policy.batch_size();
    //let num_batches = 3072;
    let num_batches = 1;
    //let num_batches = 12;
    let max_steps = 200;
    for batch in (0 .. num_batches) {
      //println!("DEBUG: search: on batch: {}", batch);
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
      let mut steps_count = 0;
      loop {
        //println!("DEBUG: search:   on batch step: {}", steps_count);
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
        steps_count += 1;
        if !running || steps_count >= max_steps {
          break;
        }
      }
      //println!("DEBUG: search: ran batch steps: {}", steps_count);
      let mut num_success = 0;
      for batch_idx in (0 .. batch_size) {
        // FIXME(20151011): do terminal nodes result in a backup?
        if !terminal.contains(&batch_idx) {
          trajs[batch_idx].end_rollout();
          num_success += tree_policy.backup(tree, &trajs[batch_idx]);
        }
      }
      println!("DEBUG: search: batch {}: {}/{} success backups", batch, num_success, batch_size);
    }
    println!("DEBUG: search: ran {} batches of {} steps", num_batches, max_steps);
    tree_policy.execute_best(tree.get_root())
  }
}
