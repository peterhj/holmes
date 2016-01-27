extern crate holmes;

use holmes::data::{GogodbEpisodePreproc, EpisodeIter};
use holmes::pg::eval::{RolloutEvalOutcomeConfig, RolloutEvalOutcomeMachine};

use std::path::{PathBuf};

fn main() {
  let cfg = RolloutEvalOutcomeConfig{
    fixed_t:        100,
    num_rollouts:   512, // FIXME(20160126): unused.
  };
  let mut machine = RolloutEvalOutcomeMachine::new(cfg);
  let mut episodes = EpisodeIter::new(
      PathBuf::from("train_index_1024_a18.v2"),
      GogodbEpisodePreproc,
  );
  machine.estimate(&mut episodes);
}
