extern crate holmes;

use holmes::data::{GogodbEpisodePreproc, EpisodeIter};
use holmes::pg::balance::{PgBalanceConfig, PgBalanceMachine};

use std::path::{PathBuf};

fn main() {
  let cfg: PgBalanceConfig = Default::default();
  let mut machine = PgBalanceMachine::new(cfg);
  let mut episodes = EpisodeIter::new(
      PathBuf::from("train_index_1024_a18.v2"),
      GogodbEpisodePreproc,
  );
  machine.estimate(&mut episodes);
}
