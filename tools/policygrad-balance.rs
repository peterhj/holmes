extern crate holmes;

use holmes::data::{GogodbEpisodePreproc, EpisodeIter};
//use holmes::pg::balance::{PgBalanceConfig, PgBalanceMachine};
use holmes::pg::balance2::{PolicyGradBalanceConfig, PolicyGradBalanceMachine};

use std::path::{PathBuf};

fn main() {
  let cfg: PolicyGradBalanceConfig = Default::default();
  let mut machine = PolicyGradBalanceMachine::new(cfg);
  let mut episodes = EpisodeIter::new(
      PathBuf::from("train_index_1024_a18.v2"),
      GogodbEpisodePreproc,
  );
  machine.train(&mut episodes);
}
