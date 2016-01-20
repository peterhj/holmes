use board::{Action};
use txnstate::{TxnState};
use txnstate::extras::{TxnStateNodeData};

use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng};
use std::io::{Read};
use std::path::{PathBuf};

pub struct Episode {
  history:  Vec<(TxnState<TxnStateNodeData>, Action)>,
  value:    f32,
}

pub trait EpisodePreproc {
  fn try_load(reader: &mut Read) -> Option<Episode>;
}

pub struct GogodbEpisodePreproc;

impl EpisodePreproc for GogodbEpisodePreproc {
  fn try_load(reader: &mut Read) -> Option<Episode> {
    // TODO(20160119)
    unimplemented!();
  }
}

pub struct EpisodeIter {
  rng:      Xorshiftplus128Rng,
  episodes: Vec<Episode>,
}

impl EpisodeIter {
  pub fn new(sgf_list: PathBuf) -> EpisodeIter {
    // TODO(20160119)
    unimplemented!();
  }

  pub fn num_episodes(&self) -> usize {
    self.episodes.len()
  }

  pub fn for_each_random_sample<F>(&mut self, mut f: F)
  where F: FnMut(usize, &TxnState<TxnStateNodeData>, Action, f32) {
    let num_episodes = self.episodes.len();
    for idx in 0 .. num_episodes {
      let i = self.rng.gen_range(0, num_episodes);
      let episode = &self.episodes[i];
      let episode_len = episode.history.len();
      let t = self.rng.gen_range(0, episode_len);
      let sample = &episode.history[t];
      f(idx, &sample.0, sample.1, episode.value);
    }
  }
}
