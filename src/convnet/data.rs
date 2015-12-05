use random::{XorShift128PlusRng};

use array::{ArrayDeserialize, Array3d};
use byteorder::{ReadBytesExt, BigEndian, LittleEndian};
use episodb::{EpisoDb};
use rembrandt::data::{DataSourceConfig, DataSource, SampleDatum, SampleLabel, SampleLabelConfig};

use rand::{Rng, SeedableRng, thread_rng};
use rand::distributions::{IndependentSample};
use rand::distributions::range::{Range};
use std::io::{Read, BufReader, Cursor};

pub struct RandomEpisodeSamples {
  rng:  XorShift128PlusRng,
  data: Box<DataSource>,
}

impl RandomEpisodeSamples {
  pub fn new(data: Box<DataSource>) -> RandomEpisodeSamples {
    RandomEpisodeSamples{
      rng:  XorShift128PlusRng::from_seed([thread_rng().gen(), thread_rng().gen()]),
      data: data,
    }
  }
}

impl DataSource for RandomEpisodeSamples {
  fn len(&self) -> usize {
    self.num_samples()
  }

  fn num_samples(&self) -> usize {
    self.data.num_samples()
  }

  fn num_episodes(&self) -> usize {
    self.data.num_episodes()
  }

  fn each_sample(&mut self, label_cfg: SampleLabelConfig, f: &mut FnMut(usize, &SampleDatum, Option<SampleLabel>)) {
    let mut epoch_idx = 0;
    let ep_idx_range = Range::new(0, self.num_episodes());
    for _ in (0 .. self.num_samples()) {
      let ep_idx = ep_idx_range.ind_sample(&mut self.rng);
      let (start_idx, end_idx) = self.data.get_episode_range(ep_idx);
      let sample_idx = self.rng.gen_range(start_idx, end_idx);
      if let Some((datum, maybe_label)) = self.data.get_episode_sample(label_cfg, ep_idx, sample_idx) {
        f(epoch_idx, &datum, maybe_label);
        epoch_idx += 1;
      }
    }
  }
}

pub struct CyclicRandomEpisodeSamples {
  rng:  XorShift128PlusRng,
  data: Box<DataSource>,
}

impl CyclicRandomEpisodeSamples {
  pub fn new(data: Box<DataSource>) -> CyclicRandomEpisodeSamples {
    CyclicRandomEpisodeSamples{
      rng:  XorShift128PlusRng::from_seed([thread_rng().gen(), thread_rng().gen()]),
      data: data,
    }
  }
}

impl DataSource for CyclicRandomEpisodeSamples {
  fn len(&self) -> usize {
    self.num_samples()
  }

  fn num_samples(&self) -> usize {
    self.data.num_samples()
  }

  fn num_episodes(&self) -> usize {
    self.data.num_episodes()
  }

  fn each_sample(&mut self, label_cfg: SampleLabelConfig, f: &mut FnMut(usize, &SampleDatum, Option<SampleLabel>)) {
    println!("DEBUG: cyclic: begin epoch: {} {}", self.num_episodes(), self.num_samples());
    let mut epoch_idx = 0;
    let mut num_skipped = 0;
    let num_eps = self.num_episodes();
    for i in (0 .. self.num_samples()) {
      let ep_idx = i % num_eps;
      let (start_idx, end_idx) = self.data.get_episode_range(ep_idx);
      let sample_idx = self.rng.gen_range(start_idx, end_idx);
      if let Some((datum, maybe_label)) = self.data.get_episode_sample(label_cfg, ep_idx, sample_idx) {
        f(epoch_idx, &datum, maybe_label);
        epoch_idx += 1;
      } else {
        num_skipped += 1;
      }
    }
    println!("DEBUG: cyclic: end epoch: {} {}", epoch_idx, num_skipped);
  }
}
