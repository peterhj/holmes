use board::{RuleSet, PlayerRank, Coord, Stone, Point, Action};
use sgf::{Sgf, parse_raw_sgf};
use txnstate::{TxnStateConfig, TxnState};
use txnstate::extras::{TxnStateNodeData};

use array_new::{Shape};
use rembrandt::data_new::{Augment, SampleDatum, SampleLabel};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, thread_rng};
use rand::distributions::{IndependentSample};
use rand::distributions::range::{Range};
use std::fs::{File};
use std::io::{Read, BufRead, BufReader};
use std::path::{PathBuf};

pub struct SymmetryAugment {
  range:    Range<u8>,
  rng:      Xorshiftplus128Rng,
}

impl SymmetryAugment {
  pub fn new<R: Rng>(seed_rng: &mut R) -> SymmetryAugment {
    SymmetryAugment{
      range:    Range::new(0, 8),
      rng:      Xorshiftplus128Rng::new(seed_rng),
    }
  }
}

impl Augment for SymmetryAugment {
  fn transform(&mut self, datum: SampleDatum, maybe_label: Option<SampleLabel>) -> (SampleDatum, Option<SampleLabel>) {
    let rot = self.range.ind_sample(&mut self.rng);
    let mut new_datum = datum.clone();
    let mut new_label = maybe_label.clone();
    match (datum, &mut new_datum) {
      (SampleDatum::WHCBytes(ref datum), &mut SampleDatum::WHCBytes(ref mut new_datum)) => {
        let bound = datum.bound();
        let stride = datum.stride();
        {
          let old_data = datum.as_slice();
          let mut new_data = new_datum.as_mut_slice();
          //for (i, j, k) in bound.major_iter() {
          for k in 0 .. bound.2 {
            for j in 0 .. bound.1 {
              for i in 0 .. bound.0 {
                let c = Coord{x: i as u8, y: j as u8};
                let nc = c.rotate(rot);
                let (ni, nj) = (nc.x as usize, nc.y as usize);
                /*let old_idx = (i, j, k);
                let new_idx = (ni, nj, k);
                new_data[new_idx.offset(stride)] = old_data[old_idx.offset(stride)];*/
                new_data[ni + nj * stride.0 + k * stride.0 * stride.1] = old_data[i + j * stride.0 + k * stride.0 * stride.1];
              }
            }
          }
        }
      }
    }
    if let Some(label) = maybe_label {
      match label {
        SampleLabel::Category{category} => {
          let c = Coord::from_idx(category as usize);
          let nc = c.rotate(rot);
          new_label = Some(SampleLabel::Category{category: nc.idx() as i32});
        }
        SampleLabel::MultiCategory{ref categories} => {
          let lookahead = categories.len();
          let mut new_categories = Vec::with_capacity(lookahead);
          for k in 0 .. lookahead {
            let c = Coord::from_idx(categories[k] as usize);
            let nc = c.rotate(rot);
            new_categories.push(nc.idx() as i32);
          }
          new_label = Some(SampleLabel::MultiCategory{categories: new_categories});
        }
      }
    }
    (new_datum, new_label)
  }
}

pub struct Episode {
  history:  Vec<(TxnState<TxnStateNodeData>, Stone, Action)>,
  outcome:  Stone,
}

pub trait EpisodePreproc {
  fn try_load(&self, reader: &mut Read) -> Option<Episode>;
}

pub struct GogodbEpisodePreproc;

impl EpisodePreproc for GogodbEpisodePreproc {
  fn try_load(&self, reader: &mut Read) -> Option<Episode> {
    let mut text = vec![];
    reader.read_to_end(&mut text).unwrap();
    let raw_sgf = parse_raw_sgf(&text);
    let sgf = Sgf::from_raw(&raw_sgf);

    let outcome = {
      let result_toks: Vec<_> = sgf.result.split("+").collect();
      //println!("DEBUG: outcome tokens: {:?}", result_toks);
      if result_toks.is_empty() {
        None
      } else {
        let first_tok = result_toks[0].split_whitespace().next();
        match first_tok {
          Some("B") => {
            Some(Stone::Black)
          }
          Some("W") => {
            Some(Stone::White)
          }
          Some("?") | Some("Jigo") | Some("Left") | Some("Void") => {
            None
          }
          _ => {
            println!("WARNING: try_load: unimplemented outcome: \"{}\"", sgf.result);
            None
          }
        }
      }
    };
    if outcome.is_none() {
      return None;
    }

    let mut history = vec![];
    let mut state = TxnState::new(
        // FIXME(20160308): infer correct state cfg (namely, komi and ranks).
        TxnStateConfig::default(),
        TxnStateNodeData::new(),
        //TxnStateLibFeaturesData::new(),
    );
    state.reset();
    for (t, &(ref player, ref mov)) in sgf.moves.iter().enumerate() {
      let turn = match player as &str {
        "B" => Stone::Black,
        "W" => Stone::White,
        _ => unreachable!(),
      };
      // XXX(20160119): Set state turn the first time if necessary.
      if turn != state.current_turn() {
        if t == 0 {
          state.unsafe_set_current_turn(turn);
        } else {
          println!("WARNING: try_load: repeated move!");
          history.clear();
          break;
        }
      }
      let action = match mov as &str {
        "Pass"    => Action::Pass,
        "Resign"  => Action::Resign,
        x         => Action::Place{point: Point::from_coord(Coord::from_code_str(x))},
      };
      history.push((state.clone(), turn, action));
      match state.try_action(turn, action) {
        Ok(_) => {
          state.commit();
        }
        Err(_) => {
          println!("WARNING: try_load: found an illegal action!");
          history.clear();
          break;
        }
      }
    }
    if history.is_empty() {
      return None;
    }

    Some(Episode{
      history:  history,
      outcome:  outcome.unwrap(),
    })
  }
}

pub struct LazyEpisodeLoader<P> where P: EpisodePreproc {
  rng:      Xorshiftplus128Rng,
  paths:    Vec<PathBuf>,
  preproc:  P,
}

impl<P> LazyEpisodeLoader<P> where P: EpisodePreproc {
  pub fn new(index_path: PathBuf, preproc: P) -> LazyEpisodeLoader<P> {
    let index_file = BufReader::new(File::open(&index_path).unwrap());
    let mut index_lines = Vec::new();
    for line in index_file.lines() {
      let line = line.unwrap();
      index_lines.push(line);
    }
    let sgf_paths: Vec<_> = index_lines.iter()
      .map(|line| PathBuf::from(line)).collect();

    let rng = Xorshiftplus128Rng::new(&mut thread_rng());
    LazyEpisodeLoader{
      rng:      rng,
      paths:    sgf_paths,
      preproc:  preproc,
    }

    /*for sgf_path in sgf_paths.iter() {
      //let mut sgf_file = match File::open(sgf_path) {
    }*/
  }

  pub fn for_each_episode<F>(&mut self, mut f: F)
  where F: FnMut(&Episode) {
    loop {
      let idx = self.rng.gen_range(0, self.paths.len());
      let sgf_path = self.paths[idx].clone();
      let mut sgf_file = match File::open(&sgf_path) {
        Ok(file) => file,
        Err(_) => {
          println!("WARNING: failed to open sgf file: {:?}", &sgf_path);
          continue;
        }
      };
      match self.preproc.try_load(&mut sgf_file) {
        Some(episode) => {
          f(&episode);
        }
        None => {
          println!("WARNING: failed to load: {:?}", &sgf_path);
        }
      }
    }
  }

  pub fn for_each_random_pair<F>(&mut self, mut f: F)
  where F: FnMut(&(TxnState<TxnStateNodeData>, Stone, Action), &(TxnState<TxnStateNodeData>, Stone, Action)) {
    loop {
      let idx = self.rng.gen_range(0, self.paths.len());
      let sgf_path = self.paths[idx].clone();
      let mut sgf_file = match File::open(&sgf_path) {
        Ok(file) => file,
        Err(_) => {
          println!("WARNING: failed to open sgf file: {:?}", &sgf_path);
          continue;
        }
      };
      match self.preproc.try_load(&mut sgf_file) {
        Some(episode) => {
          if episode.history.len() < 2 {
            println!("WARNING: episode is too short: {:?}", &sgf_path);
            continue;
          }
          let k = self.rng.gen_range(0, episode.history.len() - 1);
          f(&episode.history[k], &episode.history[k+1]);
        }
        None => {
          println!("WARNING: failed to load: {:?}", &sgf_path);
          continue;
        }
      }
    }
  }
}

pub struct EpisodeFrame<'a> {
  pub state:    &'a TxnState<TxnStateNodeData>,
  pub turn:     Stone,
  pub action:   Action,
  pub value:    f32,
  pub outcome:  Stone,
}

pub struct EpisodeIter {
  rng:      Xorshiftplus128Rng,
  episodes: Vec<Episode>,
}

impl EpisodeIter {
  pub fn new<P>(index_path: PathBuf, preproc: P) -> EpisodeIter
  where P: EpisodePreproc {
    let index_file = BufReader::new(File::open(&index_path).unwrap());
    let mut index_lines = Vec::new();
    for line in index_file.lines() {
      let line = line.unwrap();
      index_lines.push(line);
    }
    let sgf_paths: Vec<_> = index_lines.iter()
      .map(|line| PathBuf::from(line)).collect();

    let mut episodes = vec![];
    for sgf_path in sgf_paths.iter() {
      let mut sgf_file = match File::open(sgf_path) {
        Ok(file) => file,
        Err(_) => {
          println!("WARNING: failed to open sgf file: '{:?}'", sgf_path);
          continue;
        }
      };
      match preproc.try_load(&mut sgf_file) {
        Some(episode) => {
          episodes.push(episode)
        }
        None => {}
      }
    }
    println!("DEBUG: loaded {} episodes", episodes.len());

    EpisodeIter{
      rng:      Xorshiftplus128Rng::new(&mut thread_rng()),
      episodes: episodes,
    }
  }

  pub fn num_episodes(&self) -> usize {
    self.episodes.len()
  }

  pub fn for_each_random_sample<F>(&mut self, mut f: F)
  where F: FnMut(usize, EpisodeFrame) {
    let num_episodes = self.episodes.len();
    for idx in 0 .. num_episodes {
      let i = self.rng.gen_range(0, num_episodes);
      let episode = &self.episodes[i];
      let episode_len = episode.history.len();
      let t = self.rng.gen_range(0, episode_len);
      let sample = &episode.history[t];
      let value = match (sample.0.current_turn(), episode.outcome) {
        (Stone::Black, Stone::Black) | (Stone::White, Stone::White) => 1.0,
        (Stone::Black, Stone::White) | (Stone::White, Stone::Black) => 0.0,
        _ => unreachable!(),
      };
      let outcome = episode.outcome;
      let frame = EpisodeFrame{
        state:  &sample.0,
        turn:   sample.1,
        action: sample.2,
        value:  value,
        outcome:    outcome,
      };
      f(idx, frame);
    }
  }

  pub fn cyclic_frame_at<F>(&mut self, t: usize, mut f: F)
  where F: FnMut(usize, EpisodeFrame) {
    let num_episodes = self.episodes.len();
    let mut epoch_idx = 0;
    for i in 0 .. num_episodes {
      let episode = &self.episodes[i];
      let episode_len = episode.history.len();
      if t >= episode_len {
        continue;
      }
      let sample = &episode.history[t];
      let value = match (sample.0.current_turn(), episode.outcome) {
        (Stone::Black, Stone::Black) | (Stone::White, Stone::White) => 1.0,
        (Stone::Black, Stone::White) | (Stone::White, Stone::Black) => 0.0,
        _ => unreachable!(),
      };
      let outcome = episode.outcome;
      let frame = EpisodeFrame{
        state:  &sample.0,
        turn:   sample.1,
        action: sample.2,
        value:  value,
        outcome:    outcome,
      };
      f(epoch_idx, frame);
      epoch_idx += 1;
    }
  }
}
