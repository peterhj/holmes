use board::{RuleSet, PlayerRank, Coord, Stone, Point, Action};
use sgf::{Sgf, parse_raw_sgf};
use txnstate::{TxnState};
use txnstate::extras::{TxnStateNodeData};

use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, thread_rng};
use std::fs::{File};
use std::io::{Read, BufRead, BufReader};
use std::path::{PathBuf};

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
        [PlayerRank::Dan(9), PlayerRank::Dan(9)],
        RuleSet::KgsJapanese.rules(),
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
