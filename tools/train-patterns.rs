extern crate holmes;
extern crate rembrandt;

#[macro_use]
extern crate log;
extern crate env_logger;
extern crate rand;
//extern crate scoped_threadpool;
extern crate time;
extern crate vec_map;

use holmes::array_util::{array_argmax};
use holmes::board::{Board, Point};
use holmes::data::{SymmetryAugment};
use holmes::pattern::{LibPattern3x3, InvariantLibPattern3x3};
use holmes::txnstate::{for_each_x8};
use rembrandt::data_new::{
  SampleDatumConfig, SampleLabelConfig,
  DatasetConfig,
  DataIterator, SampleIterator, RandomEpisodeIterator,
  AugmentDataSource, PartitionDataSource,
  SampleDatum, SampleLabel,
};

use rand::{Rng, thread_rng};
use std::cmp::{max};
use std::collections::{HashMap};
use std::path::{PathBuf};
use std::sync::{Arc};
use time::{get_time};
use vec_map::{VecMap};

fn main() {
  let datum_cfg = SampleDatumConfig::Bits3d{scale: 255};
  let label_cfg = SampleLabelConfig::Category{
    num_categories: 361,
  };
  let dataset_cfg = DatasetConfig::open(&PathBuf::from("datasets/gogodb_w2015-preproc-alphapatternnokov3m_19x19x7.data"));
  let mut train_data =
      RandomEpisodeIterator::new(
          //dataset_cfg.build_with_cfg(datum_cfg, label_cfg, "train"),
          Box::new(AugmentDataSource::new(SymmetryAugment::new(&mut thread_rng()), dataset_cfg.build_with_cfg(datum_cfg, label_cfg, "train"))),
      );
  let mut valid_data =
      SampleIterator::new(
          //Box::new(PartitionDataSource::new(tid, num_workers, dataset_cfg.build_with_cfg(datum_cfg, label_cfg, "valid")))
          dataset_cfg.build_with_cfg(datum_cfg, label_cfg, "valid")
      );

  let step_size = 0.001;
  //let epoch_size = train_data.max_num_samples();

  let mut prev_move_gamma: f32 = 0.0;
  let mut all_gammas: HashMap<LibPattern3x3, f32> = HashMap::with_capacity(32768);
  let mut invariant_gammas: HashMap<InvariantLibPattern3x3, f32> = HashMap::with_capacity(32768);

  let mut matched_prev_adj = VecMap::with_capacity(361);
  let mut matched_pats = VecMap::with_capacity(361);
  //let mut matched_factors = VecMap::with_capacity(361);

  loop {
    let mut start_time = get_time();

    let mut interval_counter = 0;
    let mut interval_acc = 0;
    let mut interval_loss = 0.0;

    let mut epoch = 0;
    let mut local_idx = 0;

    train_data.each_sample(datum_cfg, label_cfg, &mut |epoch_idx, datum, maybe_label| {
      let repr_bytes = match datum {
        &SampleDatum::WHCBytes(ref repr_bytes) => {
          repr_bytes.as_slice()
        }
      };
      let category = match (label_cfg, maybe_label) {
        (SampleLabelConfig::Category{num_categories}, Some(&SampleLabel::Category{category})) => {
          assert!(category >= 0);
          assert!(category < num_categories);
          category
        }
        _ => panic!("unsupported label"),
      };

      let target_p = category as usize;

      matched_prev_adj.clear();
      matched_pats.clear();
      //matched_factors.clear();
      let mut prev_move_p = None;
      for prev_p in 0 .. Board::SIZE {
        if repr_bytes[prev_p + 6 * Board::SIZE] != 0 {
          prev_move_p = Some(prev_p);
          break;
        }
      }
      //assert!(prev_move_p.is_some());
      if let Some(prev_move_p) = prev_move_p {
        for_each_x8(Point::from_idx(prev_move_p), |_, adj_pt| {
          matched_prev_adj.insert(adj_pt.idx(), true);
        });
      }
      for p in 0 .. Board::SIZE {
        if repr_bytes[p] == 0 {
          continue;
        }
        let pat = LibPattern3x3::from_repr_bytes(repr_bytes, p);
        let invariant_pat = pat.to_invariant();
        if !all_gammas.contains_key(&pat) {
          all_gammas.insert(pat, 0.0);
        }
        if !invariant_gammas.contains_key(&invariant_pat) {
          invariant_gammas.insert(invariant_pat, 0.0);
        }
        //matched_pats.insert(p, pat);
        matched_pats.insert(p, invariant_pat);
      }

      let mut max_gamma = Some(prev_move_gamma);
      for (p, pat) in matched_pats.iter() {
        //let p_gamma = all_gammas[pat];
        let p_gamma = invariant_gammas[pat];
        match max_gamma {
          None => max_gamma = Some(p_gamma),
          Some(g) => max_gamma = Some(g.max(p_gamma)),
        }
      }

      let mut sum_factors = 0.0;
      sum_factors += (prev_move_gamma - max_gamma.unwrap()).exp();
      for (p, pat) in matched_pats.iter() {
        //let p_gamma = all_gammas[pat];
        let p_gamma = invariant_gammas[pat];
        let p_stab_gamma = p_gamma - max_gamma.unwrap();
        let p_factor = p_stab_gamma.exp();
        //matched_factors.insert(p, p_factor);
        sum_factors += p_factor;
      }

      let mut out_indexes = vec![];
      let mut out_probs = vec![];
      for (p, pat) in matched_pats.iter() {
        //let p_gamma = all_gammas[pat];
        let p_gamma = invariant_gammas[pat];
        let p_stab_gamma = p_gamma - max_gamma.unwrap();
        let p_factor = if matched_prev_adj.contains_key(p) {
          p_stab_gamma.exp() + prev_move_gamma.exp()
        } else {
          p_stab_gamma.exp()
        };
        let p_prob = p_factor / sum_factors;
        out_indexes.push(p);
        out_probs.push(p_prob);
        let gradient = if p == target_p {
          interval_loss += -(p_prob.ln());
          p_prob - 1.0
        } else {
          p_prob
        };
        if matched_prev_adj.contains_key(p) {
          prev_move_gamma -= step_size * gradient;
        }
        //*all_gammas.get_mut(&pat).unwrap() = p_gamma - step_size * gradient;
        *invariant_gammas.get_mut(&pat).unwrap() = p_gamma - step_size * gradient;
      }
      //println!("DEBUG: train: probs: {:?}", probs);
      let argmax_rank = array_argmax(&out_probs).unwrap();
      let argmax_p = out_indexes[argmax_rank];
      if argmax_p == target_p {
        interval_acc += 1;
      }

      interval_counter += 1;
      local_idx += 1;
      if local_idx % 10000 == 0 {
        let lap_time = get_time();
        let elapsed_ms = (lap_time - start_time).num_milliseconds();
        interval_loss /= interval_counter as f32;
        println!("DEBUG: train: epoch: {}, samples: {}, patterns found: {}/{} acc: {:.3} loss: {:.6} elapsed: {:.3} s",
            epoch, local_idx,
            all_gammas.len(), invariant_gammas.len(),
            interval_acc as f32 / interval_counter as f32,
            interval_loss,
            elapsed_ms as f32 * 0.001,
        );
        start_time = lap_time;
        interval_counter = 0;
        interval_acc = 0;
        interval_loss = 0.0;
      }
    });
    epoch += 1;
    //break; // FIXME(20160411)
  }
}
