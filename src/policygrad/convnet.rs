use board::{Board, Action, Stone, Point};
use convnet_new::{
  //build_2layer16_5x5_19x19x16_arch_nodir,
  //build_2layer16_9x9_19x19x16_arch_nodir,
  //build_3layer32_19x19x16_arch_nodir,
  build_13layer384_19x19x32_arch_nodir,
  //build_13layer384multi3_19x19x32_arch_nodir,
};
use discrete::{DiscreteFilter};
use discrete::bfilter::{BFilter};
use random::{choose_without_replace};
use search::{translate_score_to_reward};
use txnstate::{TxnState};
use txnstate::extras::{TxnStateNodeData, TxnStateRolloutLegalityData};

use array_cuda::device::{DeviceContext, for_all_devices};
use float::ord::{F32SupNan};
use rembrandt::arch_new::{
  Worker, ArchWorker,
  PipelineArchConfig, PipelineArchSharedData, PipelineArchWorker,
};
use rembrandt::data_new::{SampleLabel};
use rembrandt::layer_new::{Phase};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, thread_rng};
use rand::distributions::{IndependentSample};
use rand::distributions::range::{Range};
use std::iter::{repeat};
use std::path::{Path, PathBuf};
use std::rc::{Rc};
use std::sync::{Arc};

#[derive(Clone)]
pub struct ConvnetBiPolicyBuilder {
  batch_capacity:       usize,
  green_arch_cfg:       PipelineArchConfig,
  green_save_path:      PathBuf,
  green_shared:         Arc<PipelineArchSharedData>,
  green_shared2:        Arc<()>,
  red_arch_cfg:         PipelineArchConfig,
  red_save_path:        PathBuf,
  red_shared:           Arc<PipelineArchSharedData>,
  red_shared2:          Arc<()>,
}

impl ConvnetBiPolicyBuilder {
  pub fn new(num_workers: usize, worker_batch_size: usize) -> ConvnetBiPolicyBuilder {
    //let green_arch_cfg = build_2layer16_5x5_19x19x16_arch_nodir(worker_batch_size);
    //let green_save_path = PathBuf::from("models/gogodb_w2015-preproc-alphaminiv3m_19x19x16_2layer16-5x5.saved");

    //let arch_cfg = build_3layer32_19x19x16_arch_nodir(worker_batch_size);
    //let save_path = PathBuf::from("models/tmp_gogodb_w2015-preproc-alphaminiv3m_19x19x16_3layer32_pg");

    let arch_cfg = build_13layer384_19x19x32_arch_nodir(worker_batch_size);
    let save_path = PathBuf::from("models/tmp_gogodb_w2015_alphav3m_new_action_13layer384m3reshape_19x19x32_pg");

    let green_shared = for_all_devices(num_workers, |contexts| {
      Arc::new(PipelineArchSharedData::new(num_workers, &arch_cfg, contexts))
    });

    let red_shared = for_all_devices(num_workers, |contexts| {
      Arc::new(PipelineArchSharedData::new(num_workers, &arch_cfg, contexts))
    });

    ConvnetBiPolicyBuilder{
      batch_capacity:       worker_batch_size,
      green_arch_cfg:       arch_cfg.clone(),
      green_save_path:      save_path.clone(),
      green_shared:         green_shared,
      green_shared2:        Arc::new(()),
      red_arch_cfg:         arch_cfg,
      red_save_path:        save_path,
      red_shared:           red_shared,
      red_shared2:          Arc::new(()),
    }
  }

  pub fn into_policy(self, tid: usize) -> ConvnetBiPolicy {
    //let context = Rc::new(DeviceContext::new(tid));
    //let ctx = (*context).as_ref();
    let context = DeviceContext::new(tid);

    let (green_arch, red_arch) = {
      let ctx = context.as_ref();

      let mut green_arch = PipelineArchWorker::new(
          self.batch_capacity,
          self.green_arch_cfg,
          self.green_save_path,
          tid,
          [thread_rng().next_u64(), thread_rng().next_u64()],
          &self.green_shared,
          self.green_shared2.clone(),
          &ctx,
      );
      green_arch.load_layer_params(None, &ctx);
      green_arch.reset_gradients(0.0, &ctx);

      let mut red_arch = PipelineArchWorker::new(
          self.batch_capacity,
          self.red_arch_cfg,
          self.red_save_path,
          tid,
          [thread_rng().next_u64(), thread_rng().next_u64()],
          &self.red_shared,
          self.red_shared2.clone(),
          &ctx,
      );
      red_arch.load_layer_params(None, &ctx);

      (green_arch, red_arch)
    };

    ConvnetBiPolicy{
      context:      context,
      green_arch:   green_arch,
      red_arch:     red_arch,
    }
  }
}

#[derive(Clone)]
pub struct Traj {
  pub green_stone:  Option<Stone>,
  //pub traj_state:   TxnState<TxnStateRolloutLegalityData>,
  pub traj_state:   TxnState<TxnStateNodeData>,
  pub path_actions: Vec<(Stone, Action)>,
  pub score:        Option<f32>,
}

impl Traj {
  pub fn new() -> Traj {
    Traj{
      green_stone:  None,
      //traj_state:   TxnState::new(Default::default(), TxnStateRolloutLegalityData::new()),
      traj_state:   TxnState::new(Default::default(), TxnStateNodeData::new()),
      path_actions: vec![],
      score:        None,
    }
  }

  pub fn reset(&mut self) {
    self.green_stone = None;
    self.traj_state.reset();
    self.path_actions.clear();
    self.score = None;
  }
}

pub struct ConvnetBiPolicy {
  //context:      Rc<DeviceContext>,
  context:      DeviceContext,
  green_arch:   PipelineArchWorker<()>,
  red_arch:     PipelineArchWorker<()>,
}

impl ConvnetBiPolicy {
  pub fn rollout_batch(&mut self,
      batch_size:   usize,
      trajs:        &mut [Traj],
      rng:          &mut Xorshiftplus128Rng,
      ) -> usize
  {
    //let ctx = (*self.context).as_ref();
    let ctx = self.context.as_ref();
    assert!(batch_size <= trajs.len());

    for batch_idx in 0 .. batch_size {
      trajs[batch_idx].reset()
    }

    let range2 = Range::new(0, 2);

    // TODO(20160310)
    let mut green_start = Vec::with_capacity(batch_size);
    let mut terminated_count = 0;
    let mut terminated = Vec::with_capacity(batch_size);
    //let mut legal_masks = vec![];
    let mut filters = vec![];
    for batch_idx in 0 .. batch_size {
      // Choose some elements of the batch to start in the first evaluation.
      // By convention, green is evaluated first.
      match range2.ind_sample(rng) {
        0 => {
          trajs[batch_idx].green_stone = Some(Stone::White);
          green_start.push(false);
        }
        1 => {
          trajs[batch_idx].green_stone = Some(Stone::Black);
          green_start.push(true);
        }
        _ => unreachable!(),
      }
      terminated.push(false);
      //let mask_buf: Vec<_> = repeat(0).take(Board::SIZE).collect();
      //legal_masks.push(mask_buf);
      filters.push(BFilter::with_capacity(Board::SIZE));
    }

    let mut stats_first_term_t = 0;
    let mut stats_last_term_t = 0;
    let mut t = 0;
    loop {
      for batch_idx in 0 .. batch_size {
        if t == 0 && !green_start[batch_idx] || terminated[batch_idx] {
          continue;
        }
        let turn = trajs[batch_idx].traj_state.current_turn();
        trajs[batch_idx].traj_state.get_data().features
          .extract_relative_features(
              turn, self.green_arch.input_layer().expose_host_frame_buf(batch_idx));
        trajs[batch_idx].traj_state.get_data().legality
          .extract_legal_mask(
              turn, self.green_arch.loss_layer().expose_host_mask_buf(batch_idx));
      }

      self.green_arch.input_layer().load_frames(batch_size, &ctx);
      self.green_arch.loss_layer().load_masks(batch_size, &ctx);
      self.green_arch.forward(batch_size, Phase::Inference, &ctx);
      self.green_arch.loss_layer().apply_masks(batch_size, &ctx);
      self.green_arch.loss_layer().store_probs(batch_size, &ctx);

      {
        let batch_probs = self.green_arch.loss_layer().get_probs(batch_size).as_slice();
        for batch_idx in 0 .. batch_size {
          if t == 0 && !green_start[batch_idx] || terminated[batch_idx] {
            continue;
          }
          filters[batch_idx].reset(&batch_probs[batch_idx * Board::SIZE .. (batch_idx + 1) * Board::SIZE]);
        }
      }

      for batch_idx in 0 .. batch_size {
        if t == 0 && !green_start[batch_idx] || terminated[batch_idx] {
          continue;
        }
        let turn = trajs[batch_idx].traj_state.current_turn();
        let action = if let Some(p) = filters[batch_idx].sample(rng) {
          let point = Point::from_idx(p);
          Action::Place{point: point}
        } else {
          Action::Pass
        };
        if trajs[batch_idx].traj_state.try_action(turn, action).is_err() {
          panic!("illegal action during green trajectory");
        } else {
          trajs[batch_idx].traj_state.commit();
        }
        trajs[batch_idx].path_actions.push((turn, action));
        let path_len = trajs[batch_idx].path_actions.len();
        if path_len >= 2 {
          if  trajs[batch_idx].path_actions[path_len - 2].1 == Action::Pass &&
              trajs[batch_idx].path_actions[path_len - 1].1 == Action::Pass
          {
            if terminated_count == 0 {
              stats_first_term_t = t;
            } else if terminated_count == batch_size - 1 {
              stats_last_term_t = t;
            }
            terminated_count += 1;
            terminated[batch_idx] = true;
          }
        }
      }

      t += 1;
      if terminated_count >= batch_size {
        break;
      }

      for batch_idx in 0 .. batch_size {
        if terminated[batch_idx] {
          continue;
        }
        let turn = trajs[batch_idx].traj_state.current_turn();
        trajs[batch_idx].traj_state.get_data().features
          .extract_relative_features(
              turn, self.red_arch.input_layer().expose_host_frame_buf(batch_idx));
        trajs[batch_idx].traj_state.get_data().legality
          .extract_legal_mask(
              turn, self.red_arch.loss_layer().expose_host_mask_buf(batch_idx));
      }

      self.red_arch.input_layer().load_frames(batch_size, &ctx);
      self.red_arch.loss_layer().load_masks(batch_size, &ctx);
      self.red_arch.forward(batch_size, Phase::Inference, &ctx);
      self.red_arch.loss_layer().apply_masks(batch_size, &ctx);
      self.red_arch.loss_layer().store_probs(batch_size, &ctx);

      {
        let batch_probs = self.red_arch.loss_layer().get_probs(batch_size).as_slice();
        for batch_idx in 0 .. batch_size {
          if terminated[batch_idx] {
            continue;
          }
          filters[batch_idx].reset(&batch_probs[batch_idx * Board::SIZE .. (batch_idx + 1) * Board::SIZE]);
        }
      }

      for batch_idx in 0 .. batch_size {
        if terminated[batch_idx] {
          continue;
        }
        let turn = trajs[batch_idx].traj_state.current_turn();
        let action = if let Some(p) = filters[batch_idx].sample(rng) {
          let point = Point::from_idx(p);
          if trajs[batch_idx].traj_state.try_place(turn, point).is_err() {
            panic!("illegal action during red trajectory");
          } else {
            trajs[batch_idx].traj_state.commit();
          }
          Action::Place{point: point}
        } else {
          Action::Pass
        };
        trajs[batch_idx].path_actions.push((turn, action));
        let path_len = trajs[batch_idx].path_actions.len();
        if path_len >= 2 {
          if  trajs[batch_idx].path_actions[path_len - 2].1 == Action::Pass &&
              trajs[batch_idx].path_actions[path_len - 1].1 == Action::Pass
          {
            if terminated_count == 0 {
              stats_first_term_t = t;
            } else if terminated_count == batch_size - 1 {
              stats_last_term_t = t;
            }
            terminated_count += 1;
            terminated[batch_idx] = true;
          }
        }
      }

      t += 1;
      if terminated_count >= batch_size {
        break;
      }
    }

    /*println!("DEBUG: rollout batch stats: first t: {} last t: {}",
        stats_first_term_t, stats_last_term_t);*/

    let mut green_succ_count = 0;
    let mut scratch: Vec<_> = repeat(0).take(Board::SIZE).collect();
    for batch_idx in 0 .. batch_size {
      let score = trajs[batch_idx].traj_state.current_score_tromp_taylor_undead(7.5, &mut scratch);
      trajs[batch_idx].score = Some(score);
      let green_stone = trajs[batch_idx].green_stone.unwrap();
      match translate_score_to_reward(green_stone, score) {
        0 => {}
        1 => green_succ_count += 1,
        _ => unreachable!(),
      }
    }
    green_succ_count
  }

  pub fn update_params(&mut self, batch_size: usize, step_size: f32, baseline: f32, trajs: &mut [Traj]) {
    let ctx = self.context.as_ref();
    assert!(batch_size <= trajs.len());

    for batch_idx in 0 .. batch_size {
      let green_turn = trajs[batch_idx].green_stone.unwrap();
      let value = translate_score_to_reward(green_turn, trajs[batch_idx].score.unwrap()) as f32;
      let path_len = trajs[batch_idx].path_actions.len();
      trajs[batch_idx].traj_state.reset();
      let mut fill_idx = 0;
      for t in 0 .. path_len {
        let turn = trajs[batch_idx].path_actions[t].0;
        let action = trajs[batch_idx].path_actions[t].1;
        if trajs[batch_idx].traj_state.try_action(turn, action).is_err() {
          println!("illegal action during trajectory descent");
        } else {
          trajs[batch_idx].traj_state.commit();
        }
        if turn != green_turn {
          continue;
        }
        let p = match action {
          Action::Place{point} => point.idx(),
          _ => continue,
        };
        trajs[batch_idx].traj_state.get_data().features
          .extract_relative_features(
              turn, self.green_arch.input_layer().expose_host_frame_buf(fill_idx));
        trajs[batch_idx].traj_state.get_data().legality
          .extract_legal_mask(
              turn, self.green_arch.loss_layer().expose_host_mask_buf(fill_idx));
        self.green_arch.loss_layer().preload_label(fill_idx, &SampleLabel::Category{category: p as i32}, Phase::Training);
        self.green_arch.loss_layer().preload_weight(fill_idx, value - baseline);
        fill_idx += 1;
        let fill_size = self.green_arch.batch_capacity();
        if fill_idx % fill_size == 0 {
          self.green_arch.input_layer().load_frames(fill_size, &ctx);
          self.green_arch.loss_layer().load_masks(fill_size, &ctx);
          self.green_arch.loss_layer().load_labels(fill_size, &ctx);
          self.green_arch.loss_layer().load_weights(fill_size, &ctx);
          self.green_arch.forward(fill_size, Phase::Training, &ctx);
          self.green_arch.loss_layer().apply_masks(fill_size, &ctx);
          self.green_arch.backward(fill_size, 1.0, &ctx);
          fill_idx = 0;
        }
      }
      let fill_size = fill_idx;
      if fill_size > 0 {
        self.green_arch.input_layer().load_frames(fill_size, &ctx);
        self.green_arch.loss_layer().load_masks(fill_size, &ctx);
        self.green_arch.loss_layer().load_labels(fill_size, &ctx);
        self.green_arch.loss_layer().load_weights(fill_size, &ctx);
        self.green_arch.forward(fill_size, Phase::Training, &ctx);
        self.green_arch.loss_layer().apply_masks(fill_size, &ctx);
        self.green_arch.backward(fill_size, 1.0, &ctx);
      }
    }

    let num_workers = self.green_arch.num_workers();
    self.green_arch.dev_allreduce_sum_gradients(&ctx);
    self.green_arch.descend(step_size / (batch_size * num_workers) as f32, 0.0, &ctx);
    self.green_arch.reset_gradients(0.0, &ctx);
  }

  pub fn load_params_into_mem(&mut self, iter: usize) -> Result<Vec<u8>, ()> {
    let params_path = self.green_arch.params_path();
    self.red_arch.load_layer_params_into_mem(&params_path, Some(iter))
  }

  pub fn load_red_params(&mut self, blob: &[u8]) {
    let ctx = self.context.as_ref();
    self.red_arch.load_layer_params_from_mem(blob, &ctx);
  }

  pub fn save_params(&mut self, iter: usize) {
    let ctx = self.context.as_ref();
    self.green_arch.save_layer_params(iter, &ctx);
  }

  pub fn save_params_to_mem(&mut self) -> Vec<u8> {
    let ctx = self.context.as_ref();
    self.green_arch.save_layer_params_to_mem(&ctx)
  }
}
