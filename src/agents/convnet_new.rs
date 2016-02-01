use agents::{Agent};
use board::{Board, RuleSet, PlayerRank, Stone, Point, Action};
use convnet_new::{
  build_12layer384_19x19x37_arch_nodir,
};
use txnstate::{TxnState};
use txnstate::features::{
  TxnStateAlphaFeatsV1Data,
};

use array_cuda::device::{DeviceContext, for_all_devices};
use float::ord::{F32SupNan};
use rembrandt::arch_new::{
  Worker, ArchWorker,
  PipelineArchConfig, PipelineArchSharedData, PipelineArchWorker,
};
//use rembrandt::data_new::{SampleLabel};
use rembrandt::layer_new::{Phase};

use rand::{Rng, thread_rng};
use std::path::{PathBuf};
use std::sync::{Arc};

pub struct ConvnetAgent {
  komi:     f32,
  player:   Option<Stone>,

  history:  Vec<(TxnState<TxnStateAlphaFeatsV1Data>, Action)>,
  state:    TxnState<TxnStateAlphaFeatsV1Data>,

  context:  DeviceContext,
  arch:     PipelineArchWorker<()>,
}

impl ConvnetAgent {
  pub fn new() -> ConvnetAgent {
    let context = DeviceContext::new(0);
    let arch_cfg = build_12layer384_19x19x37_arch_nodir(1);
    let save_path = PathBuf::from("models/kgs_ugo_201505_new_action_12layer384_19x19x37.v3.saved");
    let shared = for_all_devices(1, |contexts| {
      Arc::new(PipelineArchSharedData::new(1, &arch_cfg, contexts))
    });
    let mut arch = PipelineArchWorker::new(
        1,
        arch_cfg,
        save_path,
        0,
        [thread_rng().next_u64(), thread_rng().next_u64()],
        &shared,
        Arc::new(()),
        &context.as_ref(),
    );
    arch.load_layer_params(None, &context.as_ref());
    ConvnetAgent{
      komi:     0.0,
      player:   None,
      history:  vec![],
      state:    TxnState::new(
          [PlayerRank::Dan(9), PlayerRank::Dan(9)],
          RuleSet::KgsJapanese.rules(),
          TxnStateAlphaFeatsV1Data::new(),
      ),
      context:  context,
      arch:     arch,
    }
  }
}

impl Agent for ConvnetAgent {
  fn reset(&mut self) {
    self.komi = 7.5;
    self.player = None;

    self.history.clear();
    self.state.reset();
  }

  fn board_dim(&mut self, board_dim: usize) {
    assert_eq!(Board::DIM, board_dim);
  }

  fn komi(&mut self, komi: f32) {
    self.komi = komi;
  }

  fn player(&mut self, stone: Stone) {
    self.player = Some(Stone::Black);
  }

  fn apply_action(&mut self, turn: Stone, action: Action) {
    self.history.push((self.state.clone(), action));
    match self.state.try_action(turn, action) {
      Ok(_)   => { self.state.commit(); }
      Err(_)  => { panic!("agent tried to apply an illegal action!"); }
    }
  }

  fn undo(&mut self) {
    // FIXME(20151108): track ply; limit to how far we can undo.
    if let Some((prev_state, _)) = self.history.pop() {
      self.state.clone_from(&prev_state);
    } else {
      self.state.reset();
    }
  }

  fn act(&mut self, turn: Stone) -> Action {
    if self.history.len() > 0 {
      match self.history[self.history.len() - 1].1 {
        Action::Resign | Action::Pass => {
          return Action::Pass;
        }
        _ => {}
      }
    }

    let &mut ConvnetAgent{
      ref mut state, ref context, ref mut arch, .. } = self;
    let ctx = (*context).as_ref();
    state.get_data().extract_relative_features(turn, arch.input_layer().expose_host_frame_buf(0));
    arch.input_layer().load_frames(1, &ctx);
    arch.forward(1, Phase::Inference, &ctx);
    arch.loss_layer().store_probs(1, &ctx);
    let pred_probs = arch.loss_layer().get_probs(1);

    // FIXME(20160131): for 3-lookahead, but should use 1-lookahead for inference.
    let mut ranked_probs: Vec<_> = pred_probs.as_slice().iter().enumerate()
      .filter_map(|(k, &x)| {
        if k < 361 {
          Some((F32SupNan(-x), k))
        } else {
          None
        }
      })
      .collect()
    ;
    ranked_probs.sort();
    assert_eq!(361, ranked_probs.len());

    let mut action = Action::Pass;
    for k in 0 .. Board::SIZE {
      let place_point = Point(ranked_probs[k].1 as i16);
      let res = state.try_place(turn, place_point);
      state.undo();
      match res {
        Ok(_) => {
          action = Action::Place{point: place_point};
          break;
        }
        Err(_) => {}
      }
    }
    action
  }
}
