use agents::{Agent};
use board::{Board, RuleSet, Stone, Point, Action};
use txnstate::{TxnState, TxnStateFeaturesData};

use async_cuda::context::{DeviceContext};
use rembrandt::layer::{
  ActivationFunction, ParamsInitialization, Layer,
  DataLayerConfig, Conv2dLayerConfig, SoftmaxLossLayerConfig,
  DataLayer, Conv2dLayer, SoftmaxLossLayer,
};
use rembrandt::net::{NetArch, LinearNetArch};
use rembrandt::opt::{OptPhase};

use std::path::{PathBuf};

pub struct ConvnetAgent {
  komi:     f32,
  player:   Option<Stone>,

  history:  Vec<(TxnState<TxnStateFeaturesData>, Action)>,
  state:    TxnState<TxnStateFeaturesData>,

  ctx:      DeviceContext,
  arch:     LinearNetArch,
}

impl ConvnetAgent {
  pub fn new() -> ConvnetAgent {
    let ctx = DeviceContext::new(0);

    let batch_size = 1;
    let num_hidden = 64;
    let data_layer_cfg = DataLayerConfig{
      raw_width: 19, raw_height: 19,
      crop_width: 19, crop_height: 19,
      channels: 4,
    };
    let conv1_layer_cfg = Conv2dLayerConfig{
      in_width: 19, in_height: 19, in_channels: 4,
      conv_size: 9, conv_stride: 1, conv_pad: 4,
      out_channels: num_hidden,
      act_fun: ActivationFunction::Rect,
      init_weights: ParamsInitialization::None,
    };
    let hidden_conv_layer_cfg = Conv2dLayerConfig{
      in_width: 19, in_height: 19, in_channels: num_hidden,
      conv_size: 3, conv_stride: 1, conv_pad: 1,
      out_channels: num_hidden,
      act_fun: ActivationFunction::Rect,
      init_weights: ParamsInitialization::None,
    };
    let final_conv_layer_cfg = Conv2dLayerConfig{
      in_width: 19, in_height: 19, in_channels: num_hidden,
      conv_size: 3, conv_stride: 1, conv_pad: 1,
      out_channels: 1,
      act_fun: ActivationFunction::Identity,
      init_weights: ParamsInitialization::None,
    };
    let loss_layer_cfg = SoftmaxLossLayerConfig{
      num_categories: 361,
      do_mask: false,
    };

    let data_layer = DataLayer::new(0, data_layer_cfg, batch_size);
    let conv1_layer = Conv2dLayer::new(0, conv1_layer_cfg, batch_size, Some(&data_layer), &ctx);
    let conv2_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv1_layer), &ctx);
    let conv3_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv2_layer), &ctx);
    let conv4_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv3_layer), &ctx);
    let conv5_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv4_layer), &ctx);
    let conv6_layer = Conv2dLayer::new(0, final_conv_layer_cfg, batch_size, Some(&conv5_layer), &ctx);
    let softmax_layer = SoftmaxLossLayer::new(0, loss_layer_cfg, batch_size, Some(&conv6_layer));
    let mut arch = LinearNetArch::new(
        PathBuf::from("experiments/models/tmp_19x19x4.v2"),
        batch_size,
        data_layer,
        softmax_layer,
        vec![
          Box::new(conv1_layer),
          Box::new(conv2_layer),
          Box::new(conv3_layer),
          Box::new(conv4_layer),
          Box::new(conv5_layer),
          Box::new(conv6_layer),
        ],
    );
    arch.load_layer_params(None, &ctx);

    ConvnetAgent{
      komi:     0.0,
      player:   None,
      history:  vec![],
      state:    TxnState::new(RuleSet::KgsJapanese.rules(), TxnStateFeaturesData::new()),
      ctx:      ctx,
      arch:     arch,
    }
  }
}

impl Agent for ConvnetAgent {
  fn reset(&mut self) {
    self.komi = 6.5;
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
    if let Some((prev_state, _)) = self.history.pop() {
      self.state.clone_from(&prev_state);
    } else {
      self.state.reset();
    }
  }

  fn act(&mut self, turn: Stone) -> Action {
    if self.history.len() > 0 {
      if let Action::Pass = self.history[self.history.len() - 1].1 {
        return Action::Pass;
      }
    }
    let &mut ConvnetAgent{
      ref mut state, ref ctx, ref mut arch, .. } = self;
    state.get_data().extract_relative_features(turn, arch.data_layer().expose_host_frame_buf(0));
    arch.data_layer().load_frames(1, ctx);
    arch.evaluate(OptPhase::Evaluation, ctx);
    arch.loss_layer().store_ranked_labels(1, ctx);
    let ranked_labels = arch.loss_layer().predict_ranked_labels(1);
    let mut action = Action::Pass;
    for k in (0 .. Board::SIZE) {
      let place_point = Point(ranked_labels[k] as i16);
      match state.try_place(turn, place_point) {
        Ok(_) => {
          state.undo();
          action = Action::Place{point: place_point};
          break;
        }
        Err(_) => {
          state.undo();
        }
      }
    }
    action
  }
}
