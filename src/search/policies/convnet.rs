use board::{Point};
use convnet::{
  build_action_3layer_arch,
  build_action_6layer_arch,
};
use search::policies::{PriorPolicy};
use txnstate::{TxnState};
use txnstate::extras::{TxnStateNodeData};

use async_cuda::context::{DeviceContext};
use rembrandt::layer::{Layer};
use rembrandt::net::{NetArch, LinearNetArch};
use rembrandt::opt::{OptPhase};

pub struct ConvnetPriorPolicy {
  ctx:  DeviceContext,
  arch: LinearNetArch,
}

impl ConvnetPriorPolicy {
  pub fn new() -> ConvnetPriorPolicy {
    let ctx = DeviceContext::new(0);
    //let arch = build_action_3layer_arch(1, &ctx);
    let arch = build_action_6layer_arch(1, &ctx);
    ConvnetPriorPolicy{
      ctx:  ctx,
      arch: arch,
    }
  }
}

impl PriorPolicy for ConvnetPriorPolicy {
  //type Ctx = DeviceContext;

  /*fn init_prior(&mut self, node: &mut Node, ctx: &DeviceContext) {
    // TODO(20151113)
    /*let turn = node_state.current_turn();
    node_state.get_data().features.extract_relative_features(
        turn, self.arch.data_layer().expose_host_frame_buf(0));
    self.arch.data_layer().load_frames(1, ctx);
    self.arch.evaluate(OptPhase::Evaluation, ctx);*/
  }*/

  fn fill_prior_probs(&mut self, state: &TxnState<TxnStateNodeData>, valid_moves: &[Point], prior_moves: &mut Vec<(Point, f32)>) {
    // TODO(20151113)
    let turn = state.current_turn();
    state.get_data().features.extract_relative_features(turn, self.arch.data_layer().expose_host_frame_buf(0));
    self.arch.data_layer().load_frames(1, &self.ctx);
    self.arch.evaluate(OptPhase::Evaluation, &self.ctx);
    self.arch.loss_layer().store_probs(1, &self.ctx);
    let pred_probs = self.arch.loss_layer().predict_probs(1, &self.ctx);
    for &point in valid_moves.iter() {
      //prior_moves.push((point, 0.5));
      prior_moves.push((point, pred_probs.as_slice()[point.0 as usize]));
    }
  }
}
