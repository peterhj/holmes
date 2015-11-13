use convnet::{
  build_action_3layer_arch,
  //build_action_6layer_arch,
};
use search_policies::{PriorPolicy};
use txnstate::{TxnState, TxnStateNodeData};

use async_cuda::context::{DeviceContext};
use rembrandt::layer::{Layer};
use rembrandt::net::{NetArch, LinearNetArch};
use rembrandt::opt::{OptPhase};

pub struct ConvnetPriorPolicy {
  arch: LinearNetArch,
}

impl ConvnetPriorPolicy {
  pub fn new(ctx: &DeviceContext) -> ConvnetPriorPolicy {
    let arch = build_action_3layer_arch(1, &ctx);
    ConvnetPriorPolicy{
      arch: arch,
    }
  }
}

impl PriorPolicy for ConvnetPriorPolicy {
  type Ctx = DeviceContext;

  fn init_prior(&mut self,
      node_state: &TxnState<TxnStateNodeData>,
      total_trials: &mut f32,
      num_trials: &mut [f32],
      num_succs: &mut [f32],
      ctx: &DeviceContext)
  {
    let turn = node_state.current_turn();
    node_state.get_data().features.extract_relative_features(
        turn, self.arch.data_layer().expose_host_frame_buf(0));
    self.arch.data_layer().load_frames(1, ctx);
    self.arch.evaluate(OptPhase::Evaluation, ctx);
    // TODO(20151112)
  }
}
