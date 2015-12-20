use board::{Action};
use txnstate::{TxnState};
use txnstate::features::{TxnStateLibFeaturesData};

use array_cuda::device::{DeviceCtxRef};
use rembrandt::arch_new::{ArchWorker, PipelineArchWorker};
use rembrandt::layer_new::{Phase};

pub struct Trace {
  pairs:    Vec<(TxnState<TxnStateLibFeaturesData>, Action)>,
  _score:   Option<f32>,
  reward:   Option<f32>,
}

pub struct PolicyBalanceMachine {
  value_minibatch_size: usize,
  grad_minibatch_size:  usize,
  learning_rate:        f32,

  arch:     PipelineArchWorker,
  trace:    Trace,

  target_value: f32,
  mean_value:   f32,
}

impl PolicyBalanceMachine {
  pub fn estimate(&mut self, initial_state: &TxnState<TxnStateLibFeaturesData>, ctx: &DeviceCtxRef) {
    // TODO(20151219): estimate target value using search.

    let mut mean_value_accum = 0.0;
    for i in (0 .. self.value_minibatch_size) {
      // TODO(20151219): invoke rollout policy.

      let reward = self.trace.reward.unwrap();
      mean_value_accum += reward;
    }
    self.mean_value = mean_value_accum / (self.value_minibatch_size as f32);

    for i in (0 .. self.grad_minibatch_size) {
      // TODO(20151219): invoke rollout policy.

      let trace_len = self.trace.pairs.len();
      let reward = self.trace.reward.unwrap();
      let baseline = 0.5;
      for t in (0 .. trace_len) {
        let state = &self.trace.pairs[t].0;
        let turn = state.current_turn();
        state.get_data()
          .extract_relative_features(
              turn, self.arch.input_layer().expose_host_frame_buf(t),
          );
      }
      self.arch.input_layer().load_frames(trace_len, ctx);
      self.arch.forward(trace_len, Phase::Training, ctx);
      self.arch.backward(trace_len, (reward - baseline) / ((self.grad_minibatch_size * trace_len) as f32), ctx);
    }

    self.arch.descend(self.learning_rate * (self.mean_value - self.target_value), 0.0, ctx);
    self.arch.reset_gradients(0.0, ctx);
  }
}
