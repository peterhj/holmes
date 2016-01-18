use board::{Action};
use search::parallel_policies::{
  SearchPolicyWorker,
  RolloutMode,
};
use search::parallel_policies::convnet::{
  ConvnetPolicyWorker,
  ConvnetRolloutPolicy,
};
use search::parallel_tree::{
  ParallelMonteCarloSearch,
  ParallelMonteCarloSearchServer,
  ParallelMonteCarloEval,
  ParallelMonteCarloEvalServer,
};
use txnstate::{TxnState};
use txnstate::extras::{TxnStateNodeData};
use txnstate::features::{TxnStateExtLibFeatsData};

use array_cuda::device::{DeviceCtxRef};
use rembrandt::arch_new::{AtomicData, ArchWorker, PipelineArchWorker};
use rembrandt::layer_new::{Phase};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng};

pub struct Trace {
  pairs:    Vec<(TxnState<TxnStateExtLibFeatsData>, Action)>,
  _score:   Option<f32>,
  reward:   Option<f32>,
}

pub struct PgBalanceMachine {
  target_num_rollouts:  usize,
  value_minibatch_size: usize,
  grad_minibatch_size:  usize,
  learning_rate:        f32,

  rng:              Xorshiftplus128Rng,
  search_server:    ParallelMonteCarloSearchServer<ConvnetPolicyWorker>,
  eval_server:      ParallelMonteCarloEvalServer<ConvnetRolloutPolicy>,

  target_value: f32,
  eval_value:   f32,
}

impl PgBalanceMachine {
  pub fn estimate_sample(&mut self, init_state: &TxnState<TxnStateNodeData>, ctx: &DeviceCtxRef) {
    // XXX(20160106): Estimate target value using search.
    let mut search = ParallelMonteCarloSearch::new();
    let (search_res, _) = search.join(5120, &self.search_server, init_state, None, &mut self.rng);
    self.target_value = search_res.expected_value;

    // XXX(20160106): Invoke Monte Carlo evaluation with rollout policy to
    // compute values.
    let eval = ParallelMonteCarloEval::new();
    let eval_res = eval.join(self.value_minibatch_size, &self.eval_server, init_state, RolloutMode::Simulation, &mut self.rng);
    self.eval_value = eval_res.expected_value;

    // XXX(20160106): Invoke Monte Carlo evaluation with rollout policy to
    // compute gradients and update policy parameters.
    let grad_eval = ParallelMonteCarloEval::new();
    let grad_eval_res = eval.join(self.grad_minibatch_size, &self.eval_server, init_state, RolloutMode::BalanceTraining, &mut self.rng);

    // TODO(20160116)
    unimplemented!();
  }
}
