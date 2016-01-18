use board::{Action};
use search::parallel_policies::{
  SearchPolicyWorker,
  RolloutMode,
};
use search::parallel_policies::convnet::{
  ConvnetPolicyWorkerBuilder,
  ConvnetPolicyWorker,
  //ConvnetRolloutPolicy,
};
use search::parallel_tree::{
  ParallelMonteCarloSearchServer,
  ParallelMonteCarloSearch,
  ParallelMonteCarloEval,
  ParallelMonteCarloBackup,
  //ParallelMonteCarloEvalServer,
};
use txnstate::{TxnState};
use txnstate::extras::{TxnStateNodeData};
//use txnstate::features::{TxnStateExtLibFeatsData};

use array_cuda::device::{DeviceCtxRef};
use cuda::runtime::{CudaDevice};
use rembrandt::arch_new::{AtomicData, ArchWorker, PipelineArchWorker};
use rembrandt::layer_new::{Phase};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, thread_rng};

pub struct PgBalanceConfig {
  pub target_num_rollouts:  usize,
  pub value_minibatch_size: usize,
  pub grad_minibatch_size:  usize,
  pub learning_rate:        f32,
}

impl Default for PgBalanceConfig {
  fn default() -> PgBalanceConfig {
    PgBalanceConfig{
      target_num_rollouts:  5120,
      value_minibatch_size: 512,
      grad_minibatch_size:  512,
      learning_rate:        0.05,
    }
  }
}

pub struct PgBalanceMachine {
  config:           PgBalanceConfig,
  rng:              Xorshiftplus128Rng,
  search_server:    ParallelMonteCarloSearchServer<ConvnetPolicyWorker>,
  /*target_value: Option<f32>,
  eval_value:   Option<f32>,*/
}

impl PgBalanceMachine {
  pub fn new(config: PgBalanceConfig) -> PgBalanceMachine {
    let batch_size = 256;
    let num_workers = CudaDevice::count().unwrap();
    let worker_batch_size = batch_size / num_workers;
    PgBalanceMachine{
      config:           config,
      rng:              Xorshiftplus128Rng::new(&mut thread_rng()),
      search_server:    ParallelMonteCarloSearchServer::new(
          num_workers, worker_batch_size,
          ConvnetPolicyWorkerBuilder::new(num_workers, worker_batch_size),
      ),
    }
  }

  pub fn estimate_sample(&mut self, init_state: &TxnState<TxnStateNodeData>, ctx: &DeviceCtxRef) {
    // XXX(20160106): Estimate target value using search.
    let search = ParallelMonteCarloSearch::new();
    let (search_res, _) = search.join(
        self.config.target_num_rollouts,
        &self.search_server,
        init_state,
        None,
        &mut self.rng);
    let target_value = search_res.expected_value;

    // XXX(20160106): Invoke Monte Carlo evaluation with rollout policy to
    // compute values.
    let eval = ParallelMonteCarloEval::new();
    let (eval_res, _) = eval.join(
        self.config.value_minibatch_size,
        &self.search_server,
        init_state,
        &mut self.rng);
    let eval_value = eval_res.expected_value;

    // XXX(20160106): Invoke Monte Carlo evaluation with rollout policy to
    // compute gradients and update policy parameters.
    let backup = ParallelMonteCarloBackup::new();
    let _ = backup.join(
        self.config.value_minibatch_size,
        self.config.learning_rate,
        0.5,
        &self.search_server,
        &mut self.rng);
  }
}
