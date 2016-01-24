use board::{Action};
use data::{EpisodeIter};
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
  ParallelMonteCarloSave,
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
use std::path::{PathBuf};

use rand::{Rng, thread_rng};
use time::{get_time};

pub struct PgBalanceConfig {
  pub num_epochs:           usize,
  pub save_interval:        usize,
  pub target_num_rollouts:  usize,
  pub target_batch_size:    usize,
  pub value_minibatch_size: usize,
  pub grad_minibatch_size:  usize,
  pub learning_rate:        f32,
}

impl Default for PgBalanceConfig {
  fn default() -> PgBalanceConfig {
    PgBalanceConfig{
      num_epochs:           10,
      save_interval:        50,
      target_num_rollouts:  5120,
      target_batch_size:    256,
      value_minibatch_size: 512,
      grad_minibatch_size:  512,
      learning_rate:        0.001,
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
    let num_workers = CudaDevice::count().unwrap();
    let worker_batch_size = 544;
    PgBalanceMachine{
      config:           config,
      rng:              Xorshiftplus128Rng::new(&mut thread_rng()),
      search_server:    ParallelMonteCarloSearchServer::new(
          num_workers, worker_batch_size,
          ConvnetPolicyWorkerBuilder::new(num_workers, worker_batch_size),
      ),
    }
  }

  pub fn estimate(&mut self, episodes: &mut EpisodeIter) {
    // FIXME(20160121): temporarily using existing balance run.
    let mut idx = 1000;

    if idx == 0 {
      let save = ParallelMonteCarloSave::new();
      save.join(
          &PathBuf::from("experiments/models_balance/test"),
          0,
          &self.search_server);
    }

    loop {
      episodes.for_each_random_sample(|_, state, _, _, _| {
        self.estimate_sample(idx, state);

        idx += 1;
        if idx % self.config.save_interval == 0 {
          println!("DEBUG: pg::balance: saving params (iter {})", idx);
          let save = ParallelMonteCarloSave::new();
          save.join(
              &PathBuf::from("experiments/models_balance/test"),
              idx,
              &self.search_server);
        }
      });
    }
  }

  fn estimate_sample(&mut self, idx: usize, init_state: &TxnState<TxnStateNodeData>) {
    let start_time = get_time();

    // XXX(20160106): Estimate target value using search.
    let search = ParallelMonteCarloSearch::new();
    let (search_res, _) = search.join(
        self.config.target_num_rollouts,
        self.config.target_batch_size,
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
        self.config.value_minibatch_size,
        &self.search_server,
        init_state,
        &mut self.rng);
    let eval_value = eval_res.expected_value;

    // XXX(20160106): Invoke Monte Carlo evaluation with rollout policy to
    // compute gradients and update policy parameters.
    let backup = ParallelMonteCarloBackup::new();
    let _ = backup.join(
        self.config.grad_minibatch_size,
        self.config.grad_minibatch_size,
        self.config.learning_rate,
        target_value,
        eval_value,
        0.5,
        &self.search_server,
        init_state,
        &mut self.rng);

    let lap_time = get_time();
    let elapsed_ms = (lap_time - start_time).num_milliseconds();
    println!("DEBUG: sample {}: target: {:.4} eval: {:.4} elapsed: {:.3} s",
        idx, target_value, eval_value, elapsed_ms as f32 * 0.001);
  }
}
