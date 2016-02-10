use board::{Stone, Action};
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

pub struct RolloutEvalOutcomeConfig {
  pub fixed_t:      usize,
  pub num_rollouts: usize,
}

pub struct RolloutEvalOutcomeMachine {
  config:           RolloutEvalOutcomeConfig,
  batch_size:       usize,
  rng:              Xorshiftplus128Rng,
  search_server:    ParallelMonteCarloSearchServer<ConvnetPolicyWorker>,
}

impl RolloutEvalOutcomeMachine {
  pub fn new(config: RolloutEvalOutcomeConfig) -> RolloutEvalOutcomeMachine {
    let num_workers = 1; // CudaDevice::count().unwrap();
    let worker_batch_size = 512;
    RolloutEvalOutcomeMachine{
      config:           config,
      batch_size:       worker_batch_size,
      rng:              Xorshiftplus128Rng::new(&mut thread_rng()),
      search_server:    ParallelMonteCarloSearchServer::new(
          num_workers, worker_batch_size,
          ConvnetPolicyWorkerBuilder::new(num_workers, worker_batch_size),
      ),
    }
  }

  pub fn estimate(&mut self, episodes: &mut EpisodeIter) {
    let mut mean_pred_value: f32 = 0.5;
    let mut mean_count: f32 = 1.0;

    let mut correct_count: i32 = 0;
    let mut total_count: i32 = 0;

    let mut idx = 0;
    loop {
      episodes.cyclic_frame_at(self.config.fixed_t, |_, frame| {
        let start_time = get_time();

        let init_state = frame.state;
        let init_turn = frame.turn;
        assert_eq!(init_turn, init_state.current_turn());
        let value = frame.value;
        let outcome = frame.outcome;

        // XXX(20160106): Invoke Monte Carlo evaluation with rollout policy to
        // compute values.
        let eval = ParallelMonteCarloEval::new();
        let (eval_res, _) = eval.join(
            self.batch_size,
            self.batch_size,
            &self.search_server,
            // FIXME(20160208): what is our player color?
            init_state.current_turn(),
            init_state,
            &mut self.rng);
        let eval_value = eval_res.expected_value;

        let threshold = mean_pred_value;
        let correct = match (init_turn, outcome) {
          (Stone::Black, Stone::Black) => eval_value > threshold,
          (Stone::Black, Stone::White) => eval_value <= threshold,
          (Stone::White, Stone::White) => eval_value > threshold,
          (Stone::White, Stone::Black) => eval_value <= threshold,
          _ => unreachable!(),
        };
        if correct {
          correct_count += 1;
        }
        total_count += 1;

        mean_count += 1.0;
        mean_pred_value += (eval_value - mean_pred_value) / mean_count;

        let lap_time = get_time();
        let elapsed_ms = (lap_time - start_time).num_milliseconds();
        println!("DEBUG: sample {}: accuracy: {:.3} mean: {:.4} target: {:.4} eval: {:.4} elapsed: {:.3} s",
            idx,
            correct_count as f32 / total_count as f32,
            mean_pred_value, value, eval_value,
            elapsed_ms as f32 * 0.001);

        idx += 1;
        /*if idx % self.config.save_interval == 0 {
          println!("DEBUG: pg::balance: saving params (iter {})", idx);
          let save = ParallelMonteCarloSave::new();
          save.join(
              &PathBuf::from("experiments/models_balance/test"),
              idx,
              &self.search_server);
        }*/
      });
    }
  }
}
