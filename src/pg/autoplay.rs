use board::{Stone};
use search::parallel_policies::{
  SearchPolicyWorker,
  RolloutMode,
};
use search::parallel_policies::convnet::{
  BiConvnetPolicyWorkerBuilder,
  BiConvnetPolicyWorker,
};
use search::parallel_tree::{
  SearchWorkerOutput,
  ParallelMonteCarloSearchServer,
  ParallelMonteCarloRollout,
  ParallelMonteCarloRolloutBackup,
  ParallelMonteCarloSaveFriendlyParams,
  ParallelMonteCarloLoadOpponentParams,
};
use txnstate::{TxnState};
use txnstate::extras::{TxnStateNodeData};

use cuda::runtime::{CudaDevice};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, thread_rng};
use rand::distributions::{IndependentSample};
use rand::distributions::range::{Range};
use std::sync::{Arc};
use time::{Duration, get_time};

pub struct PolicyGradAutoplayConfig {
  pub minibatch_size:       usize,
  pub opponent_interval:    usize,
  pub step_size:            f32,
  pub baseline_value:       f32,
}

pub struct PolicyGradAutoplayMachine {
  config:           PolicyGradAutoplayConfig,
  init_params_blob: Option<Vec<u8>>,

  rng:              Xorshiftplus128Rng,
  search_server:    ParallelMonteCarloSearchServer<BiConvnetPolicyWorker>,
  opponent_pool:    Vec<(usize, Arc<Vec<u8>>)>,
}

impl PolicyGradAutoplayMachine {
  pub fn new(config: PolicyGradAutoplayConfig/*, init_params_blob: Vec<u8>*/) -> PolicyGradAutoplayMachine {
    let num_workers = CudaDevice::count().unwrap();
    let worker_batch_capacity = 544;
    let rng = Xorshiftplus128Rng::new(&mut thread_rng());
    //let opponent_pool = vec![(0, Arc::new(init_params_blob.clone()))];
    PolicyGradAutoplayMachine{
      config:           config,
      init_params_blob: None, //init_params_blob,
      rng:              rng,
      search_server:    ParallelMonteCarloSearchServer::new(
          num_workers, worker_batch_capacity,
          BiConvnetPolicyWorkerBuilder::new(num_workers, worker_batch_capacity),
      ),
      opponent_pool:    vec![], //opponent_pool,
    }
  }

  pub fn train(&mut self) {
    self.opponent_pool.clear();
    ParallelMonteCarloSaveFriendlyParams.join(
        &self.search_server,
    );
    match self.search_server.recv_output() {
      SearchWorkerOutput::OpponentRolloutParams{params_blob} => {
        self.init_params_blob = Some(params_blob.clone());
        self.opponent_pool.push((0, Arc::new(params_blob)));
      }
      //_ => unreachable!(),
    }
    let mut opponent_pool_range = Range::new(0, 1);

    let mut init_state = TxnState::new(
        Default::default(),
        TxnStateNodeData::new(),
    );
    init_state.reset();

    let mut prev_opponent_idx = 0;
    let mut t = 0;
    loop {
      let start_time = get_time();

      // FIXME(20160206): how to do policy gradient training:
      // - load a single set of params from the opponent pool
      // - initiate a single batch of rollouts only, specifying which stones
      //   are friend and enemy
      // - compute gradients and update params one trace at a time
      // FIXME(20160207): trace gradients should NOT be evaluated when the
      // policy probabilities are zero and the uniform fallback is invoked.

      // FIXME(20160210): pick colors for us and opponent.
      let green_stone = Stone::Black;

      // TODO(20160207): pick an opponent from the pool.
      let opponent_idx = opponent_pool_range.ind_sample(&mut self.rng);
      if opponent_idx != prev_opponent_idx {
        ParallelMonteCarloLoadOpponentParams.join(
            &self.search_server,
            self.opponent_pool[opponent_idx].1.clone(),
        );
      }

      let (num_wins, num_trials) = ParallelMonteCarloRollout.join(
          &self.search_server,
          self.config.minibatch_size,
          green_stone,
          init_state.clone(),
          &mut self.rng,
      );

      ParallelMonteCarloRolloutBackup.join(
          &self.search_server,
          self.config.step_size,
          self.config.baseline_value,
          self.config.minibatch_size,
          green_stone,
          &mut self.rng,
      );

      let lap_time = get_time();
      let elapsed_ms = (lap_time - start_time).num_milliseconds() as f32 * 0.001;
      println!("DEBUG: t: {}, record: {}/{} ({:.4}), elapsed: {:.3} s",
          t, num_wins, num_trials, num_wins as f32 / num_trials as f32, elapsed_ms);

      t += 1;
      if t % self.config.opponent_interval == 0 {
        // FIXME(20160207)

        //opponent_pool_range = Range::new(0, self.opponent_pool.len());
      }

      prev_opponent_idx = opponent_idx;
    }
  }
}
