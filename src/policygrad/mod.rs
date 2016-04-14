//extern crate scoped_threadpool;

use policygrad::convnet::{ConvnetBiPolicyBuilder, Traj};
use rng::xorshift::{Xorshiftplus128Rng};
use scoped_threadpool::{Pool};

use rand::{Rng, thread_rng};
use std::cmp::{max};
use std::iter::{repeat};
use std::sync::{Arc};
use std::sync::atomic::{AtomicUsize, Ordering, fence};
use time::{get_time};

pub mod convnet;

#[derive(Clone, Copy)]
pub struct PolicyGradConfig {
  pub step_size:    f32,
  pub baseline:     f32,
  pub minibatch_size:   usize,
  pub display_interval: usize,
  pub save_interval:    usize,
  pub pool_interval:    usize,
}

pub struct PolicyGradDriver {
  num_workers:  usize,
  init_iter:    Option<usize>,
  cfg:      PolicyGradConfig,
  builder:  ConvnetBiPolicyBuilder,
}

impl PolicyGradDriver {
  pub fn new(num_workers: usize, init_iter: Option<usize>, cfg: PolicyGradConfig) -> PolicyGradDriver {
    PolicyGradDriver{
      num_workers:  num_workers,
      init_iter:    init_iter,
      cfg:      cfg,
      builder:  ConvnetBiPolicyBuilder::new(num_workers, max(32, cfg.minibatch_size / num_workers)),
    }
  }

  pub fn train(&mut self) {
    let mut pool = Pool::new(self.num_workers as u32);
    //let barrier = Arc::new(Barrier::new(self.num_workers));
    pool.scoped(|scope| {
      let cfg = self.cfg;
      let init_iter = self.init_iter;
      let num_workers = self.num_workers;
      let batch_size = cfg.minibatch_size / self.num_workers;
      let shared_succ_count = Arc::new(AtomicUsize::new(0));
      for tid in 0 .. num_workers {
        //let barrier = barrier.clone();
        let builder = self.builder.clone();
        let shared_succ_count = shared_succ_count.clone();
        scope.execute(move || {
          let mut rng = Xorshiftplus128Rng::new(&mut thread_rng());
          let mut policy = builder.into_policy(tid);
          let mut trajs: Vec<_> = repeat(Traj::new()).take(batch_size).collect();

          let mut red_pool = vec![];
          let mut k = 0;
          loop {
            // FIXME(20160310)
            match policy.load_params_into_mem(k) {
              Ok(blob) => {
                red_pool.push(blob);
              }
              Err(_) => break,
            }
            k += cfg.pool_interval;
          }

          let mut start_time = get_time();
          let mut prev_red_idx = None;
          //let mut iter = 100; // FIXME(20160311)
          let mut iter = init_iter.unwrap_or(0);

          loop {
            let red_idx = rng.gen_range(0, red_pool.len());
            match prev_red_idx {
              Some(prev_red_idx) => {
                if prev_red_idx != red_idx {
                  policy.load_red_params(&red_pool[red_idx]);
                }
              }
              None => {
                policy.load_red_params(&red_pool[red_idx]);
              }
            }
            prev_red_idx = Some(red_idx);

            let succ_count = policy.rollout_batch(batch_size, &mut trajs, &mut rng);
            shared_succ_count.fetch_add(succ_count, Ordering::AcqRel);
            policy.update_params(batch_size, cfg.step_size, cfg.baseline, &mut trajs);
            //barrier.wait();
            fence(Ordering::AcqRel);
            iter += 1;

            if iter % cfg.display_interval == 0 {
              if tid == 0 {
                let accum_succ_count = shared_succ_count.load(Ordering::Acquire);
                let accum_trial_count = cfg.display_interval * cfg.minibatch_size;
                let ratio = accum_succ_count as f32 / accum_trial_count as f32;
                let lap_time = get_time();
                let elapsed_ms = (lap_time - start_time).num_milliseconds();
                println!("DEBUG: iter: {} ratio: {}/{} {:.6} elapsed: {:.3} s",
                    iter,
                    accum_succ_count, accum_trial_count,
                    ratio,
                    0.001 * elapsed_ms as f32,
                );
                start_time = lap_time;
                shared_succ_count.store(0, Ordering::Release);
              }
              fence(Ordering::AcqRel);
            }

            if iter % cfg.save_interval == 0 {
              if tid == 0 {
                policy.save_params(iter);
              }
            }

            if iter % cfg.pool_interval == 0 {
              let blob = policy.save_params_to_mem();
              red_pool.push(blob);
            }
          }
        });
      }
      scope.join_all();
    });
  }
}
