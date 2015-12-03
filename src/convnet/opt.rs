use async_cuda::context::{DeviceContext};
use rembrandt::data::{DataSource, SampleLabelConfig};
use rembrandt::net::{NetArch};
use rembrandt::opt::{OptConfig, OptState, DescentSchedule};

pub struct SimulationBalancing;

impl SimulationBalancing {
  pub fn train(&self,
      opt_cfg: &OptConfig,
      state: &mut OptState,
      arch: &mut NetArch,
      train_data: &mut DataSource,
      ctx: &DeviceContext)
  {
    //let descent = DescentSchedule::new(*opt_cfg);
    train_data.each_episode(SampleLabelConfig::Category, &mut |ep_idx, episode| {
      // TODO(20151129):
      // - pick a (precomputed) random position in the episode
      // - estimate "true" value using full search policy (5120 rollouts)
      // - save "true" value, possibly on disk
      // - run a 512-batch to compute the rollout value
      // - run another 512-batch to compute the rollout gradient
      // - descend
    });
  }
}
