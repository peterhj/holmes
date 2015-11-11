extern crate array;
extern crate async;
extern crate async_cuda;
extern crate rembrandt;
extern crate rand;
extern crate time;

use rembrandt::config::{ModelConfigFile};
use rembrandt::data::{DatasetConfiguration, DataSourceBuilder};
use rembrandt::layer::{
  Layer,
  ParamsInitialization, ActivationFunction,
  DataLayerConfig, Conv2dLayerConfig, SoftmaxLossLayerConfig,
};
use rembrandt::layer::{
  DataLayer, Conv2dLayer, SoftmaxLossLayer,
};
use rembrandt::net::{NetArch, LinearNetArch};
use rembrandt::opt::{
  OptPhase, OptConfig, DescentSchedule, AnnealingPolicy,
  OptState, Optimizer, SgdOptimizer,
};

use array::*;
use async::*;
use async_cuda::array_types::*;
use async_cuda::context::{DeviceContext};

use rand::{Rng, thread_rng};
use std::path::{PathBuf};
use time::{Duration, get_time};

fn main() {
  bench_net();
}

fn bench_net() {
  let mut rng = thread_rng();
  let ctx = DeviceContext::new(0);

  /*let opt_cfg = OptConfig{
    minibatch_size: 256,
    max_iters:      100000,
    init_step_size: 0.01,
    momentum:       0.9,
    l2_reg_coef:    0.0,
    anneal:         AnnealingPolicy::Step{step_iters: 6144, decay: 0.1},
    display_interval:   400,
    validate_interval:  400,
    save_interval:      None,
  };
  let descent = DescentSchedule::new(opt_cfg);*/

  let batch_size = 256;
  let num_planes = 8;
  let num_hidden = 16;

  let data_layer_cfg = DataLayerConfig{
    raw_width: 19, raw_height: 19,
    crop_width: 19, crop_height: 19,
    channels: num_planes,
  };
  let conv1_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: num_planes,
    conv_size: 9, conv_stride: 1, conv_pad: 4,
    out_channels: num_hidden,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::Normal{mean: 0.0, std: 0.01},
  };
  let conv2_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: num_hidden,
    conv_size: 5, conv_stride: 1, conv_pad: 2,
    out_channels: 1,
    act_fun: ActivationFunction::Identity,
    init_weights: ParamsInitialization::Normal{mean: 0.0, std: 0.01},
  };
  let loss_layer_cfg = SoftmaxLossLayerConfig{
    num_categories: 361,
    do_mask: false,
  };
  let data_layer = DataLayer::new(0, data_layer_cfg, batch_size);
  let conv1_layer = Conv2dLayer::new(0, conv1_layer_cfg, batch_size, Some(&data_layer), &ctx);
  let conv2_layer = Conv2dLayer::new(0, conv2_layer_cfg, batch_size, Some(&conv1_layer), &ctx);
  let softmax_layer = SoftmaxLossLayer::new(0, loss_layer_cfg, batch_size, Some(&conv2_layer));

  let mut arch = LinearNetArch::new(
      PathBuf::from(""),
      batch_size,
      data_layer,
      Box::new(softmax_layer),
      vec![
        Box::new(conv1_layer),
        Box::new(conv2_layer),
      ],
  );
  for layer in arch.hidden_layers_forward() {
    layer.initialize_params(&ctx);
  }

  let host_input_buf = Array2d::<u8>::with_zeros((361*num_planes, 256));
  let mut device_input_buf = DeviceBuf::<u8>::with_zeros(361*num_planes*256);
  let mut host_output_buf = Array2d::<f32>::with_zeros((361, 256));
  let mut device_output_buf = DeviceBuf::<f32>::with_zeros(361*256);

  let trials = 1000;
  ctx.blocking_sync();
  let start_time = get_time();
  for _ in (0 .. trials) {
    let host_view = host_input_buf.as_view();
    let mut device_view = device_input_buf.as_mut_view_2d((361*num_planes, 256));
    let guard = device_view.sync_load(&host_view, &ctx);
  }
  ctx.blocking_sync();
  let lap_time = get_time();
  let elapsed_ms = (lap_time - start_time).num_milliseconds();
  println!("round-way transfer:");
  println!("  elapsed:    {:.3} ms", elapsed_ms as f32);
  println!("  throughput: {:.3}/s", (trials * arch.batch_size()) as f32 / (elapsed_ms as f32 * 0.001));

  let trials = 1000;
  ctx.blocking_sync();
  let start_time = get_time();
  for _ in (0 .. trials) {
    let host_input_view = host_input_buf.as_view();
    let mut device_input_view = device_input_buf.as_mut_view_2d((361*num_planes, 256));
    device_input_view.sync_load(&host_input_view, &ctx);
    arch.evaluate(OptPhase::Evaluation, &ctx);
    //let _ = arch.predict_labels(&ctx);
    let _ = arch.loss_layer().predict_probs(batch_size, &ctx);
    /*let device_output_view = device_output_buf.as_view_2d((361, 256));
    let mut host_output_view = host_output_buf.as_mut_view();
    device_output_view.sync_store(&mut host_output_view, &ctx);*/
  }
  ctx.blocking_sync();
  let lap_time = get_time();
  let elapsed_ms = (lap_time - start_time).num_milliseconds();
  println!("net evaluation:");
  println!("  elapsed:    {:.3} ms", elapsed_ms as f32);
  println!("  throughput: {:.3}/s", (trials * arch.batch_size()) as f32 / (elapsed_ms as f32 * 0.001));
}
