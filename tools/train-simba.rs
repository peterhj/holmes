extern crate async_cuda;
extern crate rembrandt;
extern crate rand;

use rembrandt::config::{ModelConfigFile};
use rembrandt::data::{DatasetConfiguration, DataSourceBuilder};
use rembrandt::layer::{
  Layer,
  ParamsInitialization, ActivationFunction,
  DataLayerConfig, Conv2dLayerConfig, SoftmaxLossLayerConfig,
};
use rembrandt::layer::device::{
  DataLayer, Conv2dLayer, SoftmaxLossLayer,
};
use rembrandt::net::{NetArch, LinearNetArch};
use rembrandt::opt::{
  OptConfig, DescentSchedule, AnnealingPolicy,
  OptState, Optimizer, SgdOptimizer,
};

use async_cuda::context::{DeviceContext};

use rand::{Rng, thread_rng};
use std::path::{PathBuf};

fn main() {
  train_simba_2_layers();
}

fn train_simba_2_layers() {
  let mut rng = thread_rng();
  let ctx = DeviceContext::new(0);

  let opt_cfg = OptConfig{
    minibatch_size: 256,
    max_iters:      100000,
    init_step_size: 0.01,
    momentum:       0.9,
    l2_reg_coef:    0.0,
    anneal:         AnnealingPolicy::Step{step_iters: 6144, decay: 0.1},
    interval_size:  400,
  };
  let descent = DescentSchedule::new(opt_cfg);
  let batch_size = 256;

  let data_layer_cfg = DataLayerConfig{
    raw_width: 19, raw_height: 19,
    crop_width: 19, crop_height: 19,
    channels: 4,
  };
  let conv1_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: 4,
    conv_size: 9, conv_stride: 1, conv_pad: 4,
    out_channels: 16,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::Normal{mean: 0.0, std: 0.01},
  };
  let conv2_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: 16,
    conv_size: 3, conv_stride: 1, conv_pad: 1,
    out_channels: 1,
    act_fun: ActivationFunction::Identity,
    init_weights: ParamsInitialization::Normal{mean: 0.0, std: 0.01},
  };
  let data_layer = DataLayer::new(0, data_layer_cfg, batch_size);
  let conv1_layer = Conv2dLayer::new(0, conv1_layer_cfg, batch_size, Some(&data_layer), &ctx);
  let conv2_layer = Conv2dLayer::new(0, conv2_layer_cfg, batch_size, Some(&conv1_layer), &ctx);
  let softmax_layer = SoftmaxLossLayer::new(0, 361, batch_size, Some(&conv2_layer));

  let mut arch = LinearNetArch::new(batch_size,
      Box::new(data_layer),
      Box::new(softmax_layer),
      vec![
        Box::new(conv1_layer),
        Box::new(conv2_layer),
      ],
  );
  for layer in arch.hidden_layers_forward() {
    layer.initialize_params(&ctx);
  }

  let mut state = OptState{epoch: 0, t: 0};

  let dataset_cfg = DatasetConfiguration::open(&PathBuf::from("experiments/gogodb.data"));
  let mut train_data_source = if let Some(&(ref name, ref cfg)) = dataset_cfg.datasets.get("train") {
    DataSourceBuilder::build(name, cfg.clone())
  } else {
    panic!("missing train data source!");
  };
  let mut test_data_source = if let Some(&(ref name, ref cfg)) = dataset_cfg.datasets.get("valid") {
    DataSourceBuilder::build(name, cfg.clone())
  } else {
    panic!("missing validation data source!");
  };

  let sgd = SgdOptimizer;
  sgd.train(&opt_cfg, &mut state, &mut arch, &mut *train_data_source, &mut *test_data_source, &ctx);
}
