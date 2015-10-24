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
use rembrandt::layer::{
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

  let batch_size = 128;
  let num_hidden = 128;

  let opt_cfg = OptConfig{
    minibatch_size: batch_size,
    max_iters:      100000,
    init_step_size: 0.1,
    momentum:       0.9,
    l2_reg_coef:    0.0,
    anneal:         AnnealingPolicy::StepOnce{
      step_iters: 2048, new_rate: 0.01,
    },
    /*anneal:         AnnealingPolicy::StepTwice{
                      first_step: 2048, second_step: 12288,
                      first_rate: 0.01, second_rate: 0.001,
                    },*/
    display_interval:   1024,
    validate_interval:  2048,
    save_interval:      Some(2048),
  };
  let descent = DescentSchedule::new(opt_cfg);

  let data_layer_cfg = DataLayerConfig{
    raw_width: 19, raw_height: 19,
    crop_width: 19, crop_height: 19,
    channels: 4,
  };
  let conv1_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: 4,
    conv_size: 5, conv_stride: 1, conv_pad: 2,
    out_channels: num_hidden,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::Normal{mean: 0.0, std: 0.01},
  };
  let inner_conv_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: num_hidden,
    conv_size: 3, conv_stride: 1, conv_pad: 1,
    out_channels: num_hidden,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::Normal{mean: 0.0, std: 0.01},
  };
  let final_conv_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: num_hidden,
    conv_size: 3, conv_stride: 1, conv_pad: 1,
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
  let conv2_layer = Conv2dLayer::new(0, inner_conv_layer_cfg, batch_size, Some(&conv1_layer), &ctx);
  let conv3_layer = Conv2dLayer::new(0, final_conv_layer_cfg, batch_size, Some(&conv2_layer), &ctx);
  let softmax_layer = SoftmaxLossLayer::new(0, loss_layer_cfg, batch_size, Some(&conv3_layer));

  let mut arch = LinearNetArch::new(
      PathBuf::from("experiments/models/maddison_3_layers"),
      batch_size,
      data_layer,
      softmax_layer,
      vec![
        Box::new(conv1_layer),
        Box::new(conv2_layer),
        Box::new(conv3_layer),
      ],
  );
  for layer in arch.hidden_layers_forward() {
    layer.initialize_params(&ctx);
  }

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
  let mut state = OptState{epoch: 0, t: 0};
  sgd.train(&opt_cfg, &mut state, &mut arch, &mut *train_data_source, &mut *test_data_source, &ctx);
}
