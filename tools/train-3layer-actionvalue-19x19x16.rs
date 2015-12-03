extern crate async_cuda;
extern crate rembrandt;
extern crate rand;

use rembrandt::config::{ModelConfigFile};
use rembrandt::data::{DatasetConfiguration, DataSourceBuilder};
use rembrandt::layer::{
  Layer,
  ParamsInitialization, ActivationFunction,
  DataLayerConfig, Conv2dLayerConfig,
  SoftmaxLossLayerConfig,
  BinaryLogisticKLLossLayerConfig,
};
use rembrandt::layer::{
  DataLayer, Conv2dLayer, SoftmaxLossLayer, BinaryLogisticKLLossLayer,
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
    max_iters:      500000,
    momentum:       0.9,
    l2_reg_coef:    0.0,

    init_step_size: 0.01,
    anneal:         AnnealingPolicy::StepTwice{
                      /*first_step: 100000, second_step: 200000,
                      first_rate: 0.001, second_rate: 0.0001,*/
                      first_step:  100000, first_rate:  0.001,
                      second_step: 200000, second_rate: 0.0001,
                    },

    display_interval:   20,
    validate_interval:  800,
    save_interval:      Some(3200),
  };
  let descent = DescentSchedule::new(opt_cfg);

  let batch_size = 256;
  let num_hidden = 128;

  let data_layer_cfg = DataLayerConfig{
    raw_width: 19, raw_height: 19,
    crop_width: 19, crop_height: 19,
    channels: 16,
  };
  let conv1_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: 16,
    //conv_size: 5, conv_stride: 1, conv_pad: 2,
    conv_size: 9, conv_stride: 1, conv_pad: 4,
    out_channels: num_hidden,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::Uniform{half_range: 0.05},
  };
  let hidden_conv_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: num_hidden,
    conv_size: 3, conv_stride: 1, conv_pad: 1,
    out_channels: num_hidden,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::Uniform{half_range: 0.05},
  };
  let final_conv_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: num_hidden,
    conv_size: 3, conv_stride: 1, conv_pad: 1,
    out_channels: 1,
    act_fun: ActivationFunction::Identity,
    init_weights: ParamsInitialization::Uniform{half_range: 0.05},
  };
  /*let loss_layer_cfg = SoftmaxLossLayerConfig{
    num_categories: 361,
    do_mask: false,
  };*/
  let loss_layer_cfg = BinaryLogisticKLLossLayerConfig{
    num_categories: 361,
  };

  let data_layer = DataLayer::new(0, data_layer_cfg, batch_size);
  let conv1_layer = Conv2dLayer::new(0, conv1_layer_cfg, batch_size, Some(&data_layer), &ctx);
  let conv2_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv1_layer), &ctx);
  let conv3_layer = Conv2dLayer::new(0, final_conv_layer_cfg, batch_size, Some(&conv2_layer), &ctx);
  //let loss_layer = SoftmaxLossLayer::new(0, loss_layer_cfg, batch_size, Some(&conv3_layer));
  let loss_layer = BinaryLogisticKLLossLayer::new(0, loss_layer_cfg, batch_size, Some(&conv3_layer));
  let mut arch = LinearNetArch::new(
      //PathBuf::from("experiments/models/action_3layer_19x19x16.v2"),
      PathBuf::from("experiments/models/actionvalue_3layer_19x19x16.v2"),
      batch_size,
      data_layer,
      Box::new(loss_layer),
      vec![
        Box::new(conv1_layer),
        Box::new(conv2_layer),
        Box::new(conv3_layer),
      ],
  );
  arch.initialize_layer_params(&ctx);

  //let dataset_cfg = DatasetConfiguration::open(&PathBuf::from("experiments/gogodb_19x19x16.v2.data"));
  let dataset_cfg = DatasetConfiguration::open(&PathBuf::from("experiments/gogodb_actionvalue_19x19x16.v2.data"));
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
