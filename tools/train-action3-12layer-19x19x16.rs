extern crate async_cuda;
extern crate holmes;
extern crate rembrandt;
extern crate rand;

use holmes::convnet::data::{CyclicRandomEpisodeSamples};
use rembrandt::config::{ModelConfigFile};
use rembrandt::data::{DatasetConfiguration, DataSource, DataSourceBuilder, SampleLabelConfig};
use rembrandt::layer::{
  Layer,
  ParamsInitialization, ActivationFunction,
  DataLayerConfig, Conv2dLayerConfig,
};
use rembrandt::layer::{
  DataLayer, Conv2dLayer,
  MultiSoftmaxKLLossLayer, MultiSoftmaxKLLossLayerConfig,
};
use rembrandt::net::{NetArch, LinearNetArch};
use rembrandt::opt::{
  OptConfig, DescentSchedule, AnnealingPolicy,
  OptState, Optimizer, SgdOptimizer2,
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

  let batch_size = 256;
  let num_hidden = 128;

  let opt_cfg = OptConfig{
    minibatch_size: batch_size,
    max_iters:      1500000,
    momentum:       0.0,
    l2_reg_coef:    0.0,

    init_step_size: 0.05,
    anneal:         AnnealingPolicy::StepTwice{
                      first_step:  500000, first_rate:  0.005,
                      second_step: 1000000, second_rate: 0.0005,
                    },

    display_interval:   20,
    validate_interval:  800,
    save_interval:      Some(3200),
  };
  let descent = DescentSchedule::new(opt_cfg);

  let data_layer_cfg = DataLayerConfig{
    raw_width: 19, raw_height: 19,
    crop_width: 19, crop_height: 19,
    channels: 16,
  };
  let conv1_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: 16,
    //conv_size: 9, conv_stride: 1, conv_pad: 4,
    conv_size: 5, conv_stride: 1, conv_pad: 2,
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
    out_channels: 3,
    act_fun: ActivationFunction::Identity,
    init_weights: ParamsInitialization::Uniform{half_range: 0.05},
  };
  //let loss_layer_cfg = SoftmaxLossLayerConfig{
  let loss_layer_cfg = MultiSoftmaxKLLossLayerConfig{
    num_categories: 361,
    num_train_labels: 3,
    //do_mask: false,
  };

  let data_layer = DataLayer::new(0, data_layer_cfg, batch_size);
  let conv1_layer = Conv2dLayer::new(0, conv1_layer_cfg, batch_size, Some(&data_layer), &ctx);
  let conv2_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv1_layer), &ctx);
  let conv3_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv2_layer), &ctx);
  let conv4_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv3_layer), &ctx);
  let conv5_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv4_layer), &ctx);
  let conv6_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv5_layer), &ctx);
  let conv7_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv6_layer), &ctx);
  let conv8_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv7_layer), &ctx);
  let conv9_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv8_layer), &ctx);
  let conv10_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv9_layer), &ctx);
  let conv11_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv10_layer), &ctx);
  let conv12_layer = Conv2dLayer::new(0, final_conv_layer_cfg, batch_size, Some(&conv11_layer), &ctx);
  let softmax_layer = MultiSoftmaxKLLossLayer::new(0, loss_layer_cfg, batch_size, Some(&conv12_layer));
  let mut arch = LinearNetArch::new(
      PathBuf::from("experiments/models/tmp_action3_12layer_19x19x16.v2"),
      batch_size,
      data_layer,
      Box::new(softmax_layer),
      vec![
        Box::new(conv1_layer),
        Box::new(conv2_layer),
        Box::new(conv3_layer),
        Box::new(conv4_layer),
        Box::new(conv5_layer),
        Box::new(conv6_layer),
        Box::new(conv7_layer),
        Box::new(conv8_layer),
        Box::new(conv9_layer),
        Box::new(conv10_layer),
        Box::new(conv11_layer),
        Box::new(conv12_layer),
      ],
  );
  arch.initialize_layer_params(&ctx);

  let dataset_cfg = DatasetConfiguration::open(&PathBuf::from("experiments/gogodb_19x19x16_episode.v2.data"));
  let bare_train_data_source = if let Some(&(ref name, ref cfg)) = dataset_cfg.datasets.get("train") {
    DataSourceBuilder::build(name, cfg.clone())
  } else {
    panic!("missing train data source!");
  };
  let mut test_data_source = if let Some(&(ref name, ref cfg)) = dataset_cfg.datasets.get("valid") {
    DataSourceBuilder::build(name, cfg.clone())
  } else {
    panic!("missing validation data source!");
  };

  let mut train_data_source = CyclicRandomEpisodeSamples::new(bare_train_data_source);

  let sgd = SgdOptimizer2;
  let mut state = OptState{epoch: 0, t: 0};
  sgd.train(SampleLabelConfig::Lookahead{lookahead: 3}, &opt_cfg, &mut state, &mut arch, &mut train_data_source, &mut *test_data_source, &ctx);
}
