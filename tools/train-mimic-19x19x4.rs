extern crate async_cuda;
extern crate rembrandt;
extern crate rand;

use rembrandt::config::{ModelConfigFile};
use rembrandt::data::{DatasetConfiguration, DataSourceBuilder};
use rembrandt::layer::{
  Layer,
  ParamsInitialization, ActivationFunction,
  DataLayerConfig, MimicAffineLayerConfig, Conv2dLayerConfig, SoftmaxLossLayerConfig, L2LossLayerConfig,
  DataLayer, MimicAffineLayer, Conv2dLayer, SoftmaxLossLayer, L2LossLayer,
};
use rembrandt::net::{NetArch, LinearNetArch};
use rembrandt::opt::{
  OptConfig, DescentSchedule, AnnealingPolicy,
  OptState, Optimizer, SgdOptimizer, MimicSgdOptimizer,
};

use async_cuda::context::{DeviceContext};

use rand::{Rng, thread_rng};
use std::path::{PathBuf};

const BATCH_SIZE: usize = 128;

fn main() {
  train_simba_2_layers();
}

fn build_target_arch(ctx: &DeviceContext) -> LinearNetArch {
  let batch_size = BATCH_SIZE;
  let num_hidden = 64;

  let data_layer_cfg = DataLayerConfig{
    raw_width: 19, raw_height: 19,
    crop_width: 19, crop_height: 19,
    channels: 4,
  };
  let conv1_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: 4,
    conv_size: 9, conv_stride: 1, conv_pad: 4,
    out_channels: num_hidden,
    act_fun: ActivationFunction::Rect,
    //init_weights: ParamsInitialization::Normal{mean: 0.0, std: 0.01},
    init_weights: ParamsInitialization::Uniform{half_range: 0.05},
  };
  let hidden_conv_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: num_hidden,
    conv_size: 3, conv_stride: 1, conv_pad: 1,
    out_channels: num_hidden,
    act_fun: ActivationFunction::Rect,
    //init_weights: ParamsInitialization::Normal{mean: 0.0, std: 0.01},
    init_weights: ParamsInitialization::Uniform{half_range: 0.05},
  };
  let final_conv_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: num_hidden,
    conv_size: 3, conv_stride: 1, conv_pad: 1,
    out_channels: 1,
    act_fun: ActivationFunction::Identity,
    //init_weights: ParamsInitialization::Normal{mean: 0.0, std: 0.01},
    init_weights: ParamsInitialization::Uniform{half_range: 0.05},
  };
  let loss_layer_cfg = SoftmaxLossLayerConfig{
    num_categories: 361,
    do_mask: false,
  };

  let data_layer = DataLayer::new(0, data_layer_cfg, batch_size);
  let conv1_layer = Conv2dLayer::new(0, conv1_layer_cfg, batch_size, Some(&data_layer), &ctx);
  let conv2_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv1_layer), &ctx);
  let conv3_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv2_layer), &ctx);
  let conv4_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv3_layer), &ctx);
  let conv5_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv4_layer), &ctx);
  let conv6_layer = Conv2dLayer::new(0, final_conv_layer_cfg, batch_size, Some(&conv5_layer), &ctx);
  //let conv6_layer = Conv2dLayer::new(0, final_conv_layer_cfg, batch_size, Some(&conv2_layer), &ctx);
  let softmax_layer = SoftmaxLossLayer::new(0, loss_layer_cfg, batch_size, Some(&conv6_layer));
  let mut arch = LinearNetArch::new(
      PathBuf::from("experiments/models/action_6layer_19x19x4.v2"),
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
      ],
  );
  arch.load_layer_params(None, &ctx);
  arch
}

fn build_mimic_arch(ctx: &DeviceContext) -> LinearNetArch {
  let batch_size = BATCH_SIZE;
  let low_rank = 256;
  let num_hidden = 16 * 1024;

  let data_layer_cfg = DataLayerConfig{
    raw_width: 19, raw_height: 19,
    crop_width: 19, crop_height: 19,
    channels: 4,
  };
  let mimic_layer_cfg = MimicAffineLayerConfig{
    in_channels: 19 * 19 * 4,
    bottleneck_rank: low_rank,
    hidden_channels: num_hidden,
    hidden_act_fun: ActivationFunction::Rect,
    out_channels: 361,
    init_weights: ParamsInitialization::Normal{mean: 0.0, std: 0.1},
  };
  let loss_layer_cfg = L2LossLayerConfig{
    num_channels: 361,
  };

  let data_layer = DataLayer::new(0, data_layer_cfg, batch_size);
  let mimic_layer = MimicAffineLayer::new(0, mimic_layer_cfg, batch_size, Some(&data_layer));
  let loss_layer = L2LossLayer::new(0, loss_layer_cfg, batch_size, Some(&mimic_layer));
  let mut arch = LinearNetArch::new(
      PathBuf::from("experiments/models/tmp_action_mimic_19x19x4.v2"),
      batch_size,
      data_layer,
      Box::new(loss_layer),
      vec![
        Box::new(mimic_layer),
      ],
  );
  arch.initialize_layer_params(&ctx);
  arch
}

fn train_simba_2_layers() {
  let mut rng = thread_rng();
  let ctx = DeviceContext::new(0);

  let opt_cfg = OptConfig{
    minibatch_size: BATCH_SIZE,
    max_iters:      500000,
    momentum:       0.0,
    l2_reg_coef:    0.0,
    //init_step_size: 0.01,
    //anneal:         AnnealingPolicy::Step{step_iters: 6144, decay: 0.1},
    //anneal:         AnnealingPolicy::Step{step_iters: 12288, decay: 0.1},
    //anneal:         AnnealingPolicy::StepOnce{step_iters: 3072, new_rate: 0.01},

    /*init_step_size: 0.1,
    anneal:         AnnealingPolicy::StepTwice{
                      first_step: 3072, second_step: 6144,
                      first_rate: 0.01, second_rate: 0.001,
                    },*/

    init_step_size: 0.01,
    anneal:         AnnealingPolicy::StepTwice{
                      /*first_step: 100000, second_step: 200000,
                      first_rate: 0.001, second_rate: 0.0001,*/
                      first_step:  100000, first_rate:  0.0001,
                      second_step: 200000, second_rate: 0.0001,
                    },

    display_interval:   20,
    validate_interval:  800,
    save_interval:      Some(12800),
  };
  let descent = DescentSchedule::new(opt_cfg);

  let mut target_arch = build_target_arch(&ctx);
  let mut mimic_arch = build_mimic_arch(&ctx);

  let dataset_cfg = DatasetConfiguration::open(&PathBuf::from("experiments/gogodb_19x19x4.v2.data"));
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

  let sgd = MimicSgdOptimizer;
  let mut state = OptState{epoch: 0, t: 0};
  sgd.train(&opt_cfg, &mut state, &mut target_arch, &mut mimic_arch, &mut *train_data_source, &mut *test_data_source, &ctx);
}
