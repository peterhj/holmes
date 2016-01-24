use async_cuda::context::{DeviceContext};
use rembrandt::layer::{
  ActivationFunction, ParamsInitialization, Layer,
  DataLayerConfig, DataLayer,
  Conv2dLayerConfig, Conv2dLayer,
  SoftmaxLossLayerConfig, SoftmaxLossLayer,
  MultiSoftmaxKLLossLayer, MultiSoftmaxKLLossLayerConfig,
};
use rembrandt::net::{NetArch, LinearNetArch};

use std::path::{PathBuf};

pub fn build_pgbalance_action_2layer_19x19x16_arch(batch_size: usize, ctx: &DeviceContext) -> LinearNetArch {
  let num_hidden = 16;
  let data_layer_cfg = DataLayerConfig{
    raw_width: 19, raw_height: 19,
    crop_width: 19, crop_height: 19,
    channels: 16,
  };
  let conv1_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: 16,
    conv_size: 9, conv_stride: 1, conv_pad: 4,
    out_channels: num_hidden,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::None,
  };
  let final_conv_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: num_hidden,
    conv_size: 3, conv_stride: 1, conv_pad: 1,
    out_channels: 1,
    act_fun: ActivationFunction::Identity,
    init_weights: ParamsInitialization::None,
  };
  let loss_layer_cfg = SoftmaxLossLayerConfig{
    num_categories: 361,
    do_mask: false,
  };

  let data_layer = DataLayer::new(0, data_layer_cfg, batch_size);
  let conv1_layer = Conv2dLayer::new(0, conv1_layer_cfg, batch_size, Some(&data_layer), &ctx);
  let conv2_layer = Conv2dLayer::new(0, final_conv_layer_cfg, batch_size, Some(&conv1_layer), &ctx);
  let softmax_layer = SoftmaxLossLayer::new(0, loss_layer_cfg, batch_size, Some(&conv2_layer));
  let mut arch = LinearNetArch::new(
      PathBuf::from("models/rollout_2layer_pgbalance.saved"),
      batch_size,
      data_layer,
      Box::new(softmax_layer),
      vec![
        Box::new(conv1_layer),
        Box::new(conv2_layer),
      ],
  );
  arch.load_layer_params(None, &ctx);
  arch
}

pub fn build_action_3layer_arch(batch_size: usize, ctx: &DeviceContext) -> LinearNetArch {
  let num_hidden = 128;

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
  let loss_layer_cfg = SoftmaxLossLayerConfig{
    num_categories: 361,
    do_mask: false,
  };

  let data_layer = DataLayer::new(0, data_layer_cfg, batch_size);
  let conv1_layer = Conv2dLayer::new(0, conv1_layer_cfg, batch_size, Some(&data_layer), &ctx);
  let conv2_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv1_layer), &ctx);
  let conv3_layer = Conv2dLayer::new(0, final_conv_layer_cfg, batch_size, Some(&conv2_layer), &ctx);
  let softmax_layer = SoftmaxLossLayer::new(0, loss_layer_cfg, batch_size, Some(&conv3_layer));
  let mut arch = LinearNetArch::new(
      PathBuf::from("experiments/models/action_3layer_19x19x4.v2.saved"),
      batch_size,
      data_layer,
      Box::new(softmax_layer),
      vec![
        Box::new(conv1_layer),
        Box::new(conv2_layer),
        Box::new(conv3_layer),
      ],
  );
  arch.load_layer_params(None, &ctx);
  arch
}

pub fn build_action_6layer_arch(batch_size: usize, ctx: &DeviceContext) -> LinearNetArch {
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
    init_weights: ParamsInitialization::None,
  };
  let hidden_conv_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: num_hidden,
    conv_size: 3, conv_stride: 1, conv_pad: 1,
    out_channels: num_hidden,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::None,
  };
  let final_conv_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: num_hidden,
    conv_size: 3, conv_stride: 1, conv_pad: 1,
    out_channels: 1,
    act_fun: ActivationFunction::Identity,
    init_weights: ParamsInitialization::None,
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
  let softmax_layer = SoftmaxLossLayer::new(0, loss_layer_cfg, batch_size, Some(&conv6_layer));
  let mut arch = LinearNetArch::new(
      PathBuf::from("experiments/models/action_6layer_19x19x4.v2.saved"),
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

pub fn build_action_2layer_19x19x16_arch(batch_size: usize, ctx: &DeviceContext) -> LinearNetArch {
  let num_hidden = 16;
  let data_layer_cfg = DataLayerConfig{
    raw_width: 19, raw_height: 19,
    crop_width: 19, crop_height: 19,
    channels: 16,
  };
  let conv1_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: 16,
    conv_size: 9, conv_stride: 1, conv_pad: 4,
    out_channels: num_hidden,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::None,
  };
  let final_conv_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: num_hidden,
    conv_size: 3, conv_stride: 1, conv_pad: 1,
    out_channels: 1,
    act_fun: ActivationFunction::Identity,
    init_weights: ParamsInitialization::None,
  };
  let loss_layer_cfg = SoftmaxLossLayerConfig{
    num_categories: 361,
    do_mask: false,
  };

  let data_layer = DataLayer::new(0, data_layer_cfg, batch_size);
  let conv1_layer = Conv2dLayer::new(0, conv1_layer_cfg, batch_size, Some(&data_layer), &ctx);
  let conv2_layer = Conv2dLayer::new(0, final_conv_layer_cfg, batch_size, Some(&conv1_layer), &ctx);
  let softmax_layer = SoftmaxLossLayer::new(0, loss_layer_cfg, batch_size, Some(&conv2_layer));
  let mut arch = LinearNetArch::new(
      //PathBuf::from("experiments/models/action_2layer_19x19x16.v2.saved"),
      PathBuf::from("models/action_2layer_19x19x16.v2.saved"),
      batch_size,
      data_layer,
      Box::new(softmax_layer),
      vec![
        Box::new(conv1_layer),
        Box::new(conv2_layer),
      ],
  );
  arch.load_layer_params(None, &ctx);
  arch
}

pub fn build_action_narrow_3layer_19x19x16_arch(batch_size: usize, ctx: &DeviceContext) -> LinearNetArch {
  let num_hidden = 16;
  let data_layer_cfg = DataLayerConfig{
    raw_width: 19, raw_height: 19,
    crop_width: 19, crop_height: 19,
    channels: 16,
  };
  let conv1_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: 16,
    conv_size: 9, conv_stride: 1, conv_pad: 4,
    out_channels: num_hidden,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::None,
  };
  let hidden_conv_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: num_hidden,
    conv_size: 3, conv_stride: 1, conv_pad: 1,
    out_channels: num_hidden,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::None,
  };
  let final_conv_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: num_hidden,
    conv_size: 3, conv_stride: 1, conv_pad: 1,
    out_channels: 1,
    act_fun: ActivationFunction::Identity,
    init_weights: ParamsInitialization::None,
  };
  let loss_layer_cfg = SoftmaxLossLayerConfig{
    num_categories: 361,
    do_mask: false,
  };

  let data_layer = DataLayer::new(0, data_layer_cfg, batch_size);
  let conv1_layer = Conv2dLayer::new(0, conv1_layer_cfg, batch_size, Some(&data_layer), &ctx);
  let conv2_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv1_layer), &ctx);
  let conv3_layer = Conv2dLayer::new(0, final_conv_layer_cfg, batch_size, Some(&conv2_layer), &ctx);
  let softmax_layer = SoftmaxLossLayer::new(0, loss_layer_cfg, batch_size, Some(&conv3_layer));
  let mut arch = LinearNetArch::new(
      PathBuf::from("experiments/models/action_narrow_3layer_19x19x16.v2.saved"),
      batch_size,
      data_layer,
      Box::new(softmax_layer),
      vec![
        Box::new(conv1_layer),
        Box::new(conv2_layer),
        Box::new(conv3_layer),
      ],
  );
  arch.load_layer_params(None, &ctx);
  arch
}

pub fn build_action_3layer_19x19x16_arch(batch_size: usize, ctx: &DeviceContext) -> LinearNetArch {
  let num_hidden = 128;
  let data_layer_cfg = DataLayerConfig{
    raw_width: 19, raw_height: 19,
    crop_width: 19, crop_height: 19,
    channels: 16,
  };
  let conv1_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: 16,
    conv_size: 9, conv_stride: 1, conv_pad: 4,
    out_channels: num_hidden,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::None,
  };
  let hidden_conv_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: num_hidden,
    conv_size: 3, conv_stride: 1, conv_pad: 1,
    out_channels: num_hidden,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::None,
  };
  let final_conv_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: num_hidden,
    conv_size: 3, conv_stride: 1, conv_pad: 1,
    out_channels: 1,
    act_fun: ActivationFunction::Identity,
    init_weights: ParamsInitialization::None,
  };
  let loss_layer_cfg = SoftmaxLossLayerConfig{
    num_categories: 361,
    do_mask: false,
  };

  let data_layer = DataLayer::new(0, data_layer_cfg, batch_size);
  let conv1_layer = Conv2dLayer::new(0, conv1_layer_cfg, batch_size, Some(&data_layer), &ctx);
  let conv2_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv1_layer), &ctx);
  let conv3_layer = Conv2dLayer::new(0, final_conv_layer_cfg, batch_size, Some(&conv2_layer), &ctx);
  let softmax_layer = SoftmaxLossLayer::new(0, loss_layer_cfg, batch_size, Some(&conv3_layer));
  let mut arch = LinearNetArch::new(
      PathBuf::from("experiments/models/action_3layer_19x19x16.v2.saved"),
      batch_size,
      data_layer,
      Box::new(softmax_layer),
      vec![
        Box::new(conv1_layer),
        Box::new(conv2_layer),
        Box::new(conv3_layer),
      ],
  );
  arch.load_layer_params(None, &ctx);
  arch
}

pub fn build_action_6layer_19x19x16_arch(batch_size: usize, ctx: &DeviceContext) -> LinearNetArch {
  let num_hidden = 128;
  let data_layer_cfg = DataLayerConfig{
    raw_width: 19, raw_height: 19,
    crop_width: 19, crop_height: 19,
    channels: 16,
  };
  let conv1_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: 16,
    conv_size: 9, conv_stride: 1, conv_pad: 4,
    out_channels: num_hidden,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::None,
  };
  let hidden_conv_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: num_hidden,
    conv_size: 3, conv_stride: 1, conv_pad: 1,
    out_channels: num_hidden,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::None,
  };
  let final_conv_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: num_hidden,
    conv_size: 3, conv_stride: 1, conv_pad: 1,
    out_channels: 1,
    act_fun: ActivationFunction::Identity,
    init_weights: ParamsInitialization::None,
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
  let softmax_layer = SoftmaxLossLayer::new(0, loss_layer_cfg, batch_size, Some(&conv6_layer));
  let mut arch = LinearNetArch::new(
      PathBuf::from("experiments/models/action_6layer_19x19x16.v2.saved"),
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

pub fn build_action_12layer_19x19x16_arch(batch_size: usize, ctx: &DeviceContext) -> LinearNetArch {
  let num_hidden = 128;
  let data_layer_cfg = DataLayerConfig{
    raw_width: 19, raw_height: 19,
    crop_width: 19, crop_height: 19,
    channels: 16,
  };
  let conv1_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: 16,
    conv_size: 5, conv_stride: 1, conv_pad: 2,
    out_channels: num_hidden,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::None,
  };
  let hidden_conv_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: num_hidden,
    conv_size: 3, conv_stride: 1, conv_pad: 1,
    out_channels: num_hidden,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::None,
  };
  let final_conv_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: num_hidden,
    conv_size: 3, conv_stride: 1, conv_pad: 1,
    out_channels: 1,
    act_fun: ActivationFunction::Identity,
    init_weights: ParamsInitialization::None,
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
  let conv6_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv5_layer), &ctx);
  let conv7_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv6_layer), &ctx);
  let conv8_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv7_layer), &ctx);
  let conv9_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv8_layer), &ctx);
  let conv10_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv9_layer), &ctx);
  let conv11_layer = Conv2dLayer::new(0, hidden_conv_layer_cfg, batch_size, Some(&conv10_layer), &ctx);
  let conv12_layer = Conv2dLayer::new(0, final_conv_layer_cfg, batch_size, Some(&conv11_layer), &ctx);
  let softmax_layer = SoftmaxLossLayer::new(0, loss_layer_cfg, batch_size, Some(&conv12_layer));
  let mut arch = LinearNetArch::new(
      PathBuf::from("models/action_12layer_19x19x16.v2.saved"),
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
  arch.load_layer_params(None, &ctx);
  arch
}

pub fn build_action3_narrow_3layer_19x19x16_arch(batch_size: usize, ctx: &DeviceContext) -> LinearNetArch {
  let num_hidden = 16;

  let data_layer_cfg = DataLayerConfig{
    raw_width: 19, raw_height: 19,
    crop_width: 19, crop_height: 19,
    channels: 16,
  };
  let conv1_layer_cfg = Conv2dLayerConfig{
    in_width: 19, in_height: 19, in_channels: 16,
    conv_size: 9, conv_stride: 1, conv_pad: 4,
    //conv_size: 5, conv_stride: 1, conv_pad: 2,
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
  let conv12_layer = Conv2dLayer::new(0, final_conv_layer_cfg, batch_size, Some(&conv2_layer), &ctx);
  let softmax_layer = MultiSoftmaxKLLossLayer::new(0, loss_layer_cfg, batch_size, Some(&conv12_layer));
  let mut arch = LinearNetArch::new(
      PathBuf::from("experiments/models/action3_narrow_3layer_19x19x16.v2.saved"),
      batch_size,
      data_layer,
      Box::new(softmax_layer),
      vec![
        Box::new(conv1_layer),
        Box::new(conv2_layer),
        Box::new(conv12_layer),
      ],
  );
  arch.load_layer_params(None, &ctx);
  arch
}

pub fn build_action3_12layer_19x19x16_arch(batch_size: usize, ctx: &DeviceContext) -> LinearNetArch {
  let num_hidden = 128;

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
      PathBuf::from("experiments/models/action3_12layer_19x19x16.v2.saved"),
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
  arch.load_layer_params(None, &ctx);
  arch
}
