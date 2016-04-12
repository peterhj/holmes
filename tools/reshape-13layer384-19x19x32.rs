extern crate rembrandt;

use rembrandt::arch_new::{ReshapePipelineArchConfigs, PipelineArchConfig};
use rembrandt::layer_new::{
  ActivationFunction, ParamsInitialization,
  Data3dLayerConfig,
  Conv2dLayerConfig,
  CategoricalLossLayerConfig,
  MultiCategoricalLossLayerConfig,
};

use std::path::{PathBuf};

fn main() {
  let input_channels = 32;
  let conv1_channels = 96;
  let hidden_channels = 384;
  let lookahead = 3;
  let data_layer_cfg = Data3dLayerConfig{
    dims:           (19, 19, input_channels),
    normalize:      true,
  };
  let conv1_layer_cfg = Conv2dLayerConfig{
    in_dims:        (19, 19, input_channels),
    conv_size:      5,
    conv_stride:    1,
    conv_pad:       2,
    out_channels:   conv1_channels,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInitialization::Uniform{half_range: 0.05},
  };
  let conv2_layer_cfg = Conv2dLayerConfig{
    in_dims:        (19, 19, conv1_channels),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   hidden_channels,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInitialization::Uniform{half_range: 0.05},
  };
  let inner_conv_layer_cfg = Conv2dLayerConfig{
    in_dims:        (19, 19, hidden_channels),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   hidden_channels,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInitialization::Uniform{half_range: 0.05},
  };
  let final_conv_layer_cfg = Conv2dLayerConfig{
    in_dims:        (19, 19, hidden_channels),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   1,
    act_func:       ActivationFunction::Identity,
    init_weights:   ParamsInitialization::Uniform{half_range: 0.05},
  };
  let multi_final_conv_layer_cfg = Conv2dLayerConfig{
    in_dims:        (19, 19, hidden_channels),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   lookahead,
    act_func:       ActivationFunction::Identity,
    init_weights:   ParamsInitialization::Uniform{half_range: 0.05},
  };
  let loss_layer_cfg = CategoricalLossLayerConfig{
    num_categories:     361,
  };
  let multi_loss_layer_cfg = MultiCategoricalLossLayerConfig{
    num_categories:     361,
    train_lookahead:    lookahead,
    infer_lookahead:    1,
  };

  let mut new_arch = PipelineArchConfig::new();
  new_arch
    .data3d(data_layer_cfg)
    .conv2d(conv1_layer_cfg)
    .conv2d(conv2_layer_cfg)
    .conv2d(inner_conv_layer_cfg)
    .conv2d(inner_conv_layer_cfg)
    .conv2d(inner_conv_layer_cfg)
    .conv2d(inner_conv_layer_cfg)
    .conv2d(inner_conv_layer_cfg)
    .conv2d(inner_conv_layer_cfg)
    .conv2d(inner_conv_layer_cfg)
    .conv2d(inner_conv_layer_cfg)
    .conv2d(inner_conv_layer_cfg)
    .conv2d(inner_conv_layer_cfg)
    .conv2d(final_conv_layer_cfg)
    .softmax_kl_loss(loss_layer_cfg);

  let mut old_arch = PipelineArchConfig::new();
  old_arch
    .data3d(data_layer_cfg)
    .conv2d(conv1_layer_cfg)
    .conv2d(conv2_layer_cfg)
    .conv2d(inner_conv_layer_cfg)
    .conv2d(inner_conv_layer_cfg)
    .conv2d(inner_conv_layer_cfg)
    .conv2d(inner_conv_layer_cfg)
    .conv2d(inner_conv_layer_cfg)
    .conv2d(inner_conv_layer_cfg)
    .conv2d(inner_conv_layer_cfg)
    .conv2d(inner_conv_layer_cfg)
    .conv2d(inner_conv_layer_cfg)
    .conv2d(inner_conv_layer_cfg)
    .conv2d(multi_final_conv_layer_cfg)
    .multi_softmax_kl_loss(multi_loss_layer_cfg);

  let reshape = ReshapePipelineArchConfigs::new(old_arch, new_arch);
  reshape.reshape_params(
      &PathBuf::from("models/gogodb_w2015-preproc-alphav3m_19x19x32_13layer384multi3.saved/layer_params.latest.blob"),
      &PathBuf::from("models/layer_params.blob"),
  );
}
