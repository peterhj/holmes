use array_cuda::device::{DeviceCtxRef};
use rembrandt::arch_new::{
  PipelineArchConfig, PipelineArchWorker,
};
use rembrandt::layer_new::{
  ActivationFunction,
  ParamsInitialization,
  ConvBackend,
  Data3dLayerConfig,
  Conv2dLayerConfig,
  Conv2dLayerConfigV2,
  CategoricalLossLayerConfig,
  MultiCategoricalLossLayerConfig,
};

use std::path::{PathBuf};

pub fn build_2layer16_19x19x16_arch(batch_size: usize) -> (PipelineArchConfig, PathBuf) {
  let arch_cfg = build_2layer16_19x19x16_arch_nodir(batch_size);
  let save_path = PathBuf::from("experiments/models/action_2layer_19x19x16.v2.saved");
  (arch_cfg, save_path)
}

pub fn build_2layer16_19x19x16_arch_nodir(batch_size: usize) -> PipelineArchConfig {
  let input_channels = 16;

  let data_layer_cfg = Data3dLayerConfig{
    dims:           (19, 19, input_channels),
    normalize:      false,
  };
  let conv1_layer_cfg = Conv2dLayerConfig{
    in_dims:        (19, 19, input_channels),
    conv_size:      9,
    conv_stride:    1,
    conv_pad:       4,
    out_channels:   16,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInitialization::None,
  };
  let conv2_layer_cfg = Conv2dLayerConfig{
    in_dims:        (19, 19, 16),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   1,
    act_func:       ActivationFunction::Identity,
    init_weights:   ParamsInitialization::None,
  };
  let loss_layer_cfg = CategoricalLossLayerConfig{
    num_categories: 361,
  };

  let mut arch_cfg = PipelineArchConfig::new();
  arch_cfg
    .data3d(data_layer_cfg)
    .conv2d(conv1_layer_cfg)
    .conv2d(conv2_layer_cfg)
    .softmax_kl_loss(loss_layer_cfg);

  arch_cfg
}

pub fn build_12layer128_19x19x16_arch(batch_size: usize) -> (PipelineArchConfig, PathBuf) {
  let input_channels = 16;
  let conv1_channels = 128;
  let hidden_channels = 128;

  let data_layer_cfg = Data3dLayerConfig{
    dims:           (19, 19, input_channels),
    normalize:      false,
  };
  let conv1_layer_cfg = Conv2dLayerConfig{
    in_dims:        (19, 19, input_channels),
    conv_size:      5,
    conv_stride:    1,
    conv_pad:       2,
    out_channels:   conv1_channels,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInitialization::None,
  };
  let conv2_layer_cfg = Conv2dLayerConfig{
    in_dims:        (19, 19, conv1_channels),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   hidden_channels,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInitialization::None,
  };
  let inner_conv_layer_cfg = Conv2dLayerConfig{
    in_dims:        (19, 19, hidden_channels),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   hidden_channels,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInitialization::None,
  };
  let final_conv_layer_cfg = Conv2dLayerConfig{
    in_dims:        (19, 19, hidden_channels),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   1,
    act_func:       ActivationFunction::Identity,
    init_weights:   ParamsInitialization::None,
  };
  let loss_layer_cfg = CategoricalLossLayerConfig{
    num_categories: 361,
  };

  let mut arch_cfg = PipelineArchConfig::new();
  arch_cfg
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
    .conv2d(final_conv_layer_cfg)
    .softmax_kl_loss(loss_layer_cfg);

  let save_path = PathBuf::from("experiments/models/action_12layer_19x19x16.v2.saved");

  (arch_cfg, save_path)
}

pub fn build_12layer384_19x19x37_arch_nodir(batch_size: usize) -> PipelineArchConfig {
  let input_channels = 37;
  let conv1_channels = 96;
  let hidden_channels = 384;

  let data_layer_cfg = Data3dLayerConfig{
    dims:           (19, 19, input_channels),
    normalize:      false,
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
    out_channels:   3,
    act_func:       ActivationFunction::Identity,
    init_weights:   ParamsInitialization::Uniform{half_range: 0.05},
  };
  let loss_layer_cfg = MultiCategoricalLossLayerConfig{
    num_categories:     361,
    train_lookahead:    3,
    infer_lookahead:    1,
  };

  let mut arch_cfg = PipelineArchConfig::new();
  arch_cfg
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
    .conv2d(final_conv_layer_cfg)
    .multi_softmax_kl_loss(loss_layer_cfg);

  arch_cfg
}

pub fn build_2layer16_19x19x44_arch_nodir(batch_size: usize) -> PipelineArchConfig {
  let input_channels = 44;
  let hidden_channels = 16;

  let data_layer_cfg = Data3dLayerConfig{
    dims:           (19, 19, input_channels),
    normalize:      true,
  };
  let conv1_layer_cfg = Conv2dLayerConfig{
    in_dims:        (19, 19, input_channels),
    conv_size:      9,
    conv_stride:    1,
    conv_pad:       4,
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
  /*let loss_layer_cfg = MultiCategoricalLossLayerConfig{
    num_categories:     361,
    train_lookahead:    3,
    infer_lookahead:    1,
  };*/
  let loss_layer_cfg = CategoricalLossLayerConfig{
    num_categories:     361,
  };

  let mut arch_cfg = PipelineArchConfig::new();
  arch_cfg
    .data3d(data_layer_cfg)
    .conv2d(conv1_layer_cfg)
    .conv2d(final_conv_layer_cfg)
    //.multi_softmax_kl_loss(loss_layer_cfg);
    .softmax_kl_loss(loss_layer_cfg);

  arch_cfg
}

pub fn build_3layer32_19x19x44_arch_nodir(batch_size: usize) -> PipelineArchConfig {
  let input_channels = 44;
  let hidden_channels = 32;

  let data_layer_cfg = Data3dLayerConfig{
    dims:           (19, 19, input_channels),
    normalize:      true,
  };
  let conv1_layer_cfg = Conv2dLayerConfig{
    in_dims:        (19, 19, input_channels),
    conv_size:      5,
    conv_stride:    1,
    conv_pad:       2,
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
  let loss_layer_cfg = CategoricalLossLayerConfig{
    num_categories:     361,
  };

  let mut arch_cfg = PipelineArchConfig::new();
  arch_cfg
    .data3d(data_layer_cfg)
    .conv2d(conv1_layer_cfg)
    .conv2d(inner_conv_layer_cfg)
    .conv2d(final_conv_layer_cfg)
    .softmax_kl_loss(loss_layer_cfg);

  arch_cfg
}

pub fn build_12layer384_19x19x44_arch_nodir(batch_size: usize) -> PipelineArchConfig {
  let input_channels = 44;
  let conv1_channels = 96;
  let hidden_channels = 384;

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
    out_channels:   3,
    act_func:       ActivationFunction::Identity,
    init_weights:   ParamsInitialization::Uniform{half_range: 0.05},
  };
  let loss_layer_cfg = MultiCategoricalLossLayerConfig{
    num_categories:     361,
    train_lookahead:    3,
    infer_lookahead:    1,
  };

  let mut arch_cfg = PipelineArchConfig::new();
  arch_cfg
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
    .conv2d(final_conv_layer_cfg)
    .multi_softmax_kl_loss(loss_layer_cfg);

  arch_cfg
}

pub fn build_2layer16_5x5_19x19x16_arch_nodir(batch_size: usize) -> PipelineArchConfig {
  let input_channels = 16;
  let hidden_channels = 16;

  let data_layer_cfg = Data3dLayerConfig{
    dims:           (19, 19, input_channels),
    normalize:      true,
  };
  let conv1_layer_cfg = Conv2dLayerConfigV2{
    in_dims:        (19, 19, input_channels),
    conv_size:      5,
    conv_stride:    1,
    conv_pad:       2,
    out_channels:   hidden_channels,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInitialization::Uniform{half_range: 0.05},
    backend:        ConvBackend::CudnnFftTiling,
  };
  /*let inner_conv_layer_cfg = Conv2dLayerConfig{
    in_dims:        (19, 19, hidden_channels),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   hidden_channels,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInitialization::Uniform{half_range: 0.05},
  };*/
  let final_conv_layer_cfg = Conv2dLayerConfigV2{
    in_dims:        (19, 19, hidden_channels),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   1,
    act_func:       ActivationFunction::Identity,
    init_weights:   ParamsInitialization::Uniform{half_range: 0.05},
    backend:        ConvBackend::CudnnImplicitPrecompGemm,
  };
  let loss_layer_cfg = CategoricalLossLayerConfig{
    num_categories:     361,
  };

  let mut arch_cfg = PipelineArchConfig::new();
  arch_cfg
    .data3d(data_layer_cfg)
    .conv2d_v2(conv1_layer_cfg)
    //.conv2d(inner_conv_layer_cfg)
    .conv2d_v2(final_conv_layer_cfg)
    .softmax_kl_loss(loss_layer_cfg);

  arch_cfg
}

pub fn build_2layer16_9x9_19x19x16_arch_nodir(batch_size: usize) -> PipelineArchConfig {
  let input_channels = 16;
  let hidden_channels = 16;

  let data_layer_cfg = Data3dLayerConfig{
    dims:           (19, 19, input_channels),
    normalize:      true,
  };
  let conv1_layer_cfg = Conv2dLayerConfig{
    in_dims:        (19, 19, input_channels),
    conv_size:      9,
    conv_stride:    1,
    conv_pad:       4,
    out_channels:   hidden_channels,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInitialization::Uniform{half_range: 0.05},
  };
  /*let inner_conv_layer_cfg = Conv2dLayerConfig{
    in_dims:        (19, 19, hidden_channels),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   hidden_channels,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInitialization::Uniform{half_range: 0.05},
  };*/
  let final_conv_layer_cfg = Conv2dLayerConfig{
    in_dims:        (19, 19, hidden_channels),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   1,
    act_func:       ActivationFunction::Identity,
    init_weights:   ParamsInitialization::Uniform{half_range: 0.05},
  };
  let loss_layer_cfg = CategoricalLossLayerConfig{
    num_categories:     361,
  };

  let mut arch_cfg = PipelineArchConfig::new();
  arch_cfg
    .data3d(data_layer_cfg)
    .conv2d(conv1_layer_cfg)
    //.conv2d(inner_conv_layer_cfg)
    .conv2d(final_conv_layer_cfg)
    .softmax_kl_loss(loss_layer_cfg);

  arch_cfg
}

pub fn build_3layer32_19x19x16_arch_nodir(batch_size: usize) -> PipelineArchConfig {
  let input_channels = 16;
  let hidden_channels = 32;

  let data_layer_cfg = Data3dLayerConfig{
    dims:           (19, 19, input_channels),
    normalize:      true,
  };
  let conv1_layer_cfg = Conv2dLayerConfig{
    in_dims:        (19, 19, input_channels),
    conv_size:      5,
    conv_stride:    1,
    conv_pad:       2,
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
  let loss_layer_cfg = CategoricalLossLayerConfig{
    num_categories:     361,
  };

  let mut arch_cfg = PipelineArchConfig::new();
  arch_cfg
    .data3d(data_layer_cfg)
    .conv2d(conv1_layer_cfg)
    .conv2d(inner_conv_layer_cfg)
    .conv2d(final_conv_layer_cfg)
    .softmax_kl_loss(loss_layer_cfg);

  arch_cfg
}

pub fn build_3layer32_19x19x16_ind_arch_nodir(batch_size: usize) -> PipelineArchConfig {
  let input_channels = 16;
  let hidden_channels = 32;

  let data_layer_cfg = Data3dLayerConfig{
    dims:           (19, 19, input_channels),
    normalize:      true,
  };
  let conv1_layer_cfg = Conv2dLayerConfig{
    in_dims:        (19, 19, input_channels),
    conv_size:      5,
    conv_stride:    1,
    conv_pad:       2,
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
  let loss_layer_cfg = CategoricalLossLayerConfig{
    num_categories:     361,
  };

  let mut arch_cfg = PipelineArchConfig::new();
  arch_cfg
    .data3d(data_layer_cfg)
    .conv2d(conv1_layer_cfg)
    .conv2d(inner_conv_layer_cfg)
    .conv2d(final_conv_layer_cfg)
    .softmax_ind_loss(loss_layer_cfg);

  arch_cfg
}

pub fn build_13layer384_19x19x32_arch_nodir(batch_size: usize) -> PipelineArchConfig {
  let input_channels = 32;
  let conv1_channels = 96;
  let hidden_channels = 384;

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
  let loss_layer_cfg = CategoricalLossLayerConfig{
    num_categories:     361,
  };

  let mut arch_cfg = PipelineArchConfig::new();
  arch_cfg
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

  arch_cfg
}

pub fn build_13layer384multi3_19x19x32_arch_nodir(batch_size: usize) -> PipelineArchConfig {
  let input_channels = 32;
  let conv1_channels = 96;
  let hidden_channels = 384;

  let data_layer_cfg = Data3dLayerConfig{
    dims:           (19, 19, input_channels),
    normalize:      true,
  };
  let conv1_layer_cfg = Conv2dLayerConfigV2{
    in_dims:        (19, 19, input_channels),
    conv_size:      5,
    conv_stride:    1,
    conv_pad:       2,
    out_channels:   conv1_channels,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInitialization::Uniform{half_range: 0.05},
    backend:        ConvBackend::CudnnImplicitPrecompGemm,
  };
  let conv2_layer_cfg = Conv2dLayerConfigV2{
    in_dims:        (19, 19, conv1_channels),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   hidden_channels,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInitialization::Uniform{half_range: 0.05},
    backend:        ConvBackend::CudnnImplicitPrecompGemm,
  };
  let inner_conv_layer_cfg = Conv2dLayerConfigV2{
    in_dims:        (19, 19, hidden_channels),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   hidden_channels,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInitialization::Uniform{half_range: 0.05},
    backend:        ConvBackend::CudnnImplicitPrecompGemm,
  };
  let final_conv_layer_cfg = Conv2dLayerConfigV2{
    in_dims:        (19, 19, hidden_channels),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   3,
    act_func:       ActivationFunction::Identity,
    init_weights:   ParamsInitialization::Uniform{half_range: 0.05},
    backend:        ConvBackend::CudnnImplicitPrecompGemm,
  };
  let loss_layer_cfg = MultiCategoricalLossLayerConfig{
    num_categories:     361,
    train_lookahead:    3,
    infer_lookahead:    1,
  };

  let mut arch_cfg = PipelineArchConfig::new();
  arch_cfg
    .data3d(data_layer_cfg)
    .conv2d_v2(conv1_layer_cfg)
    .conv2d_v2(conv2_layer_cfg)
    .conv2d_v2(inner_conv_layer_cfg)
    .conv2d_v2(inner_conv_layer_cfg)
    .conv2d_v2(inner_conv_layer_cfg)
    .conv2d_v2(inner_conv_layer_cfg)
    .conv2d_v2(inner_conv_layer_cfg)
    .conv2d_v2(inner_conv_layer_cfg)
    .conv2d_v2(inner_conv_layer_cfg)
    .conv2d_v2(inner_conv_layer_cfg)
    .conv2d_v2(inner_conv_layer_cfg)
    .conv2d_v2(inner_conv_layer_cfg)
    .conv2d_v2(final_conv_layer_cfg)
    .multi_softmax_kl_loss(loss_layer_cfg);

  arch_cfg
}

pub fn build_value_3layer64_19x19x32_arch_nodir(batch_size: usize) -> PipelineArchConfig {
  let input_channels = 32;
  let hidden_channels = 64;

  let data_layer_cfg = Data3dLayerConfig{
    dims:           (19, 19, input_channels),
    normalize:      true,
  };
  let conv1_layer_cfg = Conv2dLayerConfig{
    in_dims:        (19, 19, input_channels),
    conv_size:      5,
    conv_stride:    1,
    conv_pad:       2,
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
  let loss_layer_cfg = CategoricalLossLayerConfig{
    num_categories:     361,
  };

  let mut arch_cfg = PipelineArchConfig::new();
  arch_cfg
    .data3d(data_layer_cfg)
    .conv2d(conv1_layer_cfg)
    .conv2d(inner_conv_layer_cfg)
    .conv2d(final_conv_layer_cfg)
    .logistic_ind_loss(loss_layer_cfg);

  arch_cfg
}

pub fn build_value_13layer384multi3_19x19x32_arch_nodir(batch_size: usize) -> PipelineArchConfig {
  let input_channels = 32;
  let conv1_channels = 96;
  let hidden_channels = 384;

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
    // FIXME(20160304): conv layer config with 2 variants of out channels.
    out_channels:   3,
    //out_channels:   1,
    act_func:       ActivationFunction::Identity,
    init_weights:   ParamsInitialization::Uniform{half_range: 0.05},
  };
  /*let loss_layer_cfg = MultiCategoricalLossLayerConfig{
    num_categories:     361,
    train_lookahead:    3,
    infer_lookahead:    1,
  };*/
  let loss_layer_cfg = CategoricalLossLayerConfig{
    num_categories:     361,
  };

  let mut arch_cfg = PipelineArchConfig::new();
  arch_cfg
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
    //.multi_softmax_kl_loss(loss_layer_cfg);
    .logistic_ind_loss(loss_layer_cfg);

  arch_cfg
}
