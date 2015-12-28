extern crate array_cuda;
extern crate rembrandt;
//extern crate rand;
extern crate scoped_threadpool;

use array_cuda::device::{DeviceContext};
use array_cuda::device::comm::{for_all_devices};
use rembrandt::arch_new::{
  AtomicData, ArchWorker,
  PipelineArchConfig, PipelineArchSharedData, PipelineArchWorker,
};
use rembrandt::data_new::{
  SampleLabelConfig, DatasetConfig, DataSource,
  SampleIterator, RandomEpisodeIterator, CyclicEpisodeIterator,
  PartitionDataSource,
};
use rembrandt::layer_new::{
  LayerConfig, ActivationFunction, ParamsInitialization,
  Data3dLayerConfig, Data3dLayer,
  Conv2dLayerConfig, Conv2dLayer,
  CategoricalLossLayerConfig, SoftmaxKLLossLayer,
};
use rembrandt::opt_new::{
  LearningRateSchedule,
  SupervisedData,
  SgdOptConfig, SgdOptimization,
};
use scoped_threadpool::{Pool};

//use rand::{Rng, thread_rng};
use std::path::{PathBuf};
use std::sync::{Arc};

fn main() {
  train();
}

fn train() {
  //let mut rng = thread_rng();

  let batch_size = 256;
  let input_channels = 16;
  let conv1_channels = 96;
  let hidden_channels = 384;

  let sgd_opt_cfg = SgdOptConfig{
    minibatch_size: batch_size,
    learning_rate:  LearningRateSchedule::StepTwice{
      lr0:      0.05,   lr0_iters:  200000,
      lr1:      0.005,  lr1_iters:  400000,
      lr_final: 0.0005,
    },
    momentum:       0.0,
    l2_reg_coef:    0.0,
    display_iters:  20,
    valid_iters:    800,
    save_iters:     1600,
  };

  let label_cfg = SampleLabelConfig::Category;

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
    out_channels:   1,
    act_func:       ActivationFunction::Identity,
    init_weights:   ParamsInitialization::Uniform{half_range: 0.05},
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

  let num_workers = 1;
  let arch_shared = for_all_devices(num_workers, |contexts| {
    Arc::new(PipelineArchSharedData::new(num_workers, &arch_cfg, contexts))
  });
  let atomic_data = Arc::new(SupervisedData::new());

  let mut pool = Pool::new(num_workers as u32);
  pool.scoped(|scope| {
    for tid in (0 .. num_workers) {
      let arch_cfg = arch_cfg.clone();
      let arch_shared = arch_shared.clone();
      let atomic_data = atomic_data.clone();
      scope.execute(move || {
        let context = DeviceContext::new(0);
        let ctx = context.as_ref();
        let mut arch_worker = PipelineArchWorker::new(
            batch_size,
            arch_cfg,
            PathBuf::from("experiments/models/tmp_new_action_12layer384_19x19x16.v2"),
            tid,
            &arch_shared,
            atomic_data,
            &ctx,
        );

        let dataset_cfg = DatasetConfig::open(&PathBuf::from("experiments/gogodb_19x19x16_episode.v2.data"));
        let mut train_data =
            //SampleIterator::new(
            CyclicEpisodeIterator::new(
            //RandomEpisodeIterator::new(
              //Box::new(PartitionDataSource::new(tid, num_workers, dataset_cfg.build("train")))
              dataset_cfg.build("train")
            );
        let mut valid_data =
            SampleIterator::new(
              //Box::new(PartitionDataSource::new(tid, num_workers, dataset_cfg.build("valid")))
              dataset_cfg.build("valid")
            );

        let sgd_opt = SgdOptimization;
        sgd_opt.train(sgd_opt_cfg, label_cfg, &mut arch_worker, &mut train_data, &mut valid_data, &ctx);
      });
    }
    scope.join_all();
  });
}
