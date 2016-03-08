extern crate array_cuda;
extern crate holmes;
extern crate rembrandt;

#[macro_use]
extern crate log;
extern crate env_logger;
extern crate rand;
extern crate scoped_threadpool;

use array_cuda::device::{DeviceContext};
use array_cuda::device::comm::{for_all_devices};
use holmes::data::{SymmetryAugment};
use rembrandt::arch_new::{
  PipelineArchConfig, PipelineArchSharedData, PipelineArchWorker,
};
use rembrandt::data_new::{
  SampleDatumConfig, SampleLabelConfig,
  DatasetConfig,
  SampleIterator, RandomEpisodeIterator,
  AugmentDataSource, PartitionDataSource,
};
use rembrandt::layer_new::{
  ActivationFunction, ParamsInitialization,
  Data3dLayerConfig,
  Conv2dLayerConfig,
  CategoricalLossLayerConfig,
  MultiCategoricalLossLayerConfig,
};
use rembrandt::opt_new::{
  StepSizeSchedule,
  SupervisedData,
  SgdOptConfig, SgdOptimization,
};
use scoped_threadpool::{Pool};

use rand::{Rng, thread_rng};
use std::path::{PathBuf};
use std::sync::{Arc};

fn main() {
  train();
}

fn train() {
  env_logger::init().unwrap();
  //let mut rng = thread_rng();

  /*let num_workers = 1;
  let batch_size = 256;*/

  /*let num_workers = 2;
  let batch_size = 128;*/

  let num_workers = 4;
  let batch_size = 64;

  /*let num_workers = 8;
  let batch_size = 32;*/

  let input_channels = 32;
  let lookahead = 3;

  let conv1_channels = 96;
  let hidden_channels = 384;

  let sgd_opt_cfg = SgdOptConfig{
    //init_t:         0,
    init_t:         84000,
    minibatch_size: num_workers * batch_size,
    step_size:      StepSizeSchedule::Decay{
      init_step:    0.015625,
      //init_step:    0.0625,
      decay_rate:   0.25,
      //decay_iters:  540000,
      decay_iters:  780000,
    },
    momentum:       0.0,
    l2_reg_coef:    0.0,
    display_iters:  100,
    valid_iters:    3000,
    save_iters:     3000,
  };
  let datum_cfg = SampleDatumConfig::BitsThenBytes3d{scale: 255};
  let train_label_cfg = SampleLabelConfig::LookaheadCategories{
    num_categories: 361,
    lookahead:      lookahead,
  };
  /*let train_label_cfg = SampleLabelConfig::Category{
    num_categories: 361,
  };*/
  /*let valid_label_cfg = SampleLabelConfig::Category{
    num_categories: 361,*/
  let valid_label_cfg = SampleLabelConfig::LookaheadCategories{
    num_categories: 361,
    lookahead:      lookahead,
  };

  info!("sgd cfg: {:?}", sgd_opt_cfg);
  info!("datum cfg: {:?}", datum_cfg);
  info!("train label cfg: {:?}", train_label_cfg);
  info!("valid label cfg: {:?}", valid_label_cfg);

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
    out_channels:   lookahead,
    //out_channels:   1,
    //act_func:       ActivationFunction::Identity,
    act_func:       ActivationFunction::ScaledSigmoid{beta: 8.0},
    init_weights:   ParamsInitialization::Uniform{half_range: 0.05},
  };
  let loss_layer_cfg = MultiCategoricalLossLayerConfig{
    num_categories:     361,
    train_lookahead:    lookahead,
    infer_lookahead:    1,
  };
  /*let loss_layer_cfg = CategoricalLossLayerConfig{
    num_categories:     361,
  };*/

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
    .multi_softmax_kl_loss(loss_layer_cfg);
    //.softmax_kl_loss(loss_layer_cfg);

  let shared_seed = [thread_rng().next_u64(), thread_rng().next_u64()];
  let arch_shared = for_all_devices(num_workers, |contexts| {
    Arc::new(PipelineArchSharedData::new(num_workers, &arch_cfg, contexts))
  });
  let atomic_data = Arc::new(SupervisedData::new());

  let mut pool = Pool::new(num_workers as u32);
  pool.scoped(|scope| {
    for tid in 0 .. num_workers {
      let arch_cfg = arch_cfg.clone();
      let arch_shared = arch_shared.clone();
      let atomic_data = atomic_data.clone();
      scope.execute(move || {
        let context = DeviceContext::new(tid);
        let ctx = context.as_ref();
        let mut arch_worker = PipelineArchWorker::new(
            batch_size,
            arch_cfg,
            //PathBuf::from("models/tmp_gogodb_w2015_alphav3m_new_value_13layer384multi3_19x19x32"),
            PathBuf::from("models/tmp_gogodb_w2015-preproc-alphav3m_19x19x32_value_13layer384multi3"),
            tid,
            shared_seed,
            &arch_shared,
            atomic_data,
            &ctx,
        );

        let dataset_cfg = DatasetConfig::open(&PathBuf::from("data/gogodb_w2015_alphav3m_19x19x32_episode.data"));

        let mut train_data =
            RandomEpisodeIterator::new(
              Box::new(AugmentDataSource::new(SymmetryAugment::new(&mut thread_rng()), dataset_cfg.build_with_cfg(datum_cfg, train_label_cfg, "train"))),
            );
        let mut valid_data =
            SampleIterator::new(
              Box::new(PartitionDataSource::new(tid, num_workers, dataset_cfg.build_with_cfg(datum_cfg, valid_label_cfg, "valid")))
            );

        let sgd_opt = SgdOptimization;
        sgd_opt.train(sgd_opt_cfg, datum_cfg, train_label_cfg, valid_label_cfg, &mut arch_worker, &mut train_data, &mut valid_data, &ctx);
      });
    }
    scope.join_all();
  });
}