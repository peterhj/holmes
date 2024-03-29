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
  AtomicData, ArchWorker,
  PipelineArchConfig, PipelineArchSharedData, PipelineArchWorker,
};
use rembrandt::data_new::{
  SampleDatumConfig, SampleLabelConfig,
  DatasetConfig, DataSource,
  SampleIterator, RandomEpisodeIterator, CyclicEpisodeIterator,
  AugmentDataSource, PartitionDataSource,
};
use rembrandt::layer_new::{
  LayerConfig, ActivationFunction, ParamsInitialization,
  Data3dLayerConfig, Data3dLayer,
  Conv2dLayerConfig, Conv2dLayer,
  CategoricalLossLayerConfig, SoftmaxKLLossLayer,
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

  //let num_workers = 1;
  //let batch_size = 256;

  let num_workers = 3;
  let batch_size = 86;

  //let input_channels = 16;
  //let input_channels = 28;
  let input_channels = 37;

  let conv1_channels = 96;
  let hidden_channels = 256;

  // XXX(20160104): combos that work (maybe?):
  // - LR 0.01, momentum 0.9, init 0.05 (only worked once, unstable init?)
  // - LR 0.005, momentum 0.9, init 0.01 (currently running; failed once, worked 2nd time?)
  // - LR 0.005, momentum 0.1, init 0.05 (works)
  // - LR 0.01, momentum 0.1, init 0.05 (also works)
  // - LR 0.05, momentum 0.1, init 0.05 (does not work)

  let sgd_opt_cfg = SgdOptConfig{
    init_t:         30000, // FIXME(20160128)
    minibatch_size: num_workers * batch_size,
    step_size:      StepSizeSchedule::Decay{
      init_step:    0.03125,
      //init_step:    0.01,
      decay_rate:   0.5,
      decay_iters:  120000,
    },
    momentum:       0.1,
    l2_reg_coef:    0.0,
    display_iters:  20,
    valid_iters:    3000,
    save_iters:     60000,
  };
  //let datum_cfg = SampleDatumConfig::Bytes3d;
  let datum_cfg = SampleDatumConfig::Bits3d{scale: 1};
  let label_cfg = SampleLabelConfig::Category{num_categories: 361};

  info!("sgd cfg: {:?}", sgd_opt_cfg);
  info!("datum cfg: {:?}", datum_cfg);
  info!("label cfg: {:?}", label_cfg);

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

  let shared_seed = [thread_rng().next_u64(), thread_rng().next_u64()];
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
        let context = DeviceContext::new(tid);
        let ctx = context.as_ref();
        let mut arch_worker = PipelineArchWorker::new(
            batch_size,
            arch_cfg,
            //PathBuf::from("experiments/models/tmp_new_action_12layer384_19x19x16.v2"),
            //PathBuf::from("experiments/models/tmp2_new_action_12layer384_19x19x28.v3"),
            PathBuf::from("models/tmp_new_action_12layer256_19x19x37.v3"),
            tid,
            shared_seed,
            &arch_shared,
            atomic_data,
            &ctx,
        );

        //let dataset_cfg = DatasetConfig::open(&PathBuf::from("experiments/gogodb_19x19x16_episode.v2.data"));
        //let dataset_cfg = DatasetConfig::open(&PathBuf::from("experiments/gogodb_19x19x28_episode.v3.data"));

        //let dataset_cfg = DatasetConfig::open(&PathBuf::from("data/gogodb_19x19x16_episode.v2.data"));
        let dataset_cfg = DatasetConfig::open(&PathBuf::from("data/gogodb_19x19x37_episode.v3.data"));

        let mut train_data =
            //SampleIterator::new(
            //CyclicEpisodeIterator::new(
              //Box::new(PartitionDataSource::new(tid, num_workers, dataset_cfg.build("train")))
            RandomEpisodeIterator::new(
              //dataset_cfg.build("train")
              Box::new(AugmentDataSource::new(SymmetryAugment::new(&mut thread_rng()), dataset_cfg.build("train"))),
            );
        let mut valid_data =
            SampleIterator::new(
              //dataset_cfg.build("valid")
              Box::new(PartitionDataSource::new(tid, num_workers, dataset_cfg.build("valid")))
            );

        let sgd_opt = SgdOptimization;
        sgd_opt.train(sgd_opt_cfg, datum_cfg, label_cfg, label_cfg, &mut arch_worker, &mut train_data, &mut valid_data, &ctx);
      });
    }
    scope.join_all();
  });
}
