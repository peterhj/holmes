extern crate hydra;

use hydra::api::{Experiment, TrialConfig, Resource, Asset};
use hydra::control::{ControlClient, ControlCmd};

use std::path::{PathBuf};

fn main() {
  let trial_cfg = TrialConfig{
    resources:  vec![
      (Resource::RandomSeed32, 2),
      (Resource::Port, 3),
      (Resource::CudaGpu, 1),
    ],
    assets:     vec![
      //Asset::Copy{src: PathBuf::from("default.hyperparam")},
      Asset::Symlink{src: PathBuf::from("patterns.prob")},
      Asset::Symlink{src: PathBuf::from("patterns.spat")},
      Asset::SymlinkAs{
        /*src: PathBuf::from("models/kgs_ugo_201505_new_action_12layer384_19x19x37.v3.saved/layer_params.latest.blob"),
        dst: PathBuf::from("models/kgs_ugo_201505_new_action_12layer384_19x19x37.v3.saved/layer_params.latest.blob"),*/
        /*src: PathBuf::from("models/gogodb_w2015_alphav2_new_action_12layer384_19x19x44.saved/layer_params.latest.blob"),
        dst: PathBuf::from("models/gogodb_w2015_alphav2_new_action_12layer384_19x19x44.saved/layer_params.latest.blob"),*/
        /*src: PathBuf::from("models/gogodb_w2015_alphav2_new_action_12layer384_19x19x44_run2.saved/layer_params.latest.blob"),
        dst: PathBuf::from("models/gogodb_w2015_alphav2_new_action_12layer384_19x19x44_run2.saved/layer_params.latest.blob"),*/
        /*src: PathBuf::from("models/gogodb_w2015_alphav3_new_action_13layer384_19x19x32.saved/layer_params.latest.blob"),
        dst: PathBuf::from("models/gogodb_w2015_alphav3_new_action_13layer384_19x19x32.saved/layer_params.latest.blob"),*/
        src: PathBuf::from("models/gogodb_w2015-preproc-alphav3m_19x19x32_13layer384multi3.saved"),
        dst: PathBuf::from("models/gogodb_w2015-preproc-alphav3m_19x19x32_13layer384multi3.saved"),
      },
    ],
    programs:   vec![
      // TODO(20160115)
      (PathBuf::from("gtp_ref_ctrl"),
        vec![
          "${HYDRA.PORT.0}".to_string(),
          "${HYDRA.PORT.1}".to_string(),
          "${HYDRA.PORT.2}".to_string(),
          "7.5".to_string(),
        ],
        vec![
          ("LD_LIBRARY_PATH".to_string(), "/nscratch/phj/local/cudnn_v4rc1/lib64:/usr/local/cuda-7.0/lib64".to_string()),
          ("RUST_BACKTRACE".to_string(), "1".to_string()),
        ]),
      (PathBuf::from("bin/gnugo-3.8"),
        vec![
          "--level".to_string(), "10".to_string(),
          "--chinese-rules".to_string(),
          "--seed".to_string(), "${HYDRA.SEED32.0}".to_string(),
          "--mode".to_string(), "gtp".to_string(),
          "--gtp-connect".to_string(), "127.0.0.1:${HYDRA.PORT.2}".to_string(),
        ],
        vec![]),
      (PathBuf::from("bin/pachi-11.ivybridge"),
        vec![
          "-e".to_string(), "uct".to_string(),
          "-r".to_string(), "chinese".to_string(),
          "-s".to_string(), "${HYDRA.SEED32.1}".to_string(),
          "-t".to_string(), "=10000".to_string(), "threads=1,pondering=0".to_string(),
          "-g".to_string(), "127.0.0.1:${HYDRA.PORT.0}".to_string(),
        ],
        vec![]),
      (PathBuf::from("holmes-convnet-gtp"),
        vec![
          "-h".to_string(), "127.0.0.1".to_string(),
          "-p".to_string(), "${HYDRA.PORT.1}".to_string(),
        ],
        vec![
          ("LD_LIBRARY_PATH".to_string(), "/nscratch/phj/local/cudnn_v4rc1/lib64:/usr/local/cuda-7.0/lib64".to_string()),
          ("RUST_BACKTRACE".to_string(), "1".to_string()),
          ("RUST_LOG".to_string(), "info".to_string()),
        ]),
    ],
  };
  let experiment = Experiment{
    trial_cfg:      trial_cfg,
    //trials_path:    PathBuf::from("experiments/convnet_t_630k_white.0.experiment"),
    //trials_path:    PathBuf::from("experiments/convnet_alphav2_t_561k_white.0.experiment"),
    //trials_path:    PathBuf::from("experiments/convnet_alphav2_12ppn_run2_t_960k_white.0.experiment"),
    //trials_path:    PathBuf::from("experiments/convnet_alphav3_13ppn384_t_480k_white.0.experiment"),

    //trials_path:    PathBuf::from("experiments/convnet_alphav3m_13ppn384multi3_t_786k_white.0.experiment"),
    //trials_path:    PathBuf::from("experiments/convnet_alphav3m_13ppn384multi3_t_660k_white.0.experiment"),
    trials_path:    PathBuf::from("experiments/convnet_alphav3m_13ppn384multi3_t_1569k_white.0.experiment"),

    scratch_prefix: PathBuf::from("/scratch/phj/space/holmes-project/holmes"),
    num_trials:     256,
  };
  let mut client = ControlClient::new();
  client.send_cmd(ControlCmd::SubmitExperiment{experiment: experiment});
}
