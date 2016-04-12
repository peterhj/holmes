extern crate holmes;

use holmes::data::{GogodbEpisodePreproc, LazyEpisodeLoader};
use holmes::search::parallel_trace::omega::{OmegaDriver};
use holmes::search::parallel_tree::{MonteCarloSearchConfig, TreePolicyConfig, HorizonConfig};
use holmes::txnstate::{TxnStateConfig};

use std::path::{PathBuf};

fn main() {
  let state_cfg = TxnStateConfig::default();
  let search_cfg = MonteCarloSearchConfig{
    num_rollouts:   1024,
    batch_size:     64,
  };
  let tree_cfg = TreePolicyConfig{
    //horizon_cfg:    HorizonConfig::All,
    horizon_cfg:    HorizonConfig::Fixed{max_horizon: 20},
    visit_thresh:   1,
    mc_scale:       1.0,
    prior_equiv:    16.0,
    rave:           false,
    rave_equiv:     0.0,
  };
  let mut driver = OmegaDriver::new(state_cfg, search_cfg, tree_cfg, /*save_interval*/);
  let mut loader = LazyEpisodeLoader::new(PathBuf::from("gogodb_w2015_train_index"), GogodbEpisodePreproc);
  driver.train(&mut loader);
}
