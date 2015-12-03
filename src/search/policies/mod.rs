use board::{Stone, Point};
use random::{XorShift128PlusRng};
use search::{Walk, Trajectory, Node};
use txnstate::{TxnState};
use txnstate::extras::{TxnStateNodeData};
use txnstate::features::{TxnStateFeaturesData};

use rand::{Rng};

pub mod convnet;
pub mod quasiuniform;
pub mod thompson_rave;
pub mod uct_rave;

pub trait PriorPolicy {
  //fn init_prior(&mut self, node: &mut Node, ctx: &Self::Ctx);
  fn fill_prior_probs(&mut self, state: &TxnState<TxnStateNodeData>, valid_moves: &[Point], prior_probs: &mut Vec<(Point, f32)>);
}

pub trait TreePolicy {
  type R: Rng = XorShift128PlusRng;

  fn rave(&self) -> bool;
  fn init_node(&mut self, node: &mut Node, rng: &mut Self::R);
  //fn execute_greedy(&mut self, node: &Node) -> Option<Point>;
  fn execute_search(&mut self, node: &Node, rng: &mut Self::R) -> Option<(Point, usize)>;
  fn backup_values(&mut self, node: &mut Node, rng: &mut Self::R);
}

pub trait RolloutPolicy {
  type R: Rng = XorShift128PlusRng;

  fn rollout(&self, walk: &Walk, traj: &mut Trajectory, rng: &mut Self::R);

  fn do_batch(&self) -> bool { false }
  fn batch_size(&self) -> usize { 1 }
  fn rollout_batch(&mut self, walks: &[Walk], trajs: &mut [Trajectory], rng: &mut Self::R) { unimplemented!(); }
}
