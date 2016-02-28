use std::sync::{Arc, Barrier};
use std::sync::atomic::{Ordering, fence};

pub struct WorkerData {
  pub tid:          usize,
  pub num_workers:  usize,
  pub barrier:      Arc<Barrier>,
}

impl WorkerData {
  pub fn sync(&self) {
    self.barrier.wait();
    fence(Ordering::AcqRel);
  }
}
