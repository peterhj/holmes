use std::sync::{Arc, Barrier};
use std::sync::atomic::{Ordering, fence};

#[derive(Clone)]
pub struct WorkerSharedData {
  //pub tid:          usize,
  pub num_workers:  usize,
  pub barrier:      Arc<Barrier>,
}

impl WorkerSharedData {
  pub fn new(num_workers: usize) -> WorkerSharedData {
    WorkerSharedData{
      num_workers:  num_workers,
      barrier:      Arc::new(Barrier::new(num_workers)),
    }
  }

  pub fn sync(&self) {
    self.barrier.wait();
    fence(Ordering::AcqRel);
  }
}

#[derive(Clone)]
pub struct WorkerLocalData {
  pub tid:  usize,
}
