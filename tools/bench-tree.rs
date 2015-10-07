extern crate holmes;

use holmes::fastboard::{FastBoard, FastBoardAux, FastBoardWork};
use holmes::policy::{BenchmarkTreePolicy};
use holmes::tree::{SearchTree};

fn main() {
  let mut init_state = FastBoard::new();
  init_state.reset();
  let mut init_aux_state = Some(FastBoardAux::new());
  init_aux_state.as_mut().unwrap().reset();
  let mut work = FastBoardWork::new();
  let mut policy = BenchmarkTreePolicy::new();
  let mut tree = SearchTree::<BenchmarkTreePolicy>::new();
  for _ in (0 .. 550usize) {
    tree.reset(&init_state, &init_aux_state, 0, 0.0);
    for _ in (0 .. 361usize) {
      tree.execute_path(1, &policy, &mut work);
    }
  }
}
