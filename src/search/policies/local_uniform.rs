use board::{Stone, Point};
use random::{XorShift128PlusRng, choose_without_replace};
use search::{Trajectory};
use search::policies::{RolloutPolicy};
use txnstate::{for_each_adjacent_8, check_good_move_fast};

use rand::{Rng};

pub struct LocalUniformRolloutPolicy;

impl RolloutPolicy for LocalUniformRolloutPolicy {
  type R = XorShift128PlusRng;

  fn rollout(&self, traj: &mut Trajectory, rng: &mut Self::R) {
    // TODO(20151120): try deterministic local moves (c.f. Huang sim balancing
    // local features), then try uniform random moves.

    let mut valid_moves = [vec![], vec![]];
    traj.leaf_node.as_ref().unwrap().borrow()
      .state.get_data().legality.fill_legal_points(Stone::Black, &mut valid_moves[0]);
    traj.leaf_node.as_ref().unwrap().borrow()
      .state.get_data().legality.fill_legal_points(Stone::White, &mut valid_moves[1]);
    let mut sim_turn = traj.sim_state.current_turn();
    let mut sim_pass = [false, false];

    let max_iters = 3 * 361 + rng.gen_range(0, 2);
    for t in (0 .. max_iters) {
      sim_pass[sim_turn.offset()] = false;
      let mut made_move = false;

      // FIXME(20151120): local features [Huang, 2011]:
      // - save by capture (no self-atari)
      // - 2-point semeai capture
      // - save by extend (no self-atari)
      // - 8-contiguous
      // - capture after ko

      if !valid_moves[sim_turn.offset()].is_empty() {
      }

      while !valid_moves[sim_turn.offset()].is_empty() {
        if let Some(sim_point) = choose_without_replace(&mut valid_moves[sim_turn.offset()], rng) {
          if !check_good_move_fast(&traj.sim_state.position, &traj.sim_state.chains, sim_turn, sim_point) {
            continue;
          }
          if traj.sim_state.try_place(sim_turn, sim_point).is_ok() {
            traj.sim_state.commit();
            traj.sim_pairs.push((sim_turn, sim_point));
            valid_moves[0].extend(traj.sim_state.position.last_killed[0].iter());
            valid_moves[0].extend(traj.sim_state.position.last_killed[1].iter());
            valid_moves[1].extend(traj.sim_state.position.last_killed[0].iter());
            valid_moves[1].extend(traj.sim_state.position.last_killed[1].iter());
            made_move = true;
            break;
          } else {
            traj.sim_state.undo();
          }
        }
      }
      if !made_move {
        sim_pass[sim_turn.offset()] = true;
      }
      if sim_pass[0] && sim_pass[1] {
        break;
      }
      sim_turn = sim_turn.opponent();
    }
    unimplemented!();
  }
}
