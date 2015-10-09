//extern crate holmes;
extern crate rand;
extern crate statistics_avx2;
extern crate time;

//use holmes::random::{XorShift128PlusRng};
use rand::{Rng, SeedableRng, thread_rng};
use statistics_avx2::array::{array_argmax};
use statistics_avx2::random::{XorShift128PlusStreamRng};
use time::{Duration, get_time};

fn main() {
  let mut seed: Vec<u64> = Vec::with_capacity(8);
  for _ in (0 .. 8) {
    seed.push(thread_rng().next_u64());
  }

  /*let mut rng: XorShift128PlusRng = SeedableRng::from_seed([seed[0], seed[4]]);
  let mut buf: Vec<u64> = Vec::new();
  buf.push(rng.next_u64());
  buf.push(rng.next_u64());
  buf.push(rng.next_u64());
  buf.push(rng.next_u64());
  println!("DEBUG: {:x} {:x} {:x} {:x}", buf[0], buf[1], buf[2], buf[3]);*/

  let mut rng = XorShift128PlusStreamRng::new(&seed);
  let m = 1000_000;
  let n = 1000;

  /*rng.sample32(&mut buf);
  println!("DEBUG: {:x} {:x} {:x} {:x}", buf[0], buf[1], buf[2], buf[3]);
  println!("DEBUG: {:x} {:x} {:x} {:x}", buf[4], buf[5], buf[6], buf[7]);
  println!("DEBUG: {:x} {:x} {:x} {:x}", buf[8], buf[9], buf[10], buf[11]);
  println!("DEBUG: {:x} {:x} {:x} {:x}", buf[12], buf[13], buf[14], buf[15]);*/

  let mut succ_ratio = Vec::with_capacity(n);
  unsafe { succ_ratio.set_len(n) };

  let start_t = get_time();
  for _ in (0 .. m) {
    rng.sample_uniform_f32(&mut succ_ratio);
  }
  let elapsed_ms = (get_time() - start_t).num_milliseconds();
  println!("DEBUG: uniform");
  println!("DEBUG: elapsed: {} ms", elapsed_ms);
  let off = 100;
  println!("DEBUG: {} {} {} {}", succ_ratio[off+0], succ_ratio[off+1], succ_ratio[off+2], succ_ratio[off+3]);
  println!("DEBUG: {} {} {} {}", succ_ratio[off+4], succ_ratio[off+5], succ_ratio[off+6], succ_ratio[off+7]);
  println!("DEBUG: {} {} {} {}", succ_ratio[off+8], succ_ratio[off+9], succ_ratio[off+10], succ_ratio[off+11]);
  println!("DEBUG: {} {} {} {}", succ_ratio[off+12], succ_ratio[off+13], succ_ratio[off+14], succ_ratio[off+15]);

  let mut num_trials = Vec::with_capacity(n);
  unsafe { num_trials.set_len(n) };

  let start_t = get_time();
  //rng.sample_uniform_f32(&mut num_trials);
  for i in (0 .. n) {
    num_trials[i] = 100.0;
  }
  //array_fill(1000.0, &mut num_trials);
  let elapsed_ms = (get_time() - start_t).num_milliseconds();
  println!("DEBUG: elapsed: {} ms", elapsed_ms);

  let mut buf = Vec::with_capacity(n);
  unsafe { buf.set_len(n) };
  let start_t = get_time();
  for _ in (0 .. m) {
    rng.sample_approx_beta_f32(&succ_ratio, &num_trials, &mut buf);
  }
  let elapsed_ms = (get_time() - start_t).num_milliseconds();
  println!("DEBUG: approx beta");
  println!("DEBUG: elapsed: {} ms", elapsed_ms);
  let off = 100;
  println!("DEBUG: {} {} {} {}", buf[off+0], buf[off+1], buf[off+2], buf[off+3]);
  println!("DEBUG: {} {} {} {}", buf[off+4], buf[off+5], buf[off+6], buf[off+7]);
  println!("DEBUG: {} {} {} {}", buf[off+8], buf[off+9], buf[off+10], buf[off+11]);
  println!("DEBUG: {} {} {} {}", buf[off+12], buf[off+13], buf[off+14], buf[off+15]);
  let argmax_idx = array_argmax(&buf);
  println!("DEBUG: argmax: {}", argmax_idx);
}
