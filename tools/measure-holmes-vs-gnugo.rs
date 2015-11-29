extern crate docopt;
extern crate rand;
extern crate rustc_serialize;
extern crate threadpool;

use docopt::{Docopt};
use threadpool::{ThreadPool};

use rand::{Rng, thread_rng};
use std::collections::{BTreeSet};
use std::env;
use std::fs::{File, create_dir};
use std::io::{Write};
use std::path::{PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{channel};

const USAGE: &'static str = "
measure-holmes-vs-gnu

Usage:
  measure-holmes-vs-gnu bench --log-path=<log_path> --num-jobs=<num_jobs> --num-trials=<num_trials> [--start-idx=<start_idx>] [--device-ids=<device_ids>]
";

#[derive(RustcDecodable, Debug)]
struct Args {
  cmd_bench:        bool,
  flag_log_path:    String,
  flag_num_jobs:    usize,
  flag_num_trials:  usize,
  flag_start_idx:   Option<usize>,
  flag_device_ids:  Option<String>,
}

fn main() {
  let args: Args = Docopt::new(USAGE)
    .and_then(|docopt| docopt.decode())
    .unwrap_or_else(|e| e.exit());

  let start_idx = args.flag_start_idx.unwrap_or(0);
  let n = args.flag_num_trials;

  let device_ids: Vec<usize> = match args.flag_device_ids {
    None => (0 .. 8).collect(),
    Some(ref device_ids) => device_ids.split(",").map(|s| s.parse().unwrap()).collect(),
  };

  let pool = ThreadPool::new(args.flag_num_jobs);
  let (tx, rx) = channel();

  // Set up the log directory.
  let log_dir = PathBuf::from(&args.flag_log_path);
  create_dir(&log_dir).ok();
  // TODO(20151116): copy executable to the log directory.

  let mut job_idxs = Arc::new(Mutex::new(BTreeSet::new()));
  for job_idx in (0 .. args.flag_num_jobs) {
    let mut job_idxs = job_idxs.lock().unwrap();
    job_idxs.insert(job_idx);
  }
  for i in (start_idx .. n) {
    let tx = tx.clone();
    let device_ids = device_ids.clone();
    let mut log_path = log_dir.clone();
    log_path.push(&format!("holmes-vs-gnugo.{}.log", i));
    let mut job_idxs = job_idxs.clone();
    let seed: u32 = thread_rng().gen();
    pool.execute(move || {
      println!("DEBUG: running game {} / {}...", i, n);
      // FIXME(20151116): instead of calling out to shell script, run processes
      // here and poll them for progress.
      let job_idx = {
        let mut job_idxs = job_idxs.lock().unwrap();
        let job_idx = *job_idxs.iter().next().unwrap();
        job_idxs.remove(&job_idx);
        job_idx
      };
      let device_num = device_ids[job_idx % device_ids.len()];
      let (b_port, w_port) = (6060 + 2*job_idx, 6060 + 2*job_idx + 1);
      // FIXME(20151123): manually track child process progress.
      /*if false {
        let mut ctl_child = Command::new("target/release/gtpctl")
        ;
        let mut gnugo_child = Command::new("../bin/gnugo-3.8")
          .arg("--level=10")
          .arg("--chinese-rules")
          .arg("--mode=gtp")
          .arg(&format!("--gtp-connect=127.0.0.1:{}", w_port))
        ;
        let mut holmes_child = Command::new("target/release/holmes-gtp")
          .arg("-h").arg("127.0.0.1")
          .arg("-p").arg(&format!("{}", b_port))
        ;
      }*/
      let mut child = Command::new("./gtpctl-holmes-vs-gnugo.sh")
        .env("CUDA_VISIBLE_DEVICES", &format!("{}", device_num))
        //.env("HYPERPARAM_INSTANCE_PATH", "experiments/hyperparam/ikeda_rave.instance.json")
        .env("HYPERPARAM_INSTANCE_PATH", "default.toml")
        .arg(&format!("{}", b_port))
        .arg(&format!("{}", w_port))
        .arg(&format!("{}", seed))  // XXX: Need to set seed, otherwise gnugo
                                    // tends to make the same moves.
        .stdout(Stdio::piped())
        .spawn().unwrap();
      let output = child.wait_with_output().unwrap();
      let mut file = File::create(&log_path).unwrap();
      file.write_all(&output.stdout).unwrap();
      tx.send(i).unwrap();
      {
        let mut job_idxs = job_idxs.lock().unwrap();
        job_idxs.insert(job_idx);
      }
    });
  }

  for _ in rx.iter().take(n) {
  }
}

fn old_main() {
  let args: Vec<_> = env::args().collect();
  let logs_dir = PathBuf::from(&args[1]);
  let maybe_start_idx: Option<usize> = if args.len() >= 3 {
    args[2].parse().ok()
  } else {
    None
  };

  let start_idx = maybe_start_idx.unwrap_or(0);
  let n = 200;

  // Set up the log directory.
  create_dir(&logs_dir).ok();
  // TODO(20151116): copy executable to the log directory.

  for i in (start_idx .. n) {
    println!("DEBUG: running game {} / {}...", i, n);
    // FIXME(20151116): instead of calling out to shell script, run processes
    // here and poll them for progress.
    let mut child = Command::new("./gtpctl-holmes-vs-gnugo.sh")
      .stdout(Stdio::piped())
      .spawn().unwrap();
    let output = child.wait_with_output().unwrap();
    let mut log_path = logs_dir.clone();
    log_path.push(&format!("holmes-vs-gnugo.{}.log", i));
    let mut file = File::create(&log_path).unwrap();
    file.write_all(&output.stdout).unwrap();
  }
}
