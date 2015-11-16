use std::env;
use std::fs::{File, create_dir};
use std::io::{Write};
use std::path::{PathBuf};
use std::process::{Command, Stdio};

fn main() {
  let args: Vec<_> = env::args().collect();
  let logs_dir = PathBuf::from(&args[1]);
  let maybe_start_idx: Option<usize> = if args.len() >= 3 {
    args[2].parse().ok()
  } else {
    None
  };

  let n = 200;

  // Set up the log directory.
  create_dir(&logs_dir).ok();
  // TODO(20151116): copy executable to the log directory.

  for i in (maybe_start_idx.unwrap_or(0) .. n) {
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
