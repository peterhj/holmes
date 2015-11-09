use std::env;
use std::fs::{File};
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

  for i in (maybe_start_idx.unwrap_or(0) .. n) {
    println!("DEBUG: running game {} / {}...", i, n);
    let mut child = Command::new("./gtpctl-holmes-vs-gnugo.sh")
      .stdin(Stdio::piped())
      .stdout(Stdio::piped())
      .spawn().unwrap();
    let output = child.wait_with_output().unwrap();
    let mut log_path = logs_dir.clone();
    log_path.push(&format!("holmes-vs-gnugo.{}.log", i));
    let mut file = File::create(&log_path).unwrap();
    file.write_all(&output.stdout).unwrap();
  }
}
