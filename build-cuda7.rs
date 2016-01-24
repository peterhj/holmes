fn main() {
  println!("cargo:rustc-flags=-L /usr/local/cuda-7.0/lib64 -l dylib=cudart");
  println!("cargo:rustc-flags=-L /usr/local/cuda-7.0/lib64 -l dylib=cublas");
  println!("cargo:rustc-flags=-L /usr/local/cuda-7.0/lib64 -l dylib=cudnn");
  //println!("cargo:rustc-link-search=dylib=/usr/local/cuda-7.0/lib64");
  //println!("cargo:rustc-link-search=native=/usr/local/cuda-7.0/lib64");
}
