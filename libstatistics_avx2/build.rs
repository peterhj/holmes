extern crate gcc;

fn main() {
  gcc::Config::new()
    .compiler("gcc-4.7")
    .opt_level(3)
    .flag("-std=gnu99")
    .flag("-march=native")
    .flag("-Wall")
    .flag("-Werror")
    .include("src/c")
    .file("src/c/array.c")
    .file("src/c/xorshift.c")
    .file("src/c/cephes/mtherr.c")
    .file("src/c/cephes/constf.c")
    .file("src/c/cephes/logf.c")
    .file("src/c/cephes/sinf.c")
    .file("src/c/cephes/sqrtf.c")
    .compile("libstatistics_avx2_impl.a");
}
