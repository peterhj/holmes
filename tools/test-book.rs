extern crate holmes;
extern crate rand;

use holmes::book::{OpeningBook};
use holmes::random::{XorShift128PlusRng};

use rand::{Rng, SeedableRng, thread_rng};
use std::path::{PathBuf};

fn main() {
  let path = PathBuf::from("data/book-fuego.dat");
  //let mut rng = thread_rng();
  let mut rng = XorShift128PlusRng::from_seed([1234, 5678]);
  let book = OpeningBook::load_fuego(&path, &mut rng);
}
