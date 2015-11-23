fn trans_3x3(coord: u8, t: u8) -> u8 {
  let (x, y) = (coord % 3, coord / 3);
  let (u, v) = (x - 1, y - 1);
  // XXX: Most inverse transforms are the same as the forward transforms,
  // except for two (labeled below).
  let raw_coord = match t {
    0 => coord,
    1 => (1 - u) + (1 + v) * 3,
    2 => (1 + u) + (1 - v) * 3,
    3 => (1 - u) + (1 - v) * 3,
    4 => (1 + v) + (1 + u) * 3,
    5 => (1 - v) + (1 + u) * 3,
    6 => (1 + v) + (1 - u) * 3,
    7 => (1 - v) + (1 - u) * 3,
    _ => unreachable!(),
  };
  let donut_coord = raw_coord - (raw_coord / 5);
  donut_coord
}

fn main() {
  for t in (0 .. 8) {
    let mut trans_cs = vec![];
    for coord in (0 .. 4).chain(5 .. 9) {
      let trans_c = trans_3x3(coord, t);
      trans_cs.push(trans_c);
    }
    println!("DEBUG: t: {} trans coords: {:?}", t, trans_cs);
  }
}
