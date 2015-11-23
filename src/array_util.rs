pub fn array_argmax(xs: &[f32]) -> Option<usize> {
  let xs_len = xs.len();
  let mut argmax: Option<(usize, f32)> = None;
  for j in (0 .. xs_len) {
    let x = xs[j];
    if argmax.is_none() || argmax.unwrap().1 < x {
      argmax = Some((j, x));
    }
  }
  if xs_len > 0 {
    assert!(argmax.is_some());
  }
  argmax.map(|(j, _)| j)
}
