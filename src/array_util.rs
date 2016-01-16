pub fn array_argmax(xs: &[f32]) -> Option<usize> {
  let n = xs.len();
  let mut argmax: Option<(usize, f32)> = None;
  for j in 0 .. n {
    let x = xs[j];
    if argmax.is_none() || argmax.unwrap().1 < x {
      argmax = Some((j, x));
    }
  }
  if n >= 1 {
    assert!(argmax.is_some());
  }
  argmax.map(|(j, _)| j)
}

pub fn array_argmax2(xs: &[f32]) -> (Option<usize>, Option<usize>) {
  let n = xs.len();
  let mut argmax: Option<(usize, f32)> = None;
  let mut argmax2: Option<(usize, f32)> = None;
  for j in 0 .. n {
    let x = xs[j];
    if argmax.is_none() || argmax.unwrap().1 < x {
      argmax2 = argmax;
      argmax = Some((j, x));
    } else if argmax2.is_none() || argmax2.unwrap().1 < x {
      argmax2 = Some((j, x));
    }
  }
  if n >= 1 {
    assert!(argmax.is_some());
  }
  if n >= 2 {
    assert!(argmax2.is_some());
  }
  (argmax.map(|(j, _)| j), argmax2.map(|(j, _)| j))
}
