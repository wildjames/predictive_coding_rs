// pub fn hadamard_product(a: &[f32], b: &[f32]) -> Vec<f32> {
//   a.iter().zip(b.iter())
//     .map(|(x, y)| x * y)
//     .collect()
// }

pub fn relu(x: f32) -> f32 {
  if x > 0.0 {
    x
  } else {
    0.0
  }
}


