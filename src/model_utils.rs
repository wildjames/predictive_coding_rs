use ndarray::{ArrayD, Array2, ArrayBase, Data, Dimension};


pub fn hadamard(a: &ArrayD<f32>, b: &ArrayD<f32>) -> Result<ArrayD<f32>, String> {
    // Try broadcasting b to shape of a
    if let Some(b_broadcast) = b.broadcast(a.raw_dim()) {
        return Ok(a * &b_broadcast);
    }

    // Try broadcasting a to shape of b
    if let Some(a_broadcast) = a.broadcast(b.raw_dim()) {
        return Ok(&a_broadcast * b);
    }

    Err("Shapes are not compatible for Hadamard product".into())
}

pub fn outer_product<SA, DA, SB, DB>(
  a: &ArrayBase<SA, DA>,
  b: &ArrayBase<SB, DB>
) -> Array2<f32>
where
  SA: Data<Elem = f32>,
  SB: Data<Elem = f32>,
  DA: Dimension,
  DB: Dimension,
{
  let a_values: Vec<f32> = a.iter().copied().collect();
  let b_values: Vec<f32> = b.iter().copied().collect();
  let rows = a_values.len();
  let cols = b_values.len();

  Array2::from_shape_fn((rows, cols), |(i, j)| a_values[i] * b_values[j])
}

pub fn relu(x: f32) -> f32 {
  if x > 0.0 {
    x
  } else {
    0.0
  }
}


