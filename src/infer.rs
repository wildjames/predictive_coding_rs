use ndarray::Array1;

pub fn main() {
  println!("Hello, world!");

  let a = Array1::from(vec![1.0, 2.0, 3.0]);
  let b = Array1::from(vec![4.0, 5.0, 6.0]);
  let c = a * b; // Hadamard product
  println!("Hadamard product: {:?}", c);
}
