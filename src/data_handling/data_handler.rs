//! Dataset loading and preprocessing utilities.

use ndarray::{Array1, Array2};

/// Dataset used for training a model.
pub trait TrainingDataset: Send + Sync {
  fn get_dataset_size(&self) -> usize;
  fn get_input_size(&self) -> usize;
  fn get_output_size(&self) -> usize;

  fn get_inputs(&self) -> &Array2<f32>;
  fn get_labels(&self) -> &Array2<f32>;

  fn get_random_input(&self) -> Array1<f32>;
  fn get_random_input_and_output(&self) -> (Array1<f32>, Array1<f32>);
  fn get_input(&self, index: usize) -> Array1<f32>;
  fn get_output(&self, index: usize) -> Array1<f32>;
}
