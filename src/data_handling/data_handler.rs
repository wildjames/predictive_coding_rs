//! Dataset loading and preprocessing utilities.

use ndarray::{Array2};

/// Black and white, single channel images with labels. e.g. MNIST dataset
#[derive(Clone)]
pub struct TrainingDataset {
  /// The number of samples in the dataset (e.g. 60000 for MNIST)
  pub dataset_size: usize,
  /// The number of nodes in the input layer (e.g. 28*28 for MNIST)
  pub input_size: usize,
  /// The number of nodes in the output layer (e.g. 10 for MNIST)
  pub output_size: usize,
  pub inputs: Array2<f32>,
  pub labels: Array2<f32>,
}
