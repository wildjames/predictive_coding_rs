//! Math utilities for predictive coding models.

use crate::{
  data_handling::data_handler::TrainingDataset,
  model_structure::model::PredictiveCodingModel
};

use ndarray::{Array1};

/// Choose a random index using the rng threadlocal generator, and set the model I/O accordingly.
pub fn set_rand_input_and_output(
  model: &mut PredictiveCodingModel,
  data: &TrainingDataset
) {
  let rand_index: usize = usize::from_ne_bytes(rand::random()) % data.dataset_size;
  let input_values: Array1<f32> = data.inputs
    .row(rand_index)
    .to_owned();
  let output_values: Array1<f32> = data.labels
    .row(rand_index)
    .to_owned();
  model.set_input(input_values);
  model.set_output(output_values);
}
