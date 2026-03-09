//! Math utilities for predictive coding models.

use crate::{
  data_handling::TrainingDataset,
  model_structure::model::PredictiveCodingModel
};


/// Choose a random index using the rng threadlocal generator, and set the model I/O accordingly.
pub fn set_rand_input_and_output(
  model: &mut PredictiveCodingModel,
  data: &dyn TrainingDataset
) {
  let (input_values, output_values) = data.get_random_input_and_output();

  model.set_input(input_values);
  model.set_output(output_values);
}
