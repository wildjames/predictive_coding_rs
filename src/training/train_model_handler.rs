//! Training orchestration for predictive coding models.

use crate::model::{
  model::{PredictiveCodingModel},
  model_utils::save_model
};
use crate::data_handling::data_handler;

use ndarray::Array1;
use tracing::info;

fn set_input_and_output(
  model: &mut PredictiveCodingModel,
  data: &data_handler::ImagesBWDataset
) {
  let rand_index = usize::from_ne_bytes(rand::random()) % data.num_images;

  // Normalise to the range 0..1
  let input_values: Array1<f32> = data.images
    .row(rand_index)
    .mapv(|x| x as f32 / 255.0)
    .to_owned();

  // One-hot output row with label value set to 1.0
  let output_label: usize = data.labels[rand_index] as usize;

  let output_layer_size = model.layers.last().unwrap().size;
  let mut output_values: Array1<f32> = Array1::zeros(output_layer_size);
  output_values[output_label] = 1.0;

  model.set_input(input_values);
  model.set_output(output_values);

}

/// Run inference to convergence on a single sample and update weights.
fn train_sample(
  model: &mut PredictiveCodingModel,
  data: &data_handler::ImagesBWDataset,
  convergence_threshold: f32,
  convergence_steps: u32
) {
  set_input_and_output(model, data);
  // Train on this example until convergence.
  model.converge_values_with_updates(convergence_threshold, convergence_steps);
  model.update_weights();
}


/// Train the model for a number of steps using randomly sampled data.
pub fn train(
  model: &mut PredictiveCodingModel,
  data: &data_handler::ImagesBWDataset,
  training_steps: u32,
  convergence_steps: u32,
  convergence_threshold: f32,
  snapshot_interval: u32,
  snapshot_output_prefix: &str
) {
  // Current timestamp


  for step in 0..training_steps {
    train_sample(
      model,
      data,
      convergence_threshold,
      convergence_steps
    );

    let error = model.read_total_error();
    let energy = model.read_total_energy();

    info!(
      "Sample {:.1}\terror {:.1}\tenergy {:.1}",
      step, error, energy,
    );

    if step % snapshot_interval == 0 {
      let oname = format!("{}_step_{}.json", snapshot_output_prefix, step);
      info!("Saving model snapshot {}", oname);

      save_model(model, &oname);
    }
  }

  let model_error = model.read_total_error();
  info!(
    "Final error of the model is {}",
    model_error
  );

  let model_energy = model.read_total_energy();
  info!(
    "Final energy of the model is {}",
    model_energy
  );
}
