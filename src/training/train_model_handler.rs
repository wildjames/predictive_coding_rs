//! Training orchestration for predictive coding models.

use crate::model::{
  model::{PredictiveCodingModel},
  model_utils::save_model
};
use crate::data_handling::data_handler;

use ndarray::Array1;
use tracing::info;

fn set_rand_input_and_output(
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

  let output_layer_size: usize = *model.get_layer_sizes().last().unwrap();
  let mut output_values: Array1<f32> = Array1::zeros(output_layer_size);
  output_values[output_label] = 1.0;

  model.set_input(input_values);
  model.set_output(output_values);

}

/// Run inference to convergence on a single sample and update weights.
pub fn train_and_update_model(model: &mut PredictiveCodingModel) {
  model.reinitialise_latents();
  // Train on this example until convergence.
  model.converge_values_with_updates();
  model.update_weights();
}


/// Train the model for a number of steps using randomly sampled data.
pub fn train(
  model: &mut PredictiveCodingModel,
  data: &data_handler::ImagesBWDataset,
  training_steps: u32,
  report_interval: u32,
  snapshot_interval: u32,
  snapshot_output_prefix: &str
) {
  // Supervised learning
  model.pin_input();
  model.pin_output();

  for step in 0..training_steps {
    set_rand_input_and_output(model, data);
    train_and_update_model(model);


    if step % report_interval == 0 {
      let error = model.read_total_error();
      let energy = model.read_total_energy();
      info!(
        "Sample {:.1}\terror {:.1}\tenergy {:.1}",
        step, error, energy,
      );
    }

    if step % snapshot_interval == 0 {
      let oname = format!("{}_step_{}.json", snapshot_output_prefix, step);
      info!("Saving model snapshot {}", oname);

      save_model(model, &oname);
    }
  }
}
