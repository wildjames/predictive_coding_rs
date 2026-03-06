//! Training orchestration for predictive coding models.

use crate::model_structure::{
  model::{PredictiveCodingModel},
  model_utils::save_model
};
use crate::data_handling::data_handler;
use crate::training::utils::{set_rand_input_and_output};


use ndarray::{Array2};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use tracing::info;


/// Run inference to convergence on a single sample and update weights.
pub fn train_and_update_model(model: &mut PredictiveCodingModel) {
  model.reinitialise_latents();
  // Train on this example until convergence.
  model.converge_values();
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
      let error: f32 = model.read_total_error();
      let energy: f32 = model.read_total_energy();
      info!(
        "Sample {:.1}\terror {:.1}\tenergy {:.1}",
        step, error, energy,
      );
    }

    if step % snapshot_interval == 0 {
      let oname: String = format!("{}_step_{}.json", snapshot_output_prefix, step);
      info!("Saving model snapshot {}", oname);

      save_model(model, &oname);
    }
  }
}

/// Train B models in parallel on different data, then compute the average weight update across them and apply it to the model.
pub fn mini_batch_train(
  model: &mut PredictiveCodingModel,
  data: &data_handler::ImagesBWDataset,
  batch_size: u32,
  training_steps: u32,
  report_interval: u32,
  snapshot_interval: u32,
  snapshot_output_prefix: &str
) {
  // Supervised learning
  model.pin_input();
  model.pin_output();

  for step in 0..training_steps {
    let batch_indexes: Vec<u32> = (0..batch_size).collect();

    let batch_weight_changes: Vec<Vec<Array2<f32>>> = batch_indexes
      .into_par_iter()
      .map(|i| {
        let mut model_clone = model.clone();
        set_rand_input_and_output(&mut model_clone, data);
        train_and_update_model(&mut model_clone);

        // Do reporting on the 0th sample in the batch, since the outer model never gets inference run on it
        // and so its error and energy values are never updated
        if (i == 0) && (step % report_interval == 0) {
            let error: f32 = model_clone.read_total_error();
            let energy: f32 = model_clone.read_total_energy();
            info!(
              "Sample {:.1}\terror {:.1}\tenergy {:.1}",
              step, error, energy,
            );
          }

        // Weight updates (Array2) for each layer in the model
        model_clone.compute_weight_updates()
      })
      .collect();

    // Average the weight changes across the batch. Sum outer Vec
    let sum_batch_weight_changes: Vec<Array2<f32>> = batch_weight_changes
      .into_iter()
      .reduce(|acc, weight_changes| {
        acc.into_iter()
          .zip(weight_changes.into_iter())
          .map(|(sum_weight_change, weight_change)| sum_weight_change + weight_change)
          .collect()
      })
      .unwrap();

    let avg_batch_weight_changes: Vec<Array2<f32>> = sum_batch_weight_changes
      .into_iter()
      .map(|sum_weight_change: Array2<f32>| sum_weight_change / batch_size as f32)
      .collect();

    // Apply the average weight changes to the model.
    model.apply_weight_updates(avg_batch_weight_changes);

    if step % snapshot_interval == 0 {
      let oname: String = format!("{}_step_{}.json", snapshot_output_prefix, step);
      info!("Saving model snapshot {}", oname);

      save_model(model, &oname);
    }
  }
}

