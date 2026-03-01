//! Training orchestration for predictive coding models.

use crate::model;
use crate::train_data_handler;

use ndarray::Array1;
use tracing::{info, debug};
use std::time::{SystemTime, UNIX_EPOCH};

use liveplot::{
  LivePlotConfig,
  PlotPoint,
  channel_plot,
  run_liveplot
};


/// Run inference to convergence on a single sample and update weights.
fn converge_sample(
  model: &mut model::PredictiveCodingModel,
  input_values: Array1<f32>,
  output_values: Array1<f32>,
  convergence_threshold: f32,
  convergence_steps: u32
) {

  model.set_input(input_values);
  model.set_output(output_values);

  // Train on this example until convergence.
  model.converge_values(convergence_threshold, convergence_steps);
  model.compute_predictions_and_errors();
  model.update_weights();
}


/// Train the model for a number of steps using randomly sampled data.
pub fn train(
  model: &mut model::PredictiveCodingModel,
  data: &train_data_handler::ImagesBWDataset,
  training_steps: u32,
  convergence_steps: u32,
  convergence_threshold: f32
) {

  let output_layer_size = model.layers.last().unwrap().size;

  for step in 0..training_steps {
    let rand_index = usize::from_ne_bytes(rand::random()) % data.num_images;

    // Normalise to the range 0..1
    let input_values: Array1<f32> = data.images
      .row(rand_index)
      .mapv(|x| x as f32 / 255.0)
      .to_owned();

    // One-hot output row with label value set to 1.0
    let output_label: usize = data.labels[rand_index] as usize;
    debug!("Training on sample {}, with label {}", rand_index, output_label);
    let mut output_values: Array1<f32> = Array1::zeros(output_layer_size);
    output_values[output_label] = 1.0;

    converge_sample(
      model,
      input_values,
      output_values,
      convergence_threshold,
      convergence_steps
    );

    debug!(
      "Step {}, error {}, energy {}",
      step,
      model.read_total_energy(),
      model.read_total_error(),
    );
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

/// Train with a local live plot of model energy.
pub fn train_plotting_local(
  model: &mut model::PredictiveCodingModel,
  data: &train_data_handler::ImagesBWDataset,
  training_steps: u32,
  convergence_steps: u32,
  convergence_threshold: f32) {

  let (sink, rx) = channel_plot();
  let trace = sink.create_trace("Model Energy", Some("Model Energy"));

  // Run the dataset within a worker thread, since the plotter wants the main one.
  std::thread::scope(|s| {
    s.spawn(move || {
      let output_layer_size = model.layers.last().unwrap().size;

      for step in 0..training_steps {
        let rand_index = usize::from_ne_bytes(rand::random()) % data.num_images;

        // Normalise to the range 0..1
        let input_values: Array1<f32> = data.images
          .row(rand_index)
          .mapv(|x| x as f32 / 255.0)
          .to_owned();

        // One-hot output row with label value set to 1.0
        let output_label: usize = data.labels[rand_index] as usize;
        debug!("Training on sample {}, with label {}", rand_index, output_label);
        let mut output_values: Array1<f32> = Array1::zeros(output_layer_size);
        output_values[output_label] = 1.0;

        converge_sample(
          model,
          input_values,
          output_values,
          convergence_threshold,
          convergence_steps
        );

        let energy = model.read_total_energy();
        let t_s = SystemTime::now()
          .duration_since(UNIX_EPOCH)
          .map(|d| d.as_secs_f64())
          .unwrap_or(0.0);
        let _ = sink.send_point(&trace, PlotPoint { x: t_s, y: energy as f64 });

        debug!(
          "Step {}, error {}, energy {}",
          step,
          model.read_total_energy(),
          model.read_total_error(),
        );
      }
    });

    // Run the plotting in the main thread
    run_liveplot(rx, LivePlotConfig::default())
      .expect("Failed to create the plot window. If you are on SSH, verify your VcXsrv configuration and DISPLAY settings.");
  });
}

/// Placeholder for gRPC-based live plotting.
pub fn train_plotting_grpc() {}
