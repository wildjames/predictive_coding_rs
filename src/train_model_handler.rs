//! Training orchestration for predictive coding models.

use crate::model;
use crate::train_data_handler;

use ndarray::Array1;
use tracing::{info, debug};

use liveplot::{
  data::x_formatter::{XFormatter, DecimalFormatter},
  LivePlotConfig,
  AutoFitConfig,
  PlotPoint,
  channel_plot,
  run_liveplot
};

fn set_input_and_output(
  model: &mut model::PredictiveCodingModel,
  data: &train_data_handler::ImagesBWDataset
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
fn converge_sample(
  model: &mut model::PredictiveCodingModel,
  data: &train_data_handler::ImagesBWDataset,
  convergence_threshold: f32,
  convergence_steps: u32
) {
  set_input_and_output(model, data);
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


  for step in 0..training_steps {
    converge_sample(
      model,
      data,
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
      for step in 0..training_steps {
        converge_sample(
          model,
          data,
          convergence_threshold,
          convergence_steps
        );

        let error = model.read_total_error();
        let energy = model.read_total_energy();
        let step_x = step as f64;
        // skip the first sample, since it will have massive energy
        if step > 0 {
          let _ = sink.send_point(&trace, PlotPoint { x: step_x, y: energy as f64 });
        }

        debug!(
          "Step {}, error {}, energy {}",
          step, error, energy,
        );
      }
    });

    // Run the plotting in the main thread
    let auto_fit = AutoFitConfig {
      auto_fit_to_view: true,
      keep_max_fit: true
    };

    let x_formatter = XFormatter::Decimal(DecimalFormatter {
      decimal_places: Some(0),
      unit: None
    });

    let cfg = LivePlotConfig {
      time_window_secs: training_steps as f64,
      max_points: training_steps as usize,
      x_formatter,
      auto_fit,
      ..Default::default()
    };

    run_liveplot(rx, cfg)
      .expect("Failed to create the plot window. If you are on SSH, verify your VcXsrv configuration and DISPLAY settings.");
  });
}

/// Placeholder for gRPC-based live plotting.
pub fn train_plotting_grpc() {}
