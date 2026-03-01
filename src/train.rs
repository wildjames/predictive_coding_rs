//! Training binary for the predictive coding model.

mod model;
mod model_utils;
mod data_handler;
mod train_model_handler;

use tracing::{Level, info};


/// Default value for enabling live plotting via environment variable.
const DEFAULT_USE_LIVE_PLOT: bool = false;


/// Entry point for loading data, building the model, and running training.
fn main() {
  setup_tracing();

  // Read environment variables
  let use_live_plot = std::env::var("USE_LIVE_PLOT")
    .map(|val| val == "true")
    .unwrap_or(DEFAULT_USE_LIVE_PLOT);


  let data: data_handler::ImagesBWDataset = data_handler::load_mnist(
      "data/mnist/train-images-idx3-ubyte",
      "data/mnist/train-labels-idx1-ubyte")
    .unwrap();
  info!(
    "Loaded the MNIST dataset. I have {} images",
    data.num_images
  );

  // Model params
  let input_layer_size = (data.image_width * data.image_height) as usize;
  let output_layer_size = 10; // for the 10 classes of digits
  let layer_sizes: Vec<usize> = vec![
    input_layer_size,
    256,
    128,
    64,
    output_layer_size
  ];

  let gamma: f32 = 0.0001;
  let alpha: f32 = 0.0001;

  // Training params
  let snapshot_interval: u32 = 100;
  let training_steps: u32 = 1000;
  let convergence_steps: u32 = 100;
  let convergence_threshold: f32 = 0.000001;

  // Build the model
  let mut model: model::PredictiveCodingModel = model::PredictiveCodingModel::new(
    &layer_sizes,
    gamma,
    alpha,
    model_utils::ActivationFunction::Relu
  );


  let model_error = model.read_total_error();
  info!(
    "Initial error of the model is {}",
    model_error
  );

  let model_energy = model.read_total_energy();
  info!(
    "Initial energy of the model is {}",
    model_energy
  );

  let training_func = if use_live_plot {
    info!("Using live plotting during training");
    train_model_handler::train_plotting_local
  } else {
    info!("Training without live plotting");
    train_model_handler::train
  };

   training_func(
      &mut model,
      &data,
      training_steps,
      convergence_steps,
      snapshot_interval,
      convergence_threshold
   );

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

  let model_ser = serde_json::to_string(&model).unwrap();
  std::fs::write("data/models/model.json", model_ser).unwrap();
}

/// Configure global tracing subscriber for logging.
fn setup_tracing() {
  // a builder for `FmtSubscriber`.
  let subscriber = tracing_subscriber::FmtSubscriber::builder()
    .with_max_level(Level::TRACE)
    .finish();

  tracing::subscriber::set_global_default(subscriber)
    .expect("Setting default subscriber failed");
}
