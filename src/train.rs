//! Training binary for the predictive coding model.

use predictive_coding::{
  data_handling::data_handler,
  model::{
    model_utils::{ActivationFunction, save_model},
    model::{PredictiveCodingModel}
  },
  training::train_model_handler,
  utils::logging
};

use tracing::info;


/// Entry point for loading data, building the model, and running training.
fn main() {
  logging::setup_tracing(false);

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
    128,
    64,
    output_layer_size
  ];


  // Training params
  let snapshot_interval: u32 = 100;
  let gamma: f32 = 0.05;
  let alpha: f32 = 0.001;
  let training_steps: u32 = data.num_images as u32; // full dataset
  let convergence_steps: u32 = 50;
  let convergence_threshold: f32 = 1e-5;
  let activation_function: ActivationFunction = ActivationFunction::Relu;

  info!(
    "Training hyperparameters:\n\ttraining steps: {}\n\tconvergence steps: {}\n\tconvergence threshold: {}\n\tgamma: {}\n\talpha: {}\n\tactivation function: {:?}\n\tsnapshot interval: {}",
    training_steps,
    convergence_steps,
    convergence_threshold,
    gamma,
    alpha,
    activation_function,
    snapshot_interval
  );

  // Build the model
  let mut model: PredictiveCodingModel = PredictiveCodingModel::new(
    &layer_sizes,
    gamma,
    alpha,
    activation_function
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

  let training_func = train_model_handler::train;
  let fname_base: String = format!("data/model_snapshots_{}/model", chrono::Utc::now().timestamp());

  training_func(
    &mut model,
    &data,
    training_steps,
    convergence_steps,
    convergence_threshold,
    snapshot_interval,
    &fname_base,
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

  save_model(&model, &format!("{}_{}", fname_base, "final.json"));
}
