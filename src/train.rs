mod training_data_handler;
mod model;
mod model_utils;

use tracing::{Level, info, debug};


fn main() {
  setup_tracing();

  let data = training_data_handler::load_mnist(
      "data/mnist/train-images-idx3-ubyte",
      "data/mnist/train-labels-idx1-ubyte")
    .unwrap();
  info!(
    "Loaded the MNIST dataset. I have {} images",
    data.num_images
  );

  // Display a random image
  let rand_index = usize::from_ne_bytes(rand::random()) % data.num_images;
  training_data_handler::output_image(
    &data,
    rand_index,
    format!("data/images/output_{}_label_{}.png", rand_index, data.labels[rand_index]))
    .unwrap();


  let layer_sizes: Vec<usize> = vec![28*28, 256, 128, 64, 10];
  let gamma: f32 = 0.0001;
  let alpha: f32 = 0.01;

  let mut model = model::PredictiveCodingModel::new(
    &layer_sizes,
    gamma,
    alpha,
    model_utils::relu
  );

  // When setting the input, normalise the pixel values to 0..1
  let input_row = data.images.row(rand_index).mapv(|x| x as f32 / 255.0).to_owned();
  model.set_input(input_row);

  let model_energy_history: Vec<f32> = Vec::new();
  let model_error_history: Vec<f32> = Vec::new();
  let model_value_change_histories: Vec<Vec<f32>> = Vec::new();

  model.compute_errors();
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

  let training_steps: u32 = 1;
  let convergence_steps: u32 = 1000;
  let convergence_threshold: f32 = 0.0;

  for step in 0..training_steps {
    let mut converged: bool = false;
    let mut convergence_count: u32 = 0;

    while !converged && convergence_count < convergence_steps {
      let value_change = model.timestep();

      debug!("Step {}, convergence count {}, value change {}", step, convergence_count, value_change);
      if value_change.abs() < convergence_threshold {
        info!("Model converged after {} timesteps", convergence_count);
        converged = true;
      }
      convergence_count += 1;
    };

    model.update_weights();
    model.compute_predictions();
    model.compute_errors();
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

fn setup_tracing() {
  // a builder for `FmtSubscriber`.
  let subscriber = tracing_subscriber::FmtSubscriber::builder()
    .with_max_level(Level::TRACE)
    .finish();

  tracing::subscriber::set_global_default(subscriber)
    .expect("Setting default subscriber failed");
}
