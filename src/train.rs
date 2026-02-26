mod training_data_handler;
mod model;
mod model_utils;

use tracing::{info, Level};


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
  let gamma: f32 = 0.01;
  let alpha: f32 = 0.01;

  let mut model = model::PredictiveCodingModel::new(
    &layer_sizes,
    gamma,
    alpha,
    model_utils::ReLU
  );
  // When setting the input, normalise the pixel values to 0..1
  model.set_input(data.images[rand_index].iter().map(|&x| x as f32 / 255.0).collect());

  model.compute_errors();
  let model_error = model.compute_total_error();
  info!(
    "Initial error of the model is {}",
    model_error
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
