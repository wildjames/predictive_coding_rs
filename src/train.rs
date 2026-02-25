mod training_data_handler;

use tracing::{info, Level};


fn main() {
  setup_tracing();

  let data = training_data_handler::load_mnist(
      "data/mnist/train-images-idx3-ubyte",
      "data/mnist/train-labels-idx1-ubyte")
    .unwrap();
  info!(
    target: "predictive-coding",
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
}

fn setup_tracing() {
  // a builder for `FmtSubscriber`.
  let subscriber = tracing_subscriber::FmtSubscriber::builder()
    .with_max_level(Level::TRACE)
    .finish();

  tracing::subscriber::set_global_default(subscriber)
    .expect("Setting default subscriber failed");
}
