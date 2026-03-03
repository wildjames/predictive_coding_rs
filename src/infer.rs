use std::env;

use tracing::{error, info};
use ndarray::Array1;

use predictive_coding::{
  data_handling::data_handler,
  model::model_utils::{load_model},
  utils::logging
};


fn main() {
  logging::setup_tracing();

  let args: Vec<String> = env::args().collect();
  if args.len() != 2 {
    error!("Usage: infer <model_file>");
    std::process::exit(1);
  }

  let model_fname = args[1].clone();
  let mut model = load_model(&model_fname);
  info!("Loaded model from {}", model_fname);


  let data: data_handler::ImagesBWDataset = data_handler::load_mnist(
      "data/mnist/t10k-images-idx3-ubyte",
      "data/mnist/t10k-labels-idx1-ubyte")
    .unwrap();
  info!(
    "Loaded the MNIST testing dataset. I have {} images",
    data.num_images
  );

  let rand_index = usize::from_ne_bytes(rand::random()) % data.num_images;
  let input_values: Array1<f32> = data.images
    .row(rand_index)
    .mapv(|x| x as f32 / 255.0)
    .to_owned();

  let output_label = data.labels[rand_index];
  info!(
    "Selected random image with index {} and label {}",
    rand_index,
    output_label
  );

  model.set_input(input_values);
  model.unpin_output();

  model.converge_values_with_updates(0.000001, 1000);

  let output_activations = model.layers.last().unwrap().values.clone();
  output_activations
    .iter()
    .enumerate()
    .for_each(|(i, &activation)| {
      info!("Output node {}: activation {}", i, activation);
    });
}
