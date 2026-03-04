use std::env;

use tracing::{error, info};
use ndarray::Array1;

use predictive_coding::{
  data_handling::data_handler,
  model::model_utils::{load_model},
  utils::logging
};


fn main() {
  logging::setup_tracing(false);

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

  model.unpin_output();

  let mut correct_predictions: usize = 0;
  let mut total_predictions: usize = 0;

  for i in 0..data.num_images {
    let input_values: Array1<f32> = data.images
      .row(i)
      .mapv(|x| x as f32 / 255.0)
      .to_owned();

    let output_label = data.labels[i];

    model.reinitialise_latents();
    model.set_input(input_values);
    model.converge_values_with_updates();

    let output_activations = model.get_output();
    let predicted_label = output_activations
      .iter()
      .enumerate()
      .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
      .map(|(i, _)| i)
      .unwrap();

    if i > 0 && i % 1000 == 0 {
      info!(
        "Current accuracy after {} samples: {:.2}%",
        i,
        correct_predictions as f32 / total_predictions as f32 * 100.0
      );
    }

    if predicted_label == output_label as usize {
      correct_predictions += 1;
    }
    total_predictions += 1;
  }

  let accuracy = correct_predictions as f32 / total_predictions as f32;
  info!(
    "Evaluation complete. Accuracy: {:.2}%",
    accuracy * 100.0
  );
}
