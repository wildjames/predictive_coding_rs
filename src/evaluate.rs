use tracing::{info};
use ndarray::Array1;
use std::time::Instant;

use predictive_coding::{
  data_handling::data_handler,
  model::model_utils::{create_from_config},
  utils::logging
};

use clap::Parser;

/// Evaluate a trained model against the MNIST test dataset, and report its accuracy.
#[derive(Parser)]
struct EvalArgs {
  /// The model file to evaluate
  #[arg(short, long)]
  model_file: String,
}


fn main() {
  logging::setup_tracing(false);

  let args = EvalArgs::parse();

  let mut model = create_from_config(&args.model_file);
  info!("Loaded model from {}", args.model_file);


  let data: data_handler::ImagesBWDataset = data_handler::load_mnist(
      "data/mnist/t10k-images-idx3-ubyte",
      "data/mnist/t10k-labels-idx1-ubyte")
    .unwrap();
  info!(
    "Loaded the MNIST testing dataset. I have {} images",
    data.num_images
  );

  // The output must be unpinned for evaluation
  // It was probably pinned during training, so just check
  model.unpin_output();

  let mut correct_predictions: usize = 0;
  let mut total_predictions: usize = 0;
  let mut confidence_sum: f32 = 0.0;
  let mut sum_covnvergence_time: f32 = 0.0;

  for i in 0..data.num_images {
    let input_values: Array1<f32> = data.images
      .row(i)
      .mapv(|x| x as f32 / 255.0)
      .to_owned();

    let output_label = data.labels[i];

    model.reinitialise_latents();
    model.set_input(input_values);
    let start_time = Instant::now();
    model.converge_values_with_updates();
    let elapsed_time = start_time.elapsed();

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
      confidence_sum += output_activations[predicted_label];
    }
    total_predictions += 1;
    sum_covnvergence_time += elapsed_time.as_millis() as f32;
  }

  let accuracy = correct_predictions as f32 / total_predictions as f32;
  let mean_convergence_time = sum_covnvergence_time / total_predictions as f32;
  let mean_confidence = if correct_predictions > 0 {
  confidence_sum / correct_predictions as f32
  } else {
    0.0
  };

  info!(
    "Evaluation complete. Accuracy: {:.2}%, convergence time on average is {:.0}ms When correct, average confidence: {:.3}",
    accuracy * 100.0,
    mean_convergence_time,
    mean_confidence
  );
  // Write to a file in the model directory
  let output_dir = std::path::Path::new(&args.model_file).parent().unwrap();
  let output_path = output_dir.join("evaluation_results.json");
  serde_json::to_writer_pretty(
    std::fs::File::create(output_path).unwrap(),
    &serde_json::json!({
      "accuracy": accuracy,
      "mean_convergence_time_ms": mean_convergence_time,
      "mean_confidence_when_correct": mean_confidence,
      "correct_predictions": correct_predictions,
      "total_predictions": total_predictions,
      "model_file": args.model_file,
      "config": model.get_config(),
    }),
  ).unwrap();
}
