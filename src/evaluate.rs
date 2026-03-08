//! This program takes a trained modek, and evaluates it against the MNIST test dataset, reporting its accuracy, convergence time, and confidence when correct. It also saves these results to a file in the same directory as the model.

use std::{path::Path, time::Instant};
use tracing::info;

use predictive_coding::{
  data_handling::{data_handler::TrainingDataset, mnist::{MnistDataset, load_mnist}},
  error::{PredictiveCodingError, Result},
  model_structure::configuration::load_model_snapshot,
  utils::logging
};

use clap::Parser;

/// Evaluate a trained model against the MNIST test dataset, and report its accuracy.
#[derive(Parser)]
struct EvalArgs {
  /// The model file to evaluate
  #[arg()]
  model_file: String,

  /// IDX image file to evaluate against.
  #[arg(long, default_value_t = String::from("data/mnist/t10k-images-idx3-ubyte"))]
  input_idx_file: String,

  /// IDX label file to evaluate against.
  #[arg(long, default_value_t = String::from("data/mnist/t10k-labels-idx1-ubyte"))]
  output_idx_file: String,
}


fn main() -> Result<()> {
  logging::setup_tracing(false);

  let args = EvalArgs::parse();

  let mut model = load_model_snapshot(&args.model_file)?;
  info!("Loaded model from {}", args.model_file);


  let data: MnistDataset = load_mnist(
      &args.input_idx_file,
      &args.output_idx_file
  )?;
  info!(
    "Loaded the MNIST testing dataset. I have {} images",
    data.get_dataset_size()
  );

  if data.get_dataset_size() == 0 {
    return Err(PredictiveCodingError::invalid_data("evaluation dataset is empty"));
  }

  // The output must be unpinned for evaluation
  // It was probably pinned during training, so just check
  model.unpin_output();

  let mut correct_predictions: usize = 0;
  let mut total_predictions: usize = 0;
  let mut confidence_sum: f32 = 0.0;
  let mut sum_covnvergence_time: f32 = 0.0;

  // TODO: Multithread this
  for i in 0..data.get_dataset_size() {
    let (input_values, output_values) = data.get_random_input_and_output();

    // MNIST is one-hot encoded on the index that the number correspoinse to
    let output_label: usize = output_values
      .iter()
      .enumerate()
      .max_by(|a, b| a.1.total_cmp(b.1))
      .map(|(i, _)| i)
      .ok_or_else(|| PredictiveCodingError::invalid_data("dataset produced an empty output label"))?;

    model.reinitialise_latents();
    model.set_input(input_values);
    let start_time = Instant::now();
    model.converge_values();
    let elapsed_time = start_time.elapsed();

    let output_activations = model.get_output();
    let predicted_label = output_activations
      .iter()
      .enumerate()
      .max_by(|a, b| a.1.total_cmp(b.1))
      .map(|(i, _)| i)
      .ok_or_else(|| PredictiveCodingError::invalid_data("model produced an empty output layer"))?;

    if (i > 0) && (i % 1000 == 0) {
      info!(
        "Current accuracy after {} samples: {:.2}%",
        i,
        correct_predictions as f32 / total_predictions as f32 * 100.0
      );
    }

    if predicted_label == output_label {
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
  let output_dir = Path::new(&args.model_file)
    .parent()
    .filter(|path| !path.as_os_str().is_empty())
    .unwrap_or_else(|| Path::new("./evaluation_results")); // fallback

  let output_path = output_dir.join("evaluation_results.json");
  let output_file = std::fs::File::create(&output_path)
    .map_err(|source| PredictiveCodingError::io("create evaluation results", &output_path, source))?;
  serde_json::to_writer_pretty(
    output_file,
    &serde_json::json!({
      "accuracy": accuracy,
      "mean_convergence_time_ms": mean_convergence_time,
      "mean_confidence_when_correct": mean_confidence,
      "correct_predictions": correct_predictions,
      "total_predictions": total_predictions,
      "model_file": args.model_file,
      "config": model.get_config(),
    }),
  ).map_err(|source| PredictiveCodingError::json_serialize(&output_path, source))?;

  Ok(())
}
