//! This program takes a trained modek, and evaluates it against the MNIST test dataset, reporting its accuracy, convergence time, and confidence when correct. It also saves these results to a file in the same directory as the model.

use std::{
  path::Path,
  time::Instant,
  sync::Arc
};

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use tracing::info;

use predictive_coding::{
  data_handling::data_handler::TrainingDataset,
  error::{
    PredictiveCodingError,
    Result
  },
  model_structure::{
    configuration::load_model_snapshot,
    model::PredictiveCodingModel
  },
  training::configuration::{
    DataSetSource,
    TrainConfig,
    load_dataset,
    load_training_config
  },
  utils::logging
};

use clap::Parser;

/// Evaluate a trained model against the evaluation dataset defined in the training configuration, and report its accuracy.
#[derive(Parser)]
struct EvalArgs {
  /// The model file to evaluate
  #[arg()]
  model_file: String,

  /// The training config file used to train the model defines an evaluation dataset.
  #[arg(long)]
  training_config: String
}

#[derive(Clone, Copy, Deserialize, Serialize, Debug, PartialEq)]
struct EvaluationSummary {
  accuracy: f32,
  mean_convergence_time_ms: f32,
  mean_confidence_when_correct: f32,
  correct_predictions: usize,
  total_predictions: usize,
}

fn summarise_evaluation_samples<F>(dataset_size: usize, mut evaluate_sample: F) -> Result<EvaluationSummary>
where
  F: FnMut(usize) -> Result<(usize, usize, f32, f32)>,
{
  if dataset_size == 0 {
    return Err(PredictiveCodingError::invalid_data("evaluation dataset is empty"));
  }

  let mut correct_predictions: usize = 0;
  let mut total_predictions: usize = 0;
  let mut confidence_sum: f32 = 0.0;
  let mut sum_covnvergence_time: f32 = 0.0;

  for i in 0..dataset_size {
    let (output_label, predicted_label, predicted_confidence, elapsed_time_ms) = evaluate_sample(i)?;

    if (i > 0) && (i % 1000 == 0) {
      let accuracy_percent = correct_predictions as f32 / total_predictions as f32 * 100.0;
      info!(
        "Current accuracy after {} samples: {:.2}%",
        i,
        accuracy_percent
      );
    }

    if predicted_label == output_label {
      correct_predictions += 1;
      confidence_sum += predicted_confidence;
    }
    total_predictions += 1;
    sum_covnvergence_time += elapsed_time_ms;
  }

  let accuracy: f32 = correct_predictions as f32 / total_predictions as f32;
  let mean_convergence_time: f32 = sum_covnvergence_time / total_predictions as f32;
  let mean_confidence: f32 = if correct_predictions > 0 {
    confidence_sum / correct_predictions as f32
  } else {
    0.0
  };

  Ok(EvaluationSummary {
    accuracy,
    mean_convergence_time_ms: mean_convergence_time,
    mean_confidence_when_correct: mean_confidence,
    correct_predictions,
    total_predictions,
  })
}


fn evaluate_sample(model: &mut PredictiveCodingModel, data: &dyn TrainingDataset, i: usize) -> Result<(usize, usize, f32, f32)> {
    let input_values: Array1<f32> = data.get_input(i);
    let output_values: Array1<f32> = data.get_output(i);

    let output_label: usize = output_values
      .iter()
      .enumerate()
      .max_by(|a, b| a.1.total_cmp(b.1))
      .map(|(index, _)| index)
      .ok_or_else(|| PredictiveCodingError::invalid_data("dataset produced an empty output label"))?;

    model.reinitialise_latents();
    model.set_input(input_values);
    let start_time = Instant::now();
    model.converge_values();
    let elapsed_time = start_time.elapsed();

    let output_activations: &Array1<f32> = model.get_output();
    let predicted_label: usize = output_activations
      .iter()
      .enumerate()
      .max_by(|a, b| a.1.total_cmp(b.1))
      .map(|(index, _)| index)
      .ok_or_else(|| PredictiveCodingError::invalid_data("model produced an empty output layer"))?;

    Ok((
      output_label,
      predicted_label,
      output_activations[predicted_label],
      elapsed_time.as_millis() as f32,
    ))
  }


fn main() -> Result<()> {
  logging::setup_tracing(false);

  let args = EvalArgs::parse();

  let mut model: PredictiveCodingModel = load_model_snapshot(&args.model_file)?;
  info!("Loaded model from {}", args.model_file);

  let training_config: TrainConfig = load_training_config(&args.training_config)?;
  let evaluation_source: DataSetSource = training_config
    .evaluation_dataset
    .ok_or_else(
      || PredictiveCodingError::invalid_data(
        "training config did not contain an evaluation dataset")
    )?;

  let data: Arc<dyn TrainingDataset>  = load_dataset(&evaluation_source)?;
  info!("Loaded evaluation dataset from config file {}", args.training_config);

  // The output must be unpinned for evaluation
  // It was probably pinned during training, so just check
  model.unpin_output();
  let summary: EvaluationSummary = summarise_evaluation_samples(
    data.get_dataset_size(),
    |i| evaluate_sample(&mut model, data.as_ref(), i)
  )?;

  info!(
    "Evaluation complete. Accuracy: {:.2}%, convergence time on average is {:.0}ms. When correct, average confidence: {:.3}",
    summary.accuracy * 100.0,
    summary.mean_convergence_time_ms,
    summary.mean_confidence_when_correct
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
      "summary": summary,
      "model_file": args.model_file,
      "config": model.get_config(),
    }),
  ).map_err(|source| PredictiveCodingError::json_serialize(&output_path, source))?;

  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn summarise_evaluation_samples_rejects_empty_datasets() {
    let result = summarise_evaluation_samples(0, |_| Ok((0, 0, 0.0, 0.0)));

    assert!(matches!(
      result,
      Err(PredictiveCodingError::InvalidData { message })
        if message == "evaluation dataset is empty"
    ));
  }

  #[test]
  fn summarise_evaluation_samples_accumulates_accuracy_confidence_and_time() {
    let summary = summarise_evaluation_samples(1001, |_| Ok((1, 1, 0.8, 2.0))).unwrap();

    assert_eq!(summary.correct_predictions, 1001);
    assert_eq!(summary.total_predictions, 1001);
    assert_eq!(summary.accuracy, 1.0);
    assert!((summary.mean_confidence_when_correct - 0.8).abs() < 1e-5);
    assert_eq!(summary.mean_convergence_time_ms, 2.0);
  }

  #[test]
  fn summarise_evaluation_samples_returns_zero_confidence_when_all_predictions_are_wrong() {
    let summary = summarise_evaluation_samples(2, |i| Ok((i, i + 1, 0.9, 3.0))).unwrap();

    assert_eq!(summary.correct_predictions, 0);
    assert_eq!(summary.total_predictions, 2);
    assert_eq!(summary.accuracy, 0.0);
    assert_eq!(summary.mean_confidence_when_correct, 0.0);
    assert_eq!(summary.mean_convergence_time_ms, 3.0);
  }
}
