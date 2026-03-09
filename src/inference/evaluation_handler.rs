use std::{
	path::{Path, PathBuf},
	sync::Arc
};

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::{
	data_handling::TrainingDataset,
  error::{PredictiveCodingError, Result},
  inference::inference_model_handler::{
    InferenceModelHandler,
    InferencePrediction,
    read_label
  },
  model_structure::model::{
    PredictiveCodingModel,
    PredictiveCodingModelConfig
  },
  training::{
    configuration::{
      load_dataset,
      load_training_config
    },
    validation::validate_model_and_dataset_shapes
  }
};


#[derive(Clone, Copy, Deserialize, Serialize, Debug, PartialEq)]
pub struct EvaluationSummary {
	pub accuracy: f32,
	pub mean_convergence_time_ms: f32,
	pub mean_confidence_when_correct: f32,
	pub correct_predictions: usize,
	pub total_predictions: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedSample {
	input_values: Array1<f32>,
	expected_label: usize,
}

impl PreparedSample {
	pub fn from_dataset(dataset: &dyn TrainingDataset, index: usize) -> Result<Self> {
		let input_values = dataset.get_input(index);
		let output_values = dataset.get_output(index);
		let expected_label = read_label(&output_values, "dataset produced an empty output label")?;

		Ok(PreparedSample {
			input_values,
			expected_label,
		})
	}

	pub fn expected_label(&self) -> usize {
		self.expected_label
	}

	pub fn input_values(&self) -> &Array1<f32> {
		&self.input_values
	}

	pub fn into_input_values(self) -> Array1<f32> {
		self.input_values
	}
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvaluationArtifacts {
	pub output_path: PathBuf,
	pub summary: EvaluationSummary,
	pub model_file: String,
	pub config: PredictiveCodingModelConfig,
}

pub struct EvaluationRun {
	model_file: String,
	training_config_path: String,
	dataset: Arc<dyn TrainingDataset>,
	handler: InferenceModelHandler,
}

impl EvaluationRun {
	pub fn load(
		model_file: impl Into<String>,
		training_config_path: impl Into<String>,
	) -> Result<Self> {
		let model_file = model_file.into();
		let training_config_path = training_config_path.into();
		let training_config = load_training_config(&training_config_path)?;
		let evaluation_source = training_config.evaluation_dataset.ok_or_else(|| {
			PredictiveCodingError::invalid_data(
				"training config did not contain an evaluation dataset"
			)
		})?;

		let dataset: Arc<dyn TrainingDataset> = load_dataset(&evaluation_source)?;
		let handler = InferenceModelHandler::load_snapshot(&model_file)?;

		info!("Loaded model from {}", model_file);
		info!(
			"Loaded evaluation dataset from config file {}",
			training_config_path
		);

		Ok(EvaluationRun {
			model_file,
			training_config_path,
			dataset,
			handler,
		})
	}

	pub fn evaluate(&mut self) -> Result<EvaluationSummary> {
		evaluate_dataset(&mut self.handler, self.dataset.as_ref())
	}

	pub fn evaluate_and_write_results(&mut self) -> Result<EvaluationArtifacts> {
		let summary = self.evaluate()?;
		let output_path = save_evaluation_results(&self.model_file, self.handler.model(), &summary)?;
		let config = self.handler.model().get_config();

		Ok(EvaluationArtifacts {
			output_path,
			summary,
			model_file: self.model_file.clone(),
			config,
		})
	}

	pub fn model_file(&self) -> &str {
		&self.model_file
	}

	pub fn training_config_path(&self) -> &str {
		&self.training_config_path
	}
}

pub fn evaluate_dataset(
	handler: &mut InferenceModelHandler,
	dataset: &dyn TrainingDataset,
) -> Result<EvaluationSummary> {
	validate_model_and_dataset_shapes(handler.model(), dataset)?;

	summarise_evaluation_samples(dataset.get_dataset_size(), |index| {
		let sample = PreparedSample::from_dataset(dataset, index)?;
		let expected_label: usize = sample.expected_label();
		let prediction: InferencePrediction = handler.infer(sample.into_input_values())?;

		Ok((
			expected_label,
			prediction.predicted_label,
			prediction.confidence,
			prediction.elapsed_time_ms,
		))
	})
}

pub fn evaluation_results_path(model_file: &str) -> PathBuf {
	let output_dir = Path::new(model_file)
		.parent()
		.filter(|path| !path.as_os_str().is_empty())
		.map(Path::to_path_buf)
		.unwrap_or_else(|| PathBuf::from("./evaluation_results"));

	output_dir.join("evaluation_results.json")
}

pub fn save_evaluation_results(
	model_file: &str,
	model: &PredictiveCodingModel,
	summary: &EvaluationSummary,
) -> Result<PathBuf> {
	let output_path = evaluation_results_path(model_file);

	if let Some(output_dir) = output_path.parent() && !output_dir.as_os_str().is_empty() {
		std::fs::create_dir_all(output_dir)
			.map_err(|source| PredictiveCodingError::io("create evaluation artifact directory", output_dir, source))?;
	}

	let output_file = std::fs::File::create(&output_path)
		.map_err(|source| PredictiveCodingError::io("create evaluation results", &output_path, source))?;
	serde_json::to_writer_pretty(
		output_file,
		&serde_json::json!({
			"summary": summary,
			"model_file": model_file,
			"config": model.get_config(),
		}),
	).map_err(|source| PredictiveCodingError::json_serialize(&output_path, source))?;

	Ok(output_path)
}

fn summarise_evaluation_samples<F>(
	dataset_size: usize,
	mut evaluate_sample: F,
) -> Result<EvaluationSummary>
where
	F: FnMut(usize) -> Result<(usize, usize, f32, f32)>,
{
	if dataset_size == 0 {
		return Err(PredictiveCodingError::invalid_data("evaluation dataset is empty"));
	}

	let mut correct_predictions: usize = 0;
	let mut total_predictions: usize = 0;
	let mut confidence_sum: f32 = 0.0;
	let mut convergence_time_sum: f32 = 0.0;

	for index in 0..dataset_size {
		let (output_label, predicted_label, predicted_confidence, elapsed_time_ms) =
			evaluate_sample(index)?;

		if (index > 0) && (index % 1000 == 0) {
			let accuracy_percent = correct_predictions as f32 / total_predictions as f32 * 100.0;
			info!(
				"Current accuracy after {} samples: {:.2}%",
				index,
				accuracy_percent
			);
		}

		if predicted_label == output_label {
			correct_predictions += 1;
			confidence_sum += predicted_confidence;
		}
		total_predictions += 1;
		convergence_time_sum += elapsed_time_ms;
	}

	let accuracy: f32 = correct_predictions as f32 / total_predictions as f32;
	let mean_convergence_time_ms: f32 = convergence_time_sum / total_predictions as f32;
	let mean_confidence_when_correct: f32 = if correct_predictions > 0 {
		confidence_sum / correct_predictions as f32
	} else {
		0.0
	};

	Ok(EvaluationSummary {
		accuracy,
		mean_convergence_time_ms,
		mean_confidence_when_correct,
		correct_predictions,
		total_predictions,
	})
}


#[cfg(test)]
mod tests {
	use crate::{error::PredictiveCodingError, inference::{evaluation_handler::{EvaluationSummary, PreparedSample, evaluation_results_path, summarise_evaluation_samples}, inference_model_handler::{InferenceModelHandler, read_prediction}}, test_utils::{DummyTrainingDataset, TempDir, tiny_relu_model}};
	use ndarray::array;

	#[test]
	fn prepared_sample_reads_expected_label_from_dataset() {
		let dataset = DummyTrainingDataset::from_arrays(
			array![[1.0, 0.0, 0.5, 0.25]],
			array![[0.0, 1.0, 0.0]],
		);

		let sample = PreparedSample::from_dataset(&dataset, 0).unwrap();

		assert_eq!(sample.expected_label(), 1);
		assert_eq!(sample.input_values(), &array![1.0, 0.0, 0.5, 0.25]);
	}

	#[test]
	fn read_prediction_uses_highest_output_activation() {
		let prediction = read_prediction(&array![0.1, 0.9, 0.4], 3.5).unwrap();

		assert_eq!(prediction.predicted_label, 1);
		assert_eq!(prediction.confidence, 0.9);
		assert_eq!(prediction.elapsed_time_ms, 3.5);
		assert_eq!(prediction.output_activations, array![0.1, 0.9, 0.4]);
	}

	#[test]
	fn inference_handler_prepares_input_and_reads_output() {
		let model: crate::model_structure::model::PredictiveCodingModel = tiny_relu_model();
		let mut handler = InferenceModelHandler::from_model(model);

		handler.prepare_input(array![1.0, 0.0, 0.5, 0.25]);

		assert_eq!(handler.model().get_input(), &array![1.0, 0.0, 0.5, 0.25]);
		assert_eq!(handler.read_output_activations().len(), 10);
	}

	#[test]
	fn summarise_evaluation_samples_rejects_empty_datasets() {
		let result: Result<EvaluationSummary, crate::error::PredictiveCodingError> = summarise_evaluation_samples(0, |_| Ok((0, 0, 0.0, 0.0)));

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
		let summary: EvaluationSummary = summarise_evaluation_samples(2, |index| Ok((index, index + 1, 0.9, 3.0))).unwrap();

		assert_eq!(summary.correct_predictions, 0);
		assert_eq!(summary.total_predictions, 2);
		assert_eq!(summary.accuracy, 0.0);
		assert_eq!(summary.mean_confidence_when_correct, 0.0);
		assert_eq!(summary.mean_convergence_time_ms, 3.0);
	}

	#[test]
	fn evaluation_results_path_uses_model_directory() {
		let temp_dir = TempDir::new("evaluation_results_path");
		let model_path = temp_dir.join("artifacts/model_final_model.json");

		let output_path = evaluation_results_path(model_path.to_str().unwrap());

		assert_eq!(output_path, temp_dir.join("artifacts/evaluation_results.json"));
	}
}
