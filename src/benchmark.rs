//! This program is used to benchmark the speed of various training configurations and model architectures. It takes a training config file as input, and times its processes, creating a series of files detailing the results.

use std::{
  path::Path,
  time::Instant
};

use predictive_coding::{
  error::{
    PredictiveCodingError,
    Result
  },
  model_structure::{
    model::PredictiveCodingModelConfig,
    configuration::save_model_config
  },
  training::{
    setup::setup_training_run_handler,
    train_handler::TrainingHandler,
    configuration::{
      TrainConfig,
      save_training_config
    }
  },
  utils::{
    logging,
    timestamp
  }
};

use tracing::info;
use clap::Parser;


/// This program is used to benchmark the speed of various training configurations and model architectures. It takes a training config file as input, and times its processes, creating a series of files detailing the results.
#[derive(Parser)]
struct BenchArgs {
  /// The model configuration to benchmark.
  #[arg(default_value_t = String::from("benchmark_data/benchmark_config.json"))]
  config: String,

  /// Optional artifact output prefix. Defaults to `benchmark_data/<timestamp>/benchmark`.
  #[arg(long, default_value_t = format!("benchmark_data/benchmark_{}/bench_", timestamp()))]
  output_prefix: String
}


fn current_git_commit_hash_with_command(command_name: &str, args: &[&str]) -> Result<String> {
  let command = if args.is_empty() {
    command_name.to_string()
  } else {
    format!("{} {}", command_name, args.join(" "))
  };
  let output = std::process::Command::new(command_name)
    .args(args)
    .output()
    .map_err(|source| PredictiveCodingError::command_io(command.clone(), source))?;

  if !output.status.success() {
    return Err(PredictiveCodingError::command_failed(
      command,
      output.status,
      String::from_utf8_lossy(&output.stderr),
    ));
  }

  Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn current_git_commit_hash() -> Result<String> {
  current_git_commit_hash_with_command("git", &["rev-parse", "HEAD"])
}

fn run_benchmark(args: BenchArgs) -> Result<()> {
  // Detect if this binary was compiled in release mode or not
  let release_mode: bool = !cfg!(debug_assertions);
  #[cfg(debug_assertions)]
  info!("Running benchmark in debug mode. For more accurate benchmarking, compile with --release");
  let mut handler: Box<dyn TrainingHandler> = setup_training_run_handler(args.config, args.output_prefix.clone())?;

  let step_data = run_benchmark_training_loop(handler.as_mut(), &format!("{}_{}", &args.output_prefix, "bench_run.csv"))?;

  // Write the training params to "{output_prefix}/params.json"
  let current_commit_hash_str = current_git_commit_hash()?;

  let result = BenchmarkResult {
    step_data,
    git_commit_hash: current_commit_hash_str,
    run_timestamp: chrono::Utc::now().to_rfc3339(),
    release_mode
   };


  // Write the benchmarking parameters, training parameters, and model configuration to files for posterity.
  // This is JSON, so probably a bit harder to read than the CSV file, but it does create a single file with all the relevant information for each benchmark run, which is nice.
  let result_path = format!("{}_{}", args.output_prefix, "result.json");
  let result_file = std::fs::File::create(&result_path)
    .map_err(|source| PredictiveCodingError::io("create benchmark result", &result_path, source))?;
  serde_json::to_writer_pretty(
    result_file,
    &result
  ).map_err(|source| PredictiveCodingError::json_serialize(&result_path, source))?;

  Ok(())

}

fn main() -> Result<()> {
  logging::setup_tracing(false);
  info!("Starting benchmark run");
  run_benchmark(BenchArgs::parse())
}

#[derive(serde::Serialize)]
struct BenchmarkResult {
  step_data: Vec<BenchmarkStepData>,
  git_commit_hash: String,
  run_timestamp: String,
  release_mode: bool
}

#[derive(serde::Serialize)]
struct BenchmarkStepData {
  step: u32,
  time_ms: f32
}

fn run_benchmark_training_loop(
  handler: &mut dyn TrainingHandler,
  bench_run_outfile: &str
) -> Result<Vec<BenchmarkStepData>> {
  let mut benchmark_data: Vec<BenchmarkStepData> = Vec::new();

  handler.pre_training_hook()?;

  // Time each training step and write to a csv file
  if let Some(parent) = Path::new(bench_run_outfile).parent().filter(|path| !path.as_os_str().is_empty()) {
    std::fs::create_dir_all(parent)
      .map_err(|source| PredictiveCodingError::io("create benchmark output directory", parent, source))?;
  }
  let mut wtr = csv::Writer::from_path(bench_run_outfile)
    .map_err(|source| PredictiveCodingError::csv("create benchmark writer", bench_run_outfile, source))?;
  wtr.write_record(["step", "time_ms"])
    .map_err(|source| PredictiveCodingError::csv("write benchmark header", bench_run_outfile, source))?;

  let training_config: &TrainConfig = handler.get_config();
  let training_steps: u32 = training_config.training_steps;

  // Write the config and training params to a file
  let model_config: &PredictiveCodingModelConfig = &handler.get_model().get_config();
  save_model_config(
    model_config,
    &format!("{}_model_config.json", &handler.get_file_output_prefix())
  )?;
  save_training_config(
    handler.get_config(),
    &format!("{}_training_config.json", &handler.get_file_output_prefix())
  )?;

  for step in 0..training_steps {

    let start_time: Instant = Instant::now();
    handler.pre_step_hook(step)?;
    handler.train_step(step)?;
    handler.post_step_hook(step)?;
    let elapsed_time = start_time.elapsed();


    let elapsed_time_ms: f32 = elapsed_time.as_secs_f32() * 1000.0;

    benchmark_data.push(BenchmarkStepData {
      step,
      time_ms: elapsed_time_ms
    });

    wtr.write_record(&[
      step.to_string(),
      elapsed_time_ms.to_string()
    ]).map_err(|source| PredictiveCodingError::csv("append benchmark row", bench_run_outfile, source))?;
    wtr.flush()
      .map_err(|source| PredictiveCodingError::io("flush benchmark CSV", bench_run_outfile, source))?;

    info!(
      "Step {}: time {:.1} ms",
      step, elapsed_time_ms,
    );
  }

  handler.post_training_hook()?;

  Ok(benchmark_data)
}

#[cfg(test)]
mod tests {
  use super::*;

  use ndarray::{Array1, Array2, array};
  use predictive_coding::{
    data_handling::data_handler,
    model_structure::maths::ActivationFunction,
    training::configuration::{
      DataSetSource,
      ModelSource,
      TrainingStrategy,
    },
  };
  use std::{
    fs,
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
  };

  struct TempDir {
    path: PathBuf,
  }

  impl TempDir {
    fn new(prefix: &str) -> Self {
      let unique_id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
      let path = std::env::temp_dir().join(format!(
        "predictive_coding_{prefix}_{}_{}",
        std::process::id(),
        unique_id
      ));
      fs::create_dir_all(&path).unwrap();
      TempDir { path }
    }

    fn join(&self, filename: &str) -> PathBuf {
      self.path.join(filename)
    }
  }

  impl Drop for TempDir {
    fn drop(&mut self) {
      let _ = fs::remove_dir_all(&self.path);
    }
  }

  struct DummyTrainingDataset {
    inputs: Array2<f32>,
    labels: Array2<f32>,
  }

  impl data_handler::TrainingDataset for DummyTrainingDataset {
    fn get_dataset_size(&self) -> usize {self.inputs.nrows()}
    fn get_input_size(&self) -> usize {self.inputs.ncols()}
    fn get_output_size(&self) -> usize {self.labels.ncols()}
    fn get_inputs(&self) -> &Array2<f32> {&self.inputs}
    fn get_labels(&self) -> &Array2<f32> {&self.labels}

    fn get_random_input(&self) -> Array1<f32> {
      self.get_input(0)
    }

    fn get_random_input_and_output(&self) -> (Array1<f32>, Array1<f32>) {
      (self.get_input(0), self.get_output(0))
    }

    fn get_input(&self, _index: usize) -> Array1<f32> {
      self.inputs.row(0).to_owned()
    }

    fn get_output(&self, _index: usize) -> Array1<f32> {
      self.labels.row(0).to_owned()
    }
  }

  struct RecordingHandler {
    config: TrainConfig,
    model: predictive_coding::model_structure::model::PredictiveCodingModel,
    data: Arc<dyn data_handler::TrainingDataset>,
    file_output_prefix: String,
    steps: Vec<u32>,
  }

  impl RecordingHandler {
    fn new(output_prefix: String) -> Self {
      let model = predictive_coding::model_structure::model::PredictiveCodingModel::new(
        &PredictiveCodingModelConfig {
          layer_sizes: vec![4, 10],
          alpha: 0.01,
          gamma: 0.05,
          convergence_threshold: 0.0,
          convergence_steps: 1,
          activation_function: ActivationFunction::Relu,
        },
      );
      let data: Arc<dyn data_handler::TrainingDataset> = Arc::new(DummyTrainingDataset {
        inputs: array![[1.0, 0.0, 0.5, 0.25]],
        labels: array![[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
      });

      RecordingHandler {
        config: TrainConfig {
          model_source: ModelSource::Config(String::from("unused.json")),
          dataset: DataSetSource::MNIST {
            input_idx_file: String::from("unused-images.idx"),
            output_idx_file: String::from("unused-labels.idx"),
          },
          training_strategy: TrainingStrategy::SingleThread,
          training_steps: 2,
          report_interval: 0,
          snapshot_interval: 0,
        },
        model,
        data,
        file_output_prefix: output_prefix,
        steps: Vec::new(),
      }
    }
  }

  impl TrainingHandler for RecordingHandler {
    fn get_config(&self) -> &TrainConfig {
      &self.config
    }

    fn get_model(&mut self) -> &mut predictive_coding::model_structure::model::PredictiveCodingModel {
      &mut self.model
    }

    fn get_data(&self) -> &dyn data_handler::TrainingDataset {
      self.data.as_ref()
    }

    fn get_file_output_prefix(&self) -> &String {
      &self.file_output_prefix
    }

    fn train_step(&mut self, step: u32) -> Result<()> {
      self.steps.push(step);
      Ok(())
    }
  }

  #[test]
  fn current_git_commit_hash_helper_handles_success_and_failure_cases() {
    let success: String = current_git_commit_hash_with_command("sh", &["-c", "printf 'abc123\\n'"]).unwrap();
    assert_eq!(success, "abc123");

    let failure = current_git_commit_hash_with_command("sh", &["-c", "printf 'boom\\n' >&2; exit 2"]);
    assert!(matches!(
      failure,
      Err(PredictiveCodingError::CommandFailed { stderr, .. }) if stderr.contains("boom")
    ));

    let spawn_failure: std::result::Result<String, PredictiveCodingError> = current_git_commit_hash_with_command("/definitely/missing/git", &[]);
    assert!(matches!(spawn_failure, Err(PredictiveCodingError::CommandIo { .. })));
  }

  #[test]
  fn benchmark_loop_writes_csv_and_artifacts() {
    let temp_dir: TempDir = TempDir::new("benchmark_loop");
    let output_prefix: String = temp_dir.join("nested/bench").display().to_string();
    let bench_csv: PathBuf = temp_dir.join("nested/bench_run.csv");
    let mut handler: RecordingHandler = RecordingHandler::new(output_prefix.clone());

    let step_data: Vec<BenchmarkStepData> = run_benchmark_training_loop(&mut handler, bench_csv.to_str().unwrap()).unwrap();

    assert_eq!(handler.steps, vec![0, 1]);
    assert_eq!(step_data.len(), 2);
    assert!(bench_csv.exists());
    assert!(Path::new(&format!("{}_model_config.json", output_prefix)).exists());
    assert!(Path::new(&format!("{}_training_config.json", output_prefix)).exists());
    assert!(Path::new(&format!("{}_final_model.json", output_prefix)).exists());

    let csv_output = fs::read_to_string(bench_csv).unwrap();
    assert!(csv_output.contains("step,time_ms"));
    assert!(csv_output.contains("0,"));
    assert!(csv_output.contains("1,"));
  }

  #[test]
  fn recording_handler_exposes_dataset_fixture() {
    let handler: RecordingHandler = RecordingHandler::new(String::from("unused/bench"));
    let data: &dyn data_handler::TrainingDataset = handler.get_data();

    assert_eq!(data.get_dataset_size(), 1);
    assert_eq!(data.get_input_size(), 4);
    assert_eq!(data.get_output_size(), 10);
    assert_eq!(data.get_inputs().dim(), (1, 4));
    assert_eq!(data.get_labels().dim(), (1, 10));
    assert_eq!(data.get_random_input(), data.get_input(0));

    let (input, output) = data.get_random_input_and_output();
    assert_eq!(input, data.get_input(0));
    assert_eq!(output, data.get_output(0));
  }

  #[test]
  fn run_benchmark_writes_result_payload_in_process() {
    let temp_dir: TempDir = TempDir::new("benchmark_run_main_path");
    let output_prefix: String = temp_dir.join("end_to_end/bench").display().to_string();

    run_benchmark(BenchArgs {
      config: String::from("test_data/bench_single_thread_config.json"),
      output_prefix: output_prefix.clone(),
    })
    .unwrap();

    let result_path: String = format!("{}_result.json", output_prefix);
    let result_json: serde_json::Value = serde_json::from_str(&fs::read_to_string(result_path).unwrap()).unwrap();
    assert_eq!(result_json["step_data"].as_array().unwrap().len(), 2);
    assert!(!result_json["git_commit_hash"].as_str().unwrap().is_empty());
  }
}
