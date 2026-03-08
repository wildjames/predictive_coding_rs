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

#[cfg(test)]
#[path = "test_utils.rs"]
mod test_utils;


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
  use super::test_utils::{
    RecordingTrainingHandler,
    TempDir,
    single_thread_train_config,
  };

  use std::{
    fs,
    path::PathBuf,
  };

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
    let mut handler = RecordingTrainingHandler::new(
      single_thread_train_config(2, 0, 0),
      output_prefix.clone(),
    );

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
    let handler = RecordingTrainingHandler::new(
      single_thread_train_config(2, 0, 0),
      String::from("unused/bench"),
    );
    let data = handler.get_data();

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
