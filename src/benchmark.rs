use std::time::Instant;

use predictive_coding::{
  data_handling::data_handler,
  model_structure::model::PredictiveCodingModel,
  training::{
    get_handler::get_handler,
    train_handler::TrainingHandler,
    utils::{
      TrainConfig,
      load_model,
      load_training_config
    }
  },
  utils::logging
};

use tracing::info;
use clap::Parser;

#[derive(Parser)]
struct BenchArgs {
  /// The model configuration to benchmark.
  #[arg(default_value_t = String::from("benchmark_data/benchmark_config.json"))]
  config: String,
}


fn main() {
  logging::setup_tracing(false);

  // Detect if this binary was compiled in release mode or not
  let release_mode: bool = !cfg!(debug_assertions);
  #[cfg(debug_assertions)]
  info!("Running benchmark in debug mode. For more accurate benchmarking, compile with --release");

  let args = BenchArgs::parse();
  let benchmark_config: TrainConfig = load_training_config(&args.config);

  // Run a benchmark model. Random input and output data, inputs are pinned
  let model: PredictiveCodingModel = load_model(&benchmark_config.model_source);
  info!(
    "Created the model with layer sizes {:?}",
    model.get_layer_sizes()
  );

  let output_prefix = format!("benchmark_data/{}/benchmark", chrono::Utc::now().timestamp());

  let data: data_handler::TrainingDataset = data_handler::load_mnist(
      "data/mnist/train-images-idx3-ubyte",
      "data/mnist/train-labels-idx1-ubyte")
    .unwrap();
  info!(
    "Loaded the MNIST dataset. I have {} images",
    data.dataset_size
  );

  // Mkae sure we have the output directory so we dont crash out later
  let output_dir: &std::path::Path = std::path::Path::new(&output_prefix).parent().unwrap();
  std::fs::create_dir_all(output_dir).unwrap();

  let mut handler: Box<dyn TrainingHandler> = get_handler(
    benchmark_config,
    data,
    output_prefix.clone()
  );

  let step_data = run_benchmark_training_loop(
    handler.as_mut(),
    &format!("{}_{}", output_prefix, "bench_run.csv")
  );

  // Write the training params to "{output_prefix}/params.json"
  let current_commit_hash = std::process::Command::new("git")
    .args(["rev-parse", "HEAD"])
    .output()
    .expect("Failed to get git commit hash")
    .stdout;
  let current_commit_hash_str: String = String::from_utf8_lossy(&current_commit_hash)
    .trim()
    .to_string();

  let result = BenchmarkResult {
    step_data,
    git_commit_hash: current_commit_hash_str,
    run_timestamp: chrono::Utc::now().to_rfc3339(),
    release_mode
   };


  // Write the benchmarking parameters, training parameters, and model configuration to files for posterity.
  // This is JSON, so probably a bit harder to read than the CSV file, but it does create a single file with all the relevant information for each benchmark run, which is nice.
  serde_json::to_writer_pretty(
    std::fs::File::create(format!("{}_{}", output_prefix, "result.json")).unwrap(),
    &result
  ).unwrap();

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
) -> Vec<BenchmarkStepData> {
  let mut benchmark_data: Vec<BenchmarkStepData> = Vec::new();

  handler.pre_training_hook();

  // Time each training step and write to a csv file
  let mut wtr = csv::Writer::from_path(bench_run_outfile).unwrap();
  wtr.write_record(["step", "time_ms"]).unwrap();

  let training_config: &TrainConfig = handler.get_config();
  let training_steps: u32 = training_config.training_steps;

  for step in 0..training_steps {

    let start_time: Instant = Instant::now();
    handler.pre_step_hook(step);
    handler.train_step(step);
    handler.post_step_hook(step);
    let elapsed_time = start_time.elapsed();


    let elapsed_time_ms: f32 = elapsed_time.as_secs_f32() * 1000.0;

    benchmark_data.push(BenchmarkStepData {
      step,
      time_ms: elapsed_time_ms
    });

    wtr.write_record(&[
      step.to_string(),
      elapsed_time_ms.to_string()
    ]).unwrap();
    wtr.flush().unwrap();

    info!(
      "Step {}: time {:.1} ms",
      step, elapsed_time_ms,
    );
  }

  handler.post_training_hook();

  benchmark_data
}
