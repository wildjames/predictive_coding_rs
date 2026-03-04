use std::time::Instant;

use predictive_coding::{
  model::{
    model::{
      PredictiveCodingModel
    },
    model_utils::{
      create_from_config
    }
  },
  training::train_model_handler,
  utils::logging
};

use tracing::info;
use clap::Parser;

#[derive(Parser)]
struct BenchArgs {
  /// The model file to benchmark. If it doesn't exist, a new model will be created with the same architecture as the MNIST model and random weights, and saved to this path for future
  #[arg(short, long, default_value_t = String::from("benchmark_data/model_config.json"))]
  config: String,
}


fn main() {
  logging::setup_tracing(false);
  let args = BenchArgs::parse();

  // Run a benchmark model. Random input and output data, inputs are pinned
  // And we run for 50 convergence steps, tolerance of 0.0 (i.e. run for all 50 steps)
  let config: &String = &args.config;
  let mut model = create_from_config(config);

  let training_steps: u32 = 100;
  let model_config = model.get_config();

  info!(
    "Benchmarking hyperparameters:\n\ttraining steps: {}\n\tconvergence steps: {}\n\tconvergence threshold: {}",
    training_steps,
    model_config.convergence_steps,
    model_config.convergence_threshold,
  );

  let output_prefix = format!("benchmark_data/{}", chrono::Utc::now().timestamp());

  // Write the training params to "{output_prefix}/params.json"
  let current_commit_hash = std::process::Command::new("git")
    .args(["rev-parse", "HEAD"])
    .output()
    .expect("Failed to get git commit hash")
    .stdout;
  let current_commit_hash_str = String::from_utf8_lossy(&current_commit_hash).trim().to_string();

  let params = serde_json::json!({
    "git_commit_hash": current_commit_hash_str,
    "run_timestamp": chrono::Utc::now().to_rfc3339(),
    "training_steps": training_steps,
    "convergence_steps": model_config.convergence_steps,
    "convergence_threshold": model_config.convergence_threshold,
  });

  std::fs::create_dir_all(output_prefix.clone()).unwrap();
  // benchmarking params dumped to file
  serde_json::to_writer_pretty(
    std::fs::File::create(format!("{}{}", output_prefix, "/params.json")).unwrap(),
    &params
  ).unwrap();
  // Model config dumped to file
  serde_json::to_writer_pretty(
    std::fs::File::create(format!("{}{}", output_prefix, "/model_config.json")).unwrap(),
    &model_config
  ).unwrap();

  benchmark(
    &mut model,
    training_steps,
    &format!("{}{}", output_prefix, "/bench_run.csv")
  );

}

pub fn benchmark(
  model: &mut PredictiveCodingModel,
  training_steps: u32,
  bench_run_outfile: &str
) {
  // Create the output directory if it doesn't exist
  std::fs::create_dir_all(std::path::Path::new(bench_run_outfile).parent().unwrap()).unwrap();

  // Time each training step and write to a csv file
  let mut wtr = csv::Writer::from_path(bench_run_outfile).unwrap();
  wtr.write_record(["step", "time_ms"]).unwrap();

  model.pin_input();
  model.pin_output();
  for step in 0..training_steps {

    // Set random input and output data for the model
    model.randomise_input();
    model.randomise_output();

    let start_time: Instant = Instant::now();
    train_model_handler::train_and_update_model(model);
    let elapsed_time = start_time.elapsed();

    let elapsed_time_ms = elapsed_time.as_secs_f32() * 1000.0;

    wtr.write_record(&[
      step.to_string(),
      elapsed_time_ms.to_string()
    ]).unwrap();
    wtr.flush().unwrap();

    info!(
      "Sample {:.1}: time {:.1} ms",
      step, elapsed_time_ms,
    );
  }
}
