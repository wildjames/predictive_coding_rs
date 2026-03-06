use std::time::Instant;

use predictive_coding::{
  model_structure::{
    model::{PredictiveCodingModel, PredictiveCodingModelConfig},
    model_utils::{create_from_config, save_model_config}
  },
  training::{
    cpu_train::train_and_update_model,
    utils::{
      TrainConfig,
      load_training_config,
      save_training_config
    }
  },
  utils::logging
};

use tracing::info;
use clap::Parser;

#[derive(Parser)]
struct BenchArgs {
  /// The model configuration to benchmark.
  #[arg(short, long, default_value_t = String::from("benchmark_data/benchmark_config.json"))]
  config: String,
}


fn main() {
  logging::setup_tracing(false);
  let args = BenchArgs::parse();
  let benchmark_config: TrainConfig = load_training_config(&args.config);

  // Run a benchmark model. Random input and output data, inputs are pinned
  let model_config_path: &String = benchmark_config.model_config.as_ref().expect(
    "Model config path must be provided in the benchmark config file"
  );
  let mut model: PredictiveCodingModel = create_from_config(model_config_path);

  let training_steps: u32 = 100;
  let model_config: PredictiveCodingModelConfig = model.get_config();

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
  });

  std::fs::create_dir_all(output_prefix.clone()).unwrap();

  // Write the benchmarking parameters, training parameters, and model configuration to files for posterity.
  serde_json::to_writer_pretty(
    std::fs::File::create(format!("{}{}", output_prefix, "/params.json")).unwrap(),
    &params
  ).unwrap();
  save_model_config(&model_config, &format!("{}{}", output_prefix, "/model_config.json"));
  save_training_config(&benchmark_config, &format!("{}{}", output_prefix, "/training_config.json"));

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
    train_and_update_model(model);
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
