use std::time::Instant;

use predictive_coding::{
  model::{
    model::PredictiveCodingModel, model_utils::load_model
  },
  training::train_model_handler,
  utils::logging
};

use tracing::info;


fn main() {
  logging::setup_tracing(false);

  // Run a benchmark model. Random input and output data, inputs are pinned
  // And we run for 50 convergence steps, tolerance of 0.0 (i.e. run for all 50 steps)
  let mut model = load_model("benchmark_data/initial_model.json");

  info!("Are the model's input and output pinned? {} {}", model.layers.first().unwrap().pinned, model.layers.last().unwrap().pinned);

  // Training params
  let training_steps: u32 = 100;
  let convergence_steps: u32 = 50;
  let convergence_threshold: f32 = 0.0;


  info!(
    "Benchmarking hyperparameters:\n\ttraining steps: {}\n\tconvergence steps: {}\n\tconvergence threshold: {}",
    training_steps,
    convergence_steps,
    convergence_threshold,
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
    "training_steps": training_steps,
    "convergence_steps": convergence_steps,
    "convergence_threshold": convergence_threshold,
  });

  std::fs::create_dir_all(output_prefix.clone()).unwrap();
  std::fs::write(format!("{}{}", output_prefix, "/params.json"), serde_json::to_string_pretty(&params).unwrap()).unwrap();

  benchmark(
    &mut model,
    training_steps,
    convergence_steps,
    convergence_threshold,
    &format!("{}{}", output_prefix, "/bench_run.csv")
  );

}

pub fn benchmark(
  model: &mut PredictiveCodingModel,
  training_steps: u32,
  convergence_steps: u32,
  convergence_threshold: f32,
  bench_run_outfile: &str
) {
  // Create the output directory if it doesn't exist
  std::fs::create_dir_all(std::path::Path::new(bench_run_outfile).parent().unwrap()).unwrap();

  // Time each training step and write to a csv file
  let mut wtr = csv::Writer::from_path(bench_run_outfile).unwrap();
  wtr.write_record(["step", "time_ms"]).unwrap();

  for step in 0..training_steps {
    let start_time: Instant = Instant::now();

    // Set random input and output data for the model
    model.randomise_input();
    model.randomise_output();

    train_model_handler::train_and_update_model(
      model,
      convergence_threshold,
      convergence_steps
    );

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
