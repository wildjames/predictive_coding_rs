mod common;

use std::{
  path::Path,
  process::Command,
};

use common::{assert_json_files_equal, prepare_smoke_root, read_json, repo_root, run_command};

fn run_bench(config: &str, output_prefix: &Path) {
  let root = repo_root();
  run_command(
    // https://doc.rust-lang.org/cargo/reference/environment-variables.html
    Command::new(env!("CARGO_BIN_EXE_bench"))
      .current_dir(&root)
      .arg(config)
      .arg("--output-prefix")
      .arg(output_prefix),
  );
}

#[test]
fn bench_single_thread_binary_smoke_on_tiny_fixtures() {
  // File stuff
  let root = repo_root();
  let (smoke_root, _cleanup) = prepare_smoke_root("bench_smoke_single");
  let bench_dir = smoke_root.join("single_thread");
  let bench_prefix = bench_dir.join("benchmark");

  run_bench("test_data/bench_single_thread_config.json", &bench_prefix);

  // Did we produce the correct artifacts?
  let benchmark_result = bench_dir.join("benchmark_result.json");
  assert!(benchmark_result.exists());
  assert!(bench_dir.join("benchmark_bench_run.csv").exists());
  assert!(bench_dir.join("benchmark_final_model.json").exists());

  // Do the produced artifacts' contents match what we expect?
  assert_json_files_equal(
    &bench_dir.join("benchmark_model_config.json"),
    &root.join("test_data/baselines/single_thread/bench/benchmark_model_config.json"),
  );
  assert_json_files_equal(
    &bench_dir.join("benchmark_training_config.json"),
    &root.join("test_data/baselines/single_thread/bench/benchmark_training_config.json"),
  );
  assert_json_files_equal(
    &bench_dir.join("benchmark_final_model.json"),
    &root.join("test_data/baselines/single_thread/bench/benchmark_final_model.json"),
  );

  // And do the benchmark results look like what we expect?
  let benchmark_json = read_json(&benchmark_result);
  assert_eq!(benchmark_json["step_data"].as_array().unwrap().len(), 2);
  assert!(!benchmark_json["git_commit_hash"].as_str().unwrap().is_empty());
}

#[test]
fn bench_minibatch_binary_smoke_on_tiny_fixtures() {
  // file system stuff
  let root = repo_root();
  let (smoke_root, _cleanup) = prepare_smoke_root("bench_smoke_minibatch");
  let bench_dir = smoke_root.join("mini_batch");
  let bench_prefix = bench_dir.join("benchmark");

  run_bench("test_data/bench_minibatch_config.json", &bench_prefix);

  // Did we produce the correct artifacts?
  let benchmark_result = bench_dir.join("benchmark_result.json");
  assert!(benchmark_result.exists());
  assert!(bench_dir.join("benchmark_bench_run.csv").exists());
  assert!(bench_dir.join("benchmark_final_model.json").exists());

  // Do the produced artifacts' contents match what we expect?
  assert_json_files_equal(
    &bench_dir.join("benchmark_model_config.json"),
    &root.join("test_data/baselines/mini_batch/bench/benchmark_model_config.json"),
  );
  assert_json_files_equal(
    &bench_dir.join("benchmark_training_config.json"),
    &root.join("test_data/baselines/mini_batch/bench/benchmark_training_config.json"),
  );
  assert_json_files_equal(
    &bench_dir.join("benchmark_final_model.json"),
    &root.join("test_data/baselines/mini_batch/bench/benchmark_final_model.json"),
  );

  // And do the benchmark results look like what we expect?
  let benchmark_json = read_json(&benchmark_result);
  assert_eq!(benchmark_json["step_data"].as_array().unwrap().len(), 2);
  assert!(!benchmark_json["git_commit_hash"].as_str().unwrap().is_empty());
}
