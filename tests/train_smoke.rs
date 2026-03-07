mod common;

use std::{
  fs,
  path::{Path, PathBuf},
  process::Command,
};

use common::{assert_json_files_equal, prepare_smoke_root, repo_root, run_command};

fn run_train(config: &str, output_prefix: &Path) {
  let root = repo_root();
  run_command(
    // https://doc.rust-lang.org/cargo/reference/environment-variables.html
    Command::new(env!("CARGO_BIN_EXE_train"))
      .current_dir(&root)
      .arg(config)
      .arg("--output-prefix")
      .arg(output_prefix),
  );
}

#[test]
fn train_single_thread_binary_smoke_on_tiny_fixtures() {
  let root: PathBuf = repo_root();
  let (smoke_root, _cleanup) = prepare_smoke_root("train_smoke_single");
  let train_dir: PathBuf = smoke_root.join("single_thread");
  let train_prefix: PathBuf = train_dir.join("model");

  run_train("test_data/train_single_thread_config.json", &train_prefix);

  let final_model: PathBuf = train_dir.join("model_final_model.json");
  assert!(final_model.exists());
  assert!(train_dir.join("model_training_config.json").exists());
  assert!(train_dir.join("model_model_config.json").exists());
  assert!(train_dir.join("model_snapshot_step_0.json").exists());
  assert!(train_dir.join("model_snapshot_step_1.json").exists());

  assert_json_files_equal(
    &train_dir.join("model_model_config.json"),
    &root.join("test_data/baselines/single_thread/train/model_model_config.json"),
  );
  assert_json_files_equal(
    &train_dir.join("model_training_config.json"),
    &root.join("test_data/baselines/single_thread/train/model_training_config.json"),
  );
  assert_json_files_equal(
    &train_dir.join("model_snapshot_step_0.json"),
    &root.join("test_data/baselines/single_thread/train/model_snapshot_step_0.json"),
  );
  assert_json_files_equal(
    &train_dir.join("model_snapshot_step_1.json"),
    &root.join("test_data/baselines/single_thread/train/model_snapshot_step_1.json"),
  );
  assert_json_files_equal(
    &final_model,
    &root.join("test_data/baselines/single_thread/train/model_final_model.json"),
  );
}

#[test]
fn train_minibatch_binary_smoke_on_tiny_fixtures() {
  let root = repo_root();
  let (smoke_root, _cleanup) = prepare_smoke_root("train_smoke_minibatch");
  let train_dir = smoke_root.join("mini_batch");
  let train_prefix = train_dir.join("model");

  run_train("test_data/train_minibatch_config.json", &train_prefix);

  let final_model = train_dir.join("model_final_model.json");
  assert!(final_model.exists());
  assert!(train_dir.join("model_training_config.json").exists());
  assert!(train_dir.join("model_model_config.json").exists());
  assert!(train_dir.join("model_snapshot_step_0.json").exists());
  assert!(train_dir.join("model_snapshot_step_1.json").exists());

  assert_json_files_equal(
    &train_dir.join("model_model_config.json"),
    &root.join("test_data/baselines/mini_batch/train/model_model_config.json"),
  );
  assert_json_files_equal(
    &train_dir.join("model_training_config.json"),
    &root.join("test_data/baselines/mini_batch/train/model_training_config.json"),
  );
  assert_json_files_equal(
    &train_dir.join("model_snapshot_step_0.json"),
    &root.join("test_data/baselines/mini_batch/train/model_snapshot_step_0.json"),
  );
  assert_json_files_equal(
    &train_dir.join("model_snapshot_step_1.json"),
    &root.join("test_data/baselines/mini_batch/train/model_snapshot_step_1.json"),
  );
  assert_json_files_equal(
    &final_model,
    &root.join("test_data/baselines/mini_batch/train/model_final_model.json"),
  );

  let _ = fs::remove_dir_all(&train_dir);
}
