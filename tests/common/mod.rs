use std::{
  fs,
  path::{Path, PathBuf},
  process::Command,
};

use tracing::debug;

pub fn repo_root() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub struct Cleanup {
  directories: Vec<PathBuf>,
}

impl Cleanup {
  pub fn new(directories: Vec<PathBuf>) -> Self {
    Cleanup { directories }
  }
}

impl Drop for Cleanup {
  fn drop(&mut self) {
    for directory in &self.directories {
      let _ = fs::remove_dir_all(directory);
    }
  }
}

pub fn prepare_smoke_root(test_name: &str) -> (PathBuf, Cleanup) {
  let smoke_root = repo_root().join("data/tests").join(test_name);
  let _ = fs::remove_dir_all(&smoke_root);
  let cleanup = Cleanup::new(vec![smoke_root.clone()]);
  (smoke_root, cleanup)
}

pub fn run_command(command: &mut Command) {
  debug!("Running command: {:?}", command);
  let output = command.output().unwrap();
  assert!(
    output.status.success(),
    "command failed with status {}\nstdout:\n{}\nstderr:\n{}",
    output.status,
    String::from_utf8_lossy(&output.stdout),
    String::from_utf8_lossy(&output.stderr)
  );
}

#[allow(dead_code)]
pub fn assert_json_files_equal(actual: &Path, expected: &Path) {
  // Check that both files exist
  assert!(actual.exists(), "actual file {} does not exist", actual.display());
  assert!(expected.exists(), "expected file {} does not exist", expected.display());

  let actual_json = read_json(actual);
  let expected_json = read_json(expected);
  assert_eq!(
    actual_json,
    expected_json,
    "json mismatch between {} and {}",
    actual.display(),
    expected.display()
  );
}

pub fn read_json(path: &Path) -> serde_json::Value {
  serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}
