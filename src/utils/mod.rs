use std::path::Path;

use crate::error::{
  PredictiveCodingError,
  Result
};
pub mod logging;

pub fn timestamp() -> String {
  chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S").to_string()
}


pub fn ensure_parent_dir(filename: &str) -> Result<()> {
  if let Some(parent) = Path::new(filename).parent() && !parent.as_os_str().is_empty() {
      std::fs::create_dir_all(parent)
        .map_err(|source| PredictiveCodingError::io("create directory", parent, source))?;
    }

  Ok(())
}
