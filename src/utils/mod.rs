pub mod logging;

pub fn timestamp() -> String {
  chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S").to_string()
}
