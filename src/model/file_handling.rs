use crate::{
  model::model_structure::{
    PredictiveCodingModel,
    PredictiveCodingModelConfig
  },
  utils::ensure_parent_dir,
  error::{PredictiveCodingError, Result}
};


pub fn load_model_config(fname: &str) -> Result<PredictiveCodingModelConfig> {
  let file = std::fs::File::open(fname)
    .map_err(|source| PredictiveCodingError::io("open model config", fname, source))?;

  serde_json::from_reader(file)
    .map_err(|source| PredictiveCodingError::json_deserialize(fname, source))
}

pub fn create_from_config(fname: &str) -> Result<PredictiveCodingModel> {
  let config = load_model_config(fname)?;
  Ok(PredictiveCodingModel::new(&config))
}

pub fn save_model_config(
  config: &PredictiveCodingModelConfig,
  filename: &str
) -> Result<()> {
  ensure_parent_dir(filename)?;

  let config_ser = serde_json::to_string(config)
    .map_err(|source| PredictiveCodingError::json_serialize(filename, source))?;
  std::fs::write(filename, config_ser)
    .map_err(|source| PredictiveCodingError::io("write model config", filename, source))?;

  Ok(())
}

pub fn save_model_snapshot(
  model: &PredictiveCodingModel,
  filename: &str
) -> Result<()> {
  ensure_parent_dir(filename)?;

  let model_ser = serde_json::to_string(&model)
    .map_err(|source| PredictiveCodingError::json_serialize(filename, source))?;
  std::fs::write(filename, model_ser)
    .map_err(|source| PredictiveCodingError::io("write model snapshot", filename, source))?;

  Ok(())
}

pub fn load_model_snapshot(filename: &str) -> Result<PredictiveCodingModel> {
  let model_ser = std::fs::read_to_string(filename)
    .map_err(|source| PredictiveCodingError::io("read model snapshot", filename, source))?;

  serde_json::from_str(&model_ser)
    .map_err(|source| PredictiveCodingError::json_deserialize(filename, source))
}


#[cfg(test)]
mod tests {
  use super::*;
  use crate::test_utils::{TempDir, write_file};

  use crate::model::maths::ActivationFunction;

  use ndarray::array;

  #[test]
  fn load_model_config_parses_expected_json_shape() {
    let temp_dir = TempDir::new("model_config_parse");
    let config_path = temp_dir.join("model_config.json");
    write_file(
      &config_path,
      r#"{
  "layer_sizes": [4, 10],
  "alpha": 0.01,
  "gamma": 0.05,
  "convergence_threshold": 0.0,
  "convergence_steps": 2,
  "activation_function": "Tanh"
}"#
    );

    let actual = load_model_config(config_path.to_str().unwrap()).unwrap();
    let expected = PredictiveCodingModelConfig {
      layer_sizes: vec![4, 10],
      alpha: 0.01,
      gamma: 0.05,
      convergence_threshold: 0.0,
      convergence_steps: 2,
      activation_function: ActivationFunction::Tanh,
    };

    assert_eq!(actual, expected);
  }

  #[test]
  fn load_model_config_rejects_missing_required_fields() {
    let temp_dir = TempDir::new("model_config_missing_field");
    let config_path = temp_dir.join("model_config.json");
    write_file(
      &config_path,
      r#"{
  "layer_sizes": [4, 10],
  "alpha": 0.01,
  "gamma": 0.05,
  "convergence_threshold": 0.0,
  "convergence_steps": 2
}"#
    );

    let result = load_model_config(config_path.to_str().unwrap());

    assert!(result.is_err());
  }

  #[test]
  fn load_model_config_error_includes_path() {
    let temp_dir = TempDir::new("missing_model_config");
    let config_path = temp_dir.join("missing_model_config.json");

    let error = load_model_config(config_path.to_str().unwrap()).unwrap_err();

    assert!(error.to_string().contains(&config_path.display().to_string()));
  }

  #[test]
  fn model_snapshot_round_trips_through_disk() {
    let temp_dir = TempDir::new("snapshot_round_trip");
    let snapshot_path = temp_dir.join("model_snapshot.json");
    let mut model = PredictiveCodingModel::new(&PredictiveCodingModelConfig {
      layer_sizes: vec![4, 10],
      alpha: 0.01,
      gamma: 0.05,
      convergence_threshold: 0.0,
      convergence_steps: 2,
      activation_function: ActivationFunction::Relu,
    });

    model.set_input(array![1.0, 0.0, 0.5, 0.25]);
    model.set_output(array![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    model.compute_predictions_and_errors();

    save_model_snapshot(&model, snapshot_path.to_str().unwrap()).unwrap();
    let loaded_model = load_model_snapshot(snapshot_path.to_str().unwrap()).unwrap();

    let original_json = serde_json::to_value(&model).unwrap();
    let loaded_json = serde_json::to_value(&loaded_model).unwrap();
    assert_eq!(loaded_json, original_json);
  }
}
