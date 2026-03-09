#![allow(dead_code)]

use chrono::Duration;
use ndarray::{Array1, Array2, array};
use predictive_coding::{
    data_handling::TrainingDataset,
    error::Result,
    model::{
        PredictiveCodingModel, PredictiveCodingModelConfig, maths::ActivationFunction,
        save_model_snapshot,
    },
    training::{
        TrainConfig, TrainingHandler, TrainingStrategy,
        configuration::{DataSetSource, ModelSource},
    },
};

use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

pub(crate) struct TempDir {
    path: PathBuf,
}

impl TempDir {
    pub(crate) fn new(prefix: &str) -> Self {
        let unique_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "predictive_coding_{prefix}_{}_{}",
            std::process::id(),
            unique_id
        ));
        fs::create_dir_all(&path).unwrap();
        TempDir { path }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn join<P: AsRef<Path>>(&self, path: P) -> PathBuf {
        self.path.join(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

pub(crate) fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).unwrap();
}

pub(crate) struct DummyTrainingDataset {
    inputs: Array2<f32>,
    labels: Array2<f32>,
}

impl DummyTrainingDataset {
    pub(crate) fn from_arrays(inputs: Array2<f32>, labels: Array2<f32>) -> Self {
        DummyTrainingDataset { inputs, labels }
    }

    pub(crate) fn zeros(dataset_size: usize, input_size: usize, output_size: usize) -> Self {
        DummyTrainingDataset {
            inputs: Array2::zeros((dataset_size, input_size)),
            labels: Array2::zeros((dataset_size, output_size)),
        }
    }
}

impl TrainingDataset for DummyTrainingDataset {
    fn get_dataset_size(&self) -> usize {
        self.inputs.nrows()
    }
    fn get_input_size(&self) -> usize {
        self.inputs.ncols()
    }
    fn get_output_size(&self) -> usize {
        self.labels.ncols()
    }

    fn get_random_input(&self) -> Array1<f32> {
        self.get_input(0)
    }

    fn get_random_input_and_output(&self) -> (Array1<f32>, Array1<f32>) {
        (self.get_input(0), self.get_output(0))
    }

    fn get_input(&self, index: usize) -> Array1<f32> {
        self.inputs.row(index).to_owned()
    }

    fn get_output(&self, index: usize) -> Array1<f32> {
        self.labels.row(index).to_owned()
    }
}

pub(crate) fn tiny_relu_model() -> PredictiveCodingModel {
    PredictiveCodingModel::new(&PredictiveCodingModelConfig {
        layer_sizes: vec![4, 10],
        alpha: 0.01,
        gamma: 0.05,
        convergence_threshold: 0.0,
        convergence_steps: 1,
        activation_function: ActivationFunction::Relu,
    })
}

pub(crate) fn single_thread_train_config(
    training_steps: u32,
    report_interval: u32,
    snapshot_interval: u32,
) -> TrainConfig {
    TrainConfig {
        model_source: ModelSource::Config(String::from("unused.json")),
        training_dataset: DataSetSource::IdxFormat {
            input_idx_file: String::from("unused-images.idx"),
            output_idx_file: String::from("unused-labels.idx"),
        },
        evaluation_dataset: Some(DataSetSource::IdxFormat {
            input_idx_file: String::from("unused-images.idx"),
            output_idx_file: String::from("unused-labels.idx"),
        }),
        training_strategy: TrainingStrategy::SingleThread,
        training_steps,
        report_interval,
        snapshot_interval,
    }
}

pub(crate) struct RecordingTrainingHandler {
    config: TrainConfig,
    model: PredictiveCodingModel,
    data: Arc<dyn TrainingDataset>,
    file_output_prefix: String,
    pub(crate) steps: Vec<u32>,
    pub(crate) events: Vec<String>,
}

impl RecordingTrainingHandler {
    pub(crate) fn new(config: TrainConfig, output_prefix: String) -> Self {
        let data: Arc<dyn TrainingDataset> = Arc::new(DummyTrainingDataset::from_arrays(
            array![[1.0, 0.0, 0.5, 0.25]],
            array![[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        ));

        RecordingTrainingHandler {
            config,
            model: tiny_relu_model(),
            data,
            file_output_prefix: output_prefix,
            steps: Vec::new(),
            events: Vec::new(),
        }
    }
}

impl TrainingHandler for RecordingTrainingHandler {
    fn get_config(&self) -> &TrainConfig {
        &self.config
    }

    fn get_model(&mut self) -> &mut PredictiveCodingModel {
        &mut self.model
    }

    fn get_data(&self) -> &dyn TrainingDataset {
        self.data.as_ref()
    }

    fn get_file_output_prefix(&self) -> &String {
        &self.file_output_prefix
    }

    fn pre_training_hook(&mut self) -> Result<()> {
        self.events.push(String::from("pre_training"));
        Ok(())
    }

    fn train_step(&mut self, step: u32) -> Result<()> {
        self.steps.push(step);
        self.events.push(format!("train_step:{step}"));
        Ok(())
    }

    fn report_hook(&mut self, step: u32, _mean_step_time: Duration) -> Result<()> {
        self.events.push(format!("report:{step}"));
        Ok(())
    }

    fn pre_step_hook(&mut self, step: u32) -> Result<()> {
        self.events.push(format!("pre_step:{step}"));
        Ok(())
    }

    fn post_step_hook(&mut self, step: u32) -> Result<()> {
        self.events.push(format!("post_step:{step}"));
        Ok(())
    }

    fn post_training_hook(&mut self) -> Result<()> {
        self.events.push(String::from("post_training"));
        let final_output_path = format!("{}_final_model.json", self.get_file_output_prefix());
        save_model_snapshot(self.get_model(), &final_output_path)
    }
}
