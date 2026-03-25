use std::time::Instant;

use ndarray::Array1;

use crate::{
    error::{PredictiveCodingError, Result},
    model::{CpuModelRuntime, ModelRuntime, PredictiveCodingModel, load_model_snapshot},
};

#[derive(Clone, Debug, PartialEq)]
pub struct InferencePrediction {
    pub output_activations: Array1<f32>,
    pub predicted_label: usize,
    pub confidence: f32,
    pub elapsed_time_ms: f32,
}

pub struct InferenceModelHandler {
    runtime: CpuModelRuntime,
}

impl InferenceModelHandler {
    pub fn load_snapshot(model_file: &str) -> Result<Self> {
        let mut runtime = CpuModelRuntime::from_model(load_model_snapshot(model_file)?);
        runtime.unpin_output()?;

        Ok(InferenceModelHandler { runtime })
    }

    pub fn from_model(mut model: PredictiveCodingModel) -> Self {
        model.unpin_output();
        InferenceModelHandler {
            runtime: CpuModelRuntime::from_model(model),
        }
    }

    pub fn prepare_input(&mut self, input_values: Array1<f32>) {
        self.runtime.model_mut().reinitialise_latents();
        self.runtime.model_mut().set_input(input_values);
    }

    pub fn converge(&mut self) -> f32 {
        let start_time = Instant::now();
        self.runtime
            .converge_values()
            .expect("CPU runtime convergence should be infallible");
        start_time.elapsed().as_secs_f32() * 1000.0
    }

    pub fn read_output_activations(&self) -> &Array1<f32> {
        self.runtime.model().get_output()
    }

    pub fn read_prediction(&self, elapsed_time_ms: f32) -> Result<InferencePrediction> {
        read_prediction(self.read_output_activations(), elapsed_time_ms)
    }

    pub fn infer(&mut self, input_values: Array1<f32>) -> Result<InferencePrediction> {
        self.prepare_input(input_values);
        let elapsed_time_ms = self.converge();
        self.read_prediction(elapsed_time_ms)
    }

    pub fn model(&self) -> &PredictiveCodingModel {
        self.runtime.model()
    }
}

pub fn read_label(values: &Array1<f32>, empty_message: &str) -> Result<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .ok_or_else(|| PredictiveCodingError::invalid_data(empty_message))
}

pub fn read_prediction(
    output_activations: &Array1<f32>,
    elapsed_time_ms: f32,
) -> Result<InferencePrediction> {
    let predicted_label = read_label(output_activations, "model produced an empty output layer")?;

    Ok(InferencePrediction {
        output_activations: output_activations.clone(),
        predicted_label,
        confidence: output_activations[predicted_label],
        elapsed_time_ms,
    })
}
