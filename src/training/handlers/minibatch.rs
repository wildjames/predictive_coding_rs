use crate::{
    data_handling::TrainingDataset,
    error::Result,
    model::{
        CpuModelRuntime, ModelRuntime, PredictiveCodingModel, TrainableModelRuntime,
        WeightUpdateSet,
    },
    training::TrainConfig,
};

use super::impl_handler_delegation;

use chrono::TimeDelta;
use ndarray::{Array1, Array2};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info};

use super::super::StepProfile;

pub struct CpuBatchTrainHandler {
    config: TrainConfig,
    runtime: CpuModelRuntime,
    data: Arc<dyn TrainingDataset>,
    file_output_prefix: String,
    batch_size: u32,
}

impl CpuBatchTrainHandler {
    pub fn new(
        config: TrainConfig,
        model: PredictiveCodingModel,
        data: Arc<dyn TrainingDataset>,
        file_output_prefix: String,
        batch_size: u32,
    ) -> Self {
        CpuBatchTrainHandler {
            config,
            runtime: CpuModelRuntime::from_model(model),
            data,
            file_output_prefix,
            batch_size,
        }
    }
}

impl_handler_delegation!(CpuBatchTrainHandler, runtime, {
    fn pre_training_hook(&mut self) -> Result<()> {
        info!("Starting training with mini-batch strategy");
        info!("Mini batch params: batch size = {}", self.batch_size);
        Ok(())
    }

    fn profiled_train_step(&mut self, _step: u32) -> Result<StepProfile> {
        let mut profile = StepProfile::new();

        let t = Instant::now();
        let batch_inputs_and_outputs: Vec<(Array1<f32>, Array1<f32>)> = (0..self.batch_size)
            .map(|_| self.data.get_random_input_and_output())
            .collect();
        profile.record("prepare_batch_data", t.elapsed());

        let t = Instant::now();
        let batch_weight_changes: Vec<Result<WeightUpdateSet>> = batch_inputs_and_outputs
            .into_par_iter()
            .map(|(input_data, output_data)| {
                debug!(
                    "Training on batch element with input {:?} and output {:?}",
                    input_data, output_data
                );
                let mut runtime_clone = CpuModelRuntime::from_model(self.runtime.model().clone());

                runtime_clone.model_mut().set_input(input_data);
                runtime_clone.model_mut().set_output(output_data);
                runtime_clone.reinitialise_latents()?;
                runtime_clone.converge_values()?;

                runtime_clone.compute_weight_updates()
            })
            .collect();
        debug!(
            "Batch weight changes computed for {} samples",
            self.batch_size
        );
        profile.record("parallel_converge", t.elapsed());

        let t = Instant::now();
        let successful_updates: Vec<WeightUpdateSet> = batch_weight_changes
            .into_iter()
            .collect::<Result<Vec<_>>>()?;

        let first = &successful_updates[0];
        let mut sum_batch_weight_changes: Vec<Array2<f32>> = first
            .updates
            .iter()
            .zip(first.shapes.iter())
            .map(|(data, &(rows, cols))| {
                Array2::from_shape_vec((rows, cols), data.clone()).unwrap()
            })
            .collect();

        for update_set in &successful_updates[1..] {
            for (sum, (data, &(rows, cols))) in sum_batch_weight_changes
                .iter_mut()
                .zip(update_set.updates.iter().zip(update_set.shapes.iter()))
            {
                let array = Array2::from_shape_vec((rows, cols), data.clone()).unwrap();
                *sum += &array;
            }
        }

        let avg_batch_weight_changes: Vec<Array2<f32>> = sum_batch_weight_changes
            .into_iter()
            .map(|sum_weight_change| sum_weight_change / self.batch_size as f32)
            .collect();
        profile.record("aggregate_updates", t.elapsed());

        let t = Instant::now();
        let updates = WeightUpdateSet {
            updates: avg_batch_weight_changes
                .iter()
                .map(|array| array.iter().copied().collect())
                .collect(),
            shapes: avg_batch_weight_changes
                .iter()
                .map(|array| array.dim())
                .collect(),
        };
        self.runtime.apply_weight_updates(&updates)?;
        profile.record("apply_weights", t.elapsed());

        Ok(profile)
    }

    fn report_hook(&mut self, step: u32, mean_step_time: TimeDelta) -> Result<()> {
        debug!(
            "After step {}: mean step duration = {:.2?}",
            step, mean_step_time
        );

        let est_time_to_finish = mean_step_time * (self.config.training_steps - step) as i32;
        let est_finish_time = chrono::Utc::now() + est_time_to_finish;

        // The mini batch model is cloned for each batch element, so the main model never gets
        // inference run on it. Do a forward pass here to report current energy.
        let (input, output) = self.data.get_random_input_and_output();
        self.runtime
            .set_input(input.as_slice().expect("contiguous input array"))?;
        self.runtime
            .set_output(output.as_slice().expect("contiguous output array"))?;
        self.runtime.reinitialise_latents()?;
        self.runtime.converge_values()?;

        let energy = self.runtime.total_energy()?;
        info!(
            "Step {}: Current model state: energy = {:.2}\tEstimated finish time: {}",
            step,
            energy,
            est_finish_time.format("%Y-%m-%d %H:%M:%S")
        );
        Ok(())
    }
});

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_utils::{DummyTrainingDataset, tiny_relu_model};
    use crate::training::TrainingHandler;
    use crate::training::configuration::{DataSetSource, ModelSource, TrainingStrategy};
    use crate::training::handlers::singlethreaded::SingleThreadTrainHandler;
    use ndarray::{Array2, array};

    fn dummy_config() -> TrainConfig {
        TrainConfig {
            model_source: ModelSource::Snapshot(String::from("unused.json")),
            training_dataset: DataSetSource::IdxFormat {
                input_idx_file: String::from("unused-images.idx"),
                output_idx_file: String::from("unused-labels.idx"),
            },
            evaluation_dataset: Some(DataSetSource::IdxFormat {
                input_idx_file: String::from("unused-images.idx"),
                output_idx_file: String::from("unused-labels.idx"),
            }),
            training_strategy: TrainingStrategy::CpuSingleThread,
            training_steps: 1,
            report_interval: 0,
            snapshot_interval: 0,
        }
    }

    fn tiny_dataset() -> Arc<dyn TrainingDataset> {
        let mut labels: Array2<f32> = Array2::zeros((1, 10));
        labels
            .row_mut(0)
            .assign(&array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        Arc::new(DummyTrainingDataset::from_arrays(
            array![[1.0, 0.0, 0.5, 0.25]],
            labels,
        ))
    }

    fn assert_arrays_close(left: &Array2<f32>, right: &Array2<f32>, tolerance: f32) {
        assert_eq!(left.dim(), right.dim());
        for (left_value, right_value) in left.iter().zip(right.iter()) {
            assert!(
                (left_value - right_value).abs() <= tolerance,
                "expected {left_value} and {right_value} to be within {tolerance}"
            );
        }
    }

    #[test]
    fn minibatch_aggregation_matches_single_sample_update_on_deterministic_fixture() {
        let initial_model: PredictiveCodingModel = tiny_relu_model();
        let dataset: Arc<dyn TrainingDataset> = tiny_dataset();
        let config: TrainConfig = dummy_config();

        let runtime = CpuModelRuntime::from_model(initial_model.clone());
        let mut single_handler: SingleThreadTrainHandler<CpuModelRuntime> =
            SingleThreadTrainHandler::new(
                config.clone(),
                runtime,
                Arc::clone(&dataset),
                String::from("unused/single"),
            );

        let mut batch_handler: CpuBatchTrainHandler = CpuBatchTrainHandler::new(
            config,
            initial_model,
            dataset,
            String::from("unused/batch"),
            4,
        );

        single_handler.train_step(0).unwrap();
        batch_handler.train_step(0).unwrap();

        let single_snapshot = single_handler.model_snapshot().unwrap();
        let batch_snapshot = batch_handler.model_snapshot().unwrap();
        let single_layer = &single_snapshot.layers[1];
        let batch_layer = &batch_snapshot.layers[1];
        let single_weights = Array2::from_shape_vec(
            (single_layer.weight_rows, single_layer.weight_cols),
            single_layer.weights.clone(),
        )
        .unwrap();
        let batch_weights = Array2::from_shape_vec(
            (batch_layer.weight_rows, batch_layer.weight_cols),
            batch_layer.weights.clone(),
        )
        .unwrap();
        assert_arrays_close(&single_weights, &batch_weights, 1e-6);
    }

    #[test]
    fn minibatch_report_hook_and_dataset_accessors_use_fixture_sample() {
        let dataset = tiny_dataset();
        let mut handler = CpuBatchTrainHandler::new(
            dummy_config(),
            tiny_relu_model(),
            Arc::clone(&dataset),
            String::from("unused/batch"),
            2,
        );

        handler.pre_training_hook().unwrap();
        handler.report_hook(0, TimeDelta::zero()).unwrap();

        let data = handler.get_data();
        assert_eq!(data.get_dataset_size(), 1);
        assert_eq!(data.get_input_size(), 4);
        assert_eq!(data.get_output_size(), 10);
        assert_eq!(data.get_random_input(), data.get_input(0));

        let (input, output) = data.get_random_input_and_output();
        assert_eq!(input, data.get_input(0));
        assert_eq!(output, data.get_output(0));
    }
}
