use crate::{
    data_handling::TrainingDataset,
    error::Result,
    model::{
        CpuModelRuntime, ModelRuntime, PredictiveCodingModel, TrainableModelRuntime,
        WeightUpdateSet, model_utils::set_rand_input_and_output,
    },
};

use super::{TrainConfig, TrainingHandler};

use chrono::TimeDelta;
use ndarray::{Array1, Array2};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::Arc;
use tracing::{debug, info};

pub struct BatchTrainHandler {
    config: TrainConfig,
    runtime: CpuModelRuntime,
    data: Arc<dyn TrainingDataset>,
    file_output_prefix: String,
    batch_size: u32,
}

impl BatchTrainHandler {
    pub fn new(
        config: TrainConfig,
        model: PredictiveCodingModel,
        data: Arc<dyn TrainingDataset>,
        file_output_prefix: String,
        batch_size: u32,
    ) -> Self {
        BatchTrainHandler {
            config,
            runtime: CpuModelRuntime::from_model(model),
            data,
            file_output_prefix,
            batch_size,
        }
    }
}

impl TrainingHandler for BatchTrainHandler {
    fn get_config(&self) -> &TrainConfig {
        &self.config
    }
    fn get_model(&mut self) -> &mut PredictiveCodingModel {
        self.runtime.model_mut()
    }
    fn get_data(&self) -> &dyn TrainingDataset {
        self.data.as_ref()
    }
    fn get_file_output_prefix(&self) -> &String {
        &self.file_output_prefix
    }

    /// Report the training parameters and model config. Also save setup to file for future reference
    fn pre_training_hook(&mut self) -> Result<()> {
        info!("Starting training with mini-batch strategy");
        info!("Mini batch params: batch size = {}", self.batch_size);
        Ok(())
    }

    /// Train batch_size models in parallel on different data, then compute the average weight update across them and apply it to the model.
    fn train_step(&mut self, _step: u32) -> Result<()> {
        // I'll iterate over this with Rayon to parallelise the batch
        let batch_inputs_and_outputs: Vec<(Array1<f32>, Array1<f32>)> = (0..self.batch_size)
            .map(|_| self.data.get_random_input_and_output())
            .collect();

        // Each element of the batch trains on a single sample, and we collect their weight changes as a result.
        // The batch weight changes will be a Vec of length batch_size, where each element is a Vec of length
        // num_layers, and each element of THAT is an array2 of the weight changes for the relevant layer.
        let batch_weight_changes: Vec<Result<WeightUpdateSet>> = batch_inputs_and_outputs
            .into_par_iter()
            .map(|(input_data, output_data)| {
                debug!(
                    "Training on batch element with input {:?} and output {:?}",
                    input_data, output_data
                );
                // Each thread gets its own runtime clone, and trains it independently
                let mut runtime_clone: CpuModelRuntime =
                    CpuModelRuntime::from_model(self.runtime.model().clone());

                // Set the model input and output to the batch element's data
                runtime_clone.model_mut().set_input(input_data);
                runtime_clone.model_mut().set_output(output_data);

                runtime_clone.reinitialise_latents()?;

                // Train on this example until convergence.
                runtime_clone.converge_values()?;

                let updates: WeightUpdateSet = runtime_clone.compute_weight_updates()?;
                Ok(updates)
            })
            .collect();
        debug!(
            "Batch weight changes computed for {} samples",
            self.batch_size
        );

        // Unwrap results and collect into Vec<WeightUpdateSet>
        let successful_updates: Vec<WeightUpdateSet> = batch_weight_changes
            .into_iter()
            .collect::<Result<Vec<_>>>()?;

        // Reconstruct Array2s from the first element to initialise the sum
        let first = &successful_updates[0];
        let mut sum_batch_weight_changes: Vec<Array2<f32>> = first
            .updates
            .iter()
            .zip(first.shapes.iter())
            .map(|(data, &(rows, cols))| {
                Array2::from_shape_vec((rows, cols), data.clone()).unwrap()
            })
            .collect();

        // Accumulate remaining updates
        for update_set in &successful_updates[1..] {
            for (sum, (data, &(rows, cols))) in sum_batch_weight_changes
                .iter_mut()
                .zip(update_set.updates.iter().zip(update_set.shapes.iter()))
            {
                let array = Array2::from_shape_vec((rows, cols), data.clone()).unwrap();
                *sum += &array;
            }
        }

        // Average the weight changes across the batch
        let avg_batch_weight_changes: Vec<Array2<f32>> = sum_batch_weight_changes
            .into_iter()
            .map(|sum_weight_change| sum_weight_change / self.batch_size as f32)
            .collect();

        // Apply the average weight changes to the model.
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

        Ok(())
    }

    fn report_hook(&mut self, step: u32, mean_step_time: TimeDelta) -> Result<()> {
        debug!(
            "After step {}: mean step duration = {:.2?}",
            step, mean_step_time
        );

        // Estimate how much longer needed to complete the training
        let est_time_to_finish: chrono::Duration =
            mean_step_time * (self.config.training_steps - step) as i32;
        let est_finish_time: chrono::DateTime<chrono::Utc> =
            chrono::Utc::now() + est_time_to_finish;

        // The mini batch model is cloned for each batch element, so the main model never gets inference run on it
        // So, do that in the reporting step to get a sense of how the model is doing on the data.
        self.runtime.reinitialise_latents()?;
        set_rand_input_and_output(self.runtime.model_mut(), self.data.as_ref());
        self.runtime.converge_values()?;

        let energy: f32 = self.runtime.total_energy()?;
        info!(
            "Step {}: Current model state: energy = {:.2}\tEstimated finish time: {}",
            step,
            energy,
            est_finish_time.format("%Y-%m-%d %H:%M:%S")
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_utils::{DummyTrainingDataset, tiny_relu_model};
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
            training_strategy: TrainingStrategy::SingleThread,
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

        let mut single_handler: SingleThreadTrainHandler = SingleThreadTrainHandler::new(
            config.clone(),
            initial_model.clone(),
            Arc::clone(&dataset),
            String::from("unused/single"),
        );

        let mut batch_handler: BatchTrainHandler = BatchTrainHandler::new(
            config,
            initial_model,
            dataset,
            String::from("unused/batch"),
            4,
        );

        single_handler.train_step(0).unwrap();
        batch_handler.train_step(0).unwrap();

        let single_weights = &single_handler.get_model().get_layer(1).weights;
        let batch_weights = &batch_handler.get_model().get_layer(1).weights;
        assert_arrays_close(single_weights, batch_weights, 1e-6);
    }

    #[test]
    fn minibatch_report_hook_and_dataset_accessors_use_fixture_sample() {
        let dataset = tiny_dataset();
        let mut handler = BatchTrainHandler::new(
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
