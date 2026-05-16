use crate::{
    data_handling::TrainingDataset,
    error::{PredictiveCodingError, Result},
    model::{CpuModelRuntime, PredictiveCodingModel},
};

use super::{
    TrainConfig, TrainingHandler, TrainingStrategy,
    handlers::{CpuBatchTrainHandler, SingleThreadTrainHandler},
    load_dataset, load_model, load_training_config, validate_model_and_dataset_shapes,
    validate_training_config,
};

#[cfg(feature = "gpu")]
use super::handlers::GpuBatchTrainHandler;
#[cfg(feature = "gpu")]
use crate::model::GpuModelRuntime;

use std::sync::Arc;
use tracing::info;

fn get_handler(
    training_config: TrainConfig,
    model: PredictiveCodingModel,
    data: Arc<dyn TrainingDataset>,
    file_output_prefix: String,
) -> Result<Box<dyn TrainingHandler>> {
    match training_config.training_strategy.clone() {
        TrainingStrategy::CpuSingleThread => {
            let runtime = CpuModelRuntime::from_model(model);
            Ok(Box::new(SingleThreadTrainHandler::new(
                training_config,
                runtime,
                data,
                file_output_prefix,
            )))
        }
        TrainingStrategy::CpuMiniBatch { batch_size } => Ok(Box::new(CpuBatchTrainHandler::new(
            training_config,
            model,
            data,
            file_output_prefix,
            batch_size,
        ))),
        #[cfg(feature = "gpu")]
        TrainingStrategy::GpuSingleThread => {
            let snapshot = model.to_snapshot();
            let runtime = GpuModelRuntime::from_snapshot(&snapshot)?;
            Ok(Box::new(SingleThreadTrainHandler::new(
                training_config,
                runtime,
                data,
                file_output_prefix,
            )))
        }
        #[cfg(feature = "gpu")]
        TrainingStrategy::GpuMiniBatch { batch_size } => Ok(Box::new(GpuBatchTrainHandler::new(
            training_config,
            model,
            data,
            file_output_prefix,
            batch_size,
        )?)),
    }
}

/// Sets up a training run handler based on the provided config path and output prefix.
/// The handler will orchestrate the training process by providing hook functions to the training loop.
pub fn setup_training_run_handler(
    config: String,
    output_prefix: String,
) -> Result<Box<dyn TrainingHandler>> {
    let training_config: TrainConfig = load_training_config(&config)?;
    validate_training_config(&training_config)?;

    let data: Arc<dyn TrainingDataset> = load_dataset(&training_config.training_dataset)?;
    info!(
        "Loaded the dataset. I have {} samples",
        data.get_dataset_size()
    );

    // Build the model
    let model: PredictiveCodingModel = load_model(&training_config.model_source)?;
    info!(
        "Created the model with layer sizes {:?}",
        model.get_layer_sizes()
    );

    validate_model_and_dataset_shapes(&model, data.as_ref())?;

    // Make sure that the output directory exists
    if let Some(output_dir) = std::path::Path::new(&output_prefix)
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        info!("Saving training artifacts to {}", output_dir.display());
        std::fs::create_dir_all(output_dir).map_err(|source| {
            PredictiveCodingError::io("create training artifact directory", output_dir, source)
        })?;
    }

    // The handler orchestrated the training process by providing hook functions to the training loop.
    // Choose the correct one for this config.
    get_handler(training_config, model, data, output_prefix)
}
