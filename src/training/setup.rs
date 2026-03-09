use crate::{
    data_handling::TrainingDataset,
    error::{PredictiveCodingError, Result},
    model::PredictiveCodingModel,
};

use super::{
    TrainConfig, TrainingHandler, TrainingStrategy,
    handlers::{BatchTrainHandler, SingleThreadTrainHandler},
    load_dataset, load_model, load_training_config, validate_model_and_dataset_shapes,
    validate_training_config,
};

use std::sync::Arc;
use tracing::info;

fn get_handler(
    training_config: TrainConfig,
    model: PredictiveCodingModel,
    data: Arc<dyn TrainingDataset>,
    file_output_prefix: String,
) -> Box<dyn TrainingHandler> {
    match training_config.training_strategy.clone() {
        TrainingStrategy::SingleThread => Box::new(SingleThreadTrainHandler::new(
            training_config,
            model,
            data,
            file_output_prefix,
        )),
        TrainingStrategy::MiniBatch { batch_size } => Box::new(BatchTrainHandler::new(
            training_config,
            model,
            data,
            file_output_prefix,
            batch_size,
        )),
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
    Ok(get_handler(training_config, model, data, output_prefix))
}
