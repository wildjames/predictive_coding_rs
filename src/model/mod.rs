pub mod maths;
pub mod model_utils;

mod model_structure;
pub use model_structure::{Layer, PredictiveCodingModel, PredictiveCodingModelConfig};

mod file_handling;
pub use file_handling::{
    create_from_config, load_model_config, load_model_snapshot, save_model_config,
    save_model_snapshot,
};
