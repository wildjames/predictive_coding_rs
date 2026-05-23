pub mod maths;
pub mod model_utils;

mod snapshot;
pub use snapshot::{LayerSnapshot, ModelSnapshot};

mod runtime;
pub use runtime::{ExecutionBackend, ModelRuntime, TrainableModelRuntime, WeightUpdateSet};

mod cpu;
pub use cpu::CpuModelRuntime;

#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "gpu")]
pub use gpu::GpuRuntime;
#[cfg(feature = "gpu")]
pub use gpu::{GpuMemoryEstimate, estimate_batch_size};

mod model_structure;
pub use model_structure::{Layer, PredictiveCodingModel, PredictiveCodingModelConfig};

mod file_handling;
pub use file_handling::{
    create_from_config, load_model_config, load_model_snapshot, save_model_config,
    save_model_snapshot, save_snapshot,
};
