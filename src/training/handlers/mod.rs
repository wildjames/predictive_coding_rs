mod minibatch;
mod singlethreaded;

#[cfg(feature = "gpu")]
mod gpu_minibatch;

pub use minibatch::CpuBatchTrainHandler;
pub use singlethreaded::SingleThreadTrainHandler;

#[cfg(feature = "gpu")]
pub use gpu_minibatch::GpuBatchTrainHandler;

use super::{TrainConfig, TrainingHandler};
