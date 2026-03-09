mod minibatch;
mod singlethreaded;

pub use minibatch::BatchTrainHandler;
pub use singlethreaded::SingleThreadTrainHandler;

use super::{TrainConfig, TrainingHandler};
