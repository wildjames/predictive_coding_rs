mod minibatch;
mod singlethreaded;

#[cfg(feature = "gpu")]
mod gpu_minibatch;

pub use minibatch::CpuBatchTrainHandler;
pub use singlethreaded::SingleThreadTrainHandler;

#[cfg(feature = "gpu")]
pub use gpu_minibatch::GpuBatchTrainHandler;

/// Implements the boilerplate delegation methods of `TrainingHandler` for a handler struct.
///
/// Expects the struct to have fields: `config`, `data`, `file_output_prefix`,
/// and a runtime field that implements `ModelRuntime`.
///
/// Custom trait methods (e.g. `profiled_train_step`, `report_hook`) are passed
/// in the body and emitted inside the same `impl TrainingHandler` block.
macro_rules! impl_handler_delegation {
    (
      $handler:ty, // ty: type expression
      $runtime:ident, // ident: identifier
      { $($custom:tt)* } // tt: token tree, this catches the custom method implementations as a token tree and emits them verbatim inside the impl block
    ) => {
        impl $crate::training::TrainingHandler for $handler {
            fn get_config(&self) -> &$crate::training::TrainConfig {
                &self.config
            }
            fn model_snapshot(&mut self) -> $crate::error::Result<$crate::model::ModelSnapshot> {
                self.$runtime.snapshot()
            }
            fn model_config(&self) -> $crate::model::PredictiveCodingModelConfig {
                self.$runtime.config()
            }
            fn pin_input(&mut self) -> $crate::error::Result<()> {
                self.$runtime.pin_input()
            }
            fn pin_output(&mut self) -> $crate::error::Result<()> {
                self.$runtime.pin_output()
            }
            fn get_data(&self) -> &dyn $crate::data_handling::TrainingDataset {
                self.data.as_ref()
            }
            fn get_file_output_prefix(&self) -> &String {
                &self.file_output_prefix
            }

            $($custom)*
        }
    };
}

pub(crate) use impl_handler_delegation;
