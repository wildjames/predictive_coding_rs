//! Evaluate a trained model against the evaluation dataset defined in a training config.

use tracing::info;

use predictive_coding::{
  error::Result,
  inference::evaluation_handler::{
    EvaluationArtifacts,
    EvaluationRun
  },
  utils::logging
};

use clap::Parser;

/// Evaluate a trained model against the evaluation dataset defined in the training configuration, and report its accuracy.
#[derive(Parser)]
struct EvalArgs {
  /// The model file to evaluate
  #[arg()]
  model_file: String,

  /// The training config file used to train the model defines an evaluation dataset.
  #[arg(long)]
  training_config: String
}

fn main() -> Result<()> {
  logging::setup_tracing(false);

  let args = EvalArgs::parse();

  let mut evaluation_run: EvaluationRun = EvaluationRun::load(args.model_file, args.training_config)?;
  let artifacts: EvaluationArtifacts = evaluation_run.evaluate_and_write_results()?;

  info!(
    "Evaluation complete. Accuracy: {:.2}%, convergence time on average is {:.0}ms. When correct, average confidence: {:.3}",
    artifacts.summary.accuracy * 100.0,
    artifacts.summary.mean_convergence_time_ms,
    artifacts.summary.mean_confidence_when_correct
  );
  info!("Saved evaluation results to {}", artifacts.output_path.display());

  Ok(())
}
