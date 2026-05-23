//! CLI tool to estimate the ideal GPU batch size for a given model config.

use predictive_coding::{error::Result, model::load_model_config};

use clap::Parser;

#[derive(Parser)]
#[command(about = "Estimate the ideal GPU batch size for a model configuration")]
struct Args {
    /// Path to model config JSON file
    #[arg()]
    config: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = load_model_config(&args.config)?;

    println!("Model: {:?}", config.layer_sizes);
    println!(
        "Convergence: {} steps, threshold {:.1e}",
        config.convergence_steps, config.convergence_threshold
    );
    println!();

    let estimate = predictive_coding::model::estimate_batch_size(&config)?;
    println!("{estimate}");

    Ok(())
}
