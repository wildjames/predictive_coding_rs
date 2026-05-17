//! This program is used to benchmark the speed of various training configurations and model architectures. It takes a training config file as input, and times its processes, creating a series of files detailing the results.

use std::{collections::BTreeMap, path::Path, time::Instant};

use predictive_coding::{
    error::{PredictiveCodingError, Result},
    model::{PredictiveCodingModelConfig, save_model_config},
    training::{
        StepProfile, TrainConfig, TrainingHandler, save_training_config, setup_training_run_handler,
    },
    utils::{logging, timestamp},
};

use clap::Parser;
use tracing::info;

#[cfg(test)]
#[path = "test_utils.rs"]
mod test_utils;

/// This program is used to benchmark the speed of various training configurations and model architectures. It takes a training config file as input, and times its processes, creating a series of files detailing the results.
#[derive(Parser)]
struct BenchArgs {
    /// The model configuration to benchmark.
    // #[arg(default_value_t = String::from("benchmark_data/benchmark_minibatch_config.json"))]
    #[arg(default_value_t = String::from("benchmark_data/benchmark_gpu_singlethread_config.json"))]
    config: String,

    /// Optional artifact output prefix. Defaults to `benchmark_data/<timestamp>/benchmark`.
    #[arg(long, default_value_t = format!("benchmark_data/benchmark_{}/bench_", timestamp()))]
    output_prefix: String,
}

fn current_git_commit_hash_with_command(command_name: &str, args: &[&str]) -> Result<String> {
    let command = if args.is_empty() {
        command_name.to_string()
    } else {
        format!("{} {}", command_name, args.join(" "))
    };
    let output = std::process::Command::new(command_name)
        .args(args)
        .output()
        .map_err(|source| PredictiveCodingError::command_io(command.clone(), source))?;

    if !output.status.success() {
        return Err(PredictiveCodingError::command_failed(
            command,
            output.status,
            String::from_utf8_lossy(&output.stderr),
        ));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn current_git_commit_hash() -> Result<String> {
    current_git_commit_hash_with_command("git", &["rev-parse", "HEAD"])
}

fn run_benchmark(args: BenchArgs) -> Result<()> {
    // Detect if this binary was compiled in release mode or not
    let release_mode: bool = !cfg!(debug_assertions);
    #[cfg(debug_assertions)]
    info!(
        "Running benchmark in debug mode. For more accurate benchmarking, compile with --release"
    );
    let mut handler: Box<dyn TrainingHandler> =
        setup_training_run_handler(args.config, args.output_prefix.clone())?;

    let step_data: Vec<BenchmarkStepData> = run_benchmark_training_loop(
        handler.as_mut(),
        &format!("{}_{}", &args.output_prefix, "bench_run.csv"),
    )?;

    // Compute per-phase summary
    let phase_summary = compute_phase_summary(&step_data);

    // Print summary to console
    info!("--- Benchmark Phase Summary ---");
    let total_wall: f32 = step_data.iter().map(|s| s.total_ms).sum();
    info!(
        "Total wall time: {:.1} ms over {} steps ({:.1} ms/step avg)",
        total_wall,
        step_data.len(),
        if step_data.is_empty() {
            0.0
        } else {
            total_wall / step_data.len() as f32
        }
    );
    for (name, s) in &phase_summary {
        info!(
            "  {:<25} mean={:>8.2}ms  min={:>8.2}ms  max={:>8.2}ms  total={:>10.1}ms  ({:.1}%)",
            name, s.mean_ms, s.min_ms, s.max_ms, s.total_ms, s.pct_of_total
        );
    }

    // Write the training params to "{output_prefix}/params.json"
    let current_commit_hash_str: String = current_git_commit_hash()?;

    let result = BenchmarkResult {
        step_data,
        phase_summary,
        git_commit_hash: current_commit_hash_str,
        run_timestamp: chrono::Utc::now().to_rfc3339(),
        release_mode,
    };

    // Write the benchmarking parameters, training parameters, and model configuration to files for posterity.
    // This is JSON, so probably a bit harder to read than the CSV file, but it does create a single file with all the relevant information for each benchmark run, which is nice.
    let result_path: String = format!("{}_{}", args.output_prefix, "result.json");
    let result_file = std::fs::File::create(&result_path).map_err(|source| {
        PredictiveCodingError::io("create benchmark result", &result_path, source)
    })?;
    serde_json::to_writer_pretty(result_file, &result)
        .map_err(|source| PredictiveCodingError::json_serialize(&result_path, source))?;

    Ok(())
}

fn main() -> Result<()> {
    logging::setup_tracing(false);
    info!("Starting benchmark run");
    run_benchmark(BenchArgs::parse())
}

#[derive(serde::Serialize)]
struct BenchmarkResult {
    step_data: Vec<BenchmarkStepData>,
    phase_summary: BTreeMap<String, PhaseSummary>,
    git_commit_hash: String,
    run_timestamp: String,
    release_mode: bool,
}

#[derive(serde::Serialize)]
struct BenchmarkStepData {
    step: u32,
    total_ms: f32,
    phases: BTreeMap<String, f32>,
}

#[derive(serde::Serialize)]
struct PhaseSummary {
    mean_ms: f32,
    min_ms: f32,
    max_ms: f32,
    total_ms: f32,
    pct_of_total: f32,
}

fn run_benchmark_training_loop(
    handler: &mut dyn TrainingHandler,
    bench_run_outfile: &str,
) -> Result<Vec<BenchmarkStepData>> {
    let mut benchmark_data: Vec<BenchmarkStepData> = Vec::new();

    handler.pre_training_hook()?;

    // Time each training step and write to a csv file
    if let Some(parent) = Path::new(bench_run_outfile)
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).map_err(|source| {
            PredictiveCodingError::io("create benchmark output directory", parent, source)
        })?;
    }
    let mut wtr = csv::Writer::from_path(bench_run_outfile).map_err(|source| {
        PredictiveCodingError::csv("create benchmark writer", bench_run_outfile, source)
    })?;

    let training_config: &TrainConfig = handler.get_config();
    let training_steps: u32 = training_config.training_steps;

    // Write the config and training params to a file
    let model_config: PredictiveCodingModelConfig = handler.model_config();
    save_model_config(
        &model_config,
        &format!("{}_model_config.json", &handler.get_file_output_prefix()),
    )?;
    save_training_config(
        handler.get_config(),
        &format!("{}_training_config.json", &handler.get_file_output_prefix()),
    )?;

    // We discover phase names from the first step's profile, then use a
    // consistent column order for the CSV.
    let mut phase_names: Vec<String> = Vec::new();
    let mut header_written = false;

    for step in 0..training_steps {
        let start_time: Instant = Instant::now();
        handler.pre_step_hook(step)?;
        let profile: StepProfile = handler.profiled_train_step(step)?;
        handler.post_step_hook(step)?;
        let wall_time = start_time.elapsed();

        let wall_time_ms: f32 = wall_time.as_secs_f32() * 1000.0;

        // On the first step, discover phase names and write the CSV header
        if !header_written {
            phase_names = profile
                .phases
                .iter()
                .map(|(name, _)| name.clone())
                .collect();
            let mut header: Vec<String> = vec!["step".into(), "total_ms".into()];
            for name in &phase_names {
                header.push(format!("{}_ms", name));
            }
            wtr.write_record(&header).map_err(|source| {
                PredictiveCodingError::csv("write benchmark header", bench_run_outfile, source)
            })?;
            header_written = true;
        }

        // Build phase timing map
        let mut phases: BTreeMap<String, f32> = BTreeMap::new();
        for (name, dur) in &profile.phases {
            phases.insert(name.clone(), dur.as_secs_f32() * 1000.0);
        }

        // Write CSV row
        let mut row: Vec<String> = vec![step.to_string(), format!("{:.3}", wall_time_ms)];
        for name in &phase_names {
            row.push(format!("{:.3}", phases.get(name).copied().unwrap_or(0.0)));
        }
        wtr.write_record(&row).map_err(|source| {
            PredictiveCodingError::csv("append benchmark row", bench_run_outfile, source)
        })?;
        wtr.flush().map_err(|source| {
            PredictiveCodingError::io("flush benchmark CSV", bench_run_outfile, source)
        })?;

        // Log with phase breakdown
        let phase_str: String = profile
            .phases
            .iter()
            .map(|(name, dur)| format!("{}={:.1}ms", name, dur.as_secs_f32() * 1000.0))
            .collect::<Vec<_>>()
            .join("  ");
        info!(
            "Step {}: total {:.1} ms  [{}]",
            step, wall_time_ms, phase_str
        );

        benchmark_data.push(BenchmarkStepData {
            step,
            total_ms: wall_time_ms,
            phases,
        });
    }

    handler.post_training_hook()?;

    Ok(benchmark_data)
}

/// Compute per-phase summary statistics from the step data.
fn compute_phase_summary(step_data: &[BenchmarkStepData]) -> BTreeMap<String, PhaseSummary> {
    if step_data.is_empty() {
        return BTreeMap::new();
    }

    // Collect all phase names
    let mut all_phases: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut total_wall_ms: f32 = 0.0;

    for step in step_data {
        total_wall_ms += step.total_ms;
        for (name, &ms) in &step.phases {
            all_phases.entry(name.clone()).or_default().push(ms);
        }
    }

    let mut summary = BTreeMap::new();
    for (name, times) in &all_phases {
        let total: f32 = times.iter().sum();
        let mean = total / times.len() as f32;
        let min = times.iter().copied().fold(f32::INFINITY, f32::min);
        let max = times.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let pct = if total_wall_ms > 0.0 {
            (total / total_wall_ms) * 100.0
        } else {
            0.0
        };
        summary.insert(
            name.clone(),
            PhaseSummary {
                mean_ms: mean,
                min_ms: min,
                max_ms: max,
                total_ms: total,
                pct_of_total: pct,
            },
        );
    }

    summary
}

#[cfg(test)]
mod tests {
    use super::test_utils::{RecordingTrainingHandler, TempDir, single_thread_train_config};
    use super::*;

    use std::{fs, path::PathBuf};

    #[test]
    fn current_git_commit_hash_helper_handles_success_and_failure_cases() {
        let success: String =
            current_git_commit_hash_with_command("sh", &["-c", "printf 'abc123\\n'"]).unwrap();
        assert_eq!(success, "abc123");

        let failure =
            current_git_commit_hash_with_command("sh", &["-c", "printf 'boom\\n' >&2; exit 2"]);
        assert!(matches!(
          failure,
          Err(PredictiveCodingError::CommandFailed { stderr, .. }) if stderr.contains("boom")
        ));

        let spawn_failure: std::result::Result<String, PredictiveCodingError> =
            current_git_commit_hash_with_command("/definitely/missing/git", &[]);
        assert!(matches!(
            spawn_failure,
            Err(PredictiveCodingError::CommandIo { .. })
        ));
    }

    #[test]
    fn benchmark_loop_writes_csv_and_artifacts() {
        let temp_dir: TempDir = TempDir::new("benchmark_loop");
        let output_prefix: String = temp_dir.join("nested/bench").display().to_string();
        let bench_csv: PathBuf = temp_dir.join("nested/bench_run.csv");
        let mut handler = RecordingTrainingHandler::new(
            single_thread_train_config(2, 0, 0),
            output_prefix.clone(),
        );

        let step_data: Vec<BenchmarkStepData> =
            run_benchmark_training_loop(&mut handler, bench_csv.to_str().unwrap()).unwrap();

        assert_eq!(handler.steps, vec![0, 1]);
        assert_eq!(step_data.len(), 2);
        assert!(bench_csv.exists());
        assert!(Path::new(&format!("{}_model_config.json", output_prefix)).exists());
        assert!(Path::new(&format!("{}_training_config.json", output_prefix)).exists());
        assert!(Path::new(&format!("{}_final_model.json", output_prefix)).exists());

        let csv_output = fs::read_to_string(bench_csv).unwrap();
        assert!(csv_output.contains("step,total_ms"));
        assert!(csv_output.contains("\n0,"));
        assert!(csv_output.contains("\n1,"));
    }

    #[test]
    fn recording_handler_exposes_dataset_fixture() {
        let handler = RecordingTrainingHandler::new(
            single_thread_train_config(2, 0, 0),
            String::from("unused/bench"),
        );
        let data = handler.get_data();

        assert_eq!(data.get_dataset_size(), 1);
        assert_eq!(data.get_input_size(), 4);
        assert_eq!(data.get_output_size(), 10);
        assert_eq!(data.get_random_input(), data.get_input(0));

        let (input, output) = data.get_random_input_and_output();
        assert_eq!(input, data.get_input(0));
        assert_eq!(output, data.get_output(0));
    }

    #[test]
    fn run_benchmark_writes_result_payload_in_process() {
        let temp_dir: TempDir = TempDir::new("benchmark_run_main_path");
        let output_prefix: String = temp_dir.join("end_to_end/bench").display().to_string();

        run_benchmark(BenchArgs {
            config: String::from("test_data/bench_single_thread_config.json"),
            output_prefix: output_prefix.clone(),
        })
        .unwrap();

        let result_path: String = format!("{}_result.json", output_prefix);
        let result_json: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(result_path).unwrap()).unwrap();
        assert_eq!(result_json["step_data"].as_array().unwrap().len(), 2);
        assert!(!result_json["git_commit_hash"].as_str().unwrap().is_empty());
    }
}
