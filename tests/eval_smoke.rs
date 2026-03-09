mod common;

use std::{fs, path::PathBuf, process::Command};

use common::{prepare_smoke_root, read_json, repo_root, run_command};

#[test]
fn eval_binary_smoke_on_tiny_fixtures() {
    // File system stuff
    let root: PathBuf = repo_root();
    let (smoke_root, _cleanup) = prepare_smoke_root("eval_smoke");
    let eval_dir: PathBuf = smoke_root.join("eval");
    fs::create_dir_all(&eval_dir).unwrap();

    // Get the model snapshot in the correct directory
    let model_path: PathBuf = eval_dir.join("model_final_model.json");
    fs::copy(
        root.join("test_data/baselines/single_thread/train/model_final_model.json"),
        &model_path,
    )
    .unwrap();

    // Dispatch the eval binary with the arguments set up above
    run_command(
        Command::new(env!("CARGO_BIN_EXE_eval"))
            .current_dir(&root)
            .arg(&model_path)
            .arg("--training-config")
            .arg("test_data/baselines/single_thread/train/model_training_config.json"),
    );

    // Did we produce the correct artifacts?
    let eval_results_path: PathBuf = eval_dir.join("evaluation_results.json");
    assert!(eval_results_path.exists());

    // Do the artifacts' contents look right?
    let eval_results = read_json(&eval_results_path);
    assert_eq!(eval_results["summary"]["total_predictions"], 1);
}
