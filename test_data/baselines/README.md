# Baselines

These files record the current behavior of the tiny deterministic fixture runs used for e2e tests.

- `single_thread/train/` contains the artifact set from the `train` binary using the single-threaded strategy.
- `mini_batch/train/` contains the artifact set from the `train` binary using the mini-batch strategy.
- `single_thread/bench/` contains the benchmark CSV and JSON produced by the `bench` binary in release mode.
- `mini_batch/bench/` contains the benchmark CSV and JSON produced by the `bench` binary in release mode.

The training artifacts are deterministic because they start from a fixed snapshot and use a one-sample 4-to-10 fixture with no hidden layer. Benchmark timing numbers are environment-specific. Treat them as reference measurements captured on for a specific machine, not as exact values that must hold on every machine. For real benchmarking, I plan on using my homelab server so I get a consistent amount of horsepower when running that.
