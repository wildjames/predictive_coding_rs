# Test Data

This directory contains tiny, deterministic fixtures used by Phase 1 tests.

- `mnist/` contains hand-made 2x2 IDX image fixtures with a single label.
- `model_config_tiny.json` builds a tiny 4-to-10 model.
- `model_snapshot_tiny.json` is a fixed model snapshot used for deterministic training and benchmark baselines.
- `train_*` and `bench_*` configs exercise both current training strategies against the tiny MNIST-like fixture.
- `baselines/` records reference artifacts generated from these fixtures.

The JSON files are committed as text. The IDX payloads are binary and are generated directly in the workspace.
