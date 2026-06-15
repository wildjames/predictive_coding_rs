.PHONY: grab-mnist build test clean train infer bench gen

grab-mnist:
	./scripts/grab_mnist.sh

build:
	cargo build --release --all-features

test:
	cargo test --release --all-features

clean:
	cargo clean

train-gpu:
	cargo run --release --bin train --all-features data/training_gpu_batch_config.json

bench:
	cargo run --release --bin bench --all-features

coverage:
	cargo llvm-cov

estimate-batch-size:
	cargo run --release --bin batch-size-advisor --all-features data/model_config.json
