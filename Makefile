.PHONY: grab-mnist

grab-mnist:
	./scripts/grab_mnist.sh

build:
	cargo build --release

test:
	cargo test

clean:
	cargo clean

train: build
	cargo run --bin train

infer: build
	cargo run --bin infer

monitor_training: build
	cargo run --bin train_grpc_client

test-liveplot: build
	cargo run --example test-liveplot
