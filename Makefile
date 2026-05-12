.PHONY: grab-mnist build test clean train infer bench gen

grab-mnist:
	./scripts/grab_mnist.sh

build:
	cargo build --release --all-features

test: build
	cargo test --release --all-features

clean:
	cargo clean

train: build
	cargo run --release --bin train --all-features

bench: build
	cargo run --release --bin bench --all-features

coverage: build
	cargo llvm-cov
