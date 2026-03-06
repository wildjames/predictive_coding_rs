.PHONY: grab-mnist build test clean train infer bench gen

grab-mnist:
	./scripts/grab_mnist.sh

build:
	cargo build --release

test:
	cargo test

clean:
	cargo clean

train: build
	cargo run --release --bin train

bench: build
	cargo run --release --bin bench
