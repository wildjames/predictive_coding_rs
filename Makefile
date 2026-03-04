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

eval: build
	cargo run --release --bin eval $(MODEL)

infer: build
	cargo run --release --bin infer $(MODEL)

gen: build
	cargo run --release --bin generate $(MODEL) $(NUMBER)

bench: build
	cargo run --release --bin bench
