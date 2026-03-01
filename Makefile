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

test-liveplot: build
	WGPU_BACKEND=gl WINIT_UNIX_BACKEND=x11 LIBGL_ALWAYS_SOFTWARE=1 cargo run --bin test-liveplot
