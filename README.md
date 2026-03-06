# Predictive Coding in Rust

This is (for now) a basic implementation of predictive coding in Rust. The underlying algorithm is very interesting, and I've added some papers to this repo.

I have, at the time of writing, got a model architecture that converges on a roughly accurate, low confidence MNIST classifier. The code is single-threaded, CPU-only right now (I want to use the `rust-cuda` ecosystem to add GPU support) and for relatively small models (~900 node, ~100,000 connections), I'm getting ~14ms per training sample. This works out to about 14 minutes to train on the full MNIST dataset of 60k images. Slow, but alright for a first pass. Critically, the implementation does converge on valid model parameters. From here, the focus will be on optimisation.

# TODO

- [x] First correct implementation of the PC architecture
- [x] Make a ModelConfig struct to help in building models
- [x] Model evaluation binary (accuracy, average confidence, speed)
- [x] Multithreading
  - [ ] Refactor benchmark to benchmark the multithreaded version of the training too
- [ ] GPU processing
- [ ] Flesh out train binary
  - [x] allow user to resume from stored model
	- [x] store the model hyperparameters in a json file in the output directory
	- [x] allow hyperparameters to be set by the user (pass in JSON)
- [ ] Gain confidence in my implementation
	- [x] Figure out the model params for a mnist classifier
	- [ ] unit tests?
	- [ ] integration tests
- [ ] more advanced training strategies
  - [x] mini-batch sampling (optimise B samples in parallel, then for weight updates compute the mean $\Delta\bf{W}$ for the batches. 5.2. of 2506.06332)
- [ ] Benchmark tracking
	- [x] Basic benchmarking
	- [ ] Host a runner on my own known hardware (polyhymnia?) and dispatch a workflow to it on pushes to main.
	- [ ] Show a graph or something in github pages?
- [ ] Plotting model training (make a separate binary that monitors test output)
	- [ ] `egui`? `plotters`?
- [ ] Visualiser for model state - a graph with connections and nodes
	- Nodes could be two circles, one red one blue, each scaled to activation
