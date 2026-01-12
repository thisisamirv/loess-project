# Run all local checks (formatting, linting, building, tests, docs)
check: fmt clippy build test doc examples
	@echo "All checks completed successfully!"

# Coverage (requires cargo-llvm-cov and llvm)
coverage:
	@echo "Running coverage report (text)..."
	@LLVM_COV=llvm-cov LLVM_PROFDATA=llvm-profdata cargo llvm-cov --features dev
	@echo "Coverage report complete!"

# ... (formatting, linter, build, test, doc sections remain unchanged)

# Examples
examples: example_batch example_online example_streaming

example_batch:
	@echo "Building examples..."
	@cargo run --example batch_smoothing
	@echo "Examples build complete!"

example_online:
	@echo "Building examples..."
	@cargo run --example online_smoothing
	@echo "Examples build complete!"

example_streaming:
	@echo "Building examples..."
	@cargo run --example streaming_smoothing
	@echo "Examples build complete!"

# Formatting
fmt:
	@echo "Checking code formatting..."
	@cargo fmt --all -- --check
	@echo "Formatting check complete!"

# Linter
clippy: clippy-cpu clippy-dev
	@echo "All clippy checks completed successfully!"

clippy-cpu:
	@echo "Running clippy (cpu)..."
	@cargo clippy --all-targets --features cpu -- -D warnings
	@echo "Clippy check complete!"

clippy-dev:
	@echo "Running clippy (dev)..."
	@cargo clippy --all-targets --features dev -- -D warnings
	@echo "Clippy check complete!"

# Build
build: build-cpu build-dev

build-cpu:
	@echo "Building crate (cpu)..."
	@cargo build --features cpu
	@echo "Build complete!"

build-dev:
	@echo "Building crate (dev)..."
	@cargo build --features dev
	@echo "Build complete!"

# Test
test: test-cpu

test-cpu:
	@echo "Running tests (cpu)..."
	@cargo test --features cpu
	@echo "Tests complete!"

# Documentation
doc: doc-cpu doc-dev
	@echo "Documentation build complete!"

doc-cpu:
	@echo "Building documentation (cpu)..."
	@RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features cpu
	@echo "Documentation build complete!"

doc-dev:
	@echo "Building documentation (dev)..."
	@RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features dev
	@echo "Documentation build complete!"


# Clean
clean:
	@echo "Cleaning..."
	@cargo clean
	@rm -rf Cargo.lock
	@rm -rf target
	@rm -rf coverage_html
	@rm -rf benchmarks
	@echo "Clean complete!"
