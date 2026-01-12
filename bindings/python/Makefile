# Variables
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
MATURIN := $(VENV)/bin/maturin
RUFF := $(VENV)/bin/ruff
PYTEST := $(VENV)/bin/pytest

# Run all local checks (formatting, linting, building, tests, docs)
check: venv fmt clippy build maturin test doc examples
	@echo "All checks completed successfully!"

# Environment Setup
venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv $(VENV); \
		$(PIP) install --upgrade pip; \
		$(PIP) install maturin ruff pytest sphinx sphinx_rtd_theme numpydoc numpy matplotlib; \
	fi

# Formatting
fmt: fmt-rust fmt-py fmt-fix fmt-fix-rust fmt-fix-py
	@echo "Formatting check complete!"

fmt-rust:
	@echo "Checking Rust code formatting..."
	@cargo fmt --all -- --check

fmt-py: venv
	@echo "Checking Python code formatting with ruff..."
	@$(RUFF) format --check fastloess/ tests/ || true

fmt-fix: fmt-fix-rust fmt-fix-py
	@echo "Formatting complete!"

fmt-fix-rust:
	@echo "Formatting Rust code..."
	@cargo fmt --all

fmt-fix-py: venv
	@echo "Formatting Python code with ruff..."
	@$(RUFF) format fastloess/ tests/

# Linter
clippy: clippy-default clippy-serial lint-py lint-py-fix

clippy-default:
	@echo "Running clippy (default / parallel)..."
	@cargo clippy --all-targets -- -D warnings

clippy-serial:
	@echo "Running clippy (serial / no parallel)..."
	@cargo clippy --all-targets --no-default-features -- -D warnings
	@echo "Clippy check complete!"

lint-py: venv
	@echo "Linting Python code with ruff..."
	@$(RUFF) check fastloess/ tests/
	@echo "Python lint complete!"

lint-py-fix: venv
	@echo "Fixing Python lint issues with ruff..."
	@$(RUFF) check --fix fastloess/ tests/
	@echo "Python lint fix complete!"

# Build
build: build-default build-serial

build-default:
	@echo "Building crate (default / parallel)..."
	@cargo build

build-serial:
	@echo "Building crate (serial / no parallel)..."
	@cargo build --no-default-features
	@echo "Build complete!"

# Maturin (Python package)
maturin: develop develop-release wheel wheel-release

develop: venv
	@echo "Building and installing Python package (development mode)..."
	@$(MATURIN) develop
	@echo "Development install complete!"

develop-release: venv
	@echo "Building and installing Python package (release mode)..."
	@$(MATURIN) develop --release
	@echo "Development install (release) complete!"

wheel: venv
	@echo "Building Python wheel..."
	@$(MATURIN) build
	@echo "Wheel build complete!"

wheel-release: venv
	@echo "Building Python wheel (release)..."
	@$(MATURIN) build --release
	@echo "Wheel build (release) complete!"

# Test
test: test-rust test-python

test-rust:
	@echo "Running Rust tests..."
	@cargo test
	@echo "Rust tests complete!"

test-python: venv
	@echo "Running Python tests..."
	@$(PYTEST) tests -v
	@echo "Python tests complete!"

# Documentation
doc: doc-default doc-serial doc-py
	@echo "Documentation build complete!"

doc-default:
	@echo "Building documentation (default / parallel)..."
	@RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

doc-serial:
	@echo "Building documentation (serial / no parallel)..."
	@RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --no-default-features

# Python (Sphinx) Documentation
doc-py: venv
	@echo "Building Sphinx documentation..."
	@$(PIP) install -r docs/requirements.txt
	@cd docs && ../$(VENV)/bin/sphinx-build -M html source build
	@echo "Sphinx documentation build complete! Output is in docs/build/html/index.html"

# Examples
examples: example_batch example_online example_streaming

example_batch: venv
	@echo "Running batch example..."
	@$(PYTHON) examples/batch_smoothing.py
	@echo "Batch example complete!"

example_online: venv
	@echo "Running online example..."
	@$(PYTHON) examples/online_smoothing.py
	@echo "Online example complete!"

example_streaming: venv
	@echo "Running streaming example..."
	@$(PYTHON) examples/streaming_smoothing.py
	@echo "Streaming example complete!"

# Installation (into venv)
install: venv
	@echo "Installing Python package..."
	@$(PIP) install target/wheels/fastloess-*.whl --force-reinstall
	@echo "Python package installed!"

# Clean
clean: clean-rust clean-python
	@echo "Clean complete!"

clean-rust:
	@echo "Performing cargo clean..."
	@cargo clean
	@rm -rf Cargo.lock
	@rm -rf target

clean-python:
	@echo "Cleaning Python build artifacts..."
	@rm -rf coverage_html
	@rm -rf benchmarks
	@rm -rf .benchmarks
	@rm -rf target/wheels
	@rm -rf .pytest_cache
	@rm -rf __pycache__
	@rm -rf fastloess/__pycache__
	@rm -rf tests/__pycache__
	@rm -rf *.egg-info
	@rm -rf .ruff_cache
	@rm -rf *.so
	@rm -rf .venv

# Help
help:
	@echo "Available targets (using .venv):"
	@echo "  check          - Run all check (creates venv if needed)"
	@echo "  venv           - Create virtual environment and install deps"
	@echo "  fmt            - Check formatting"
	@echo "  lint-py        - Run Python linter"
	@echo "  build          - Build Rust crate"
	@echo "  test           - Run all tests"
	@echo "  doc            - Build all docs"
	@echo "  examples       - Run examples"
	@echo "  clean          - Clean artifacts (including venv)"

