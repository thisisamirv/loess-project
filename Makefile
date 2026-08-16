# ==============================================================================
# Configuration
# ==============================================================================
FEATURE_SET   ?= all
RUN_GPU_TESTS ?= auto

# Make shell commands fail on error
.SHELLFLAGS := -ec

UNAME_S := $(shell uname -s)

ifeq ($(OS),Windows_NT)
	HOST_PLATFORM  := windows
	PATH_SEPARATOR := ;
	STAT_SIZE_CMD  := stat -c%s
	NPM := npm.cmd
	NPX := npx.cmd
else ifeq ($(UNAME_S),Darwin)
	HOST_PLATFORM  := macos
	PATH_SEPARATOR := :
	STAT_SIZE_CMD  := stat -f%z
	NPM := npm
	NPX := npx
else
	HOST_PLATFORM  := linux
	PATH_SEPARATOR := :
	STAT_SIZE_CMD  := stat -c%s
	NPM := npm
	NPX := npx
endif

PYTHON      ?= python3
PYO3_PYTHON ?= $(PYTHON)
NODE        ?= node

TEMP ?= /tmp
ifeq ($(OS),Windows_NT)
	TEMP := /tmp
endif

# loess-rs crate
LOESS_PKG      := loess-rs
LOESS_DIR      := crates/loess-rs
LOESS_FEATURES := std dev
LOESS_EXAMPLES := batch_smoothing online_smoothing streaming_smoothing

# fastLoess crate
FASTLOESS_PKG      := fastLoess
FASTLOESS_DIR      := crates/fastLoess
FASTLOESS_FEATURES := dev
FASTLOESS_EXAMPLES := fast_batch_smoothing fast_online_smoothing fast_streaming_smoothing

# Python bindings
PY_PKG      := fastLoess-py
PY_DIR      := bindings/python
PY_VENV     := .venv
PY_TEST_DIR := tests/python
ifeq ($(OS),Windows_NT)
	PY_ACTIVATE    := $(PY_VENV)/Scripts/activate
	PY_VENV_PYTHON := $(PY_VENV)/Scripts/python.exe
else
	PY_ACTIVATE    := $(PY_VENV)/bin/activate
	PY_VENV_PYTHON := $(PY_VENV)/bin/python
endif

# R bindings
R_PKG_NAME := rfastloess
R_DIR      := bindings/r
R_LIB_DIR  := $(R_DIR)/.r-lib

# Julia bindings
JL_PKG  := fastloess-jl
JL_DIR  := bindings/julia

# Julia native library paths (for examples)
ifeq ($(HOST_PLATFORM),windows)
	JL_SHARED_LIB := target/release/fastloess_jl.dll
else ifeq ($(HOST_PLATFORM),macos)
	JL_SHARED_LIB := target/release/libfastloess_jl.dylib
else
	JL_SHARED_LIB := target/release/libfastloess_jl.so
endif

# Node.js bindings
NODE_PKG  := fastloess-node
NODE_DIR  := bindings/nodejs

# WebAssembly bindings
WASM_PKG  := fastloess-wasm
WASM_DIR  := bindings/wasm

# C++ bindings
CPP_PKG         := fastloess-cpp
CPP_DIR         := bindings/cpp
CPP_CARGO_PROFILE := --profile release-c
CPP_LIBRARY_DIR := target/release-c

ifeq ($(OS),Windows_NT)
	_CPP_GCC_MACHINE := $(shell gcc -dumpmachine 2>/dev/null)
	ifneq ($(findstring mingw,$(_CPP_GCC_MACHINE)),)
		CPP_LIBRARY_DIR := target/x86_64-pc-windows-gnu/release-c
	else
		CPP_LIBRARY_DIR := target/x86_64-pc-windows-msvc/release-c
	endif
endif

ifeq ($(HOST_PLATFORM),windows)
	CPP_EXAMPLE_RUN_ENV := PATH="$(CPP_LIBRARY_DIR)$(PATH_SEPARATOR)$$PATH"
else ifeq ($(HOST_PLATFORM),macos)
	CPP_EXAMPLE_RUN_ENV := DYLD_LIBRARY_PATH=$(CPP_LIBRARY_DIR)
else
	CPP_EXAMPLE_RUN_ENV := LD_LIBRARY_PATH=$(CPP_LIBRARY_DIR)
endif

# Examples directory
EXAMPLES_DIR := examples

# Documentation
DOCS_VENV := docs-venv

# ==============================================================================
# loess-rs crate
# ==============================================================================
loess-rs:
	@"$(MAKE)" -f crates/loess-rs/Makefile FEATURE_SET="$(FEATURE_SET)"

loess-rs-coverage:
	@"$(MAKE)" -f crates/loess-rs/Makefile coverage

loess-rs-clean:
	@"$(MAKE)" -f crates/loess-rs/Makefile clean

ensure-llvm-cov:
	@cargo llvm-cov --version > /dev/null 2>&1 || (echo "Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov && cargo llvm-cov install-llvm-tools)

# ==============================================================================
# fastLoess crate
# ==============================================================================
fastLoess:
	@"$(MAKE)" -f crates/fastLoess/Makefile FEATURE_SET="$(FEATURE_SET)"

fastLoess-coverage:
	@"$(MAKE)" -f crates/fastLoess/Makefile coverage

fastLoess-clean:
	@"$(MAKE)" -f crates/fastLoess/Makefile clean

# ==============================================================================
# Python bindings
# ==============================================================================
python:
	@"$(MAKE)" -f bindings/python/Makefile \
		PYTHON="$(PYTHON)" PYO3_PYTHON="$(PYO3_PYTHON)"

python-coverage:
	@"$(MAKE)" -f bindings/python/Makefile coverage PYTHON="$(PYTHON)" PYO3_PYTHON="$(PYO3_PYTHON)"

python-clean:
	@"$(MAKE)" -f bindings/python/Makefile clean

# ==============================================================================
# R bindings
# ==============================================================================
r:
	@"$(MAKE)" -f bindings/r/Makefile

r-coverage:
	@"$(MAKE)" -f bindings/r/Makefile coverage

r-clean:
	@"$(MAKE)" -f bindings/r/Makefile clean PYTHON="$(PYTHON)"

# ==============================================================================
# Julia bindings
# ==============================================================================
julia:
	@"$(MAKE)" -f bindings/julia/Makefile PYTHON="$(PYTHON)"

julia-clean:
	@"$(MAKE)" -f bindings/julia/Makefile clean

# ==============================================================================
# Node.js bindings
# ==============================================================================
nodejs:
	@"$(MAKE)" -f bindings/nodejs/Makefile

nodejs-clean:
	@"$(MAKE)" -f bindings/nodejs/Makefile clean

# ==============================================================================
# WebAssembly bindings
# ==============================================================================
wasm:
	@"$(MAKE)" -f bindings/wasm/Makefile

wasm-clean:
	@"$(MAKE)" -f bindings/wasm/Makefile clean

# ==============================================================================
# C++ bindings
# ==============================================================================
cpp:
	@"$(MAKE)" -f bindings/cpp/Makefile

cpp-clean:
	@"$(MAKE)" -f bindings/cpp/Makefile clean

# ==============================================================================
# Examples
# ==============================================================================
examples: examples-loess-rs examples-fastLoess examples-python examples-r examples-julia examples-nodejs examples-cpp
	@echo "All examples completed successfully!"

examples-loess-rs:
	@echo "Running $(LOESS_PKG) examples..."
	@echo "=============================================================================="
	@echo "Running examples (no-default-features)..."
	@for example in $(LOESS_EXAMPLES); do \
		cargo run -q -p examples --example $$example --no-default-features || exit 1; \
	done
	@for feature in $(LOESS_FEATURES); do \
		echo "Running examples ($$feature)..."; \
		for example in $(LOESS_EXAMPLES); do \
			cargo run -q -p examples --example $$example --features $$feature || exit 1; \
		done; \
	done
	@echo "=============================================================================="

examples-fastLoess:
	@echo "Running $(FASTLOESS_PKG) examples..."
	@echo "=============================================================================="
	@echo "Running examples (no-default-features)..."
	@for example in $(FASTLOESS_EXAMPLES); do \
		cargo run -q -p examples --example $$example --no-default-features > /dev/null || exit 1; \
	done
	@for feature in $(FASTLOESS_FEATURES); do \
		echo "Running examples with feature: $$feature"; \
		for example in $(FASTLOESS_EXAMPLES); do \
			if [ "$$feature" = "dev" ]; then \
				cargo run -q -p examples --example $$example --features $$feature || exit 1; \
			else \
				cargo run -q -p examples --example $$example --features $$feature > /dev/null || exit 1; \
			fi; \
		done; \
	done
	@echo "=============================================================================="

examples-python:
	@echo "Running $(PY_PKG) examples..."
	@echo "=============================================================================="
	@. $(PY_ACTIVATE) && pip install -q matplotlib
	@. $(PY_ACTIVATE) && python $(EXAMPLES_DIR)/python/batch_smoothing.py
	@. $(PY_ACTIVATE) && python $(EXAMPLES_DIR)/python/streaming_smoothing.py
	@. $(PY_ACTIVATE) && python $(EXAMPLES_DIR)/python/online_smoothing.py
	@echo "=============================================================================="

examples-r:
	@echo "Running $(R_PKG_NAME) examples..."
	@echo "=============================================================================="
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript $(EXAMPLES_DIR)/r/batch_smoothing.R
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript $(EXAMPLES_DIR)/r/streaming_smoothing.R
	@R_LIBS_USER=$(CURDIR)/$(R_LIB_DIR) Rscript $(EXAMPLES_DIR)/r/online_smoothing.R
	@echo "=============================================================================="

examples-julia:
	@echo "Running $(JL_PKG) examples..."
	@echo "=============================================================================="
	@FASTLOESS_LIB=$(CURDIR)/$(JL_SHARED_LIB) julia --project=$(JL_DIR)/julia $(EXAMPLES_DIR)/julia/batch_smoothing.jl
	@FASTLOESS_LIB=$(CURDIR)/$(JL_SHARED_LIB) julia --project=$(JL_DIR)/julia $(EXAMPLES_DIR)/julia/streaming_smoothing.jl
	@FASTLOESS_LIB=$(CURDIR)/$(JL_SHARED_LIB) julia --project=$(JL_DIR)/julia $(EXAMPLES_DIR)/julia/online_smoothing.jl
	@echo "=============================================================================="

examples-nodejs:
	@echo "Running $(NODE_PKG) examples..."
	@echo "=============================================================================="
	@cd $(NODE_DIR) && node ../../$(EXAMPLES_DIR)/nodejs/batch_smoothing.js
	@cd $(NODE_DIR) && node ../../$(EXAMPLES_DIR)/nodejs/online_smoothing.js
	@cd $(NODE_DIR) && node ../../$(EXAMPLES_DIR)/nodejs/streaming_smoothing.js
	@echo "=============================================================================="

examples-cpp:
	@echo "Running $(CPP_PKG) examples..."
	@echo "=============================================================================="
	@mkdir -p $(CPP_DIR)/bin
	@g++ -O3 $(EXAMPLES_DIR)/cpp/batch_smoothing.cpp -o $(CPP_DIR)/bin/batch_smoothing -I$(CPP_DIR)/include -L$(CPP_LIBRARY_DIR) -lfastloess_cpp -lpthread -ldl -lm
	@g++ -O3 $(EXAMPLES_DIR)/cpp/streaming_smoothing.cpp -o $(CPP_DIR)/bin/streaming_smoothing -I$(CPP_DIR)/include -L$(CPP_LIBRARY_DIR) -lfastloess_cpp -lpthread -ldl -lm
	@g++ -O3 $(EXAMPLES_DIR)/cpp/online_smoothing.cpp -o $(CPP_DIR)/bin/online_smoothing -I$(CPP_DIR)/include -L$(CPP_LIBRARY_DIR) -lfastloess_cpp -lpthread -ldl -lm
	@$(CPP_EXAMPLE_RUN_ENV) $(CPP_DIR)/bin/batch_smoothing
	@$(CPP_EXAMPLE_RUN_ENV) $(CPP_DIR)/bin/streaming_smoothing
	@$(CPP_EXAMPLE_RUN_ENV) $(CPP_DIR)/bin/online_smoothing
	@echo "=============================================================================="

# ==============================================================================
# Development checks
# ==============================================================================
check-msrv:
	@echo "Checking MSRV..."
	@$(PYTHON) dev/check_msrv.py

# ==============================================================================
# Documentation
# ==============================================================================
docs:
	@echo "Building documentation..."
	@if [ ! -d "$(DOCS_VENV)" ]; then $(PYTHON) -m venv $(DOCS_VENV); fi
	@. $(DOCS_VENV)/$(if $(filter $(HOST_PLATFORM),windows),Scripts,bin)/activate && pip install -q -r docs/requirements.txt && mkdocs build --config-file docs/mkdocs.yml

docs-serve:
	@echo "Starting documentation server..."
	@if [ ! -d "$(DOCS_VENV)" ]; then $(PYTHON) -m venv $(DOCS_VENV); fi
	@. $(DOCS_VENV)/$(if $(filter $(HOST_PLATFORM),windows),Scripts,bin)/activate && pip install -q -r docs/requirements.txt && mkdocs serve --config-file docs/mkdocs.yml

docs-test:
	@echo "Running doc snippet tests..."
	@if [ -f "$(PY_VENV_PYTHON)" ]; then \
		$(PY_VENV_PYTHON) dev/verify_snippets.py --timeout 120; \
	else \
		$(PYTHON) dev/verify_snippets.py --timeout 120; \
	fi

docs-clean:
	@echo "Cleaning documentation build..."
	@rm -rf site/ $(DOCS_VENV)/
	@echo "Documentation clean complete!"

# ==============================================================================
# All targets
# ==============================================================================
all: loess-rs fastLoess python r julia nodejs wasm cpp check-msrv docs-test
	@echo "All checks completed successfully!"

all-coverage: loess-rs-coverage fastLoess-coverage python-coverage r-coverage
	@echo "All coverage completed!"

all-clean: r-clean loess-rs-clean fastLoess-clean python-clean julia-clean nodejs-clean wasm-clean cpp-clean
	@echo "Cleaning project root..."
	@cargo clean
	@rm -rf target Cargo.lock .venv .ruff_cache .pytest_cache site docs-venv build bindings/python/.venv bindings/python/target crates/fastLoess/target crates/loess-rs/target .vscode tests/.pytest_cache local_*.tar.gz bindings/r/.r-lib bindings/r/docs
	@rm -f Rplots.pdf .gitignore~ ..gitignore.un~
	@rm -rf r.Rcheck/
	@rm -f tests/r/testthat/Rplots.pdf
	@rm -rf examples/cpp/bin/
	@rm -f bindings/nodejs/fastloess.node
	@rm -f bindings/python/python/fastloess/*.pyd bindings/python/python/fastloess/*.pdb
	@rm -rf bindings/r/tests/
	@echo "All clean completed!"

.PHONY: loess-rs loess-rs-coverage loess-rs-clean fastLoess fastLoess-coverage fastLoess-clean python python-coverage python-clean r r-coverage r-clean julia julia-clean julia-update-commit nodejs nodejs-clean wasm wasm-clean cpp cpp-clean check-msrv docs docs-serve docs-test docs-clean all all-coverage all-clean examples examples-loess-rs examples-fastLoess examples-python examples-r examples-julia examples-nodejs examples-cpp ensure-llvm-cov
