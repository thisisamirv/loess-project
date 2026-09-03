# ==============================================================================
# Configuration
# ==============================================================================
FEATURE_SET ?= all
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

ifeq ($(OS),Windows_NT)
	PYTHON ?= python
else
	PYTHON ?= python3
endif
PYO3_PYTHON ?= $(PYTHON)
NODE ?= node

TEMP ?= /tmp
ifeq ($(OS),Windows_NT)
	TEMP := /tmp
endif

# loess-rs crate
LOESS_PKG := loess-rs
LOESS_DIR := crates/loess-rs
LOESS_FEATURES := std dev

# fastLoess crate
FASTLOESS_PKG := fastLoess
FASTLOESS_DIR := crates/fastLoess
FASTLOESS_FEATURES := dev

# Python bindings
PY_PKG := fastLoess-py
PY_DIR := bindings/python
PY_VENV := .venv

ifeq ($(OS),Windows_NT)
	PY_ACTIVATE    := $(PY_VENV)/Scripts/activate
	PY_VENV_PYTHON := $(PY_VENV)/Scripts/python.exe
else
	PY_ACTIVATE    := $(PY_VENV)/bin/activate
	PY_VENV_PYTHON := $(PY_VENV)/bin/python
endif

# R bindings
R_PKG_NAME := rfastloess
R_DIR := bindings/r
R_LIB_DIR := $(R_DIR)/.r-lib

# Julia bindings
JL_PKG := fastloess-jl
JL_DIR := bindings/julia

ifeq ($(HOST_PLATFORM),windows)
	JL_SHARED_LIB := target/release/fastloess_jl.dll
else ifeq ($(HOST_PLATFORM),macos)
	JL_SHARED_LIB := target/release/libfastloess_jl.dylib
else
	JL_SHARED_LIB := target/release/libfastloess_jl.so
endif

# Node.js bindings
NODE_PKG := fastloess-node
NODE_DIR := bindings/nodejs

# WebAssembly bindings
WASM_PKG := fastloess-wasm
WASM_DIR := bindings/wasm

# C++ bindings
CPP_PKG := fastloess-cpp
CPP_DIR := bindings/cpp
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

# ==============================================================================
# loess-rs crate
# ==============================================================================
loess-rs:
	@"$(MAKE)" -f crates/loess-rs/Makefile FEATURE_SET="$(FEATURE_SET)"

loess-rs-dev:
	@"$(MAKE)" -f crates/loess-rs/Makefile dev FEATURE_SET="$(FEATURE_SET)"

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

fastLoess-dev:
	@"$(MAKE)" -f crates/fastLoess/Makefile dev FEATURE_SET="$(FEATURE_SET)"

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

python-dev:
	@"$(MAKE)" -f bindings/python/Makefile dev \
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

r-dev:
	@"$(MAKE)" -f bindings/r/Makefile dev

r-coverage:
	@"$(MAKE)" -f bindings/r/Makefile coverage

r-clean:
	@"$(MAKE)" -f bindings/r/Makefile clean PYTHON="$(PYTHON)"

# ==============================================================================
# Julia bindings
# ==============================================================================
julia:
	@"$(MAKE)" -f bindings/julia/Makefile PYTHON="$(PYTHON)"

julia-dev:
	@"$(MAKE)" -f bindings/julia/Makefile dev PYTHON="$(PYTHON)"

julia-clean:
	@"$(MAKE)" -f bindings/julia/Makefile clean

# ==============================================================================
# Node.js bindings
# ==============================================================================
nodejs:
	@"$(MAKE)" -f bindings/nodejs/Makefile

nodejs-dev:
	@"$(MAKE)" -f bindings/nodejs/Makefile dev

nodejs-clean:
	@"$(MAKE)" -f bindings/nodejs/Makefile clean

# ==============================================================================
# WebAssembly bindings
# ==============================================================================
wasm:
	@"$(MAKE)" -f bindings/wasm/Makefile

wasm-dev:
	@"$(MAKE)" -f bindings/wasm/Makefile dev

wasm-clean:
	@"$(MAKE)" -f bindings/wasm/Makefile clean

# ==============================================================================
# C++ bindings
# ==============================================================================
cpp:
	@"$(MAKE)" -f bindings/cpp/Makefile

cpp-dev:
	@"$(MAKE)" -f bindings/cpp/Makefile dev

cpp-clean:
	@"$(MAKE)" -f bindings/cpp/Makefile clean

# ==============================================================================
# Go bindings
# ==============================================================================
go:
	@"$(MAKE)" -f bindings/go/Makefile

go-dev:
	@"$(MAKE)" -f bindings/go/Makefile dev

go-clean:
	@"$(MAKE)" -f bindings/go/Makefile clean

# ==============================================================================
# Java bindings
# ==============================================================================
java:
	@"$(MAKE)" -f bindings/java/Makefile

java-dev:
	@"$(MAKE)" -f bindings/java/Makefile dev

java-clean:
	@"$(MAKE)" -f bindings/java/Makefile clean

# ==============================================================================
# Development checks
# ==============================================================================
check-msrv:
	@echo "Checking MSRV..."
	@$(PYTHON) dev/check_msrv.py

# ==============================================================================
# Documentation
# ==============================================================================
docs-test:
	@echo "Running doc snippet tests..."
	@if [ -f "$(PY_VENV_PYTHON)" ]; then \
		$(PY_VENV_PYTHON) dev/verify_snippets.py --timeout 120; \
	else \
		$(PYTHON) dev/verify_snippets.py --timeout 120; \
	fi

# ==============================================================================
# All targets
# ==============================================================================
all: loess-rs fastLoess python r julia nodejs wasm cpp go java check-msrv
	@echo "All checks completed successfully!"

all-dev: loess-rs-dev fastLoess-dev python-dev r-dev julia-dev nodejs-dev wasm-dev cpp-dev go-dev java-dev check-msrv
	@echo "All dev checks completed successfully!"

all-coverage: loess-rs-coverage fastLoess-coverage python-coverage r-coverage
	@echo "All coverage completed!"

all-clean: r-clean loess-rs-clean fastLoess-clean python-clean julia-clean nodejs-clean wasm-clean cpp-clean
	@echo "Cleaning project root..."
	@cargo clean
	@$(PYTHON) dev/kill_locked_venv.py $(PY_VENV)
	@git clean -fdX .
	@echo "All clean completed!"

.PHONY: loess-rs loess-rs-dev loess-rs-coverage loess-rs-clean fastLoess fastLoess-dev fastLoess-coverage fastLoess-clean python python-dev python-coverage python-clean r r-dev r-coverage r-clean julia julia-dev julia-clean julia-update-commit nodejs nodejs-dev nodejs-clean wasm wasm-dev wasm-clean cpp cpp-dev cpp-clean go go-dev go-clean java java-dev java-clean check-msrv docs-test all all-dev all-coverage all-clean ensure-llvm-cov