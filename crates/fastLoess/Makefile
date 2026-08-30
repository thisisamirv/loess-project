THIS_MK := $(lastword $(MAKEFILE_LIST))

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

PYTHON ?= python3
PYO3_PYTHON ?= $(PYTHON)
NODE ?= node

TEMP ?= /tmp
ifeq ($(OS),Windows_NT)
	TEMP := /tmp
endif

FASTLOESS_PKG := fastLoess
FASTLOESS_DIR := crates/fastLoess
FASTLOESS_FEATURES := dev

.PHONY: default dev clean coverage ensure-llvm-cov

default:
	@cargo build -q -p $(FASTLOESS_PKG)

dev:
	@echo "Running $(FASTLOESS_PKG) crate checks..."
	@echo "=============================================================================="
	@echo "1. Formatting..."
	@echo "=============================================================================="
	@cargo fmt -p $(FASTLOESS_PKG) -- --check
	@echo "Formatting complete!"
	@echo "=============================================================================="
	@echo "2. Linting & Building..."
	@echo "=============================================================================="
	@for feature in $(FASTLOESS_FEATURES) no-default-features; do \
		if [ "$$feature" = "no-default-features" ]; then \
			echo "Checking $(FASTLOESS_PKG) (no-default-features)..."; \
			cargo clippy -q -p $(FASTLOESS_PKG) --all-targets --no-default-features -- -D warnings || exit 1; \
			cargo build -q -p $(FASTLOESS_PKG) --no-default-features || exit 1; \
			RUSTDOCFLAGS="-D warnings" cargo doc -q -p $(FASTLOESS_PKG) --no-deps --no-default-features || exit 1; \
		else \
			echo "Checking $(FASTLOESS_PKG) ($$feature)..."; \
			cargo clippy -q -p $(FASTLOESS_PKG) --all-targets --features $$feature -- -D warnings || exit 1; \
			cargo build -q -p $(FASTLOESS_PKG) --features $$feature || exit 1; \
			RUSTDOCFLAGS="-D warnings" cargo doc -q -p $(FASTLOESS_PKG) --no-deps --features $$feature || exit 1; \
		fi; \
	done
	@echo "=============================================================================="
	@echo "3. Testing..."
	@echo "=============================================================================="
	@echo "Testing (no-default-features)..."
	@cargo test -q -p $(FASTLOESS_PKG) --test $(FASTLOESS_PKG) --no-default-features
	@for feature in $(FASTLOESS_FEATURES); do \
		echo "Testing ($$feature)..."; \
		cargo test -q -p $(FASTLOESS_PKG) --test $(FASTLOESS_PKG) --features $$feature || exit 1; \
	done
	@echo "=============================================================================="
	@echo "=============================================================================="
	@echo "Doc snippet tests (rust)..."
	@$(PYTHON) dev/verify_snippets.py --lang rust --rust-docs $(FASTLOESS_DIR)/docs --timeout 120
	@echo "Adding Rust doc snippet outputs..."
	@$(PYTHON) dev/add-rust-outputs.py $(FASTLOESS_DIR)/docs
	@$(PYTHON) dev/update_changelogs.py
	@echo "All $(FASTLOESS_PKG) crate checks completed successfully!"

ensure-llvm-cov:
	@cargo llvm-cov --version > /dev/null 2>&1 || (echo "Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov && cargo llvm-cov install-llvm-tools)

coverage: ensure-llvm-cov
	@echo "Running $(FASTLOESS_PKG) coverage..."
	@LLVM_COV=llvm-cov LLVM_PROFDATA=llvm-profdata cargo llvm-cov -p $(FASTLOESS_PKG) --test $(FASTLOESS_PKG) --all-features \
		--ignore-filename-regex 'crates[/\\]loess-rs[/\\]|bindings[/\\]|benchmarks[/\\]|examples[/\\]|tests[/\\]'

clean:
	@echo "Cleaning $(FASTLOESS_PKG) crate..."
	@cargo clean -p $(FASTLOESS_PKG)
	@git clean -fdX -- $(FASTLOESS_DIR)
	@echo "$(FASTLOESS_PKG) clean complete!"