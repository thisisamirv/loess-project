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

FEATURE_SET ?= all

LOESS_PKG      := loess-rs
LOESS_DIR      := crates/loess-rs
LOESS_FEATURES := std dev

.PHONY: default clean coverage ensure-llvm-cov

default:
	@echo "Running $(LOESS_PKG) crate checks..."
	@echo "=============================================================================="
	@echo "1. Formatting..."
	@echo "=============================================================================="
	@cargo fmt -p $(LOESS_PKG) -- --check
	@echo "Formatting complete!"
	@echo "=============================================================================="
	@echo "2. Linting & Building..."
	@echo "=============================================================================="
	@if [ "$(FEATURE_SET)" = "all" ]; then \
		echo "Checking $(LOESS_PKG) (std)..."; \
		cargo clippy -q -p $(LOESS_PKG) --all-targets -- -D warnings || exit 1; \
		cargo build -q -p $(LOESS_PKG) || exit 1; \
		RUSTDOCFLAGS="-D warnings" cargo doc -q -p $(LOESS_PKG) --no-deps || exit 1; \
		echo "Checking $(LOESS_PKG) (no-default-features)..."; \
		cargo clippy -q -p $(LOESS_PKG) --all-targets --no-default-features -- -D warnings || exit 1; \
		cargo build -q -p $(LOESS_PKG) --no-default-features || exit 1; \
		RUSTDOCFLAGS="-D warnings" cargo doc -q -p $(LOESS_PKG) --no-deps --no-default-features || exit 1; \
		for feature in $(LOESS_FEATURES); do \
			echo "Checking $(LOESS_PKG) ($$feature)..."; \
			cargo clippy -q -p $(LOESS_PKG) --all-targets --features $$feature -- -D warnings || exit 1; \
			cargo build -q -p $(LOESS_PKG) --features $$feature || exit 1; \
			RUSTDOCFLAGS="-D warnings" cargo doc -q -p $(LOESS_PKG) --no-deps --features $$feature || exit 1; \
		done; \
	else \
		cargo clippy -q -p $(LOESS_PKG) --all-targets --features $(FEATURE_SET) -- -D warnings || exit 1; \
		cargo build -q -p $(LOESS_PKG) --features $(FEATURE_SET) || exit 1; \
		RUSTDOCFLAGS="-D warnings" cargo doc -q -p $(LOESS_PKG) --no-deps --features $(FEATURE_SET) || exit 1; \
	fi
	@cargo clippy -p examples --examples --all-features -- -D warnings || exit 1
	@echo "=============================================================================="
	@echo "3. Testing..."
	@echo "=============================================================================="
	@echo "Testing (no-default-features)..."
	@cargo test -q -p loess-project-tests --test $(LOESS_PKG) --no-default-features
	@for feature in $(LOESS_FEATURES); do \
		echo "Testing ($$feature)..."; \
		cargo test -q -p loess-project-tests --test $(LOESS_PKG) --features $$feature || exit 1; \
	done
	@echo "=============================================================================="
	@echo "All $(LOESS_PKG) crate checks completed successfully!"

ensure-llvm-cov:
	@cargo llvm-cov --version > /dev/null 2>&1 || (echo "Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov && cargo llvm-cov install-llvm-tools)

coverage: ensure-llvm-cov
	@echo "Running $(LOESS_PKG) coverage..."
	@LLVM_COV=llvm-cov LLVM_PROFDATA=llvm-profdata cargo llvm-cov --workspace --test $(LOESS_PKG) --all-features \
		--ignore-filename-regex 'crates[/\\]fastLoess[/\\]|bindings[/\\]|benchmarks[/\\]|examples[/\\]|tests[/\\]'

clean:
	@echo "Cleaning $(LOESS_PKG) crate..."
	@cargo clean -p $(LOESS_PKG)
	@rm -rf $(LOESS_DIR)/coverage_html
	@rm -rf $(LOESS_DIR)/Cargo.lock
	@rm -rf $(LOESS_DIR)/benchmarks
	@rm -rf $(LOESS_DIR)/validation
	@echo "$(LOESS_PKG) clean complete!"
