# ==============================================================================
# Configuration
# ==============================================================================
FEATURE_SET ?= all

# Make shell commands fail on error
.SHELLFLAGS := -ec

# Python interpreter
PYTHON ?= python3

# loess crate
LOESS_PKG := loess-rs
LOESS_DIR := crates/loess-rs
LOESS_FEATURES := std dev
LOESS_EXAMPLES := batch_smoothing online_smoothing streaming_smoothing

# fastLoess crate
FASTLOESS_PKG := fastLoess
FASTLOESS_DIR := crates/fastLoess
FASTLOESS_FEATURES := cpu gpu dev
FASTLOESS_EXAMPLES := fast_batch_smoothing fast_online_smoothing fast_streaming_smoothing

# Python bindings
PY_PKG := fastLoess-py
PY_DIR := bindings/python
PY_VENV := .venv
PY_TEST_DIR := tests/python

# R bindings
R_PKG_NAME := rfastloess
R_PKG_VERSION = $(shell grep "^Version:" bindings/r/DESCRIPTION | sed 's/Version: //')
R_PKG_TARBALL = $(R_PKG_NAME)_$(R_PKG_VERSION).tar.gz
R_DIR := bindings/r

# Julia bindings
JL_PKG := fastloess-jl
JL_DIR := bindings/julia
JL_TEST_DIR := tests/julia

# Node.js bindings
NODE_PKG := fastloess-node
NODE_DIR := bindings/nodejs
NODE_TEST_DIR := tests/nodejs

# WebAssembly bindings
WASM_PKG := fastloess-wasm
WASM_DIR := bindings/wasm
WASM_TEST_DIR := tests/wasm

# C++ bindings
CPP_PKG := fastloess-cpp
CPP_DIR := bindings/cpp

# Examples directory
EXAMPLES_DIR := examples

# Documentation
DOCS_VENV := docs-venv

# ==============================================================================
# loess crate
# ==============================================================================
ISOLATE ?= true
all: ISOLATE := false

loess:
	@if [ "$(ISOLATE)" = "true" ]; then \
		$(PYTHON) dev/isolate_cargo.py crates/loess-rs -- $(MAKE) _loess_impl; \
	else \
		$(MAKE) _loess_impl; \
	fi

_loess_impl:
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

loess-coverage:
	@echo "Running $(LOESS_PKG) coverage..."
	@cd $(LOESS_DIR) && LLVM_COV=llvm-cov LLVM_PROFDATA=llvm-profdata cargo llvm-cov --all-targets --all-features

loess-clean:
	@echo "Cleaning $(LOESS_PKG) crate..."
	@cargo clean -p $(LOESS_PKG)
	@rm -rf $(LOESS_DIR)/coverage_html
	@rm -rf $(LOESS_DIR)/Cargo.lock
	@rm -rf $(LOESS_DIR)/benchmarks
	@rm -rf $(LOESS_DIR)/validation
	@echo "$(LOESS_PKG) clean complete!"

# ==============================================================================
# fastLoess crate
# ==============================================================================
fastLoess:
	@if [ "$(ISOLATE)" = "true" ]; then \
		$(PYTHON) dev/isolate_cargo.py crates/fastLoess -- $(MAKE) _fastLoess_impl; \
	else \
		$(MAKE) _fastLoess_impl; \
	fi

_fastLoess_impl:
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
	@cargo test -q -p loess-project-tests --test $(FASTLOESS_PKG) --no-default-features
	@for feature in $(FASTLOESS_FEATURES); do \
		echo "Testing ($$feature)..."; \
		if [ "$$feature" = "gpu" ]; then \
			cargo test -q -p loess-project-tests --test $(FASTLOESS_PKG) --features $$feature -- --test-threads=1 || exit 1; \
		else \
			cargo test -q -p loess-project-tests --test $(FASTLOESS_PKG) --features $$feature || exit 1; \
		fi; \
	done
	@echo "=============================================================================="
	@echo "All $(FASTLOESS_PKG) crate checks completed successfully!"

fastLoess-coverage:
	@echo "Running $(FASTLOESS_PKG) coverage..."
	@cd $(FASTLOESS_DIR) && LLVM_COV=llvm-cov LLVM_PROFDATA=llvm-profdata cargo llvm-cov --all-targets --all-features

fastLoess-clean:
	@echo "Cleaning $(FASTLOESS_PKG) crate..."
	@cargo clean -p $(FASTLOESS_PKG)
	@rm -rf $(FASTLOESS_DIR)/coverage_html
	@rm -rf $(FASTLOESS_DIR)/benchmarks
	@rm -rf $(FASTLOESS_DIR)/validation
	@echo "$(FASTLOESS_PKG) clean complete!"

# ==============================================================================
# Python bindings
# ==============================================================================
python:
	@if [ "$(ISOLATE)" = "true" ]; then \
		$(PYTHON) dev/isolate_cargo.py bindings/python -- $(MAKE) _python_impl; \
	else \
		$(MAKE) _python_impl; \
	fi

_python_impl:
	@echo "Running $(PY_PKG) checks..."
	@echo "=============================================================================="
	@echo "1. Formatting..."
	@echo "=============================================================================="
	@cargo fmt -p $(PY_PKG) -- --check
	@ruff format $(PY_DIR)/python/ $(PY_TEST_DIR)/ examples/python/
	@echo "=============================================================================="
	@echo "2. Linting..."
	@echo "=============================================================================="
	@VIRTUAL_ENV= PYO3_PYTHON=python3 cargo clippy -q -p $(PY_PKG) --all-targets -- -D warnings
	@ruff check $(PY_DIR)/python/ $(PY_TEST_DIR)/ examples/python/
	@echo "=============================================================================="
	@echo "3. Environment Setup..."
	@echo "=============================================================================="
	@if [ ! -d "$(PY_VENV)" ]; then python3 -m venv $(PY_VENV); fi
	@. $(PY_VENV)/bin/activate && pip install pytest numpy maturin
	@echo "=============================================================================="
	@echo "4. Building..."
	@echo "=============================================================================="
	@. $(PY_VENV)/bin/activate && cd $(PY_DIR) && maturin develop -q
	@echo "=============================================================================="
	@echo "5. Testing..."
	@echo "=============================================================================="
	@VIRTUAL_ENV= PYO3_PYTHON=python3 cargo test -q -p $(PY_PKG)
	@. $(PY_VENV)/bin/activate && python -m pytest $(PY_TEST_DIR) -q
	@echo "$(PY_PKG) checks completed successfully!"

python-coverage:
	@echo "Running $(PY_PKG) coverage..."
	@cd $(PY_DIR) && LLVM_COV=llvm-cov LLVM_PROFDATA=llvm-profdata cargo llvm-cov -p $(PY_PKG) --all-targets

python-clean:
	@echo "Cleaning $(PY_PKG)..."
	@cargo clean -p $(PY_PKG)
	@rm -rf $(PY_DIR)/coverage_html
	@rm -rf $(PY_DIR)/benchmarks
	@rm -rf $(PY_DIR)/validation
	@rm -rf $(PY_DIR)/.benchmarks
	@rm -rf $(PY_DIR)/target/wheels
	@rm -rf $(PY_DIR)/.pytest_cache
	@rm -rf $(PY_DIR)/__pycache__
	@rm -rf examples/python/plots/
	@rm -rf $(PY_DIR)/python/fastloess/__pycache__
	@rm -rf $(PY_DIR)/python/fastloess/*so
	@rm -rf $(PY_TEST_DIR)/__pycache__
	@rm -rf $(PY_DIR)/*.egg-info
	@rm -rf $(PY_DIR)/.ruff_cache
	@rm -rf $(PY_DIR)/*.so
	@rm -rf $(PY_DIR)/Cargo.lock
	@echo "$(PY_PKG) clean complete!"

# ==============================================================================
# R bindings
# ==============================================================================
r:
	@if [ "$(ISOLATE)" = "true" ]; then \
		$(PYTHON) dev/isolate_cargo.py bindings/r -- $(MAKE) _r_impl; \
	else \
		$(MAKE) _r_impl; \
	fi

_r_impl:
	@echo "Running $(R_PKG_NAME) checks..."
	@# Sync version from Cargo.toml to DESCRIPTION
	@VERSION=$$(grep "^version =" $(R_DIR)/src/Cargo.toml | head -n1 | sed 's/version = "\(.*\)"/\1/'); \
	sed -i "s/^Version: .*/Version: $$VERSION/" $(R_DIR)/DESCRIPTION; \
	echo "Synced DESCRIPTION version to $$VERSION"
	@if [ -f $(R_DIR)/src/Cargo.toml.orig ]; then \
		mv $(R_DIR)/src/Cargo.toml.orig $(R_DIR)/src/Cargo.toml; \
	elif [ ! -f $(R_DIR)/src/Cargo.toml ] && [ -f $(R_DIR)/src/Cargo.toml.test ]; then \
		cp $(R_DIR)/src/Cargo.toml.test $(R_DIR)/src/Cargo.toml; \
	fi
	@echo "=============================================================================="
	@echo "1. Patching Cargo.toml for isolated build..."
	@echo "=============================================================================="
	@cp $(R_DIR)/src/Cargo.toml $(R_DIR)/src/Cargo.toml.orig
	@# Extract values from root Cargo.toml [workspace.package] section and update R binding's Cargo.toml
	@# Metadata sync disabled by user request
	@# (Only cleaning up workspace/patch/vendor directives below)
	@sed -i '/^\[workspace\]/d' $(R_DIR)/src/Cargo.toml; \
	sed -i '/^\[patch\.crates-io\]/d' $(R_DIR)/src/Cargo.toml; \
	sed -i '/^\(loess-rs\|loess\) = { path = "vendor\/.*" }/d' $(R_DIR)/src/Cargo.toml; \
	rm -rf $(R_DIR)/*.Rcheck $(R_DIR)/*.BiocCheck $(R_DIR)/src/target $(R_DIR)/target $(R_DIR)/src/vendor; \
	echo "" >> $(R_DIR)/src/Cargo.toml
	@mkdir -p $(R_DIR)/src/.cargo && cp $(R_DIR)/src/cargo-config.toml $(R_DIR)/src/.cargo/config.toml
	@echo "Patched $(R_DIR)/src/Cargo.toml"
	@echo "=============================================================================="
	@echo "2. Installing R development packages..."
	@echo "=============================================================================="
	@Rscript -e "options(repos = c(CRAN = 'https://cloud.r-project.org')); suppressWarnings(install.packages(c('styler', 'prettycode', 'covr', 'BiocManager', 'toml', 'V8', 'visNetwork'), quiet = TRUE))" || true
	@Rscript -e "suppressWarnings(BiocManager::install('BiocCheck', quiet = TRUE, update = FALSE, ask = FALSE))" || true
	@Rscript -e "remotes::install_github ('ropensci-review-tools/pkgcheck')"
	@echo "R development packages installed!"
	@echo "=============================================================================="
	@echo "3. Vendoring..."
	@echo "=============================================================================="
	@echo "Updating and re-vendoring crates.io dependencies..."
	@# Step 1: Clean R package Cargo.toml for vendoring
	@dev/prepare_cargo.py clean $(R_DIR)/src/Cargo.toml -q
	@# Step 2: Prepare vendor directory with local crates
	@rm -rf $(R_DIR)/src/vendor $(R_DIR)/src/vendor.tar.xz
	@mkdir -p $(R_DIR)/src/vendor
	@cp -rL crates/fastLoess $(R_DIR)/src/vendor/
	@cp -rL crates/loess-rs $(R_DIR)/src/vendor/
	@rm -rf $(R_DIR)/src/vendor/fastLoess/target $(R_DIR)/src/vendor/loess-rs/target
	@rm -f $(R_DIR)/src/vendor/fastLoess/Cargo.lock $(R_DIR)/src/vendor/loess-rs/Cargo.lock
	@rm -f $(R_DIR)/src/vendor/fastLoess/README.md $(R_DIR)/src/vendor/fastLoess/CHANGELOG.md
	@rm -f $(R_DIR)/src/vendor/loess-rs/README.md $(R_DIR)/src/vendor/loess-rs/CHANGELOG.md
	@# Step 3: Patch local crates (remove workspace inheritance, strip GPU deps)
	@dev/patch_vendor_crates.py Cargo.toml $(R_DIR)/src/vendor -q
	@# Step 4: Create dummy checksum files for local crates
	@echo '{"files":{},"package":null}' > $(R_DIR)/src/vendor/loess-rs/.cargo-checksum.json
	@echo '{"files":{},"package":null}' > $(R_DIR)/src/vendor/fastLoess/.cargo-checksum.json
	@# Step 5: Add workspace isolation to R package
	@dev/prepare_cargo.py isolate $(R_DIR)/src/Cargo.toml -q
	@# Step 6: Vendor crates.io dependencies
	@(cd $(R_DIR)/src && cargo vendor -q --no-delete vendor)
	@# Step 7: Regenerate checksums after vendoring
	@dev/clean_checksums.py -q $(R_DIR)/src/vendor
	@echo "Creating vendor.tar.xz archive (including Cargo.lock)..."
	@(cd $(R_DIR)/src && tar --sort=name --mtime='1970-01-01 00:00:00Z' --owner=0 --group=0 --numeric-owner --xz --create --file=vendor.tar.xz vendor Cargo.lock)
	@rm -rf $(R_DIR)/src/vendor
	@echo "Vendor update complete. Archive: $(R_DIR)/src/vendor.tar.xz"
	@if [ -f $(R_DIR)/src/vendor.tar.xz ] && [ ! -d $(R_DIR)/src/vendor ]; then \
		echo "Extending vendor.tar.xz..."; \
		(cd $(R_DIR)/src && tar -xf vendor.tar.xz) && \
		find $(R_DIR)/src/vendor -name "CITATION.cff" -delete && \
		find $(R_DIR)/src/vendor -name "CITATION" -delete; \
	fi
	@echo "=============================================================================="
	@echo "4. Building..."
	@echo "=============================================================================="
	@(cd $(R_DIR)/src && cargo build -q --release || (mv Cargo.toml.orig Cargo.toml && exit 1))
	@rm -rf $(R_DIR)/src/.cargo
	@echo "=============================================================================="
	@echo "4a. Formatting..."
	@echo "=============================================================================="
	@cd $(R_DIR)/src && cargo fmt -q
	@cd $(R_DIR) && Rscript $(PWD)/dev/style_pkg.R || true
	@cd $(R_DIR)/src && cargo fmt -- --check || (echo "Run 'cargo fmt' to fix"; exit 1)
	@cd $(R_DIR)/src && cargo clippy -q -- -D warnings
	@Rscript -e "my_linters <- lintr::linters_with_defaults(indentation_linter = lintr::indentation_linter(indent = 4L), object_name_linter = NULL, commented_code_linter = NULL, object_usage_linter = NULL); lints <- c(lintr::lint_dir('$(R_DIR)/R', linters = my_linters), lintr::lint_dir('tests/r/testthat', linters = my_linters), lintr::lint_dir('examples/r', linters = my_linters)); print(lints); if (length(lints) > 0L) quit(status = 1)"
	@echo "=============================================================================="
	@echo "4b. Documentation..."
	@echo "=============================================================================="
	@rm -rf $(R_DIR)/*.Rcheck
	@cd $(R_DIR)/src && RUSTDOCFLAGS="-D warnings" cargo doc -q --no-deps
	@cd $(R_DIR) && Rscript -e "devtools::document(quiet = TRUE)"
	@cd $(R_DIR) && Rscript -e "devtools::build_vignettes(quiet = TRUE)" || true
	@rm -f $(R_DIR)/.gitignore
	@echo "=============================================================================="
	@echo "4c. Building..."
	@echo "=============================================================================="
	@cd $(R_DIR) && R CMD build .
	@echo "=============================================================================="
	@echo "5. Installing..."
	@echo "=============================================================================="
	@cd $(R_DIR) && R CMD INSTALL $(R_PKG_TARBALL) || true
	@cd $(R_DIR) && Rscript -e "devtools::install(quiet = TRUE)"
	@echo "=============================================================================="
	@echo "8. Testing..."
	@echo "=============================================================================="
	@cd $(R_DIR)/src && cargo test -q
	@Rscript -e "Sys.setenv(NOT_CRAN='true'); testthat::test_dir('tests/r/testthat', package = 'rfastloess')"
	@echo "=============================================================================="
	@echo "9. Submission checks..."
	@echo "=============================================================================="
	@cd $(R_DIR) && R_MAKEVARS_USER=$(PWD)/dev/Makevars.check R CMD check --as-cran $(R_PKG_TARBALL) || true
	@cd $(R_DIR) && Rscript -e "if (requireNamespace('BiocCheck', quietly=TRUE)) BiocCheck::BiocCheck('$(R_PKG_TARBALL)', new_package=FALSE)" || true
	@echo "Package size (Limit: 5MB):"
	@ls -lh $(R_DIR)/$(R_PKG_TARBALL) || true
	@Rscript -e "library(pkgstats); library(pkgcheck); pkgcheck(use_cache = FALSE)"
	@if [ -f $(R_DIR)/src/Cargo.toml.orig ]; then mv $(R_DIR)/src/Cargo.toml.orig $(R_DIR)/src/Cargo.toml; fi

	@echo "All $(R_PKG_NAME) checks completed successfully!"

r-coverage:
	@echo "Calculating $(R_PKG_NAME) coverage..."
	@cd $(R_DIR) && NOT_CRAN=true Rscript -e "\
	  if (!requireNamespace('covr', quietly = TRUE)) { message('covr missing'); quit(status=0) }; \
	  cov <- covr::package_coverage(); \
	  covr::zero_coverage(cov); \
	  print(cov)"

r-clean:
	@echo "Cleaning $(R_PKG_NAME)..."
	@if [ -d $(R_DIR)/src/target ]; then \
		rm -rf $(R_DIR)/src/target 2>/dev/null || \
		(command -v docker >/dev/null && docker run --rm -v "$(PWD)/$(R_DIR)":/pkg ghcr.io/r-universe-org/build-wasm:latest rm -rf /pkg/src/target) || \
		echo "Warning: Failed to clean src/target"; \
	fi
	@(cd $(R_DIR)/src && cargo clean 2>/dev/null || true)
	@rm -rf $(R_DIR)/src/vendor $(R_DIR)/target
	@rm -rf $(R_DIR)/$(R_PKG_NAME).Rcheck $(R_DIR)/..Rcheck $(R_DIR)/$(R_PKG_NAME).BiocCheck
	@rm -f $(R_DIR)/$(R_PKG_NAME)_*.tar.gz
	@rm -rf $(R_DIR)/src/*.o $(R_DIR)/src/*.so $(R_DIR)/src/*.dll $(R_DIR)/src/Cargo.toml.orig $(R_DIR)/src/Cargo.lock $(R_DIR)/Cargo.lock
	@rm -rf $(R_DIR)/doc $(R_DIR)/Meta $(R_DIR)/vignettes/*.html $(R_DIR)/README.html
	@find $(R_DIR) -name "*.Rout" -delete
	@Rscript -e "try(remove.packages('$(R_PKG_NAME)'), silent = TRUE)" || true
	@rm -rf $(R_DIR)/src/Makevars $(R_DIR)/rfastloess*.tgz
	@rm -rf $(R_DIR)/benchmarks $(R_DIR)/validation $(R_DIR)/docs
	@echo "$(R_PKG_NAME) clean complete!"

# ==============================================================================
# Julia bindings
# ==============================================================================
julia:
	@if [ "$(ISOLATE)" = "true" ]; then \
		$(PYTHON) dev/isolate_cargo.py bindings/julia -- $(MAKE) _julia_impl; \
	else \
		$(MAKE) _julia_impl; \
	fi

_julia_impl:
	@echo "Running $(JL_PKG) checks..."
	@echo "=============================================================================="
	@echo "0. Commit hash update..."
	@echo "=============================================================================="
	@git fetch origin main 2>/dev/null || true
	@COMMIT=$$(git rev-parse origin/main 2>/dev/null) && \
		sed -i "s/GitSource(\"[^\"]*\",\\s*\"[a-f0-9]\\+\")/GitSource(\"https:\\/\\/github.com\\/thisisamirv\\/loess-project.git\", \"$$COMMIT\")/" dev/build_tarballs_julia.jl && \
		echo "Commit: $$COMMIT"
	@echo "=============================================================================="
	@echo "1. Formatting..."
	@echo "=============================================================================="
	@cd $(JL_DIR) && cargo fmt -- --check
	@echo "Formatting complete!"
	@echo "=============================================================================="
	@echo "2. Linting..."
	@echo "=============================================================================="
	@cd $(JL_DIR) && cargo clippy -q --all-targets -- -D warnings
	@echo "Linting Julia files..."
	@julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.add("JuliaFormatter"); using JuliaFormatter; format(["bindings/julia/julia", "tests/julia", "examples/julia"], verbose=true, overwrite=false) ? exit(0) : exit(1)'
	@echo "=============================================================================="
	@echo "3. Building..."
	@echo "=============================================================================="
	@cd $(JL_DIR) && cargo build -q --release
	@cd $(JL_DIR) && RUSTDOCFLAGS="-D warnings" cargo doc -q --no-deps
	@echo "=============================================================================="
	@echo "4. Testing Rust library..."
	@echo "=============================================================================="
	@cd $(JL_DIR) && cargo test -q
	@echo "=============================================================================="
	@echo "5. Verifying library exports..."
	@echo "=============================================================================="
	@nm -D target/release/libfastloess_jl.so 2>/dev/null | grep -q jl_loess_new || \
		(echo "Error: jl_loess_new not exported"; exit 1)
	@nm -D target/release/libfastloess_jl.so 2>/dev/null | grep -q jl_streaming_loess_new || \
		(echo "Error: jl_streaming_loess_new not exported"; exit 1)
	@nm -D target/release/libfastloess_jl.so 2>/dev/null | grep -q jl_online_loess_new || \
		(echo "Error: jl_online_loess_new not exported"; exit 1)
	@nm -D target/release/libfastloess_jl.so 2>/dev/null | grep -q jl_loess_free_result || \
		(echo "Error: jl_loess_free_result not exported"; exit 1)
	@echo "All exports verified!"
	@echo "=============================================================================="
	@echo "5b. ABI size check (limit: 5 MB)..."
	@echo "=============================================================================="
	@[ $$(stat -c%s target/release/libfastloess_jl.so) -le 5242880 ] || \
		(echo "Error: libfastloess_jl.so exceeds 5 MB ABI size limit"; exit 1)
	@echo "ABI size OK."
	@echo "=============================================================================="
	@echo "6. Testing Julia bindings..."
	@echo "=============================================================================="
	@export FASTLOESS_LIB=$(PWD)/target/release/libfastloess_jl.so && \
	julia --project=$(JL_DIR)/julia -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.precompile()' && \
	julia --project=$(JL_DIR)/julia tests/julia/test_FastLOESS.jl
	@echo "=============================================================================="
	@echo "7. Aqua.jl package quality..."
	@echo "=============================================================================="
	@export FASTLOESS_LIB=$(PWD)/target/release/libfastloess_jl.so && \
	julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.develop(path="$(JL_DIR)/julia"); \
		Pkg.add("Aqua"); using Aqua, FastLOESS; \
		Aqua.test_all(FastLOESS; ambiguities=false, stale_deps=(ignore=[:Aqua],))'
	@echo "=============================================================================="
	@echo "8. JET.jl type-inference check..."
	@echo "=============================================================================="
	@export FASTLOESS_LIB=$(PWD)/target/release/libfastloess_jl.so && \
	julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.develop(path="$(JL_DIR)/julia"); \
		Pkg.add("JET"); using JET, FastLOESS; \
		report = JET.report_package(FastLOESS); \
		if length(JET.get_reports(report)) > 0; @error "JET found type errors"; exit(1); end'
	@echo "$(JL_PKG) checks completed successfully!"
	@echo ""
	@echo "To use in Julia:"
	@echo "  julia> using Pkg"
	@echo "  julia> Pkg.develop(path=\"$(JL_DIR)/julia\")"
	@echo "  julia> using FastLOESS"

julia-clean:
	@echo "Cleaning $(JL_PKG)..."
	@cargo clean -p $(JL_PKG)
	@rm -rf $(JL_DIR)/target
	@rm -rf $(JL_DIR)/julia/Manifest.toml
	@rm -rf $(JL_DIR)/Cargo.lock
	@echo "$(JL_PKG) clean complete!"

# ==============================================================================
# Node.js bindings
# ==============================================================================
nodejs:
	@if [ "$(ISOLATE)" = "true" ]; then \
		$(PYTHON) dev/isolate_cargo.py bindings/nodejs -- $(MAKE) _nodejs_impl; \
	else \
		$(MAKE) _nodejs_impl; \
	fi

_nodejs_impl:
	@echo "Running $(NODE_PKG) checks..."
	@echo "=============================================================================="
	@echo "1. Formatting..."
	@echo "=============================================================================="
	@cargo fmt -p $(NODE_PKG) -- --check
	@echo "=============================================================================="
	@echo "2. Linting & Building..."
	@echo "=============================================================================="
	@cargo clippy -q -p $(NODE_PKG) --all-targets -- -D warnings
	@echo "Linting Node.js files..."
	@cd $(NODE_DIR) && npm install
	@cd $(NODE_DIR) && npm audit
	@cd $(NODE_DIR) && npx -y license-checker --summary --failOn GPL
	@cd $(NODE_DIR) && npx -y depcheck --ignores="fastloess-*,oxlint"
	@cd $(NODE_DIR) && (npm outdated | grep -v "fastloess-" || true)
	@cd $(NODE_DIR) && npm ci --dry-run
	@cd $(NODE_DIR) && npx -y -p typescript tsc index.d.ts --noEmit --allowJs
	@npx oxlint $(NODE_DIR)/index.js tests/nodejs/test_fastloess.js examples/nodejs/*.js
	@cd $(NODE_DIR) && npm run build
	@echo "=============================================================================="
	@echo "3. Testing..."
	@echo "=============================================================================="
	@cd $(NODE_DIR) && npm test
	@echo "$(NODE_PKG) checks completed successfully!"

nodejs-clean:
	@echo "Cleaning $(NODE_PKG)..."
	@cargo clean -p $(NODE_PKG)
	@rm -rf $(NODE_DIR)/node_modules $(NODE_DIR)/fastloess.*.node
	@echo "$(NODE_PKG) clean complete!"

# ==============================================================================
# WebAssembly bindings
# ==============================================================================
wasm:
	@if [ "$(ISOLATE)" = "true" ]; then \
		$(PYTHON) dev/isolate_cargo.py bindings/wasm -- $(MAKE) _wasm_impl; \
	else \
		$(MAKE) _wasm_impl; \
	fi

_wasm_impl:
	@echo "Running $(WASM_PKG) checks..."
	@echo "=============================================================================="
	@echo "1. Formatting..."
	@echo "=============================================================================="
	@cargo fmt -p $(WASM_PKG) -- --check
	@echo "=============================================================================="
	@echo "2. Linting & Building..."
	@echo "=============================================================================="
	@cargo clippy -q -p $(WASM_PKG) --all-targets -- -D warnings
	@echo "Linting WASM JS files..."
	@cd $(WASM_DIR) && npm install -q
	@cd $(WASM_DIR) && npm audit
	@cd $(WASM_DIR) && npx -y license-checker --summary --failOn GPL
	@cd $(WASM_DIR) && npx -y depcheck --ignores="oxlint"
	@cd $(WASM_DIR) && (npm outdated | grep -v "fastloess-" || true)
	@cd $(WASM_DIR) && npm ci --dry-run
	@npx oxlint $(WASM_DIR)/src/*.js tests/wasm/*.js
	@cd $(WASM_DIR) && wasm-pack build --target nodejs --out-dir pkg
	@echo "Checking WASM size (Limit: 2MB)..."
	@[ $$(stat -c%s $(WASM_DIR)/pkg/fastloess_wasm_bg.wasm) -le 2097152 ] || (echo "Error: WASM size exceeded 2MB"; exit 1)
	@echo "Building for Web (Examples)..."
	@cd $(WASM_DIR) && wasm-pack build --target web --out-dir pkg-web
	@echo "=============================================================================="
	@echo "3. Testing..."
	@echo "=============================================================================="
	@cd $(WASM_DIR) && wasm-pack test --node
	@echo "Running JS tests..."
	@node --test tests/wasm/test_fastloess_wasm.js
	@echo "=============================================================================="
	@echo "$(WASM_PKG) checks completed successfully!"

wasm-clean:
	@echo "Cleaning $(WASM_PKG)..."
	@cargo clean -p $(WASM_PKG)
	@rm -rf $(WASM_DIR)/pkg $(WASM_DIR)/pkg-web $(WASM_DIR)/node_modules
	@echo "$(WASM_PKG) clean complete!"

# ==============================================================================
# C++ bindings
# ==============================================================================
cpp:
	@if [ "$(ISOLATE)" = "true" ]; then \
		$(PYTHON) dev/isolate_cargo.py bindings/cpp -- $(MAKE) _cpp_impl; \
	else \
		$(MAKE) _cpp_impl; \
	fi

_cpp_impl:
	@echo "Running $(CPP_PKG) checks..."
	@echo "=============================================================================="
	@echo "1. Formatting..."
	@echo "=============================================================================="
	@cargo fmt -p $(CPP_PKG) -- --check
	@echo "=============================================================================="
	@echo "2. Linting & Building..."
	@echo "=============================================================================="
	@cargo clippy -q -p $(CPP_PKG) --all-targets -- -D warnings
	@echo "Linting C++ files..."
	@clang-tidy bindings/cpp/include/fastloess.hpp tests/cpp/test_fastloess.cpp examples/cpp/*.cpp -- -I bindings/cpp/include -std=c++17 || (echo "C++ linting failed"; exit 1)
	@echo "Running cppcheck..."
	@cppcheck --error-exitcode=1 --enable=warning,performance,portability \
		--suppress=missingInclude --suppress=missingIncludeSystem \
		-I $(CPP_DIR)/include \
		$(CPP_DIR)/include/fastloess.hpp tests/cpp/test_fastloess.cpp examples/cpp/
	@cargo build -q -p $(CPP_PKG) --release
	@echo "C header generated at $(CPP_DIR)/include/fastloess.h"
	@echo "=============================================================================="
	@echo "2b. cbindgen idempotency check..."
	@echo "=============================================================================="
	@cbindgen --config $(CPP_DIR)/cbindgen.toml --crate $(CPP_PKG) --output /tmp/fastloess_new.h 2>/dev/null && \
		diff -q $(CPP_DIR)/include/fastloess.h /tmp/fastloess_new.h > /dev/null || \
		(echo "Error: fastloess.h is stale — run 'cargo build -p $(CPP_PKG) --release' to regenerate"; exit 1)
	@echo "cbindgen header is up-to-date."
	@echo "=============================================================================="
	@echo "2c. Symbol export verification..."
	@echo "=============================================================================="
	@nm -D target/release/libfastloess_cpp.so 2>/dev/null | grep -q cpp_loess_new || \
		(echo "Error: cpp_loess_new not exported"; exit 1)
	@nm -D target/release/libfastloess_cpp.so 2>/dev/null | grep -q cpp_streaming_new || \
		(echo "Error: cpp_streaming_new not exported"; exit 1)
	@nm -D target/release/libfastloess_cpp.so 2>/dev/null | grep -q cpp_online_new || \
		(echo "Error: cpp_online_new not exported"; exit 1)
	@echo "All C++ exports verified."
	@echo "=============================================================================="
	@echo "3. Testing..."
	@echo "=============================================================================="
	@rm -rf tests/cpp/build
	@mkdir -p tests/cpp/build
	@cd tests/cpp/build && cmake .. && make && ./test_fastloess_suite
	@echo "=============================================================================="
	@echo "3b. Valgrind memory check..."
	@echo "=============================================================================="
	@valgrind --leak-check=full --error-exitcode=1 --quiet \
		tests/cpp/build/test_fastloess_suite 2>&1 || \
		(echo "Error: Valgrind detected memory errors"; exit 1)
	@echo "Valgrind: no leaks."
	@echo "$(CPP_PKG) checks completed successfully!"

cpp-clean:
	@echo "Cleaning $(CPP_PKG)..."
	@cargo clean -p $(CPP_PKG)
	@rm -rf $(CPP_DIR)/include/fastloess.h $(CPP_DIR)/bin $(CPP_DIR)/build
	@rm -rf tests/cpp/build
	@echo "$(CPP_PKG) clean complete!"

# ==============================================================================
# Examples
# ==============================================================================
examples: examples-loess examples-fastLoess examples-python examples-r examples-julia examples-nodejs examples-cpp
	@echo "All examples completed successfully!"

examples-loess:
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
	@. $(PY_VENV)/bin/activate && pip install -q matplotlib
	@. $(PY_VENV)/bin/activate && python $(EXAMPLES_DIR)/python/batch_smoothing.py
	@. $(PY_VENV)/bin/activate && python $(EXAMPLES_DIR)/python/streaming_smoothing.py
	@. $(PY_VENV)/bin/activate && python $(EXAMPLES_DIR)/python/online_smoothing.py
	@echo "=============================================================================="

examples-r:
	@echo "Running $(R_PKG_NAME) examples..."
	@echo "=============================================================================="
	@Rscript $(EXAMPLES_DIR)/r/batch_smoothing.R
	@Rscript $(EXAMPLES_DIR)/r/streaming_smoothing.R
	@Rscript $(EXAMPLES_DIR)/r/online_smoothing.R
	@echo "=============================================================================="

examples-julia:
	@echo "Running $(JL_PKG) examples..."
	@echo "=============================================================================="
	@julia --project=$(JL_DIR)/julia $(EXAMPLES_DIR)/julia/batch_smoothing.jl
	@julia --project=$(JL_DIR)/julia $(EXAMPLES_DIR)/julia/streaming_smoothing.jl
	@julia --project=$(JL_DIR)/julia $(EXAMPLES_DIR)/julia/online_smoothing.jl
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
	@g++ -O3 $(EXAMPLES_DIR)/cpp/batch_smoothing.cpp -o $(CPP_DIR)/bin/batch_smoothing -I$(CPP_DIR)/include -Ltarget/release -lfastloess_cpp -lpthread -ldl -lm
	@g++ -O3 $(EXAMPLES_DIR)/cpp/streaming_smoothing.cpp -o $(CPP_DIR)/bin/streaming_smoothing -I$(CPP_DIR)/include -Ltarget/release -lfastloess_cpp -lpthread -ldl -lm
	@g++ -O3 $(EXAMPLES_DIR)/cpp/online_smoothing.cpp -o $(CPP_DIR)/bin/online_smoothing -I$(CPP_DIR)/include -Ltarget/release -lfastloess_cpp -lpthread -ldl -lm
	@LD_LIBRARY_PATH=target/release $(CPP_DIR)/bin/batch_smoothing
	@LD_LIBRARY_PATH=target/release $(CPP_DIR)/bin/streaming_smoothing
	@LD_LIBRARY_PATH=target/release $(CPP_DIR)/bin/online_smoothing
	@echo "=============================================================================="

# ==============================================================================
# Development checks
# ==============================================================================
check-msrv:
	@echo "Checking MSRV..."
	@python3 dev/check_msrv.py

# ==============================================================================
# Documentation
# ==============================================================================
docs:
	@echo "Building documentation..."
	@if [ ! -d "$(DOCS_VENV)" ]; then python3 -m venv $(DOCS_VENV); fi
	@. $(DOCS_VENV)/bin/activate && pip install -q -r docs/requirements.txt && mkdocs build

docs-serve:
	@echo "Starting documentation server..."
	@if [ ! -d "$(DOCS_VENV)" ]; then python3 -m venv $(DOCS_VENV); fi
	@. $(DOCS_VENV)/bin/activate && pip install -q -r docs/requirements.txt && mkdocs serve

docs-clean:
	@echo "Cleaning documentation build..."
	@rm -rf site/ $(DOCS_VENV)/
	@echo "Documentation clean complete!"

# ==============================================================================
# All targets
# ==============================================================================
all: loess fastLoess python r julia nodejs wasm cpp check-msrv
	@echo "All checks completed successfully!"

all-coverage: loess-coverage fastLoess-coverage python-coverage r-coverage
	@echo "All coverage completed!"

all-clean: r-clean loess-clean fastLoess-clean python-clean julia-clean nodejs-clean wasm-clean cpp-clean
	@echo "Cleaning project root..."
	@cargo clean
	@rm -rf target Cargo.lock .venv .ruff_cache .pytest_cache site docs-venv build bindings/python/.venv bindings/python/target crates/fastLoess/target crates/loess/target .vscode tests/.pytest_cache local_*.tar.gz
	@echo "All clean completed!"

.PHONY: loess loess-coverage loess-clean fastLoess fastLoess-coverage fastLoess-clean python python-coverage python-clean r r-coverage r-clean julia julia-clean julia-update-commit nodejs nodejs-clean wasm wasm-clean cpp cpp-clean check-msrv docs docs-serve docs-clean all all-coverage all-clean examples examples-loess examples-fastLoess examples-python examples-r examples-julia examples-nodejs examples-cpp
