# benchmarks/Makefile — Build, install rfastloess on system R, and run benchmarks.
#
# Targets:
#   install        (default) Build rfastloess and install to system R
#   build          Build the rfastloess tarball only
#   bench          Run both --serial and --parallel benchmarks
#   bench-serial   Run --serial benchmark only
#   bench-parallel Run --parallel benchmark only
#   clean          Remove build artefacts

PYTHON ?= python

REPO_ROOT := $(abspath ..)
R_DIR     := $(REPO_ROOT)/bindings/r
DEV_DIR   := $(REPO_ROOT)/dev

R_PKG_NAME    := rfastloess
R_PKG_VERSION  = $(shell grep "^Version:" $(R_DIR)/DESCRIPTION | sed 's/Version: //')
R_PKG_TARBALL  = $(R_PKG_NAME)_$(R_PKG_VERSION).tar.gz

R_CARGO_TARGET :=
R_CROSS_ENV    :=
ifeq ($(OS),Windows_NT)
	R_CARGO_TARGET := --target x86_64-pc-windows-gnu
	R_CROSS_ENV    := PATH="/c/rtools45/x86_64-w64-mingw32.static.posix/bin:$$PATH"
endif

.PHONY: all install build bench-r bench-rfastloess-serial bench-rfastloess-parallel compare clean

all: install

# ==============================================================================
# Install rfastloess to system R
# ==============================================================================
install: build
	@echo "Installing $(R_PKG_NAME) to system R..."
	@R CMD INSTALL $(R_DIR)/$(R_PKG_TARBALL)
	@echo "$(R_PKG_NAME) installed!"

# ==============================================================================
# Build the rfastloess tarball
# ==============================================================================
build:
	@echo "============================================================"
	@echo "1. Syncing version..."
	@echo "============================================================"
	@VERSION=$$(grep "^version =" $(R_DIR)/src/Cargo.toml | head -n1 | sed 's/version = "\(.*\)"/\1/'); \
	 sed -i.bak "s/^Version: .*/Version: $$VERSION/" $(R_DIR)/DESCRIPTION && rm -f $(R_DIR)/DESCRIPTION.bak; \
	 echo "  Version: $$VERSION"

	@echo "============================================================"
	@echo "2. Patching Cargo.toml for isolated build..."
	@echo "============================================================"
	@if [ -f $(R_DIR)/src/Cargo.toml.orig ]; then mv $(R_DIR)/src/Cargo.toml.orig $(R_DIR)/src/Cargo.toml; fi
	@cp $(R_DIR)/src/Cargo.toml $(R_DIR)/src/Cargo.toml.orig
	@sed -i.bak '/^\[workspace\]/d'                              $(R_DIR)/src/Cargo.toml; \
	 sed -i.bak '/^\[patch\.crates-io\]/d'                       $(R_DIR)/src/Cargo.toml; \
	 sed -i.bak '/^loess-rs = { path = "vendor\/loess-rs" }/d'   $(R_DIR)/src/Cargo.toml; \
	 rm -f $(R_DIR)/src/Cargo.toml.bak
	@mkdir -p $(R_DIR)/src/.cargo && cp $(R_DIR)/src/cargo-config.toml $(R_DIR)/src/.cargo/config.toml

	@echo "============================================================"
	@echo "3. Installing Python build tools..."
	@echo "============================================================"
	@$(PYTHON) -m pip install -q tomli tomli_w >/dev/null 2>&1 || true

	@echo "============================================================"
	@echo "4. Vendoring Cargo dependencies..."
	@echo "============================================================"
	@$(PYTHON) $(DEV_DIR)/prepare_cargo.py clean $(R_DIR)/src/Cargo.toml -q
	@rm -rf $(R_DIR)/src/vendor $(R_DIR)/src/vendor.tar.xz
	@mkdir -p $(R_DIR)/src/vendor
	@cp -rL $(REPO_ROOT)/crates/fastLoess $(R_DIR)/src/vendor/
	@cp -rL $(REPO_ROOT)/crates/loess-rs  $(R_DIR)/src/vendor/
	@rm -rf $(R_DIR)/src/vendor/fastLoess/target $(R_DIR)/src/vendor/loess-rs/target
	@rm -f  $(R_DIR)/src/vendor/fastLoess/Cargo.lock $(R_DIR)/src/vendor/loess-rs/Cargo.lock
	@rm -f  $(R_DIR)/src/vendor/fastLoess/README.md  $(R_DIR)/src/vendor/fastLoess/CHANGELOG.md
	@rm -f  $(R_DIR)/src/vendor/loess-rs/README.md   $(R_DIR)/src/vendor/loess-rs/CHANGELOG.md
	@$(PYTHON) $(DEV_DIR)/patch_vendor_crates.py $(REPO_ROOT)/Cargo.toml $(R_DIR)/src/vendor -q
	@echo '{"files":{},"package":null}' > $(R_DIR)/src/vendor/loess-rs/.cargo-checksum.json
	@echo '{"files":{},"package":null}' > $(R_DIR)/src/vendor/fastLoess/.cargo-checksum.json
	@$(PYTHON) $(DEV_DIR)/prepare_cargo.py isolate $(R_DIR)/src/Cargo.toml -q
	@(cd $(R_DIR)/src && cargo vendor -q --no-delete vendor)
	@$(PYTHON) $(DEV_DIR)/clean_checksums.py -q $(R_DIR)/src/vendor
	@echo "  Creating vendor.tar.xz..."
	@(cd $(R_DIR)/src && \
	  (tar --sort=name --mtime='1970-01-01 00:00:00Z' --owner=0 --group=0 --numeric-owner \
	       --xz -cf vendor.tar.xz vendor Cargo.lock 2>/dev/null \
	   || tar --xz -cf vendor.tar.xz vendor Cargo.lock))
	@rm -rf $(R_DIR)/src/vendor
	@(cd $(R_DIR)/src && tar -xf vendor.tar.xz) \
	 && rm -f $(R_DIR)/src/vendor/*/CITATION.cff $(R_DIR)/src/vendor/*/CITATION

	@echo "============================================================"
	@echo "5. Building Rust library..."
	@echo "============================================================"
	@(cd $(R_DIR)/src && $(R_CROSS_ENV) cargo build -q --release $(R_CARGO_TARGET) \
	  || (mv Cargo.toml.orig Cargo.toml && exit 1))
	@rm -rf $(R_DIR)/src/.cargo

	@echo "============================================================"
	@echo "6. Building R package tarball..."
	@echo "============================================================"
	@cd $(R_DIR) && R CMD build --no-build-vignettes --no-manual .
	@if [ -f $(R_DIR)/src/Cargo.toml.orig ]; then mv $(R_DIR)/src/Cargo.toml.orig $(R_DIR)/src/Cargo.toml; fi
	@echo "  Tarball: $(R_DIR)/$(R_PKG_TARBALL)"

# ==============================================================================
# Run benchmarks (rfastloess must be installed first)
# ==============================================================================
bench-rfastloess-serial:
	@echo "Running rfastloess benchmarks (serial)..."
	@Rscript rfastloess.R --serial

bench-rfastloess-parallel:
	@echo "Running rfastloess benchmarks (parallel)..."
	@Rscript rfastloess.R --parallel

bench-r:
	@echo "Running stats::loess benchmarks..."
	@Rscript stats_loess.R

# ==============================================================================
# Compare / plot
# ==============================================================================
compare:
	@echo "Generating benchmark comparison plot..."
	@$(PYTHON) compare.py

# ==============================================================================
# Clean
# ==============================================================================
clean:
	@echo "Cleaning benchmark build artefacts..."
	@if [ -f $(R_DIR)/src/Cargo.toml.orig ]; then mv $(R_DIR)/src/Cargo.toml.orig $(R_DIR)/src/Cargo.toml; fi
	@rm -rf $(R_DIR)/src/vendor $(R_DIR)/src/vendor.tar.xz $(R_DIR)/src/.cargo
	@rm -f  $(R_DIR)/$(R_PKG_NAME)_*.tar.gz
	@(cd $(R_DIR)/src && cargo clean -q 2>/dev/null || true)
	@echo "Clean complete!"
