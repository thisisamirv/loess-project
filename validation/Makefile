# validation/Makefile — Run validation and comparison for fastLoess vs R.
#
# Targets:
#   fastloess-visual    Build and run the visual Rust binary (writes output/fastLoess/)
#   fastloess-validate  Build and run the validate Rust binary (writes output/fastLoess/)
#   r-validate          Run R reference script (writes output/r/)
#   compare             Compare R and fastLoess outputs with compare.py
#   plot                Generate plots with plot.py

PYTHON ?= python
CARGO_DIR := fastLoess

.PHONY: fastloess-visual fastloess-validate r-validate compare plot

fastloess-visual:
	@echo "Building and running fastLoess visual..."
	@cargo run --manifest-path $(CARGO_DIR)/Cargo.toml --bin visual --release

fastloess-validate:
	@echo "Building and running fastLoess validate..."
	@cargo run --manifest-path $(CARGO_DIR)/Cargo.toml --bin validate --release

r-validate:
	@echo "Running R reference validation..."
	@Rscript R/stats_loess.R

compare:
	@echo "Comparing outputs..."
	@$(PYTHON) compare.py

plot:
	@echo "Generating plots..."
	@$(PYTHON) plot.py
