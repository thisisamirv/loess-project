<!-- markdownlint-disable MD024 MD025 -->
# Changelog

This changelog includes end-user changes only. For internal development notes, see the [repository changelog](https://github.com/thisisamirv/loess-project/blob/main/CHANGELOG.md).

## \[Unreleased\]

### Added

* Added an Alternative Software guide comparing Python LOESS results with `skmisc.loess`, including executable Gaussian and robust examples.
* Added a grouped `cv` dictionary to the Batch constructor, with validation and fallback to individual CV arguments.
* Added `outputs` sequences to `Loess`, `StreamingLoess`, `OnlineLoess`, and prediction for grouped optional result selection.
* Added `retain_model` and `LoessResult.predict(new_x, ...)` (a new `PredictOutput` class) for out-of-sample prediction.
* Added `return_gradient` to `Loess`, `StreamingLoess`, and `OnlineLoess`, exposing the per-point gradient via `LoessResult.gradient`/`OnlineOutput.gradient`. Only takes effect with `surface_mode="direct"`.
* Added `return_se`/`confidence_intervals`/`prediction_intervals` to `StreamingLoess` and `OnlineLoess`. `OnlineLoess` requires `update_mode="full"` or raises `ValueError`. New `OnlineOutput` bound fields.
* Added a Linux musl (Alpine) release binary.

### Changed

* Breaking change: replaced individual `return_*` output keywords, including `return_gradient` and prediction's `return_derivative`, with `outputs=[...]` for Batch, Streaming, Online, and prediction.
* Breaking change: replaced flat interval keywords and prediction levels with `intervals={"confidence": ..., "prediction": ...}`; CV uses only `cv={"fractions": ..., "method": ..., "k": ...}` with an outer `seed`, replacing flat CV keywords and the nested seed. Unknown grouped keys raise `ValueError`.

### Fixed

* Support multivariate Online updates by accepting one coordinate vector per `add_point` call while preserving scalar inputs for one-dimensional models.
* Reject unknown output names across Batch, Streaming, Online, and prediction; accept documented array-like inputs for fit, streaming, and prediction; release the GIL during Online updates.
* Validate case-weight lengths and values before dropping missing observations, so invalid weights on dropped rows are not silently ignored.
* Preserve case weights through sorted CV training subsets and multidimensional predictions. Serial and parallel CV now agree on seeded folds and held-out LOOCV predictions; K-fold counts above the retained observation count are rejected.
* Reject non-positive or non-finite Streaming/Online auto-convergence tolerances. Online auto-convergence requires full updates with robustness iterations.
* Include all observations for Gaussian smoothing and prediction while preserving the k-th-neighbor bandwidth; use the true exponential without an artificial tail floor.
* Correct case-weighted standard errors for direct, unpadded one-dimensional linear fits and retained prediction, with matching serial/parallel local moments. Span-one fits no longer substitute a kernel-free global OLS formula.
* Honor configured zero-weight fallback policies in constant-degree, zero-bandwidth, insufficient-neighbor, and coefficient-fit paths.
* Match R LOESS's even-sample bisquare MAR scale arithmetic, including extremely small residuals, while preserving the centered-MAD fallback.
* Aligned `OnlineLoess` defaults across the Rust core and bindings: `iterations` is now `0` with the default `update_mode = "incremental"`; positive robustness iterations require `update_mode = "full"`.

## 2.0.0

### Added

* Added `return_sorted` and `missing` options to `Loess`, `StreamingLoess`, and `OnlineLoess`.

### Changed

* Consolidated the Python README (merging Installation/Documentation, dropping GitHub-only alert syntax, and removing sections covered by docs pages), renamed "When to Use" to "When to Use Batch Adapter", and moved parameter docs into API option tables, removing `parameters.md`.
* Breaking: Removed `confidence_intervals`/`prediction_intervals`/`return_se` from `StreamingLoess()`, and those plus `return_diagnostics`/`return_residuals`/`parallel` from `OnlineLoess()`; neither adapter computed these options, and Online now always runs sequentially. `Loess`/`StreamingLoess` are unaffected.

### Fixed

* Fixed `OnlineLoess`/`StreamingLoess` defaults to match docs: `min_points` changed from `3` to `2`, and `update_mode` from `"full"` to `"incremental"`.
* Corrected docs to state that result `x` values follow input order after internal sorting and mapping back.
* Fixed the "Handling Outliers" quickstart example printing nothing at `fraction = 0.5` with only 6 points; bumped to `0.7`.
* Fixed the empty "API Reference" page (stale toctree references).

## 1.1.0

### Added

* Added a GitHub Pages landing page at the repository root, built from `README.md` via pandoc and deployed by `docs.yml`.

### Changed

* Moved the CHANGELOG and CONTRIBUTING guides to the project root.
* Updated the Python README to be binding-specific instead of using a generic README shared across bindings.
* Migrated Python documentation from MkDocs to Sphinx (with MyST-Parser and jupyter-sphinx). Code blocks now execute and embed output automatically via `jupyter-sphinx`.

### Fixed

* Fixed `.cargo/config.toml` hardcoding absolute `c:/rtools45/...` paths for the `x86_64-pc-windows-gnu` linker and ar tool; it now uses bare tool names resolved via `PATH`.
* Enforced keyword-only arguments beyond the first positional allowance in `Loess`, `StreamingLoess`, and `OnlineLoess`: `Loess(fraction, *, ...)`, `StreamingLoess(fraction, chunk_size, *, ...)`, and `OnlineLoess(fraction, window_capacity, min_points, *, ...)`. Updated the `.pyi` stubs accordingly.

## 1.0.0

### Changed

* Moved the `tutorials/` pages into a new `user-guide/use-cases/` section.
* Split `StreamingLoess`/`OnlineLoess` content into dedicated API reference pages and standardized API examples with expected output comments.
* Breaking: Renamed `OnlineOutput`'s `smoothed` and `std_error` properties to `y` and `standard_error`.

## 0.9.0

### Added

* Added the Python LOESS binding.
