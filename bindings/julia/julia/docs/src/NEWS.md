<!-- markdownlint-disable MD024 MD025 -->
# Changelog

This changelog includes end-user changes only. For internal development notes, see the [repository changelog](https://github.com/thisisamirv/loess-project/blob/main/CHANGELOG.md).

## \[Unreleased\]

### Added

* Added multivariate Online updates through `add_point(model, coordinates, y)` when `dimensions` is greater than one.
* Added `FastLOESS.version()` to report the installed Julia binding version from `Project.toml`.
* Added an Alternative Software guide comparing `FastLOESS.jl` with `Loess.jl`, including a runnable numerical comparison and feature matrix.
* Added the `cv` keyword to `Loess` for grouped cross-validation configuration.
* Added `outputs=[...]` to `Loess`, `StreamingLoess`, `OnlineLoess`, and `predict` for grouped optional result selection.
* Added `retain_model` and `predict(model, new_x; kwargs...)` for out-of-sample prediction.
* Added `return_gradient` to `Loess`, `StreamingLoess`, and `OnlineLoess`.
* Added `confidence_intervals`/`prediction_intervals`/`return_se` to `StreamingLoess` and `OnlineLoess`. `OnlineLoess` requires `update_mode="full"` or errors. New bound fields on `OnlineOutput`.
* Added a Linux musl (Alpine) release binary.

### Changed

* Clarified Batch `residual_sd` as `1.4826 * MAD`; Streaming reports the cumulative sample standard deviation of emitted residuals.
* Breaking change: replaced individual `return_*` output keywords with `outputs=[...]` for Batch, Streaming, Online, and prediction.
* Breaking change: replaced flat interval keywords and prediction levels with `intervals=(confidence=..., prediction=...)`; CV uses only `cv=(fractions=..., method=..., k=...)` with an outer `seed`, replacing flat CV keywords and the nested seed. Unknown grouped keys raise `ArgumentError`.
* Represent unavailable diagnostic metrics as `nothing` instead of `NaN` sentinels in the Julia binding.

### Fixed

* Correct tied-coordinate interpolation splits and normalized cell geometry, add two-dimensional neighboring-edge blending, and match R's neighborhood rounding near integer span boundaries.
* Preserve zero interpolation subdivision thresholds to match R's small-span cell trees, retaining terminal observation vertices and correcting interpolated fits on small datasets.
* Normalize local case weights by their neighborhood maximum so common scaling cannot turn valid positive weights into an epsilon-triggered unweighted fallback.
* Prevented overflow in even medians, mean/bisquare scales, Batch/Streaming diagnostics and AIC, and local/all-tied weight sums for large finite inputs.
* Reject multivariate vector calls before unsafe FFI reads, use fixed-width 64-bit CV seeds, validate interpolation caps before allocation, and reject appending results with different dimensions.
* Preserve model owners and input arrays across native calls, serialize mutable Streaming/Online operations, reject result appends with mismatched optional fields, and surface native constructor validation errors. GPU subprocess/target checks have no LOESS counterpart.
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

* Consolidated the Julia README (merging Installation/Documentation, dropping GitHub-only alert syntax, and removing sections covered by docs pages), renamed "When to Use" to "When to Use Batch Adapter", and moved parameter docs into API option tables, removing `parameters.md`.
* Reorganized API documentation with field tables, per-field Options sections, and a Result Structure section at the end; corrected constructor docstrings, dynamic `overlap` default, and explicit `weighted_metric_weights` distance-metric requirement.
* Breaking: Removed `confidence_intervals`, `prediction_intervals`, and `return_se` from `StreamingLoess` and `OnlineLoess`; also removed `return_diagnostics`, `return_residuals`, and `parallel` from `OnlineLoess`.
* Breaking: `weighted_metric_weights` now requires `distance_metric = "weighted"` explicitly.
* Breaking: Changed `StreamingLoess`'s `overlap` default from a fixed `500` to a dynamic `chunk_size / 10`.

### Fixed

* Fixed broken documentation links left over from the docs-site restructure.
* Fixed `OnlineLoess`/`StreamingLoess` defaults to match docs: `min_points` changed from `3` to `2`, and `update_mode` from `"full"` to `"incremental"`.
* Corrected docs to state that result `x` values follow input order after internal sorting and mapping back.
* Fixed the "Handling Outliers" quickstart example printing nothing at `fraction = 0.5` with only 6 points; bumped to `0.7`.
* Fixed `intervals.md` examples looping over all 100 points instead of a short sample.
* Fixed the Documenter homepage being a stale, separately-maintained `index.md`; now regenerated from `README.md` on every build.
* Fixed `cell`/`interpolation_vertices`/`boundary_degree_fallback`/`cv_seed` being silently non-functional due to no-op FFI setters, and `jl_streaming_loess_new` wrapping negative `dimensions` instead of clamping to 1.

## 1.1.0

### Added

* Added a GitHub Pages landing page at the repository root, built from `README.md` via pandoc and deployed by `docs.yml`.

### Changed

* Updated the Julia README to be binding-specific instead of using a generic README shared across bindings.
* Moved Julia documentation from ReadTheDocs to GitHub Pages, served by Documenter.jl at <https://thisisamirv.github.io/loess-project/julia/stable/>. The ReadTheDocs site no longer includes Julia-specific content. Code blocks use Documenter.jl `@example` sections, which execute and embed output automatically during the docs build.

### Fixed

* Fixed `.cargo/config.toml` hardcoding absolute `c:/rtools45/...` paths for the `x86_64-pc-windows-gnu` linker and ar tool; it now uses bare tool names resolved via `PATH`.
* Fixed `fit(l::Loess, x::Matrix{Float64}, y)` not validating that `size(x, 2) == l.dimensions` before flattening the matrix. If the column count differed from the configured dimensions, the library either silently used wrong data or produced a confusing C-level error. The `Loess` struct now stores `dimensions` as a field, and the matrix overload checks `size(x, 2) != l.dimensions` upfront with a clear message naming the parameter to fix.
* Fixed `FastLOESS.jl` never actually loading the prebuilt `fastloess_jll` binary: `find_library()` only checked the `FASTLOESS_LIB` env var and local dev-mode paths, so the package installed from the registry had no working native library for end users. Added the `fastloess_jll` dependency (`Project.toml`), a JLL-loading branch in `find_library()`, and switched from an eager `const libfastloess = find_library()` (resolved once at precompile time) to a lazy `current_library()` accessor re-resolved in `__init__()`.

## 1.0.0

### Fixed

* Fixed `LoessResult.iterations_used` returning the raw FFI sentinel `-1` instead of `nothing` when robustness iterations were not applicable.

### Changed

* Moved the `tutorials/` pages into a new `user-guide/use-cases/` section.
* Split `StreamingLoess`/`OnlineLoess` content into dedicated API reference pages and standardized API examples with expected output comments.
* Breaking: Renamed `OnlineOutput`'s `smoothed` and `std_error` fields to `y` and `standard_error`.

## 0.9.0

### Added

* Added the Julia LOESS binding.

* Added case-weighted Streaming chunks and Online points, plus Online window diagnostics and prediction.
