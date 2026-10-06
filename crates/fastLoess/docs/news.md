<!-- markdownlint-disable MD024 MD025 -->
# Changelog

This changelog includes end-user changes only. For internal development notes, see the [repository changelog](https://github.com/thisisamirv/loess-project/blob/main/CHANGELOG.md).

## \[Unreleased\]

### Added

* Added `.cv(...)` to the parallel Batch builder, re-exporting `CVBuilder` through the prelude and `CVOptions<T>` at the crate root.
* Added `outputs(names)` to the `Loess`, `StreamingLoess`, and `OnlineLoess` wrappers, forwarding grouped output selection and deferred unknown-name errors to the core builder.
* Added parallel `custom_gradient_pass` and predict passes for the `return_gradient` option and `Predict::call()`.
* Added parallel builder setters for `return_se`, `confidence_intervals`, and `prediction_intervals` to the Streaming and Online adapters.
* Added `return_sorted` and `missing` to `BuilderOptionSet`/`TypedBuilderOptionSet` and the `Loess`/`StreamingLoess`/`OnlineLoess` builders.
* Published the fastLoess crate on crates.io.

### Changed

* Clarified that Batch `residual_sd` is `1.4826 * MAD`, while Streaming reports the cumulative sample standard deviation of emitted residuals.
* Use `.outputs([...])` in place of individual `return_*` selectors for Batch, Streaming, Online, and retained-model prediction; legacy Rust selectors remain available.
* Breaking: Removed unsupported interval and diagnostics options from Streaming/Online and `parallel` from `OnlineLoess`; Online fitting always runs sequentially.
* Consolidated the fastLoess README (merging Installation/Documentation, dropping GitHub-only alert syntax, and removing sections covered by docs pages), and moved parameter docs into API option tables, removing `parameters.md`. Replaced `kernels.md`/`adapter-choice.md` mermaid flowcharts with tables because rustdoc does not render mermaid.
* Reorganized API documentation with field tables, per-field Options sections, and a Result Structure section at the end.

### Fixed

* Correct unweighted, unpadded one-dimensional linear prediction standard errors without robustness downweighting using the smoothing influence rows and global residual degrees of freedom, including interpolation; align prediction-interval residual scales.
* Correct tied-coordinate interpolation splits and normalized cell geometry, add two-dimensional neighboring-edge blending, and match R's neighborhood rounding near integer span boundaries.
* Preserve zero interpolation subdivision thresholds to match R's small-span cell trees, retaining terminal observation vertices and correcting interpolated fits on small datasets.
* Normalize local case weights by their neighborhood maximum so common scaling cannot turn valid positive weights into an epsilon-triggered unweighted fallback.
* Prevented overflow in even medians, mean/bisquare scales, Batch/Streaming diagnostics and AIC, and local/all-tied weight sums for large finite inputs.
* Require explicit `distance_metric("weighted")` selection before applying `weighted_metric_weights`; support fixed-size Rust arrays as fit inputs.
* Reuse the serial CV fold engine for parallel candidates, preserving case weights, seeded shuffling, multidimensional normalization and held-out LOOCV predictions.
* Forward case weights into parallel interval estimation and use the same local-SE moments as serial fits.
* Include full Gaussian kernel support in parallel smoothing, gradients, vertex refits and prediction without changing the k-th-neighbor bandwidth.
* Fixed `OnlineLoess`/`StreamingLoess` defaults to match docs: `min_points` changed from `3` to `2`, and `update_mode` from `"full"` to `"incremental"`.
* Corrected docs to state that result `x` values follow input order after internal sorting and mapping back.
* Fixed the "Handling Outliers" quickstart example printing nothing at `fraction = 0.5` with only 6 points; bumped to `0.7`.
* Fixed parallel standard errors collapsing to zero for downweighted observations by using the local-linear variance calculation.
* Fixed broken documentation cross-reference links and LaTeX rendering on docs.rs.

## 1.1.0

### Added

* Added a GitHub Pages landing page at the repository root, built from `README.md` via pandoc and deployed by `docs.yml`.

### Changed

* Updated the fastLoess README to be crate-specific instead of using a generic README shared across bindings/crates.
* Moved crate documentation from ReadTheDocs to <https://docs.rs/fastLoess>.

### Fixed

* Fixed `.cargo/config.toml` hardcoding absolute `c:/rtools45/...` paths for the `x86_64-pc-windows-gnu` linker and ar tool; it now uses bare tool names resolved via `PATH`.

## 1.0.0

### Changed

* Moved the `tutorials/` pages into a new `user-guide/use-cases/` section.
* Split `StreamingLoess`/`OnlineLoess` content into dedicated API reference pages and standardized API examples with expected output comments.

## 0.9.0

### Added

* Added the option to pass custom weights by the user to the algorithm.

### Changed

* Converted all documentation tables to compact single-space format.
* Added `Loess<T>`, `StreamingLoess<T>`, and `OnlineLoess<T>` type aliases as the primary user-facing constructors (e.g. `StreamingLoess::new().chunk_size(50).build()`). Mode-specific builder methods (`chunk_size`, `overlap`, `window_capacity`, `min_points`, `update_mode`) are now called directly on the type alias rather than after `.adapter()`.
* Breaking: Made `BatchLoessBuilder`, `StreamingLoessBuilder`, and `OnlineLoessBuilder` internal-only, removing their public setter methods. Smoothing configuration now flows through `LoessBuilder<T, Mode>`; code that called setters on an adapter builder must migrate.
* Breaking: Changed enum-typed builder methods (`weight_function`, `robustness_method`, `scaling_method`, `boundary_policy`, `zero_weight_fallback`, `merge_strategy`, and `update_mode`) to accept strings as well as enum variants through `impl IntoEnum<T>`; callers passing enum variants directly must update.
* Breaking: Replaced `cross_validate(CVConfig)` with the string-based `.cv_method(...)`, `.cv_k(...)`, `.cv_fractions(...)`, and `.cv_seed(...)` API; `KFold` and `LOOCV` are no longer exported from the prelude, so callers using the old API must migrate.

## 0.2.2

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 0.2.1

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 0.2.0

* Under-the-hood maintenance; no changes to the public API or runtime behavior.

## 0.1.0

### Added

* Initial release with parallel execution support.

* Forward weighted chunk/point updates, Online window diagnostics, and current-window prediction through the parallel adapter facade.
