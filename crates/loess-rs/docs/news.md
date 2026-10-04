<!-- markdownlint-disable MD024 MD025 -->
# Changelog

This changelog includes end-user changes only. For internal development notes, see the [repository changelog](https://github.com/thisisamirv/loess-project/blob/main/CHANGELOG.md).

## \[Unreleased\]

### Added

* Added grouped cross-validation configuration via `CVBuilder::method(...).fractions(...)` and `.cv(...)`; `CVBuilder` is in the prelude and the `CVOptions<T>` result type is at the crate root.
* Added `LoessBuilder::outputs(names)` as a grouped replacement for individual output toggles; unknown names are accumulated and reported together by `.build()`.
* Added `return_gradient` to the Batch, Streaming, and Online adapter builders, exposing each point's local-fit gradient (`LoessResult::gradient` / `OnlineOutput::gradient`) at no extra computation cost. Only populated when `surface_mode` is `"direct"`. `false` by default.
* Added `retain_model` and `Predict::call()` for out-of-sample prediction, with optional SE, interval, derivative, and extrapolation settings.
* Added `return_se`/`confidence_intervals`/`prediction_intervals` to the Streaming and Online adapters, mirroring Batch. Online requires `update_mode("full")`; using them under the default `"incremental"` mode now fails fast at `.build()` with a new `LoessError::StandardErrorRequiresFullUpdateMode`.

### Changed

* Use `.outputs([...])` in place of individual `return_*` selectors for Batch, Streaming, Online, and retained-model prediction; legacy Rust selectors remain available.
* Marked `WeightFunction` as non-exhaustive so downstream kernel must reject unsupported future variants explicitly.
* Matched R `stats::loess` span truncation, multivariate predictor normalization, and bisquare robustness cutoffs; MAR now uses R's uncentered median absolute residual and machine-minimum scale stop, while MAD remains the default. Near-singular local linear fits are handled by the regression solver rather than a global-range slope cutoff.

### Fixed

* Aligned `OnlineLoess` defaults across the Rust core and bindings: `iterations` is now `0` with the default `update_mode = "incremental"`; positive robustness iterations require `update_mode = "full"`.
* Cleaned up `loess_rs::prelude` of accidentally-leaked internals (`LoessBuilder`, adapter markers) — use the `Loess`/`StreamingLoess`/`OnlineLoess` type aliases directly.
* Matched LOWESS's effective-zero MAR stop and removed the absolute bisquare scale floor, while retaining the centered-MAD fallback.
* Added the classical simple-linear-regression standard-error path for one-dimensional global fits (`fraction >= 1.0`), matching `stats::lm`'s `se.fit` formula.
* Corrected serial LOESS standard errors to use the local-linear equivalent-kernel leverage and kernel-corrected residual degrees of freedom, preserving positive SEs for downweighted observations. Added Monte Carlo calibration and interval edge-case regressions.
* Fixed seeded k-fold CV with unordered test queries: batch interpolation now locates each query bracket independently with binary search instead of relying on a monotone scan pointer.
* Fixed local-linear and global OLS regression on small-magnitude predictors by using scale-relative degeneracy checks instead of absolute x-variance thresholds. Added gradient and standard-error regressions for small x scales.
* Matched Cleveland/R's local-linear degeneracy rule in one-dimensional linear fits by suppressing slopes when weighted local spread is below `0.001` of the global x-range.
* Matched R's `1e-7` span-truncation adjustment instead of rounding near-integer neighborhoods with `1e-5`.
* Matched R's normalized adjusted-weight fitted-value accumulation without parity-, sparsity-, or response-scale-specific branches.
* Separated local-weight adjustment and fitted-response accumulation into R's original loop order, avoiding platform-dependent cancellation in sparse robust fits.
* Separated robustness scale scratch storage from local kernel weights so median selection cannot contaminate the next R-equivalent smoothing pass.
* Matched R's `w * ((x - mean_x) * (x - mean_x))` spread parenthesization, preserving cancellation-scale endpoint fits during robust passes.
* Matched R's even-length `cmad = 3 * (lower + upper)` operation order instead of scaling an averaged median.
* Extended local kernel scans beyond the nominal right window edge until R's `0.999 * h` cutoff, matching `lowest()` on asymmetric neighborhoods.
* Fixed k-fold cross-validation to pool every test point's squared error before taking one RMSE, matching LOOCV instead of averaging per-fold RMSEs.

## 2.0.0

### Added

* Added `.return_sorted()` to the batch builder, returning results sorted ascending by `x` (default `false`).
* Passed configured degree, kernel, fallback, and distance settings to custom vertex passes.

### Changed

* Consolidated the loess-rs README (merging Installation/Documentation, dropping GitHub-only alert syntax, and removing sections covered by docs pages), and moved parameter docs into API option tables, removing `parameters.md`. Replaced `kernels.md`/`adapter-choice.md` mermaid flowcharts with tables because rustdoc does not render mermaid.
* Reorganized API documentation with field tables, per-field Options sections, and a Result Structure section at the end.
* Breaking: `Streaming::convert()` now resolves `overlap` dynamically to `chunk_size / 10` (clamped to `[1, chunk_size - 10]`) via `default_overlap()`; callers relying on the previous flat `500` default are affected.

### Fixed

* Fixed broken documentation cross-reference links left over from the docs-site restructure.
* Fixed `OnlineLoess`/`StreamingLoess` defaults to match docs: `min_points` changed from `3` to `2`, and `update_mode` from `"full"` to `"incremental"`.
* Fixed the direct Rust API's internal robustness-iteration defaults to match docs: Streaming `2`→`3` and Online `1`→`3`.
* Corrected docs to state that result `x` values follow input order after internal sorting and mapping back.
* Fixed the "Handling Outliers" quickstart example printing nothing at `fraction = 0.5` with only 6 points; bumped to `0.7`.
* Fixed LaTeX math rendering as literal text and cross-reference links not resolving against the rustdoc module tree, both on docs.rs.

## 1.1.0

### Added

* Added a GitHub Pages landing page at the repository root, built from `README.md` via pandoc and deployed by `docs.yml`.

### Changed

* Updated the loess-rs README and moved crate documentation to <https://docs.rs/loess-rs>.

### Fixed

* Fixed Windows source builds by resolving linker and archiver tools from `PATH` instead of hardcoded absolute paths.
* Improved `MismatchedInputs` errors to report the expected input length for one-dimensional and multivariate data.

## 1.0.0

### Changed

* Moved the `tutorials/` pages into a new `user-guide/use-cases/` section.
* Split `StreamingLoess`/`OnlineLoess` content into dedicated API reference pages and standardized API examples with expected output comments.
* Breaking: Renamed `OnlineOutput`'s `smoothed` and `std_error` fields to `y` and `standard_error`, matching `LoessResult`.

## 0.9.0

### Added

* Added the option to pass custom weights by the user to the algorithm.

### Changed

* Added `Loess<T>`, `StreamingLoess<T>`, and `OnlineLoess<T>` type aliases as the primary user-facing constructors (e.g. `StreamingLoess::new().chunk_size(50).build()`). Mode-specific builder methods (`chunk_size`, `overlap`, `window_capacity`, `min_points`, `update_mode`) are now called directly on the type alias rather than after `.adapter()`.
* Breaking: Made `BatchLoessBuilder`, `StreamingLoessBuilder`, and `OnlineLoessBuilder` internal-only, removing their public setter methods. Smoothing configuration now flows through `LoessBuilder<T, Mode>`; code that called setters on an adapter builder must migrate.
* Breaking: Changed enum-typed builder methods (`weight_function`, `robustness_method`, `scaling_method`, `boundary_policy`, `zero_weight_fallback`, `merge_strategy`, and `update_mode`) to accept strings as well as enum variants through `impl IntoEnum<T>`; callers passing enum variants directly must update.
* Breaking: Replaced `cross_validate(CVConfig)` with the string-based `.cv_method(...)`, `.cv_k(...)`, `.cv_fractions(...)`, and `.cv_seed(...)` API; `KFold` and `LOOCV` are no longer exported from the prelude, so callers using the old API must migrate.

## 0.2.2

### Fixed

* Updated license badge.
* Fixed LOESS mechanism figure path.

## 0.2.1

### Changed

* Reduced figures size significantly.
* Implement naming consistency for `auto_converge` (removed `auto_convergence`).

### Fixed

* Fixed `boundary_degree_fallback` pass to online and streaming adapters.
* Fixed `boundary_degree_fallback` pass to `custom_vertex_pass` and `VertexPassFn`.
* Fixed K-fold cross-validation with sorted training subsets and binary-search interpolation for each test point.
* Fixed `auto_converge` support for Online adapter.

## 0.2.0

### Added

* Added `VertexPassFn` and `custom_vertex_pass` support to enable parallelized/accelerated interpolation fitting.
* Added support for custom vertex pass callbacks to all adapters (`Batch`, `Streaming`, `Online`).
* Added support for custom parallel/accelerated standard error calculation via `custom_interval_pass`.
* Added `KDTreeBuilderFn` and `custom_kdtree_builder` hook to enable external parallel KD-tree construction.
* Added `KDTree::from_parts` and exposed `KDNode` and `KDTree::calculate_left_subtree_size` to support custom tree building.
* Added neighborhood caching in `InterpolationSurface` to significantly optimize performance during robustness iterations.
* Added configurable `boundary_degree_fallback` option to control polynomial degree reduction at boundary vertices during interpolation. Defaults to `true` for stability; set to `false` to match R's `loess` behavior exactly.

### Changed

* Changed license from AGPL-3.0-or-later to dual MIT OR Apache-2.0.
* Expanded `SmoothPassFn`, `CVPassFn`, and `IntervalPassFn` signatures to include full multi-dimensional context (dimensions, scaling, polynomial degree, etc.).
* Custom vertex-pass callbacks now receive the configured polynomial degree, kernel, fallback, distance metric, and scale values.
* Expanded the crate documentation for fitting options and adapters.

### Fixed

* Fixed a potential crash in parallel interpolation refinement by correctly propagating augmented data slices to vertex fitting functions.
* Fixed inconsistent parameter types in custom pass callbacks.
* Fixed missing setters for online and streaming adapters.
* Fixed incorrect standard error propagation in `BatchLoessBuilder`.
* Added `Boundary Linear Fallback` strategy to `InterpolationSurface` to prevent numerical instability ("explosions") at data boundaries when using high-degree polynomials (Quadratic, Cubic, Quartic).
* Fixed missing `max_distance` update in the KD-Tree search, which incorrectly calculated the bandwidth for tricube weights.
* Fixed incorrect fitted values caused by stale local-regression weights being reused between query points.
* Fixed horizontal phase shift in `Interpolation` mode when using boundary policies (`Extend`, `Reflect`, `Zero`). The robustness iteration loop was incorrectly using augmented data indices instead of original data for query point evaluation.

## 0.1.0

### Added

* Initial release.
