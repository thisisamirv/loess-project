---
title: Changelog
---
<!-- markdownlint-disable MD024 MD025 -->
# Changelog

This changelog includes end-user changes only. For internal development notes, see the [repository changelog](https://github.com/thisisamirv/loess-project/blob/main/CHANGELOG.md).

## \[Unreleased\]

### Added

* Added `OnlineLoess.add_point_vector()` for multivariate Online point updates.
* Added `version()` to report the WASM binding package version.
* Added `cv` to the Batch options interface for grouped cross-validation configuration alongside legacy fields.
* Added `outputs` arrays to Batch, Streaming, Online, and prediction options for grouped optional result selection.
* Added `retain_model` and `LoessResult.predict(newX, options)` for out-of-sample prediction.
* Added `return_gradient` to `SmoothOptions`, `StreamingOptions`, and `OnlineOptions`.
* Added `confidence_intervals`/`prediction_intervals`/`return_se` to Streaming and Online options; Online requires `update_mode: "full"`. Added the corresponding `OnlineOutput` bound fields and TypeScript types.

### Changed

* Breaking change: replaced individual `return_*` output booleans with `outputs: [...]` for Batch, Streaming, Online, and prediction options.
* Breaking change: replaced flat interval options and prediction levels with `intervals: { confidence, prediction }`; CV uses only `cv: { fractions, method, k }` with an outer `seed`, replacing flat CV options and `cv.seed`. Legacy and unknown option keys are rejected.

### Fixed

* Restored TypeScript declarations for retained prediction and weighted Streaming, documented and validated the JavaScript-safe CV seed range, and strengthened interval output assertions.

* Validate case-weight lengths and values before dropping missing observations, so invalid weights on dropped rows are not silently ignored.
* Preserve case weights through sorted CV training subsets and multidimensional predictions. Serial and parallel CV now agree on seeded folds and held-out LOOCV predictions; K-fold counts above the retained observation count are rejected.
* Reject non-positive or non-finite Streaming/Online auto-convergence tolerances. Online auto-convergence requires full updates with robustness iterations.
* Include all observations for Gaussian smoothing and prediction while preserving the k-th-neighbor bandwidth; use the true exponential without an artificial tail floor.
* Correct case-weighted standard errors for direct, unpadded one-dimensional linear fits and retained prediction, with matching serial/parallel local moments. Span-one fits no longer substitute a kernel-free global OLS formula.
* Honor configured zero-weight fallback policies in constant-degree, zero-bandwidth, insufficient-neighbor, and coefficient-fit paths.
* Match R LOESS's even-sample bisquare MAR scale arithmetic, including extremely small residuals, while preserving the centered-MAD fallback.
* Aligned `OnlineLoess` defaults across the Rust core and bindings: `iterations` is now `0` with the default `update_mode = "incremental"`; positive robustness iterations require `update_mode = "full"`.
* Reject unknown option keys and mode-inappropriate output names; return owned typed-array copies that remain valid after freeing result owners.

## 2.0.0

### Added

* Fixed the generated `onlineoptions.md` TypeDoc page stating stale defaults (`min_points: 3`, `update_mode: "full"`) that no longer matched the already-correct source.

## 1.1.0

### Added

* Added a GitHub Pages landing page at the repository root, built from `README.md` via pandoc and deployed by `docs.yml`.

### Changed

* Updated the WASM README to be binding-specific instead of using a generic README shared across bindings.
* Moved WASM documentation to GitHub Pages, served by Starlight at <https://thisisamirv.github.io/loess-project/wasm/>.

### Fixed

* Fixed `.cargo/config.toml` hardcoding absolute `c:/rtools45/...` paths for the `x86_64-pc-windows-gnu` linker and ar tool; it now uses bare tool names resolved via `PATH`.

## 1.0.0

### Fixed

* Fixed `OnlineLoess.add_point()` returning `undefined` instead of `null` when the sliding window has not yet accumulated enough points.

### Changed

* Moved the `tutorials/` pages into a new `user-guide/use-cases/` section.
* Split `StreamingLoess`/`OnlineLoess` content into dedicated API reference pages and standardized API examples with expected output comments.
* Breaking: Renamed `OnlineOutput`'s `smoothed` and `std_error` getters to `y` and `standard_error`.

## 0.9.0

### Added

* Added the WebAssembly binding.

* Added weighted Streaming/Online updates, Online window diagnostics, and prediction from the current window.
