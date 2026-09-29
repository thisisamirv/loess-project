---
title: Changelog
---
<!-- markdownlint-disable MD024 MD025 -->
# Changelog

This changelog includes end-user changes only. For internal development notes, see the [repository changelog](https://github.com/thisisamirv/loess-project/blob/main/CHANGELOG.md).

## \[Unreleased\]

### Added

* Added `cv` to the Batch options interface for grouped cross-validation configuration alongside legacy fields.
* Added `outputs` arrays to Batch, Streaming, Online, and prediction options for grouped optional result selection alongside existing booleans.
* Added `retain_model` and `LoessResult.predict(newX, options)` for out-of-sample prediction.
* Added `return_gradient` to `SmoothOptions`, `StreamingOptions`, and `OnlineOptions`.
* Added `confidence_intervals`/`prediction_intervals`/`return_se` to Streaming and Online options; Online requires `update_mode: "full"`. Added the corresponding `OnlineOutput` bound fields and TypeScript types.

### Changed

### Fixed

* Aligned `OnlineLoess` defaults across the Rust core and bindings: `iterations` is now `0` with the default `update_mode = "incremental"`; positive robustness iterations require `update_mode = "full"`.

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
