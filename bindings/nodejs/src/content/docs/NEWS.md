---
title: Changelog
---
<!-- markdownlint-disable MD024 MD025 -->
# Changelog

This changelog includes end-user changes only. For internal development notes, see the [repository changelog](https://github.com/thisisamirv/loess-project/blob/main/CHANGELOG.md).

## \[Unreleased\]

### Added

* Added `cv` to Batch options for grouped cross-validation configuration alongside legacy CV fields.
* Added `outputs` arrays to Batch, Streaming, Online, and prediction options for grouped optional result selection.
* Added `retain_model` and `LoessResult.predict(newX, options)` for out-of-sample prediction.
* Added `return_gradient` to `SmoothOptions`, `StreamingOptions`, and `OnlineOptions`.
* Added `confidence_intervals`/`prediction_intervals`/`return_se` to `StreamingSmoothOptions` and `OnlineSmoothOptions`. Online requires `update_mode: "full"` or throws. New `OnlineOutput` bound fields.

### Changed

* Breaking change: replaced individual `return_*` output booleans with `outputs: [...]` for Batch, Streaming, Online, and prediction options.

### Fixed

* Aligned `OnlineLoess` defaults across the Rust core and bindings: `iterations` is now `0` with the default `update_mode = "incremental"`; positive robustness iterations require `update_mode = "full"`.
* Fixed inconsistent naming of the Node.js binding as "JavaScript" across READMEs, doc-site home pages, and `CITATION.cff`.
* Fixed `cv_seed` silently accepting negative values by validating the signed input before converting it to `u64`.

## 2.0.0

### Added

* Added `aarch64-unknown-linux-musl` and `armv7-unknown-linux-gnueabihf` prebuilt targets with matching optional npm subpackages.
* Added `return_sorted` and `missing` options to `Loess`'s `SmoothOptions`/`StreamingSmoothOptions`/`OnlineSmoothOptions`.

### Changed

* Consolidated the Node.js README (merging Installation/Documentation, dropping GitHub-only alert syntax, and removing sections covered by docs pages), renamed "When to Use" to "When to Use Batch Adapter", and moved parameter docs into API option tables, removing `parameters.md`.
* Reorganized API documentation with field tables, per-field Options sections, and a Result Structure section at the end; documented the dynamic `overlap` default.
* Breaking: `StreamingLoess`/`OnlineLoess` now use dedicated `StreamingSmoothOptions`/`OnlineSmoothOptions` types instead of Batch's `SmoothOptions`, dropping Batch-only fields (`confidence_intervals`, `prediction_intervals`, `return_se`, `cv_*`) from both and `return_diagnostics`/`return_residuals`/`parallel` from `OnlineSmoothOptions`. This affects TypeScript consumers; `Loess`'s `SmoothOptions` is unchanged.

### Fixed

* Fixed `OnlineLoess`/`StreamingLoess` defaults to match docs: `min_points` changed from `3` to `2`, and `update_mode` from `"full"` to `"incremental"`.
* Corrected docs to state that result `x` values follow input order after internal sorting and mapping back.
* Fixed the "Handling Outliers" quickstart example printing nothing with only 6 points at `fraction = 0.5`; bumped to `0.7` so the outlier is actually downweighted.
* Fixed the docs homepage never showing README content.
* Fixed TypeDoc/Starlight API-reference links that returned 404s.

## 1.1.0

### Added

* Added a GitHub Pages landing page at the repository root, built from `README.md` via pandoc and deployed by `docs.yml`.

### Changed

* Updated the Node.js README to be binding-specific instead of using a generic README shared across bindings.
* Moved Node.js documentation to GitHub Pages, served by Starlight at <https://thisisamirv.github.io/loess-project/nodejs/>.

### Fixed

## 1.0.0

### Changed

* Moved the `tutorials/` pages into a new `user-guide/use-cases/` section.
* Split `StreamingLoess`/`OnlineLoess` content into dedicated API reference pages and standardized API examples with expected output comments.
* Breaking: Renamed `OnlineOutput`'s `smoothed` and `std_error` fields to `y` and `standard_error`.

## 0.9.0

### Added

* Added the Node.js binding.
