---
title: "Changelog"
weight: 100
---

<!-- markdownlint-disable MD024 MD025 -->
# Changelog

This changelog includes end-user changes only. For internal development notes, see the [repository changelog](https://github.com/thisisamirv/loess-project/blob/main/CHANGELOG.md).

## \[Unreleased\]

### Added

* Added `OnlineLoess.AddPointVector()` for multivariate Online point updates.
* Added `CV *CVOptions` to Batch options for grouped cross-validation, taking precedence over individual CV fields.
* Added `Outputs []string` to `Options`, `StreamingOptions`, `OnlineOptions`, and `PredictOptions` for grouped optional result selection.
* Added `RetainModel` and `Result.PredictModel.Predict(newX, options)` for out-of-sample prediction.
* Added `ReturnGradient` to `Options`, `StreamingOptions`, and `OnlineOptions`.
* Added `ConfidenceIntervals`/`PredictionIntervals`/`ReturnSe` to `StreamingOptions` and `OnlineOptions`. `OnlineOptions` requires `UpdateMode = "full"` or errors. New bound fields on `PointResult`.

### Changed

* Breaking change: replaced individual `Return*` output fields with `Outputs: []string{...}` for Batch, Streaming, Online, and prediction options.
* Breaking change: replaced flat interval fields and prediction levels with `Intervals *IntervalsOptions`; CV uses only `CV *CVOptions` with an outer `Seed`, replacing flat CV fields and `CVOptions.Seed`.
* Represent unavailable diagnostic metrics as nil optional values instead of NaN sentinels in the Go binding.

### Fixed

* Reject explicitly empty custom weights instead of treating them as omitted.
* Keep model receivers alive during Batch, Streaming, Online, and retained-prediction cgo calls to prevent premature native-handle finalization.
* Reject unknown or unsupported output names, extra custom-weight slices, and integer counts outside the C `int` range. Invalid K-fold counts are no longer silently coerced to two.
* Preserve native array lengths and all CV seed bits on Windows with `size_t` lengths and cgo-compatible `unsigned long long` seeds. Rebuild the generated header and native library together; old Windows binaries are ABI-incompatible.
* Validate case-weight lengths and values before dropping missing observations, so invalid weights on dropped rows are not silently ignored.
* Preserve case weights through sorted CV training subsets and multidimensional predictions. Serial and parallel CV now agree on seeded folds and held-out LOOCV predictions; K-fold counts above the retained observation count are rejected.
* Reject non-positive or non-finite Streaming/Online auto-convergence tolerances. Online auto-convergence requires full updates with robustness iterations.
* Include all observations for Gaussian smoothing and prediction while preserving the k-th-neighbor bandwidth; use the true exponential without an artificial tail floor.
* Correct case-weighted standard errors for direct, unpadded one-dimensional linear fits and retained prediction, with matching serial/parallel local moments. Span-one fits no longer substitute a kernel-free global OLS formula.
* Honor configured zero-weight fallback policies in constant-degree, zero-bandwidth, insufficient-neighbor, and coefficient-fit paths.
* Match R LOESS's even-sample bisquare MAR scale arithmetic, including extremely small residuals, while preserving the centered-MAD fallback.
* Aligned `OnlineLoess` defaults across the Rust core and bindings: `iterations` is now `0` with the default `update_mode = "incremental"`; positive robustness iterations require `update_mode = "full"`.
* Breaking: The Go module's import path now includes the required `/v2` major-version suffix; a new release is required for pkg.go.dev to resolve versions correctly.

## 2.0.0

### Added

* Added the Go binding.
