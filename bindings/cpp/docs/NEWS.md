\page news Changelog

This changelog includes end-user changes only. For internal development notes, see the [repository changelog](https://github.com/thisisamirv/loess-project/blob/main/CHANGELOG.md).

<!-- markdownlint-disable MD024 MD025 -->
## \[Unreleased\]

### Added

* Added Windows x64 MinGW, Linux x86/ARMv7, Android ABI, and iOS device/simulator release binaries with matching cross-target CI builds.
* Added compile-time C++ version macros and runtime `cpp_version()` reporting; generated and distributed the version header with CMake, Spack, and prebuilt release assets.
* Added `CVOptions cv` to Batch options for grouped cross-validation while preserving legacy CV fields.
* Added `retain_model`, `LoessResult::predict_model()`, and new `PredictModel`/`PredictOptions`/`PredictResult` RAII classes for out-of-sample prediction.
* Added `return_gradient` to `LoessOptions` and `OnlineOptions`.
* Added `confidence_intervals`/`prediction_intervals`/`return_se` to `OnlineOptions` (already present on `StreamingOptions` via inheritance, now forwarded). Online requires `update_mode == "full"`. New `OnlineOutput` accessors.
* Added a Linux musl (Alpine) release binary.

### Changed

* Breaking change: replaced the individual output booleans in `LoessOptions`, `OnlineOptions`, and `PredictOptions` with grouped `outputs` vectors.
* Breaking change: replaced flat interval levels with `intervals`, removed flat CV fields in favor of `cv`, and moved CV seeding to optional outer `seed`; `seed = 0` is now reproducible.
* Native C ABI lengths now use `size_t` and CV seeds use `uint64_t` instead of Windows-truncated `unsigned long`. Rebuild the wrapper/header and native library together; old Windows binaries are not layout-compatible.
* C++ musl release jobs now build dynamic x86_64 and ARM64 shared libraries, allowing the musl assets to be published reliably.
* The public CMake target propagates the wrapper's C++17 requirement to consumers. Unavailable diagnostics are empty `std::optional<double>` values, including default-constructed diagnostics and native NaN sentinels; finite values are preserved.

### Fixed

* Forward the selected distance metric when per-dimension weights are supplied, and support multivariate Online points through a vector-coordinate `add_point` overload.
* Free retained prediction handles and zero-length error results, reset freed native results, and make wrapper error paths exception-safe. Multidimensional predictor buffers are freed with their full length.
* Make empty/moved result accessors safe, bounds-check indexed access, preserve all predictor coordinates, and keep unavailable diagnostics/statistics absent.
* Reject unknown or mode-inappropriate outputs and unsupported Batch-only Streaming options. Surface invalid Batch configuration at construction, reject negative counts and invalid active CV folds instead of coercing them, and preserve CV seed bits above 32 bits.
* Validate case-weight lengths and values before dropping missing observations, so invalid weights on dropped rows are not silently ignored.
* Preserve case weights through sorted CV training subsets and multidimensional predictions. Serial and parallel CV now agree on seeded folds and held-out LOOCV predictions; K-fold counts above the retained observation count are rejected.
* Reject non-positive or non-finite Streaming/Online auto-convergence tolerances. Online auto-convergence requires full updates with robustness iterations.
* Include all observations for Gaussian smoothing and prediction while preserving the k-th-neighbor bandwidth; use the true exponential without an artificial tail floor.
* Correct case-weighted standard errors for direct, unpadded one-dimensional linear fits and retained prediction, with matching serial/parallel local moments. Span-one fits no longer substitute a kernel-free global OLS formula.
* Honor configured zero-weight fallback policies in constant-degree, zero-bandwidth, insufficient-neighbor, and coefficient-fit paths.
* Match R LOESS's even-sample bisquare MAR scale arithmetic, including extremely small residuals, while preserving the centered-MAD fallback.
* Aligned `OnlineLoess` defaults across the Rust core and bindings: `iterations` is now `0` with the default `update_mode = "incremental"`; positive robustness iterations require `update_mode = "full"`.
* Fixed `bindings/cpp/spack/package.py` building/installing from the wrong directory (`bindings/cpp` instead of the workspace-root `target/release`), which broke `spack install fastloess-cpp` on every platform. Now builds by package name. Also moved the pyright suppression out of the recipe into a new root `pyrightconfig.json`.

## 2.0.0

### Added

* Added CMake package-config support for downstream `find_package(fastloess)` use.
* Added ARM64 release binaries for Linux/Windows/macOS; fixed the macOS x64 job silently shipping a mislabeled arm64 binary.
* Added `return_sorted` and `missing` options to `LoessOptions`/`OnlineOptions`.

### Changed

* Documented `x_values`/`y_values` params (fixing a Doxygen warning) and restructured Doxygen nav from ~20 flat pages into 5 hub pages.
* Consolidated the C++ README (merging Installation/Documentation, dropping GitHub-only alert syntax, and removing sections covered by docs pages), renamed "When to Use" to "When to Use Batch Adapter", and moved parameter docs into API option tables, removing `parameters.md`.
* Vendored doxygen-awesome-css v2.4.2 for a modern Doxygen theme.
* Replaced the `kernels.md`/`adapter-choice.md` mermaid flowcharts with tables because Doxygen does not render mermaid.
* Reorganized API documentation with field tables, a per-field `## Options` section, and a `## Result Structure` section at the end.
* Updated docs to require explicitly setting `weighted_metric_weights`'s distance metric to `"weighted"` and to show the dynamic `StreamingOptions.overlap` default.
* Added Spack installation support for `fastloess-cpp`.
* Breaking: Removed the unused `confidence_intervals`, `prediction_intervals`, `return_diagnostics`, `return_residuals`, and `return_se` fields from standalone `OnlineOptions`, and removed `parallel` for consistency with `fastLowess`; Online now always runs sequentially.
* `StreamingOptions` no longer forwards `confidence_intervals`/`prediction_intervals`/`return_se` to the native constructor, since Streaming never computed them.
* Breaking: Changed `StreamingOptions::overlap` from a fixed `500` default to a sentinel (`-1`) that resolves dynamically to `chunk_size / 10`; callers relying on the previous flat default are affected.
* Breaking: `weighted_metric_weights` no longer auto-selects the `"weighted"` distance metric; callers must set it explicitly.

### Fixed

* Fixed `OnlineLoess`/`StreamingLoess` defaults to match docs: `min_points` changed from `3` to `2`, and `update_mode` from `"full"` to `"incremental"`.
* Corrected docs to state that result `x` values follow input order after internal sorting and mapping back.
* Fixed the "Handling Outliers" quickstart example printing nothing with only 6 points at `fraction = 0.5`; bumped to `0.7` so the outlier is actually downweighted.
* Fixed several Doxygen rendering bugs (wrong homepage, broken blockquotes/math/admonitions); `README.md` is now the native homepage.
* Fixed Doxygen rendering and broken API documentation links.

## 1.1.0

### Added

* Added a GitHub Pages landing page at the repository root, built from `README.md` via pandoc and deployed by `docs.yml`.

### Changed

* Updated the C++ README to be binding-specific instead of using a generic README shared across bindings.
* Moved C++ documentation from ReadTheDocs to GitHub Pages, served by Doxygen at <https://thisisamirv.github.io/loess-project/cpp/>. The ReadTheDocs site no longer includes C++-specific content.

### Fixed

* Fixed Windows source builds to use the MSVC toolchain instead of selecting an incompatible MinGW linker.

## 1.0.0

### Changed

* Moved the `tutorials/` pages into a new `user-guide/use-cases/` section.
* Split `StreamingLoess`/`OnlineLoess` content into dedicated API reference pages and standardized API examples with expected output comments.
* Breaking: Renamed `OnlineOutput`'s `smoothed()` and `std_error()` methods to `y()` and `standard_error()`.

## 0.9.0

### Added

* Added the C++ binding.
