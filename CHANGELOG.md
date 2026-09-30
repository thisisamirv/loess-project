<!-- markdownlint-disable MD024 MD046 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

**Monorepo:**

- Added R and original Cleveland LOESS references under `validation/reference/`.
- Updated Julia documentation snippet verification to use the docs environment for examples under the Julia docs source tree.
- Curated package NEWS files to include end-user changes only, using a maintenance note for releases with no public API or runtime changes.

**loess-rs:**

- Added grouped cross-validation configuration via `CVBuilder::method(...).fractions(...)` and `.cv(...)`; `CVBuilder` is in the prelude and the `CVOptions<T>` result type is at the crate root.
- Added `LoessBuilder::outputs(names)` as a grouped replacement for individual output toggles; unknown names are accumulated and reported together by `.build()`.
- Added `return_gradient` to the Batch, Streaming, and Online adapter builders, exposing each point's local-fit gradient (`LoessResult::gradient` / `OnlineOutput::gradient`) at no extra computation cost. Only populated when `surface_mode` is `"direct"`. `false` by default.
- Added `retain_model` and `Predict::call()` for out-of-sample prediction, with optional SE, interval, derivative, and extrapolation settings.
- Added `return_se`/`confidence_intervals`/`prediction_intervals` to the Streaming and Online adapters, mirroring Batch. Online requires `update_mode("full")`; using them under the default `"incremental"` mode now fails fast at `.build()` with a new `LoessError::StandardErrorRequiresFullUpdateMode`.

**fastLoess:**

- Added `.cv(...)` to the parallel Batch builder, re-exporting `CVBuilder` through the prelude and `CVOptions<T>` at the crate root.
- Added `outputs(names)` to the `Loess`, `StreamingLoess`, and `OnlineLoess` wrappers, forwarding grouped output selection and deferred unknown-name errors to the core builder.
- Added parallel `custom_gradient_pass` and predict passes for the `return_gradient` option and `Predict::call()`.
- Added parallel builder setters for `return_se`, `confidence_intervals`, and `prediction_intervals` to the Streaming and Online adapters.

**C++:**

- Added `CVOptions cv` to Batch options for grouped cross-validation while preserving legacy CV fields.
- Added `retain_model`, `LoessResult::predict_model()`, and new `PredictModel`/`PredictOptions`/`PredictResult` RAII classes for out-of-sample prediction.
- Added `return_gradient` to `LoessOptions` and `OnlineOptions`.
- Added `confidence_intervals`/`prediction_intervals`/`return_se` to `OnlineOptions` (already present on `StreamingOptions` via inheritance, now forwarded). Online requires `update_mode == "full"`. New `OnlineOutput` accessors.
- Added a Linux musl (Alpine) release binary.

**Go:**

- Added `CV *CVOptions` to Batch options for grouped cross-validation, taking precedence over individual CV fields.
- Added `Outputs []string` to `Options`, `StreamingOptions`, `OnlineOptions`, and `PredictOptions` for grouped optional result selection; existing boolean output fields remain supported.
- Added `RetainModel` and `Result.PredictModel.Predict(newX, options)` for out-of-sample prediction.
- Added `ReturnGradient` to `Options`, `StreamingOptions`, and `OnlineOptions`.
- Added `ConfidenceIntervals`/`PredictionIntervals`/`ReturnSe` to `StreamingOptions` and `OnlineOptions`. `OnlineOptions` requires `UpdateMode = "full"` or errors. New bound fields on `PointResult`.

**Java:**

- Added an Alternative Software guide with runnable Gaussian and robust comparisons to R's `stats::loess()` and a LOESS feature matrix.
- Added `CVOptions.builder()` and `Options.Builder.cv(...)` for grouped Batch cross-validation.
- Added `outputs(String...)` to `Options.Builder`, `StreamingOptions.Builder`, `OnlineOptions.Builder`, and `PredictOptions.Builder` for grouped optional result selection.
- Added `retainModel` and `Result.predictModel()` (a `PredictModel` class) for out-of-sample prediction.
- Added `returnGradient(boolean)` to `Options` and `OnlineOptions`.
- Added `confidenceIntervals(double)`/`predictionIntervals(double)`/`returnSe(boolean)` to `StreamingOptions` and `OnlineOptions`. `OnlineOptions` requires `updateMode("full")` or throws. New accessors on `PointResult`.
- Added prebuilt native libraries across 8 platforms (Linux/macOS/Windows x64/arm64, plus Linux musl variants), bundled into the jar; `NativeBridge` detects musl and extracts the matching library automatically.

**Julia:**

- Added an Alternative Software guide comparing `FastLOESS.jl` with `Loess.jl`, including a runnable numerical comparison and feature matrix.
- Added the `cv` keyword to `Loess` for grouped cross-validation configuration.
- Added `outputs=[...]` to `Loess`, `StreamingLoess`, `OnlineLoess`, and `predict` for grouped optional result selection; existing individual output keywords remain supported.
- Added `retain_model` and `predict(model, new_x; kwargs...)` for out-of-sample prediction.
- Added `return_gradient` to `Loess`, `StreamingLoess`, and `OnlineLoess`.
- Added `confidence_intervals`/`prediction_intervals`/`return_se` to `StreamingLoess` and `OnlineLoess`. `OnlineLoess` requires `update_mode="full"` or errors. New bound fields on `OnlineOutput`.
- Added a Linux musl (Alpine) release binary.

**Node.js:**

- Added `cv` to Batch options for grouped cross-validation configuration alongside legacy CV fields.
- Added `outputs` arrays to Batch, Streaming, Online, and prediction options for grouped optional result selection alongside existing booleans.
- Added `retain_model` and `LoessResult.predict(newX, options)` for out-of-sample prediction.
- Added `return_gradient` to `SmoothOptions`, `StreamingOptions`, and `OnlineOptions`.
- Added `confidence_intervals`/`prediction_intervals`/`return_se` to `StreamingSmoothOptions` and `OnlineSmoothOptions`. Online requires `update_mode: "full"` or throws. New `OnlineOutput` bound fields.

**Python:**

- Added an Alternative Software guide comparing Python LOESS results with `skmisc.loess`, including executable Gaussian and robust examples.
- Added a grouped `cv` dictionary to the Batch constructor, with validation and fallback to individual CV arguments.
- Added `outputs` sequences to `Loess`, `StreamingLoess`, `OnlineLoess`, and prediction for grouped optional result selection alongside existing booleans.
- Added `retain_model` and `LoessResult.predict(new_x, ...)` (a new `PredictOutput` class) for out-of-sample prediction.
- Added `return_gradient` to `Loess`, `StreamingLoess`, and `OnlineLoess`, exposing the per-point gradient via `LoessResult.gradient`/`OnlineOutput.gradient`. Only takes effect with `surface_mode="direct"`.
- Added `return_se`/`confidence_intervals`/`prediction_intervals` to `StreamingLoess` and `OnlineLoess`. `OnlineLoess` requires `update_mode="full"` or raises `ValueError`. New `OnlineOutput` bound fields.
- Added a Linux musl (Alpine) release binary.

**R:**

- Added `quickcheck` properties for randomized `stats::loess()` parity, sorted output, robust iterations through 12 passes, and sparse one-spike initial fits; fixed regressions cover 12- and 24-iteration robust fits.
- Added an Alternative Software vignette with runnable Gaussian and robust comparisons to `stats::loess()` and a guide to LOESS-specific defaults.
- Added `cv_opts()` and the `cv` argument on `Loess()` for grouped Batch cross-validation.
- Added `outputs` to `Loess()`, `StreamingLoess()`, `OnlineLoess()`, and `predict.Loess()` for grouped optional results with mode-specific name validation; existing `return_*` arguments remain supported.
- Added `retain_model` and a `predict.Loess()` S3 method for out-of-sample prediction.
- Added `return_gradient` to `Loess()`, `StreamingLoess()`, and `OnlineLoess()`.
- Added `confidence_intervals`/`prediction_intervals`/`return_se` to `StreamingLoess()` and `OnlineLoess()`. `OnlineLoess()` requires `update_mode = "full"` or errors. New bound fields on `add_point()`'s result.
- Added stored golden reference fixtures under `tests/testthat/fixtures/` (with a `make_reference.R` regeneration script and a `PROVENANCE.txt` provenance stamp) pinning the output of LOESS engine paths with no external reference - the default `boundary_policy = "extend"`, intervals and gradients, robustness weights, Streaming, and Online - verified by `test-golden.R` within a `1e-10` tolerance (srrstats G5.4c).
- Added an explicit G5.9a test showing that `.Machine$double.eps`-scale noise in `y` does not meaningfully change the smoothed output.
- Added explicit RE7.0/RE7.0a tests for repeated and constant predictors and RE7.1/RE7.1a tests for noiseless exact predictor-response relationships, including exact-fit diagnostics and timing against noisy data. Removed the obsolete RE7 `@srrstatsNA` declaration and added the corresponding `@srrstats` claims in `R/srr-stats-standards.R` and test headers.

**WASM:**

- Added `cv` to the Batch options interface for grouped cross-validation configuration alongside legacy fields.
- Added `outputs` arrays to Batch, Streaming, Online, and prediction options for grouped optional result selection alongside existing booleans.
- Added `retain_model` and `LoessResult.predict(newX, options)` for out-of-sample prediction.
- Added `return_gradient` to `SmoothOptions`, `StreamingOptions`, and `OnlineOptions`.
- Added `confidence_intervals`/`prediction_intervals`/`return_se` to Streaming and Online options; Online requires `update_mode: "full"`. Added the corresponding `OnlineOutput` bound fields and TypeScript types.

### Changed

**Monorepo:**

- Updated the vendored `doxygen-awesome-css` theme to v2.5.0 and the Hugo docs build to v0.167.0.

**loess-rs:**

- Hoisted inline fully-qualified paths to top-level `use` imports; genuine name collisions stay qualified with a comment. No behavior changes.
- Flattened `tests/loess-rs/` into `tests/` directly: each test file is now its own integration test binary. No behavior changes.
- Bumped the vendored KaTeX CDN version from `0.18.7` to `0.18.9`, updating SRI hashes to match.
- Removed unused `pub use` re-exports with no consumer via that path. No behavior changes.
- Marked `WeightFunction` as non-exhaustive so downstream kernel must reject unsupported future variants explicitly.
- Matched R `stats::loess` span truncation, multivariate predictor normalization, and bisquare robustness cutoffs; MAR now uses R's uncentered median absolute residual and machine-minimum scale stop, while MAD remains the default. Near-singular local linear fits are handled by the regression solver rather than a global-range slope cutoff.

**fastLoess:**

- Hoisted inline fully-qualified paths to top-level `use` imports; genuine name collisions stay qualified with a comment. No behavior changes.
- Flattened `tests/fastLoess/` into `tests/` directly: each test file is now its own integration test binary. No behavior changes.
- Bumped the vendored KaTeX CDN version from `0.18.7` to `0.18.9`, updating SRI hashes to match.
- Removed unused `pub use` re-exports with no consumer via that path. No behavior changes.
- Replaced `std::mem::forget` with `Box::into_raw` in `vec_to_raw_ptr`, making the FFI ownership transfer explicit; bindings still release it through `free_raw_f64_buffer`.
- Implemented `std::error::Error` for `BindingError`.

**C++:**

- Breaking change: replaced the individual output booleans in `LoessOptions`, `OnlineOptions`, and `PredictOptions` with grouped `outputs` vectors; interval levels remain separate fields.
- C++ musl release jobs now build dynamic x86_64 and ARM64 shared libraries, allowing the musl assets to be published reliably.
- Hoisted inline fully-qualified Rust paths to top-level `use` imports; genuine name collisions stay qualified with a comment. No behavior changes.
- Declared the public wrapper's C++17 requirement and represented unavailable diagnostics as empty `std::optional<double>` values instead of NaN sentinels.

**Go:**

- Bumped the pinned `golangci-lint` install-script version from `v2.13.2` to `v2.14.0`.
- Hoisted inline fully-qualified Rust paths to top-level `use` imports; genuine name collisions stay qualified with a comment. No behavior changes.
- Represent unavailable diagnostic metrics as nil optional values instead of NaN sentinels in the Go binding.

**Java:**

- Bumped the pinned Checkstyle standalone jar version from `14.1.0` to `14.3.0`.
- Java's musl JNI release jobs now build dynamic x86_64 and ARM64 shared libraries, so the bundled resources selected by `NativeBridge` are published reliably.
- Hoisted inline fully-qualified Rust paths to top-level `use` imports; genuine name collisions stay qualified with a comment. No behavior changes.

**Julia:**

- Hoisted inline fully-qualified Rust paths to top-level `use` imports; genuine name collisions stay qualified with a comment. No behavior changes.
- Represent unavailable diagnostic metrics as `nothing` instead of `NaN` sentinels in the Julia binding.

**Node.js:**

- Hoisted inline fully-qualified Rust paths to top-level `use` imports; genuine name collisions stay qualified with a comment. No behavior changes.

**Python:**

- Hoisted inline fully-qualified Rust paths to top-level `use` imports; genuine name collisions stay qualified with a comment. No behavior changes.

**R:**

- Hoisted inline fully-qualified Rust paths to top-level `use` imports; genuine name collisions stay qualified with a comment. No behavior changes.
- Replaced the local `Result` alias with `extendr_api::error::Result`, mapped unavailable diagnostics to R `NA`, added retry cleanup for transient Windows `pak` move failures, and added the root/binding `r-tests` workflow.

**WASM:**

- Hoisted inline fully-qualified Rust paths to top-level `use` imports; genuine name collisions stay qualified with a comment. No behavior changes.

### Fixed

**Monorepo:**

- `dev/bump_version.py` now also updates the Go module's `/vN` major-version-suffix path and the Maven dependency example version, both previously left stale after a version bump.
- Aligned `OnlineLoess` defaults across the Rust core and bindings: `iterations` is now `0` with the default `update_mode = "incremental"`; positive robustness iterations require `update_mode = "full"`.
- Replaced the stale "JavaScript" binding label with "Node.js" in the Julia README and docs-site homepage.

**loess-rs:**

- Used standard ceiling division for multivariate normalization trimming so strict Clippy passes without changing the trim count.
- Cleaned up `loess_rs::prelude` of accidentally-leaked internals (`LoessBuilder`, adapter markers) — use the `Loess`/`StreamingLoess`/`OnlineLoess` type aliases directly.
- Matched LOWESS's effective-zero MAR stop and removed the absolute bisquare scale floor, while retaining the centered-MAD fallback.
- `make loess-rs-dev` now also runs `cargo test --doc`, previously never checked by any `make` target.
- Added the classical simple-linear-regression standard-error path for one-dimensional global fits (`fraction >= 1.0`), matching `stats::lm`'s `se.fit` formula.
- Corrected serial LOESS standard errors to use the local-linear equivalent-kernel leverage and kernel-corrected residual degrees of freedom, preserving positive SEs for downweighted observations. Added Monte Carlo calibration and interval edge-case regressions.
- Fixed seeded k-fold CV with unordered test queries: batch interpolation now locates each query bracket independently with binary search instead of relying on a monotone scan pointer.
- Fixed local-linear and global OLS regression on small-magnitude predictors by using scale-relative degeneracy checks instead of absolute x-variance thresholds. Added gradient and standard-error regressions for small x scales.
- Matched Cleveland/R's local-linear degeneracy rule in one-dimensional linear fits by suppressing slopes when weighted local spread is below `0.001` of the global x-range.
- Matched R's `1e-7` span-truncation adjustment instead of rounding near-integer neighborhoods with `1e-5`.
- Matched R's normalized adjusted-weight fitted-value accumulation without parity-, sparsity-, or response-scale-specific branches.
- Separated local-weight adjustment and fitted-response accumulation into R's original loop order, avoiding platform-dependent cancellation in sparse robust fits.
- Separated robustness scale scratch storage from local kernel weights so median selection cannot contaminate the next R-equivalent smoothing pass.
- Matched R's `w * ((x - mean_x) * (x - mean_x))` spread parenthesization, preserving cancellation-scale endpoint fits during robust passes.
- Matched R's even-length `cmad = 3 * (lower + upper)` operation order instead of scaling an averaged median.
- Extended local kernel scans beyond the nominal right window edge until R's `0.999 * h` cutoff, matching `lowest()` on asymmetric neighborhoods.
- Fixed k-fold cross-validation to pool every test point's squared error before taking one RMSE, matching LOOCV instead of averaging per-fold RMSEs.

**fastLoess:**

- `make fastLoess-dev` now also runs `cargo test --doc`, previously never checked by any `make` target.
- Fixed parallel direct 1D standard errors collapsing to zero for observations with zero robustness weight. The interval pass now uses the exact local-linear equivalent-kernel variance multiplier and kernel-corrected residual degrees of freedom, matching the serial calculation.

**C++:**

- Fixed `bindings/cpp/spack/package.py` building/installing from the wrong directory (`bindings/cpp` instead of the workspace-root `target/release`), which broke `spack install fastloess-cpp` on every platform. Now builds by package name. Also moved the pyright suppression out of the recipe into a new root `pyrightconfig.json`.
- Force-stage the tracked Spack recipe in the C++ release workflow so ignore rules cannot block automated version updates.
- Fixed the C++ valgrind memory check being silently skipped in Linux CI because valgrind was not installed. The Linux matrix, Clang, and Intel oneAPI jobs now install it, as does the Linux `bindings/cpp/Makefile` `install-tools` target.
- Fixed C++ doc-snippet verification skipping on Windows ARM by locating MSVC for the built library's target architecture, searching the ARM64 MSVC output directory, and caching `vcvarsall.bat` environments by script path and target architecture.

**Go:**

- Breaking: The Go module's import path now includes the required `/v2` major-version suffix; a new release is required for pkg.go.dev to resolve versions correctly.

**Java:**

- Completed Streaming/Online builder Javadocs so the strict `failOnWarnings` documentation build passes; the Makefile now surfaces warning details if the gate regresses.
- Fixed `cv_seed` silently accepting negative values and reinterpreting them as a huge unsigned seed instead of raising an error. Now validated before the cast.
- Fixed intermittent macOS `mvn clean test` resolution failures involving `commons-io:2.6` by pinning `maven-clean-plugin` to 3.5.0, which removes the old `maven-shared-utils`/`commons-io` dependency path.

**Julia:**

- Fixed Julia 1.13 FFI loading by switching native calls to tuple-based `ccall` with a plain-string library path.

**Node.js:**

- Fixed inconsistent naming of the Node.js binding as "JavaScript" across READMEs, doc-site home pages, and `CITATION.cff`.
- Fixed `cv_seed` silently accepting negative values by validating the signed input before converting it to `u64`.

**R:**

- Removed the `fnd` role from the individual maintainer in `Authors@R`; pkgcheck treats individual funder names as institutions and requires an institutional ROR.
- Fixed `cv_seed` silently accepting negative values and reinterpreting them as a huge unsigned seed instead of raising an error. Now validated before the cast.

## 2.0.0

### Added

**Monorepo:**

- Added an "Ideas for Contribution" section to `CONTRIBUTING.md`, listing concrete Batch/Streaming/Online feature gaps (out-of-sample prediction, per-point local gradients, adaptive fraction selection, `cell`/`interpolation_vertices` tuning, bootstrap intervals, GPU backend, concurrent chunk processing, checkpointable streaming state, `OnlineOutput.standard_error`, distance-based window eviction, configurable warm-up).
- Added `dev/bump_version.py --version X.Y.Z` to bump every crate/binding version file, `CITATION.cff`, the Spack recipe, and `CONTRIBUTING.md`'s example version in one pass (supports `--dry-run`).
- Added `dev/check_pinned_versions.py` and a weekly `check-versions.yml` to catch hardcoded version pins Dependabot can't see.
- Added `.github/dependabot.yml`, covering every dependency ecosystem in one weekly PR per directory.
- Added an optional `commit` input to release workflows' `workflow_dispatch` trigger, to pin the built commit for manual runs.
- Added `dev/check_links.py` to validate Markdown cross-reference links across all docs.

**loess-rs:**

- Added `.return_sorted()` to the batch builder, returning results sorted ascending by `x` (default `false`).
- Added a `missing` option (`"error"`/`"drop"`) for non-finite (NaN/Inf) values: Batch/Streaming drop non-finite rows and matching `custom_weights`; Online skips them via `Ok(None)`. Length mismatches always error.
- Added `release-rust.yml` to publish to crates.io on release.

**fastLoess:**

- Added `return_sorted` and `missing` to `BuilderOptionSet`/`TypedBuilderOptionSet` and the `Loess`/`StreamingLoess`/`OnlineLoess` builders.
- Published to crates.io via the `release-rust.yml` workflow.

**C++:**

- Added CMake package-config support (`find_package(fastloess)`) and CI coverage for `clang-cl`, `clang`, MinGW-w64, and Intel oneAPI.
- Added ARM64 release binaries for Linux/Windows/macOS; fixed the macOS x64 job silently shipping a mislabeled arm64 binary.
- Renamed `cpp_loess_fit`/`cpp_streaming_process`'s `x`/`y` params to `x_values`/`y_values`, avoiding a collision with `CppLoessResult`'s own fields.
- Added `return_sorted` and `missing` options to `LoessOptions`/`OnlineOptions`.

**Go:**

- Added a new Go binding (`bindings/go`): `cgo`-based `fastloess` package with `Loess`/`StreamingLoess`/`OnlineLoess` types (`StreamingOptions`/`OnlineOptions` each declare only the fields they support; Online always runs sequentially), a Hugo docs site, CI/release workflows, and full doc-snippet/test coverage.
- Added `ReturnSorted` and `Missing` options to `Options`/`StreamingOptions`/`OnlineOptions`.
- `WeightedMetricWeights` requires `DistanceMetric = "weighted"` to be set explicitly or returns an error.

**Java:**

- Added a new Java binding (`bindings/java`): JNI-based `fastloess` Maven package with `Loess`/`StreamingLoess`/`OnlineLoess` classes (LOESS-specific options like `degree`, `dimensions`, `distanceMetric`, `surfaceMode`, hat-matrix stats via `Result.hatMatrix()`; `StreamingOptions`/`OnlineOptions` each declare only the fields they support, and Online always runs sequentially), an Antora docs site, CI/release workflows, and full doc-snippet/test coverage.
- Added `returnSorted` and `missing` options to `Options`/`StreamingOptions`/`OnlineOptions`.
- `weightedMetricWeights` requires `distanceMetric("weighted")` to be set explicitly or throws an exception.

**Julia:**

- Added `return_sorted` and `missing` options to `Loess`, `StreamingLoess`, and `OnlineLoess`.

**Node.js:**

- Added `aarch64-unknown-linux-musl` and `armv7-unknown-linux-gnueabihf` prebuilt targets with matching optional npm subpackages.
- Added `return_sorted` and `missing` options to `Loess`'s `SmoothOptions`/`StreamingSmoothOptions`/`OnlineSmoothOptions`.

**Python:**

- Added `return_sorted` and `missing` options to `Loess`, `StreamingLoess`, and `OnlineLoess`.

**R:**

- Added `return_sorted` and `missing` options to `Loess()`, `StreamingLoess()`, and `OnlineLoess()`.

**WASM:**

- Added `return_sorted` and `missing` options to `Loess`'s `SmoothOptions`/`StreamingSmoothOptions`/`OnlineSmoothOptions`.

### Changed

**Monorepo:**

- Merged the standalone `dev/add-{cpp,rust,nodejs,wasm}-outputs` scripts into `dev/verify_snippets.py --update-outputs`.
- Harmonized the docs-site directory structure across every binding/crate, and fixed doc-tooling scripts that missed snippets in the newly-nested pages.
- Removed automatic changelog generation; maintain each binding/crate's `NEWS.md`/`news.md` by hand.
- Added `dev/add-readme-to-docs.py` to auto-embed `README.md` as the docs homepage (Starlight/Sphinx-aware); not yet wired into Python's `Makefile`.

**loess-rs:**

- Added a `large` benchmark category (exact-fit, high-iteration, high-fraction) to the Rust benchmarks.
- Replaced Unicode super/subscript stand-ins (for example, `R²` and `xᵢ`) with plain ASCII in crate docs and comments.
- Consolidated the loess-rs README (merging Installation/Documentation, dropping GitHub-only alert syntax, and removing sections covered by docs pages), and moved parameter docs into API option tables, removing `parameters.md`. Replaced `kernels.md`/`adapter-choice.md` mermaid flowcharts with tables because rustdoc does not render mermaid.
- Reorganized API documentation with field tables, per-field Options sections, and a Result Structure section at the end.
- Updated `wide` to v1.7.
- Removed the dead `compute_residuals`/`backend` fields from `OnlineLoessBuilder` (always computed/never read) and the unused `backend` field from `StreamingLoessBuilder`. `Backend` currently has only a `CPU` variant, read only by the Batch adapter as a GPU placeholder.
- Breaking: `Streaming::convert()` now resolves `overlap` dynamically to `chunk_size / 10` (clamped to `[1, chunk_size - 10]`) via `default_overlap()`; callers relying on the previous flat `500` default are affected.

**fastLoess:**

- Added a `large` benchmark category (exact-fit, high-iteration, high-fraction) to the Rust benchmarks.
- Replaced Unicode super/subscript stand-ins (for example, `R²` and `xᵢ`) with plain ASCII in crate docs and comments.
- Consolidated the fastLoess README (merging Installation/Documentation, dropping GitHub-only alert syntax, and removing sections covered by docs pages), and moved parameter docs into API option tables, removing `parameters.md`. Replaced `kernels.md`/`adapter-choice.md` mermaid flowcharts with tables because rustdoc does not render mermaid.
- Reorganized API documentation with field tables, per-field Options sections, and a Result Structure section at the end.
- Removed the dead `compute_residuals` and `backend` fields from `OnlineLoessBuilder` and the unused `backend` field from `StreamingLoessBuilder`. `Backend` currently has only a `CPU` variant, read only by the Batch adapter as a GPU placeholder.
- Breaking: Removed `.confidenceIntervals()`, `.predictionIntervals()`, and `.returnSe()` from the `StreamingLoess`/`OnlineLoess` wrapper structs (unused leftovers from the shared builder macro), and removed `parallel` from `OnlineLoess`, which now always runs sequentially. This affects direct Rust consumers; `Loess`/`StreamingLoess` are unaffected.
- Corrected a misleading comment on `binding_support::default_overlap()` to accurately describe its dynamic default-overlap formula. No behavior changed.

**C++:**

- Replaced Unicode super/subscript stand-ins (for example, `R²` and `xᵢ`) with plain ASCII in the C++ docs and comments.
- Documented `x_values`/`y_values` params (fixing a Doxygen warning) and restructured Doxygen nav from ~20 flat pages into 5 hub pages.
- Consolidated the C++ README (merging Installation/Documentation, dropping GitHub-only alert syntax, and removing sections covered by docs pages), renamed "When to Use" to "When to Use Batch Adapter", and moved parameter docs into API option tables, removing `parameters.md`.
- Vendored doxygen-awesome-css v2.4.2 for a modern Doxygen theme.
- Replaced the `kernels.md`/`adapter-choice.md` mermaid flowcharts with tables because Doxygen does not render mermaid.
- Reorganized API documentation with field tables, a per-field `## Options` section, and a `## Result Structure` section at the end.
- Updated docs to require explicitly setting `weighted_metric_weights`'s distance metric to `"weighted"` and to show the dynamic `StreamingOptions.overlap` default.
- Added a Spack recipe (auto-updated by `release-cpp.yml`) and bumped the vendored Corrosion CMake module to v0.6.1.
- Breaking: Removed the unused `confidence_intervals`, `prediction_intervals`, `return_diagnostics`, `return_residuals`, and `return_se` fields from standalone `OnlineOptions`, and removed `parallel` for consistency with `fastLowess`; Online now always runs sequentially.
- `StreamingOptions` no longer forwards `confidence_intervals`/`prediction_intervals`/`return_se` to the native constructor, since Streaming never computed them.
- Breaking: Changed `StreamingOptions::overlap` from a fixed `500` default to a sentinel (`-1`) that resolves dynamically to `chunk_size / 10`; callers relying on the previous flat default are affected.
- Breaking: `weighted_metric_weights` no longer auto-selects the `"weighted"` distance metric; callers must set it explicitly.

**Go:**

- Replaced Unicode super/subscript stand-ins (for example, `R²` and `xᵢ`) with plain ASCII in Go docs and comments.
- Consolidated the Go README (merging Installation/Documentation, dropping GitHub-only alert syntax, and removing sections covered by docs pages), renamed "When to Use" to "When to Use Batch Adapter", and moved parameter docs into API option tables, removing `parameters.md`.
- Reorganized API documentation with field tables, per-field Options sections, and a Result Structure section at the end; documented the explicit `weighted_metric_weights` distance-metric requirement and dynamic `overlap` default.

**Java:**

- Replaced Unicode super/subscript stand-ins (for example, `R²` and `xᵢ`) with plain ASCII in Java docs and comments.
- Consolidated the Java README (merging Installation/Documentation, dropping GitHub-only alert syntax, and removing sections covered by docs pages), renamed "When to Use" to "When to Use Batch Adapter", and moved parameter docs into API option tables, removing `parameters.md`.
- Reorganized API documentation with field tables, per-field Options sections, and a Result Structure section at the end; corrected the `api-online.adoc` disclaimer, dynamic `overlap` default, and explicit `weighted_metric_weights` distance-metric requirement.

**Julia:**

- Replaced Unicode super/subscript stand-ins (for example, `R²` and `xᵢ`) with plain ASCII in Julia docs and comments.
- Consolidated the Julia README (merging Installation/Documentation, dropping GitHub-only alert syntax, and removing sections covered by docs pages), renamed "When to Use" to "When to Use Batch Adapter", and moved parameter docs into API option tables, removing `parameters.md`.
- Reorganized API documentation with field tables, per-field Options sections, and a Result Structure section at the end; corrected constructor docstrings, dynamic `overlap` default, and explicit `weighted_metric_weights` distance-metric requirement.
- Breaking: Removed `confidence_intervals`, `prediction_intervals`, and `return_se` from `StreamingLoess` and `OnlineLoess`; also removed `return_diagnostics`, `return_residuals`, and `parallel` from `OnlineLoess`.
- Breaking: `weighted_metric_weights` now requires `distance_metric = "weighted"` explicitly.
- Breaking: Changed `StreamingLoess`'s `overlap` default from a fixed `500` to a dynamic `chunk_size / 10`.

**Node.js:**

- Replaced Unicode super/subscript stand-ins (for example, `R²` and `xᵢ`) with plain ASCII in Node.js docs and comments.
- Consolidated the Node.js README (merging Installation/Documentation, dropping GitHub-only alert syntax, and removing sections covered by docs pages), renamed "When to Use" to "When to Use Batch Adapter", and moved parameter docs into API option tables, removing `parameters.md`.
- Reorganized API documentation with field tables, per-field Options sections, and a Result Structure section at the end; documented the dynamic `overlap` default.
- Updated `oxlint`, `napi`/`napi-derive`/`@napi-rs/cli`/`napi-build`, and `typedoc-plugin-markdown`; `make nodejs-dev` now runs `npm update` after `npm install`.
- Breaking: `StreamingLoess`/`OnlineLoess` now use dedicated `StreamingSmoothOptions`/`OnlineSmoothOptions` types instead of Batch's `SmoothOptions`, dropping Batch-only fields (`confidence_intervals`, `prediction_intervals`, `return_se`, `cv_*`) from both and `return_diagnostics`/`return_residuals`/`parallel` from `OnlineSmoothOptions`. This affects TypeScript consumers; `Loess`'s `SmoothOptions` is unchanged.

**Python:**

- Replaced Unicode super/subscript stand-ins (for example, `R²` and `xᵢ`) with plain ASCII in Python docs and comments.
- Consolidated the Python README (merging Installation/Documentation, dropping GitHub-only alert syntax, and removing sections covered by docs pages), renamed "When to Use" to "When to Use Batch Adapter", and moved parameter docs into API option tables, removing `parameters.md`.
- Breaking: Removed `confidence_intervals`/`prediction_intervals`/`return_se` from `StreamingLoess()`, and those plus `return_diagnostics`/`return_residuals`/`parallel` from `OnlineLoess()`; neither adapter computed these options, and Online now always runs sequentially. `Loess`/`StreamingLoess` are unaffected.

**R:**

- Added a `large` benchmark category (exact-fit, high-iteration, high-fraction) to the R benchmarks.
- Replaced Unicode super/subscript stand-ins (for example, `R²` and `xᵢ`) with plain ASCII in R docs and comments.
- Consolidated the R README (merging Installation/Documentation, dropping GitHub-only alert syntax, and removing sections covered by docs pages), renamed "When to Use" to "When to Use Batch Adapter", and moved parameter docs into API option tables, removing `parameters.md`.
- Reorganized API documentation with field tables, per-field Options sections, and a Result Structure section at the end; documented the dynamic `overlap` default.
- Removed the redundant `rfastloess-package` pkgdown topic and the internal `Nullable()` helper.
- Fixed `_pkgdown.yml` mislabeling the S3-based interface as "R6 classes".
- Merged `parameters.Rmd`/`batch.Rmd`/`streaming.Rmd`/`online.Rmd` into the constructors' roxygen docs, removing the now-redundant vignettes.
- Breaking: Removed `confidence_intervals`/`prediction_intervals`/`return_se` from `StreamingLoess()`, and those plus `return_diagnostics`/`return_residuals`/`parallel` from `OnlineLoess()`; neither adapter computed these options, and Online now always runs sequentially. `StreamingLoess()`'s `parallel` is unaffected.

- Fixed `bindings/r/R/StreamingLoess.R`, which was corrupted (a duplicate of `OnlineLoess.R` with a mangled fragment appended), making `StreamingLoess()` uncallable. Reconstructed from `man/StreamingLoess.Rd`/`utils.R`, verified via `roxygen2::roxygenise()` and the full `testthat` suite (187 passed). Also fixed a stale `test-extendr-wrappers.R` fixture with 3 extra positional args.

**WASM:**

- Replaced Unicode super/subscript stand-ins (for example, `R²` and `xᵢ`) with plain ASCII in WASM docs and comments.
- Consolidated the WASM README (merging Installation/Documentation, dropping GitHub-only alert syntax, and removing sections covered by docs pages), renamed "When to Use" to "When to Use Batch Adapter", and moved parameter docs into API option tables, removing `parameters.md`.
- Reorganized API documentation with field tables, per-field Options sections, and a Result Structure section at the end; documented the dynamic `overlap` default.
- Updated `oxlint` and `typedoc-plugin-markdown`; `make wasm-dev` now runs `npm update` after `npm install`.
- Split Batch, Streaming, and Online option types into dedicated `SmoothOptions`, `StreamingSmoothOptions`, and `OnlineSmoothOptions` interfaces. Streaming options omit `confidence_intervals`, `prediction_intervals`, `return_se`, and `cv_*`; Online options also omit `return_diagnostics`, `return_residuals`, and `parallel`.

### Fixed

**Monorepo:**

- Fixed `CONTRIBUTING.md`'s example crate version (`0.9.0` → `1.2.0`).
- Fixed `release-conda.yml`'s version-line `sed` pattern to match any indentation.
- Fixed benchmark vendoring nulling every crate's checksum instead of just the two local path crates.
- Fixed the benchmark README's inaccurate "Iterations" scenario count.
- Fixed `docs.yml`'s Pages deployment: merged per-language jobs into one artifact upload/deploy job using `upload/deploy-pages` actions instead of legacy branch-based deployment.
- Fixed 51 broken doc cross-reference links left over from the docs-site restructure (found via `dev/check_links.py`).

**loess-rs:**

- Fixed `OnlineLoess`/`StreamingLoess` defaults to match docs: `min_points` changed from `3` to `2`, and `update_mode` from `"full"` to `"incremental"`.
- Fixed the direct Rust API's internal robustness-iteration defaults to match docs: Streaming `2`→`3` and Online `1`→`3`.
- Corrected docs to state that result `x` values follow input order after internal sorting and mapping back.
- Fixed the "Handling Outliers" quickstart example printing nothing at `fraction = 0.5` with only 6 points; bumped to `0.7`.
- Fixed LaTeX math rendering as literal text and cross-reference links not resolving against the rustdoc module tree, both on docs.rs.

**fastLoess:**

- Fixed `OnlineLoess`/`StreamingLoess` defaults to match docs: `min_points` changed from `3` to `2`, and `update_mode` from `"full"` to `"incremental"`.
- Fixed the direct Rust API's internal robustness-iteration defaults to match docs: Streaming `2`→`3` and Online `1`→`3`.
- Corrected docs to state that result `x` values follow input order after internal sorting and mapping back.
- Fixed the "Handling Outliers" quickstart example printing nothing at `fraction = 0.5` with only 6 points; bumped to `0.7`.
- Fixed cross-reference links not resolving against the rustdoc module tree and LaTeX math rendering as literal text, both on docs.rs.
- Added `#[allow(clippy::excessive_precision)]` to kernel constants.
- Fixed `build_streaming`/`build_online` to use named default constants instead of hardcoded numeric fallbacks for `chunk_size`, `window_capacity`, and `min_points`.

**C++:**

- Fixed `OnlineLoess`/`StreamingLoess` defaults to match docs: `min_points` changed from `3` to `2`, and `update_mode` from `"full"` to `"incremental"`.
- Corrected docs to state that result `x` values follow input order after internal sorting and mapping back.
- Fixed the "Handling Outliers" quickstart example printing nothing with only 6 points at `fraction = 0.5`; bumped to `0.7` so the outlier is actually downweighted.
- Fixed several Doxygen rendering bugs (wrong homepage, broken blockquotes/math/admonitions); `README.md` is now the native homepage.
- Fixed `ci-cpp.yml`'s untrusted Homebrew tap warning and a broken Windows `cppcheck` install.
- Fixed `Doxyfile`'s wrong `PROJECT_NAME` and a malformed `FILE_PATTERNS` glob.

**Go:**

- Fixed `CONTRIBUTING.md`'s stale Go prerequisite (`1.21+` → `1.23+`).
- Fixed `OnlineLoess`/`StreamingLoess` defaults to match docs: `min_points` changed from `3` to `2`, and `update_mode` from `"full"` to `"incremental"`.
- Corrected docs to state that result `x` values follow input order after internal sorting and mapping back.
- Fixed the "Handling Outliers" quickstart example printing nothing at `fraction = 0.5` with only 6 points; bumped to `0.7`.

**Java:**

- Fixed `OnlineLoess`/`StreamingLoess` defaults to match docs: `min_points` changed from `3` to `2`, and `update_mode` from `"full"` to `"incremental"`.
- Corrected docs to state that result `x` values follow input order after internal sorting and mapping back.

**Julia:**

- Fixed `OnlineLoess`/`StreamingLoess` defaults to match docs: `min_points` changed from `3` to `2`, and `update_mode` from `"full"` to `"incremental"`.
- Corrected docs to state that result `x` values follow input order after internal sorting and mapping back.
- Fixed the "Handling Outliers" quickstart example printing nothing at `fraction = 0.5` with only 6 points; bumped to `0.7`.
- Fixed `intervals.md` examples looping over all 100 points instead of a short sample.
- Fixed the Documenter homepage being a stale, separately-maintained `index.md`; now regenerated from `README.md` on every build.
- Fixed `release-julia-register.yml` pulling release notes from the full changelog instead of the Julia-filtered `NEWS.md`.
- Fixed `make julia-dev` resolving an outdated `fastloess_jll`, Windows mojibake in `dev/runners/julia.py`, inconsistent tab indentation, and stale `lowess-project` links.
- Added a missing custom-weights test case to the Julia suite.
- Fixed `cell`/`interpolation_vertices`/`boundary_degree_fallback`/`cv_seed` being silently non-functional due to no-op FFI setters, and `jl_streaming_loess_new` wrapping negative `dimensions` instead of clamping to 1.
- Simplified redundant null-pointer comparisons in `FastLOESS.jl`.

**Node.js:**

- Fixed `OnlineLoess`/`StreamingLoess` defaults to match docs: `min_points` changed from `3` to `2`, and `update_mode` from `"full"` to `"incremental"`.
- Corrected docs to state that result `x` values follow input order after internal sorting and mapping back.
- Fixed the "Handling Outliers" quickstart example printing nothing with only 6 points at `fraction = 0.5`; bumped to `0.7` so the outlier is actually downweighted.
- Fixed the docs homepage never showing README content.
- Fixed an `@astrojs/sitemap` warning, TypeDoc/Starlight "API Reference" 404s, and an `astro build` failure from a missing dependency.

**Python:**

- Fixed `OnlineLoess`/`StreamingLoess` defaults to match docs: `min_points` changed from `3` to `2`, and `update_mode` from `"full"` to `"incremental"`.
- Corrected docs to state that result `x` values follow input order after internal sorting and mapping back; strengthened `test_unsorted_input` to assert this.
- Fixed the "Handling Outliers" quickstart example printing nothing at `fraction = 0.5` with only 6 points; bumped to `0.7`.
- Fixed the empty "API Reference" page (stale toctree references).
- Fixed noisy pip version-check output in `release-pypi.yml` and a Pyright false-positive warning.
- Converted 3 plain comments to doc comments and added 2 custom-weights test cases.

**R:**

- Fixed `bindings/r/Makefile`'s Air auto-install target (`make r` → `make r-dev`).
- Fixed the R benchmark script calling `fit` as a field instead of the S3 generic `fit(model, x, y)`.
- Fixed `OnlineLoess`/`StreamingLoess` defaults to match docs: `min_points` changed from `3` to `2`, and `update_mode` from `"full"` to `"incremental"`.
- Corrected docs to state that result `x` values follow input order after internal sorting and mapping back.
- Fixed the "Handling Outliers" quickstart example printing nothing at `fraction = 0.5` with only 6 points; bumped to `0.7`.
- Fixed two roxygen examples printing too much/nothing (`OnlineLoess()`, `add_point()`).
- Reformatted `configure` to tabs and fixed `.Rbuildignore` missing exclusions.
- Removed the empty `R/params.R` stub and simplified `plot.LoessResult()` to return `NULL` invisibly.
- Inlined the `.make_*` constructor helpers, and consolidated `utils.R`'s parameter validators into two generic helpers.
- Fixed `utils.R`'s internal `validate_min_points()` guard hardcoding a stricter minimum of 3 points, diverging from the Rust core's actual minimum of 2; lowered to 2.

**WASM:**

- Fixed `OnlineLoess`/`StreamingLoess` defaults to match docs: `min_points` changed from `3` to `2`, and `update_mode` from `"full"` to `"incremental"`.
- Corrected docs to state that result `x` values follow input order after internal sorting and mapping back.
- Fixed the "Handling Outliers" quickstart example printing nothing at `fraction = 0.5` with only 6 points; bumped to `0.7`.
- Fixed the docs homepage not displaying the README, an `@astrojs/sitemap` warning, TypeDoc/Starlight "API Reference" 404s, and an `astro build` failure caused by a missing dependency.
- Fixed `concepts.md` figures not rendering and LaTeX math rendering as literal text.
- Fixed generated `.d.ts` doc comments showing literal backslashes instead of quotes.
- Fixed the generated `onlineoptions.md` TypeDoc page stating stale defaults (`min_points: 3`, `update_mode: "full"`) that no longer matched the already-correct source.

## 1.1.0

### Added

**Monorepo:**

- Added a GitHub Pages landing page at the repository root, built from `README.md` via pandoc and deployed by `docs.yml`.
- Added a GitHub workflow for running validation scripts.

**C++:**

- Added clang-tidy and cppcheck installation to Makefile.

**Julia:**

- `release-julia-register.yml` now automatically extracts the matching changelog section and appends it as release notes in the JuliaRegistrator comment, enabling auto-merge on major version bumps.

**Node.js:**

- Added `npm run lint` to the `Lint` step in `ci-nodejs.yml`, so JavaScript source and test files are linted via `oxlint` on every CI run.

**R:**

- Added `lenght` gaurds for extra arguments.

**WASM:**

- Added `npm run lint` to the `Lint` step in `ci-wasm.yml`, so JavaScript source and test files are linted via `oxlint` on every CI run.

### Changed

**Monorepo:**

- Moved the CHANGELOG and CONTRIBUTING guides to the project root.
- Split the monolithic `.github/workflows/ci.yml` into seven per-language workflow files: `ci-rust.yml`, `ci-python.yml`, `ci-julia.yml`, `ci-nodejs.yml`, `ci-wasm.yml`, `ci-cpp.yml`, and `ci-r.yml`. Each file carries the relevant `ci` (multi-OS matrix), `asan`, and `gpu` jobs for its language.
- Each crate/binding sub-Makefile now runs `dev/verify_snippets.py --lang <lang>` for its own language as the final step of `make default`. The root `docs-test` target remains as a convenience to run all languages at once.
- Split every sub-Makefile `default:` into `default:` (build and system install) and `dev:` (full quality-check workflow). Both root Makefiles gain `<name>-dev` targets for each binding and crate, and an `all-dev` aggregate target.
- Split `dev/verify_snippets.py` into a lean orchestrator and a `dev/runners/` package. Each language has its own module (`python.py`, `julia.py`, `nodejs.py`, `r.py`, `wasm.py`, `rust.py`, `cpp.py`) containing its `run_<lang>()` function and a `skip_reason()` predicate. Shared types (`Snippet`, `RunResult`) and utilities live in `runners/base.py`; the registry (`RUNNERS`, `SKIP_CHECKS`) is exported from `runners/__init__.py`.

**loess-rs:**

- Updated the loess-rs README to be crate-specific instead of using a generic README shared across bindings/crates.
- Moved crate documentation from ReadTheDocs to <https://docs.rs/loess-rs>.
- `make loess-rs` (`default:`) now only runs `cargo build`. The full dev workflow moves to `make loess-rs-dev`.

**fastLoess:**

- Updated the fastLoess README to be crate-specific instead of using a generic README shared across bindings/crates.
- Moved crate documentation from ReadTheDocs to <https://docs.rs/fastLoess>.
- `make fastLoess` (`default:`) now only runs `cargo build`. The full dev workflow moves to `make fastLoess-dev`.

**C++:**

- Updated the C++ README to be binding-specific instead of using a generic README shared across bindings.
- Moved C++ documentation from ReadTheDocs to GitHub Pages, served by Doxygen at <https://thisisamirv.github.io/loess-project/cpp/>. The ReadTheDocs site no longer includes C++-specific content.
- `make cpp` (`default:`) now only runs `cargo build`. The full dev workflow (formatting, linting, cbindgen idempotency, symbol export verification, cmake tests, valgrind, doc-snippet verification) moves to `make cpp-dev`.

**Julia:**

- Updated the Julia README to be binding-specific instead of using a generic README shared across bindings.
- Moved Julia documentation from ReadTheDocs to GitHub Pages, served by Documenter.jl at <https://thisisamirv.github.io/loess-project/julia/stable/>. The ReadTheDocs site no longer includes Julia-specific content. Code blocks use Documenter.jl `@example` sections, which execute and embed output automatically during the docs build.
- `make julia` (`default:`) now builds the Rust library and installs the Julia package via `Pkg.develop`. The full dev workflow moves to `make julia-dev`.

**Node.js:**

- Updated the Node.js README to be binding-specific instead of using a generic README shared across bindings.
- Moved Node.js documentation from ReadTheDocs to GitHub Pages, served by Starlight at <https://thisisamirv.github.io/loess-project/nodejs/>. The ReadTheDocs site no longer includes Node.js-specific content. `dev/add-nodejs-outputs.js` runs as part of `make nodejs-dev`, executing each JavaScript code block in the docs and injecting its output back into the Markdown source.
- `make nodejs` (`default:`) now builds the native addon and links it globally via `npm link`. The full dev workflow moves to `make nodejs-dev`.
- Updated `oxlint` dependency to 1.80.

**Python:**

- Updated the Python README to be binding-specific instead of using a generic README shared across bindings.
- Migrated Python documentation from MkDocs to Sphinx (with MyST-Parser and jupyter-sphinx). Code blocks now execute and embed output automatically via `jupyter-sphinx`.
- `make python` (`default:`) now installs to the user Python environment via `pip install --user`. The full dev workflow (venv setup, formatting, linting, testing, doc-snippet verification) moves to `make python-dev`.

**R:**

- Updated the R README to be binding-specific instead of using a generic README shared across bindings.
- Moved R documentation from ReadTheDocs to GitHub Pages, served by pkgdown at <https://thisisamirv.github.io/loess-project/r/>. The ReadTheDocs site no longer includes R-specific content.
- Simplified `bindings/r/Makefile`: replaced `Cargo.toml.orig` save/restore vendoring with `src/vendor-update.sh`; made `[workspace]` permanent in `src/Cargo.toml`; removed Bioconductor dependencies, redundant `cargo fmt --check`, `NAMESPACE` indentation post-processing, and `pkgdown::build_site` from the dev workflow.
- Changed R version dependency to 4.4.0 due to issues with installing Bioconducter packages on R < 4.4.0.
- Replaced the multi-step `install.packages` / `BiocManager::install` package installation logic in `bindings/r/Makefile` with a single [`pak`](https://pak.r-lib.org/)-based block. `pak` handles RSPM binary vs source selection automatically (including Linux), skips already-installed packages, and installs CRAN, Bioconductor (`bioc::` prefix), and R-universe packages in one call.
- `make r` (`default:`) now runs `R CMD INSTALL $(R_DIR)` directly; R's `configure` script handles Rust compilation from the committed `vendor.tar.xz`. The full dev workflow moves to `make r-dev`.

**WASM:**

- Updated the WASM README to be binding-specific instead of using a generic README shared across bindings.
- Moved WASM documentation from ReadTheDocs to GitHub Pages, served by Starlight at <https://thisisamirv.github.io/loess-project/wasm/>. The ReadTheDocs site no longer includes WASM-specific content. `dev/add-wasm-outputs.js` runs as part of `make wasm-dev`, executing each JavaScript code block in the docs and injecting its output back into the Markdown source.
- `make wasm` (`default:`) now builds both the Node.js and web WASM targets and links the Node.js package globally via `npm link`. The full dev workflow moves to `make wasm-dev`.
- Updated `oxlint` dependency to 1.80.
- Replace the outdated `jetli/wasm-pack-action` workflow with `taiki-e/install-action`.

### Fixed

**Monorepo:**

- Fixed `.cargo/config.toml` hardcoding absolute `c:/rtools45/...` paths for the `x86_64-pc-windows-gnu` linker and ar tool; it now uses bare tool names resolved via `PATH`.

**loess-rs:**

- Improved `MismatchedInputs` error: added a `dimensions` field and updated the message to show the expected x length (`y_len × dimensions`), making it self-explanatory for both 1-D and multi-dimensional mismatches.

**C++:**

- Fixed `make cpp` Windows CI failure (`cannot find -lgcc_eh`): the C++ binding's Makefile detected MinGW via `gcc -dumpmachine` and selected the GNU target, which then used the Rtools cross-compiler from the workspace `.cargo/config.toml`; that compiler delegated to `C:\mingw64\bin\ld.exe`, which lacks `lgcc_eh`. Fixed by always targeting `x86_64-pc-windows-msvc` on Windows, removing the MinGW detection branch entirely.

**Julia:**

- Fixed `fit(l::Loess, x::Matrix{Float64}, y)` not validating that `size(x, 2) == l.dimensions` before flattening the matrix. If the column count differed from the configured dimensions, the library either silently used wrong data or produced a confusing C-level error. The `Loess` struct now stores `dimensions` as a field, and the matrix overload checks `size(x, 2) != l.dimensions` upfront with a clear message naming the parameter to fix.
- Fixed `FastLOESS.jl` never actually loading the prebuilt `fastloess_jll` binary: `find_library()` only checked the `FASTLOESS_LIB` env var and local dev-mode paths, so the package installed from the registry had no working native library for end users. Added the `fastloess_jll` dependency (`Project.toml`), a JLL-loading branch in `find_library()`, and switched from an eager `const libfastloess = find_library()` (resolved once at precompile time) to a lazy `current_library()` accessor re-resolved in `__init__()`.

**Python:**

- Enforced keyword-only arguments beyond the first positional allowance in `Loess`, `StreamingLoess`, and `OnlineLoess`: `Loess(fraction, *, ...)`, `StreamingLoess(fraction, chunk_size, *, ...)`, and `OnlineLoess(fraction, window_capacity, min_points, *, ...)`. Updated the `.pyi` stubs accordingly.

**R:**

- Fixed Windows arm64 (R-Universe) build: `ar x` without a member name correctly resolves long-name archive entries (>16 chars stored as `/<offset>`); named extraction silently fails for such entries. Used `objcopy --remove-section=.idata$4` on each extracted `.dll` stub to strip the invalid relocations that lld 19 rejects, then `ar r` to re-insert.
- Fixed `ld.lld` crashing or dropping symbols (`WakeByAddressSingle`, `WaitOnAddress`) on Windows arm64: `--whole-archive` pulls every crate's raw-dylib stub for a given DLL into the link, but different crates' stubs cover different, non-overlapping symbols of that DLL — `--allow-multiple-definition` works on x86_64 but crashes lld's arm64pe backend. Fixed by dropping `--whole-archive` on `gnullvm`; normal archive resolution applies and nothing is lost since `entrypoint.c` already references the extendr init symbol directly.
- Fixed CRAN Windows build (`error: linker not found`): `cargo-config.toml` hardcoded linker/ar as `c:/rtools45/...` absolute paths, which break when Rtools is installed on a different drive. Fixed by using bare tool names resolved via `PATH`.
- Fixed CRAN Windows build (`cannot find -lgcc_eh`): the Rtools gcc lib directory is not writable on CRAN's server, and config-file `rustflags` does not reach build-script linker invocations. `Makevars.win` creates an empty stub via `touch` in `$(TARGET_DIR)/libgcc_mock/` and passes `LIBRARY_PATH` inline on `cargo build`. The path is resolved to an absolute path via `$(pwd)` at shell execution time — a relative path silently fails because Cargo invokes GCC to link build scripts from its own temp directory, not from `src/`.
- Fixed `Loess(fraction = 0.3, 4)` incorrectly succeeding: `reject_extra_positional_args()` counted unnamed arguments but did not check their position, so a single unnamed arg in any non-first slot passed validation. The check now rejects any unnamed argument that is not in position 1.
- Fixed `fit()` and `process_chunk()` silently flattening a matrix `x` and producing a confusing Rust-level length-mismatch error when `dimensions` was not set to match `ncol(x)`. Both methods now raise an informative error at the R level, naming the `dimensions` parameter to fix.

## 1.0.0

### Added

**R:**

- Introduced S3 generics `fit()`, `process_chunk()`, `finalize()`, and `add_point()`, replacing the previous list-closure API.
- `bindings/r/Makefile` now auto-installs [Air](https://posit-dev.github.io/air/) if missing, before running `air format`.
- Added a `reject_extra_positional_args()` helper to reject extra unnamed arguments.

### Fixed

**Julia:**

- Fixed `LoessResult.iterations_used` returning the raw FFI sentinel `-1` instead of `nothing` when robustness iterations were not applicable.

**R:**

- Fixed incorrect URLs in R binding docs.

**WASM:**

- Fixed `OnlineLoess.add_point()` returning `undefined` instead of `null` when the sliding window has not yet accumulated enough points.

### Changed

**Monorepo:**

- Moved the `tutorials/` pages into a new `user-guide/use-cases/` section.
- Removed `dev/isolate_cargo.py`, `dev/check_root_cargo.py`, `dev/fix_doc_snippets.py`, and `check_js_licenses.js` — workspace isolation, doc-snippet transformation, and license checks are no longer needed.
- Split the monolithic root `Makefile` into per-crate/binding sub-Makefiles (e.g. `crates/loess-rs/Makefile`, `bindings/r/Makefile`), each invokable directly via `make -f path/Makefile`. The root `Makefile` now only aggregates (`docs`, `check-msrv`, `all*`).
- Moved Rust and binding tests into their respective crate/binding directories (e.g. `tests/loess-rs/` → `crates/loess-rs/tests/loess-rs/`, `tests/cpp/` → `bindings/cpp/tests/`). Removed the standalone `tests/` workspace packages.

**loess-rs:**

- Split `StreamingLoess`/`OnlineLoess` content into dedicated API reference pages and standardized API examples with expected output comments.
- Breaking: Renamed `OnlineOutput`'s `smoothed` and `std_error` fields to `y` and `standard_error`, matching `LoessResult`.
- Updated `wide` to v1.6.

**fastLoess:**

- Split `StreamingLoess`/`OnlineLoess` content into dedicated API reference pages and standardized API examples with expected output comments.

**C++:**

- Split `StreamingLoess`/`OnlineLoess` content into dedicated API reference pages and standardized API examples with expected output comments.
- Breaking: Renamed `OnlineOutput`'s `smoothed()` and `std_error()` methods to `y()` and `standard_error()`.

**Julia:**

- Split `StreamingLoess`/`OnlineLoess` content into dedicated API reference pages and standardized API examples with expected output comments.
- Breaking: Renamed `OnlineOutput`'s `smoothed` and `std_error` fields to `y` and `standard_error`.
- Removed `dev/format_julia.jl`; formatting is now inlined in `bindings/julia/Makefile`.

**Node.js:**

- Split `StreamingLoess`/`OnlineLoess` content into dedicated API reference pages and standardized API examples with expected output comments.
- Breaking: Renamed `OnlineOutput`'s `smoothed` and `std_error` fields to `y` and `standard_error`.
- Updated `@napi-rs/cli` to v3.8 and `oxlint` to v1.79.

**Python:**

- Split `StreamingLoess`/`OnlineLoess` content into dedicated API reference pages and standardized API examples with expected output comments.
- Breaking: Renamed `OnlineOutput`'s `smoothed` and `std_error` properties to `y` and `standard_error`.

**R:**

- Split `StreamingLoess`/`OnlineLoess` content into dedicated API reference pages and standardized API examples with expected output comments. `dev/verify_snippets.py` now also runs the R code chunks in vignettes.
- Breaking: Renamed the `smoothed` and `std_error` fields returned by `OnlineLoess`'s `add_point()` to `y` and `standard_error`.
- Replaced `dev/style_pkg.R` with [Air](https://posit-dev.github.io/air/) for formatting.
- Removed `dev/fix_rd_style.R`, `dev/prepare_cargo.py`, `dev/patch_vendor_crates.py`, `dev/clean_checksums.py`, and `dev/prepare_cran.sh` — their logic is now inlined directly in `bindings/r/Makefile`, so the R build no longer requires any Python scripts.
- Added `...` to `Loess()`, `StreamingLoess()`, and `OnlineLoess()` to force named arguments for optional parameters.
- Added `Depends: R (>= 4.6)` to `DESCRIPTION` and a matching CI matrix entry.
- Expanded roxygen2 `@param` docs and added a `See Also` section linking to <https://loess.readthedocs.io/>.
- Expanded `rfastloess-intro.Rmd` vignettes.

**WASM:**

- Split `StreamingLoess`/`OnlineLoess` content into dedicated API reference pages and standardized API examples with expected output comments.
- Breaking: Renamed `OnlineOutput`'s `smoothed` and `std_error` getters to `y` and `standard_error`.
- Updated `oxlint` to v1.79.

## 0.9.0

### Added

**Monorepo:**

- Added Python, R, WASM, Node.js, C++, and Julia bindings.

**loess-rs:**

- Added the option to pass custom weights by the user to the algorithm.

**fastLoess:**

- Added the option to pass custom weights by the user to the algorithm.

### Changed

**Monorepo:**

- Implement monorepo structure.
- Converted all documentation tables to compact single-space format.
- Moved `BENCHMARKS.md`, `CHANGELOG.md`, and `CONTRIBUTING.md` from the repository root into `docs/` and added them to the documentation site navigation.

**loess-rs:**

- Added `Loess<T>`, `StreamingLoess<T>`, and `OnlineLoess<T>` type aliases as the primary user-facing constructors (e.g. `StreamingLoess::new().chunk_size(50).build()`). Mode-specific builder methods (`chunk_size`, `overlap`, `window_capacity`, `min_points`, `update_mode`) are now called directly on the type alias rather than after `.adapter()`.
- Breaking: Made `BatchLoessBuilder`, `StreamingLoessBuilder`, and `OnlineLoessBuilder` internal-only, removing their public setter methods. Smoothing configuration now flows through `LoessBuilder<T, Mode>`; code that called setters on an adapter builder must migrate.
- Breaking: Changed enum-typed builder methods (`weight_function`, `robustness_method`, `scaling_method`, `boundary_policy`, `zero_weight_fallback`, `merge_strategy`, and `update_mode`) to accept strings as well as enum variants through `impl IntoEnum<T>`; callers passing enum variants directly must update.
- Added a `parse` module defining `IntoEnum<E>` and macro-generated impls for all enum-typed builder parameters; builder methods accept typed enum values or string names such as `"tricube"`.
- Breaking: Replaced `cross_validate(CVConfig)` with the string-based `.cv_method(...)`, `.cv_k(...)`, `.cv_fractions(...)`, and `.cv_seed(...)` API; `KFold` and `LOOCV` are no longer exported from the prelude, so callers using the old API must migrate.
- Removed `smooth()`, `smooth_streaming()`, and `smooth_online()` convenience function stubs from `_core.pyi`.

**fastLoess:**

- Added `Loess<T>`, `StreamingLoess<T>`, and `OnlineLoess<T>` type aliases as the primary user-facing constructors (e.g. `StreamingLoess::new().chunk_size(50).build()`). Mode-specific builder methods (`chunk_size`, `overlap`, `window_capacity`, `min_points`, `update_mode`) are now called directly on the type alias rather than after `.adapter()`.
- Breaking: Made `BatchLoessBuilder`, `StreamingLoessBuilder`, and `OnlineLoessBuilder` internal-only, removing their public setter methods. Smoothing configuration now flows through `LoessBuilder<T, Mode>`; code that called setters on an adapter builder must migrate.
- Breaking: Changed enum-typed builder methods (`weight_function`, `robustness_method`, `scaling_method`, `boundary_policy`, `zero_weight_fallback`, `merge_strategy`, and `update_mode`) to accept strings as well as enum variants through `impl IntoEnum<T>`; callers passing enum variants directly must update.
- Added a `parse` module defining `IntoEnum<E>` and macro-generated impls for all enum-typed builder parameters; builder methods accept typed enum values or string names such as `"tricube"`.
- Breaking: Replaced `cross_validate(CVConfig)` with the string-based `.cv_method(...)`, `.cv_k(...)`, `.cv_fractions(...)`, and `.cv_seed(...)` API; `KFold` and `LOOCV` are no longer exported from the prelude, so callers using the old API must migrate.
- Removed `smooth()`, `smooth_streaming()`, and `smooth_online()` convenience function stubs from `_core.pyi`.

**C++:**

- Updated `.clang-tidy` to configure `lower_case` as the required naming convention for functions and member functions, matching the new snake_case public API.

## 0.2.2

### Fixed

**loess-rs:**

- Updated license badge.
- Fixed LOESS mechanism figure path.

## 0.2.1

### Added

**loess-rs:**

- Added visual validation to the bench branch.

### Changed

**loess-rs:**

- Reduced figures size significantly.
- Implement naming consistency for `auto_converge` (removed `auto_convergence`).

### Fixed

**loess-rs:**

- Fixed `boundary_degree_fallback` pass to online and streaming adapters.
- Fixed `boundary_degree_fallback` pass to `custom_vertex_pass` and `VertexPassFn`.
- Fixed KFold CV bug through adding explicit sorting of training subsets and using robust binary-search interpolation for each test point.
- Fixed `auto_converge` support for Online adapter.

## 0.2.0

### Added

**loess-rs:**

- Added `VertexPassFn` and `custom_vertex_pass` support to enable parallelized/accelerated interpolation fitting.
- Added support for custom vertex pass callbacks to all adapters (`Batch`, `Streaming`, `Online`).
- Added support for custom parallel/accelerated standard error calculation via `custom_interval_pass`.
- Added `KDTreeBuilderFn` and `custom_kdtree_builder` hook to enable external parallel KD-tree construction.
- Added `KDTree::from_parts` and exposed `KDNode` and `KDTree::calculate_left_subtree_size` to support custom tree building.
- Added neighborhood caching in `InterpolationSurface` to significantly optimize performance during robustness iterations.
- Added configurable `boundary_degree_fallback` option to control polynomial degree reduction at boundary vertices during interpolation. Defaults to `true` for stability; set to `false` to match R's `loess` behavior exactly.

### Changed

**loess-rs:**

- Changed license from AGPL-3.0-or-later to dual MIT OR Apache-2.0.
- Expanded `SmoothPassFn`, `CVPassFn`, and `IntervalPassFn` signatures to include full multi-dimensional context (dimensions, scaling, polynomial degree, etc.).
- Improved data propagation in `InterpolationSurface` to ensure all necessary coordinate and value slices are available to custom pass implementations.
- Updated `LoessExecutor` to correctly handle augmented data when switching between direct and interpolation modes.
- Updated `InterpolationSurface::build` to accept and propagate `polynomial_degree`, `weight_function`, `zero_weight_fallback`, `distance_metric`, and `scales` for `custom_vertex_pass`. Also, updated `LoessExecutor` to pass these configured values correctly.
- Improved documentation.

### Fixed

**loess-rs:**

- Fixed a potential crash in parallel interpolation refinement by correctly propagating augmented data slices to vertex fitting functions.
- Fixed inconsistent parameter types in custom pass callbacks.
- Fixed missing setters for online and streaming adapters.
- Fixed incorrect standard error propagation in `BatchLoessBuilder`.
- Added `Boundary Linear Fallback` strategy to `InterpolationSurface` to prevent numerical instability ("explosions") at data boundaries when using high-degree polynomials (Quadratic, Cubic, Quartic).
- Fixed missing `max_distance` update in the KD-Tree search, which incorrectly calculated the bandwidth for tricube weights.
- Fixed cumulative cross-contamination in regression buffers, which were not being zeroed between query points.
- Delegated 2D Cubic and 3D Quadratic from context to specialized accumulators.
- Fixed horizontal phase shift in `Interpolation` mode when using boundary policies (`Extend`, `Reflect`, `Zero`). The robustness iteration loop was incorrectly using augmented data indices instead of original data for query point evaluation.

## 0.1.0

### Added

**loess-rs:**

- Initial release.

**fastLoess:**

- Initial release with parallel execution support.

**Python:**

- Added the python binding for `fastLoess`.
