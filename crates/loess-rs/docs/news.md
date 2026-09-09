<!-- markdownlint-disable MD024 MD025 -->
# loess-rs Unreleased

## Added

* Added a `return_gradient` option to the Batch, Streaming, and Online adapters' builders: each local polynomial fit (degree >= linear) already computes per-dimension coefficients internally via `RegressionContext::fit_with_coefficients()`, but only the fitted value was normally kept; `.return_gradient()` exposes that per-point gradient (`dimensions` values per point, flattened in Batch/Streaming) in `LoessResult::gradient` (Batch/Streaming) or `OnlineOutput::gradient` (Online, the latest point's gradient), enabling sensitivity/rate-of-change analysis at effectively no extra computation cost. In Streaming, gradient values in the overlap region are merged across chunk boundaries the same way `y` is, via `merge_strategy`. Only supported when `surface_mode` is `"direct"` — the default `"interpolation"` mode only stores value+gradient at a sparse grid of vertices, not enough to reconstruct an exact per-point gradient, so `gradient` stays `None` there (same limitation as the existing exact `leverage`/`standard_errors`). `false` by default.
* Added out-of-sample prediction to the Batch adapter: `.retain_model(true)` on the builder retains the fitted model's (boundary-padded) training data, final robustness weights, residual SD, and normalization scales, enabling `Predict::new()...build()?` and `.call(&result, new_x)` to evaluate the local polynomial fit at arbitrary out-of-sample query points not in the training set (like R's `predict.loess(model, newdata)`). Supports the full nD / polynomial-degree / distance-metric generality of the Batch adapter, reusing the same `RegressionContext` and `KDTree` neighbor search used during fitting; `new_x` is flattened (`dimensions` values per query point). `Predict` is a fluent, string-based builder (an alias for `PredictBuilder`, mirroring `Loess`/`LoessBuilder`'s convention, e.g. `.extrapolation("linear")`) controlling `return_se`/`confidence_intervals`/`prediction_intervals` (same z-score convention as `fit()`'s existing intervals, and same naming as `Loess`'s own `confidence_intervals`/`prediction_intervals` builder methods; prediction intervals widen using the same `sqrt(RSS / delta1)` residual scale as `LoessResult::residual_scale` when available — i.e. `.return_se()` plus an interval method were set under `.surface_mode("direct")` — otherwise a MAD-based fallback), `return_derivative` (the local fit's gradient — `dimensions` values per query point — via `RegressionContext::fit_with_coefficients()`), and `extrapolation` (`"clamp"` default: clamps each out-of-range dimension to its training boundary; `"linear"`: first-order Taylor expansion from that boundary point's own gradient; `"error"`: fails with the new `LoessError::PredictOutOfRange` if any dimension falls outside the training range). `.build()` is mandatory: it validates (failing fast on an invalid string) and produces the ready-to-call configuration, which has no public constructor of its own. `.call()` returns a `PredictOutput` struct and `LoessError::PredictionUnavailable` if called without `.retain_model(true)`. Off by default (no extra memory/clone cost unless requested). `Predict` is exported from `loess_rs::prelude`.
* Added Linux musl (Alpine) release binaries alongside the existing glibc ones: Python (`release-pypi.yml` now publishes `musllinux_1_2` wheels for x86_64/aarch64), C++ (`release-cpp.yml` builds natively inside `alpine:latest` containers on `ubuntu-latest`/`ubuntu-24.04-arm`, publishing `libfastloess-linux-{x64,arm64}-musl.so`), Go (`release-go.yml`, same container approach, publishing `libfastloess_go-linux-{x64,arm64}-musl.a`), and Julia (removed the `libc(p) != "musl"` filter from `dev/build_tarballs_julia.jl`, letting Yggdrasil build musl JLLs again). GPU wheels/libraries (`release-gpu.yml`) are not covered by this change. Java is intentionally left as-is (no prebuilt natives for any platform yet).
* Added prebuilt native libraries for the Java binding: `release-java.yml` now builds `fastloess_java` for `linux-x86_64`, `linux-x86_64-musl` (Alpine), `linux-aarch64`, `linux-aarch64-musl` (Alpine), `macos-x86_64`, `macos-aarch64`, `windows-x86_64`, and `windows-aarch64`, and bundles all eight into the published jar under `src/main/resources/native/<os>-<arch>[-musl]/`. `NativeBridge` now also detects musl at runtime (checking Alpine's `/etc/alpine-release` and musl's `ld-musl-*` dynamic linker, since the JVM has no direct API for this) in addition to its existing (previously unused) `loadFromBundledResource()` auto-extraction, so `mvn`/Gradle users on any of those eight platforms no longer need to build the native library themselves.

## Changed

* Flattened the `tests/loess-rs/` directories into `tests/` directly: each test file is now its own independent integration test binary instead of a submodule of a shared `main.rs`. No test behavior changes.
* Hoisted inline fully-qualified paths (e.g. `crate::math::distance::DistanceLinalg`, `std::slice::from_raw_parts`) to top-level `use` imports across all crates and bindings, using the bare name in the body instead. Genuine name collisions (e.g. a module-local `Result<T>`/`StreamingLoess` type alias shadowing the standard one) are kept fully-qualified with an explanatory comment. No behavior changes.
* Removed unnecessary `pub use` re-exports across `loess-rs`/`fastLoess` (`api.rs`, `binding_support.rs`) that had no consumer via their re-exported path — every actual caller already imported the type directly from its origin module (e.g. `math::boundary::BoundaryPolicy`, `engine::executor::SurfaceMode`). Changed to plain `use`. No behavior changes.

## Fixed

* `dev/bump_version.py` now also updates the Go module's `/vN` major-version-suffix path across `go.mod` files, doc snippets, the doc-snippet runner, and README/docs badges whenever a version bump crosses a major version boundary, so this doesn't regress on the next major release.
* `dev/bump_version.py` now also updates the Maven dependency example version in `bindings/java/docs/modules/ROOT/pages/introduction/installation.adoc`, which was previously left stale after a version bump.
* Fixed inconsistent naming of the Node.js binding as "JavaScript" in the shared project intro sentence (root `README.md`, every binding/crate `README.md`, their generated doc-site home pages, and `CITATION.cff`) — now says "Node.js" everywhere, matching the CI badge, installation table, and directory name (`bindings/nodejs`).
* Cleaned up `loess_rs::prelude` of accidentally-leaked internals: removed `LoessBuilder` and `Batch`/`Online`/`Streaming` adapter markers (use the `Loess`/`StreamingLoess`/`OnlineLoess` type aliases directly - each already builds without needing `.adapter(...)`).
* `make loess-rs-dev`/`make fastLoess-dev` now also run `cargo test --doc` for each tested feature set; doctests in `.rs` source files were previously never checked by any `make` target (only markdown-doc code snippets are covered by `dev/verify_snippets.py`).

# loess-rs 2.0.0

## Added

* Added an "Ideas for Contribution" section to `CONTRIBUTING.md`, listing concrete Batch/Streaming/Online feature gaps (out-of-sample prediction, per-point local gradients, adaptive fraction selection, `cell`/`interpolation_vertices` tuning, bootstrap intervals, GPU backend, concurrent chunk processing, checkpointable streaming state, `OnlineOutput.standard_error`, distance-based window eviction, configurable warm-up).
* Added `dev/bump_version.py --version X.Y.Z` to bump every crate/binding version file, `CITATION.cff`, the Spack recipe, and `CONTRIBUTING.md`'s example version in one pass (supports `--dry-run`).
* Added `dev/check_pinned_versions.py` and a weekly `check-versions.yml` to catch hardcoded version pins Dependabot can't see.
* Added `.github/dependabot.yml`, covering every dependency ecosystem in one weekly PR per directory.
* Added an optional `commit` input to release workflows' `workflow_dispatch` trigger, to pin the built commit for manual runs.
* Added `dev/check_links.py` to validate Markdown cross-reference links across all docs.
* Added `.return_sorted()` to the batch builder, returning results sorted ascending by `x` (default `false`).
* Added a `missing` option (`"error"`/`"drop"`) for non-finite (NaN/Inf) values: Batch/Streaming drop non-finite rows and matching `custom_weights`; Online skips them via `Ok(None)`. Length mismatches always error.
* Added `release-rust.yml` to publish to crates.io on release.

## Changed

* Added a `large` benchmark category (exact-fit, high-iteration, high-fraction) to the R/Rust benchmarks.
* Merged the standalone `dev/add-{cpp,rust,nodejs,wasm}-outputs` scripts into `dev/verify_snippets.py --update-outputs`.
* Replaced Unicode super/subscript stand-ins (`R²`, `xᵢ`, etc.) with plain ASCII throughout docs and comments, catching some leftover mojibake.
* Added `dev/add-readme-to-docs.py` to auto-embed `README.md` as the docs homepage (Starlight/Sphinx-aware); not yet wired into Python's `Makefile`.
* Harmonized the docs-site directory structure across every binding/crate, and fixed doc-tooling scripts that missed snippets in the newly-nested pages.
* Consolidated every README (merged Installation/Documentation sections, dropped GitHub-only alert syntax, removed sections now covered by docs-site pages) and renamed "When to Use" to "When to Use Batch Adapter" everywhere.
* Vendored doxygen-awesome-css v2.4.2 for a modern C++ Doxygen theme.
* Added `dev/update_changelogs.py` to regenerate each binding/crate's `NEWS.md`/`news.md` from the root changelog.
* Replaced the `kernels.md`/`adapter-choice.md` mermaid flowcharts with tables (Doxygen/rustdoc don't render mermaid).
* Consolidated `parameters.md`/`@autodocs` into each `api.md`'s option tables, removing `parameters.md`.
* Updated `wide` to v1.7.
* Removed the dead `compute_residuals`/`backend` fields from `OnlineLoessBuilder` (always computed/never read) and the unused `backend` field from `StreamingLoessBuilder`. `Backend` currently has only a `CPU` variant, read only by the Batch adapter as a GPU placeholder.
* `Streaming::convert()` no longer resolves `overlap` to a flat `500`; it now resolves dynamically to `chunk_size / 10` (clamped to `[1, chunk_size - 10]`) via the new `default_overlap()`, matching every binding's `build_streaming()` helper. Breaking change for callers relying on the previous flat default.
* Mirrored the Python API docs' structure (field tables, `## Options` per field, `## Result Structure` at the end) to every remaining crate/binding. Along the way, unified `weighted_metric_weights` to require explicit `distance_metric = "weighted"` on C++/Go/Java/Julia (previously auto-selected); fixed C++'s `StreamingOptions.overlap` hardcoded `500` default; corrected several bindings' docs showing a flat `500` for `overlap` when they actually use the dynamic default (Node.js, WASM, Go, Java, R); fixed Java's `api-online.adoc` disclaimer and Julia's constructor docstrings missing several accepted keyword arguments.

## Fixed

* Fixed `CONTRIBUTING.md`'s stale Go prerequisite (`1.21+` → `1.23+`), `air` auto-install target (`make r` → `make r-dev`), and example crate version (`0.9.0` → `1.2.0`).
* Fixed the R benchmark script calling `fit` as a field instead of the S3 generic `fit(model, x, y)`.
* Fixed `release-conda.yml`'s version-line `sed` pattern to match any indentation.
* Fixed benchmark vendoring nulling every crate's checksum instead of just the two local path crates.
* Fixed the benchmark README's inaccurate "Iterations" scenario count.
* Fixed `docs.yml`'s Pages deployment: merged per-language jobs into one artifact upload/deploy job using `upload/deploy-pages` actions instead of legacy branch-based deployment.
* Fixed 51 broken doc cross-reference links left over from the docs-site restructure (found via `dev/check_links.py`).
* Fixed `OnlineLoess`/`StreamingLoess` defaults silently diverging from docs: `min_points` (3→2), `update_mode` (`"full"`→`"incremental"`), and internal robustness-iteration defaults (streaming 2→3, online 1→3).
* Fixed every binding's docs describing `LoessResult.x` as "sorted"; it's actually returned in input order (sorted internally, then mapped back). Strengthened Python's `test_unsorted_input` to assert this.
* Fixed the "Handling Outliers" quickstart example printing nothing at `fraction = 0.5` with only 6 points; bumped to `0.7`.
* Fixed two R roxygen examples printing too much/nothing (`OnlineLoess()`, `add_point()`).
* Fixed Julia's `intervals.md` examples looping over all 100 points instead of a short sample.
* Fixed LaTeX math rendering as literal text and cross-reference links not resolving against the rustdoc module tree, both on docs.rs.

# loess-rs 1.1.0

## Added

* Added a GitHub Pages landing page at the repository root, built from `README.md` via pandoc and deployed by `docs.yml`.
* Added a GitHub workflow for running validation scripts.

## Changed

* Split the monolithic `.github/workflows/ci.yml` into seven per-language workflow files: `ci-rust.yml`, `ci-python.yml`, `ci-julia.yml`, `ci-nodejs.yml`, `ci-wasm.yml`, `ci-cpp.yml`, and `ci-r.yml`. Each file carries the relevant `ci` (multi-OS matrix), `asan`, and `gpu` jobs for its language.
* Each crate/binding sub-Makefile now runs `dev/verify_snippets.py --lang <lang>` for its own language as the final step of `make default`. The root `docs-test` target remains as a convenience to run all languages at once.
* Split every sub-Makefile `default:` into `default:` (build and system install) and `dev:` (full quality-check workflow). Both root Makefiles gain `<name>-dev` targets for each binding and crate, and an `all-dev` aggregate target.
* Split `dev/verify_snippets.py` into a lean orchestrator and a `dev/runners/` package. Each language has its own module (`python.py`, `julia.py`, `nodejs.py`, `r.py`, `wasm.py`, `rust.py`, `cpp.py`) containing its `run_<lang>()` function and a `skip_reason()` predicate. Shared types (`Snippet`, `RunResult`) and utilities live in `runners/base.py`; the registry (`RUNNERS`, `SKIP_CHECKS`) is exported from `runners/__init__.py`.
* Moved CHANGELOG and CONTRIBUTING guides to project root.
* Updated README files to be binding/crate specific instead of one generic README for all bindings/crates.
* Moved crate documentation from ReadTheDocs to <https://docs.rs/loess-rs>.
* `make loess-rs` (`default:`) now only runs `cargo build`. The full dev workflow moves to `make loess-rs-dev`.

## Fixed

* Fixed `.cargo/config.toml` hardcoding absolute `c:/rtools45/...` paths for the `x86_64-pc-windows-gnu` linker and ar tool. Replaced with bare tool names resolved via `PATH`, matching the existing fix in `bindings/r/src/cargo-config.toml`.
* Improved `MismatchedInputs` error: added a `dimensions` field and updated the message to show the expected x length (`y_len × dimensions`), making it self-explanatory for both 1-D and multi-dimensional mismatches.

# loess-rs 1.0.0

## Changed

* Removed `dev/isolate_cargo.py`, `dev/check_root_cargo.py`, `dev/fix_doc_snippets.py`, and `check_js_licenses.js` — workspace isolation, doc-snippet transformation, and license checks are no longer needed.
* Split the monolithic root `Makefile` into per-crate/binding sub-Makefiles (e.g. `crates/loess-rs/Makefile`, `bindings/r/Makefile`), each invokable directly via `make -f path/Makefile`. The root `Makefile` now only aggregates (`docs`, `check-msrv`, `all*`).
* Moved Rust and binding tests into their respective crate/binding directories (e.g. `tests/loess-rs/` → `crates/loess-rs/tests/loess-rs/`, `tests/cpp/` → `bindings/cpp/tests/`). Removed the standalone `tests/` workspace packages.
* Renamed `OnlineOutput`'s `smoothed` and `std_error` fields to `y` and `standard_error`, matching `LoessResult`. This is a **breaking change**.
* Updated `wide` to v1.6.

# loess-rs 0.9.0

## Added

* Added Python, R, WASM, Node.js, C++, and Julia bindings.
* Added the option to pass custom weights by the user to the algorithm.

## Changed

* Implement monorepo structure.
* Converted all documentation tables to compact single-space format.
* Updated `.clang-tidy` to configure `lower_case` as the required naming convention for functions and member functions, matching the new snake_case public API.
* Moved `BENCHMARKS.md`, `CHANGELOG.md`, and `CONTRIBUTING.md` from the repository root into `docs/` and added them to the documentation site navigation.
* Added `Loess<T>`, `StreamingLoess<T>`, and `OnlineLoess<T>` type aliases as the primary user-facing constructors (e.g. `StreamingLoess::new().chunk_size(50).build()`). Mode-specific builder methods (`chunk_size`, `overlap`, `window_capacity`, `min_points`, `update_mode`) are now called directly on the type alias rather than after `.adapter()`.
* Made `BatchLoessBuilder`, `StreamingLoessBuilder`, and `OnlineLoessBuilder` internal-only: all public setter methods have been removed from these types. All smoothing configuration now flows through `LoessBuilder<T, Mode>` (exposed via the type aliases above). This is a **breaking change** for any code that called setter methods on an adapter builder directly.
* Changed all enum-typed builder methods to accept strings instead: `weight_function`, `robustness_method`, `scaling_method`, `boundary_policy`, `zero_weight_fallback`, `merge_strategy`, and `update_mode` now take `impl IntoEnum<T>` (accepting both enum variants and strings such as `.weight_function("tricube")`) rather than requiring enum variants to be imported. This is a **breaking change** for any code passing enum variants directly.
* Added a `parse` module to both `loess` and `fastLoess` defining the `IntoEnum<E>` trait and its macro-generated impls for all enum-typed builder parameters. This allows builder methods to accept either a typed enum value (e.g. `.weight_function(WeightFunction::Tricube)`) or a string (e.g. `.weight_function("tricube")`) interchangeably.
* Replaced the `cross_validate(CVConfig)` builder method (which required importing `KFold` or `LOOCV` types) with a string-based cross-validation API: `.cv_method("kfold")` / `.cv_method("loocv")`, `.cv_k(n)`, `.cv_fractions(vec![...])`, and `.cv_seed(n)`. `KFold` and `LOOCV` are no longer exported from the prelude. This is a **breaking change** for any code using the old `cross_validate` API.
* Removed `smooth()`, `smooth_streaming()`, and `smooth_online()` convenience function stubs from `_core.pyi`.

# loess-rs 0.2.2

## Fixed

* Updated license badge.
* Fixed LOESS mechanism figure path.

# loess-rs 0.2.1

## Added

* Added visual validation to the bench branch.

## Changed

* Reduced figures size significantly.
* Implement naming consistency for `auto_converge` (removed `auto_convergence`).

## Fixed

* Fixed `boundary_degree_fallback` pass to online and streaming adapters.
* Fixed `boundary_degree_fallback` pass to `custom_vertex_pass` and `VertexPassFn`.
* Fixed KFold CV bug through adding explicit sorting of training subsets and using robust binary-search interpolation for each test point.
* Fixed `auto_converge` support for Online adapter.

# loess-rs 0.2.0

## Added

* Added `VertexPassFn` and `custom_vertex_pass` support to enable parallelized/accelerated interpolation fitting.
* Added support for custom vertex pass callbacks to all adapters (`Batch`, `Streaming`, `Online`).
* Added support for custom parallel/accelerated standard error calculation via `custom_interval_pass`.
* Added `KDTreeBuilderFn` and `custom_kdtree_builder` hook to enable external parallel KD-tree construction.
* Added `KDTree::from_parts` and exposed `KDNode` and `KDTree::calculate_left_subtree_size` to support custom tree building.
* Added neighborhood caching in `InterpolationSurface` to significantly optimize performance during robustness iterations.
* Added configurable `boundary_degree_fallback` option to control polynomial degree reduction at boundary vertices during interpolation. Defaults to `true` for stability; set to `false` to match R's `loess` behavior exactly.

## Changed

* Changed license from AGPL-3.0-or-later to dual MIT OR Apache-2.0.
* Expanded `SmoothPassFn`, `CVPassFn`, and `IntervalPassFn` signatures to include full multi-dimensional context (dimensions, scaling, polynomial degree, etc.).
* Improved data propagation in `InterpolationSurface` to ensure all necessary coordinate and value slices are available to custom pass implementations.
* Updated `LoessExecutor` to correctly handle augmented data when switching between direct and interpolation modes.
* Updated `InterpolationSurface::build` to accept and propagate `polynomial_degree`, `weight_function`, `zero_weight_fallback`, `distance_metric`, and `scales` for `custom_vertex_pass`. Also, updated `LoessExecutor` to pass these configured values correctly.
* Improved documentation.

## Fixed

* Fixed a potential crash in parallel interpolation refinement by correctly propagating augmented data slices to vertex fitting functions.
* Fixed inconsistent parameter types in custom pass callbacks.
* Fixed missing setters for online and streaming adapters.
* Fixed incorrect standard error propagation in `BatchLoessBuilder`.
* Added `Boundary Linear Fallback` strategy to `InterpolationSurface` to prevent numerical instability ("explosions") at data boundaries when using high-degree polynomials (Quadratic, Cubic, Quartic).
* Fixed missing `max_distance` update in the KD-Tree search, which incorrectly calculated the bandwidth for tricube weights.
* Fixed cumulative cross-contamination in regression buffers, which were not being zeroed between query points.
* Delegated 2D Cubic and 3D Quadratic from context to specialized accumulators.
* Fixed horizontal phase shift in `Interpolation` mode when using boundary policies (`Extend`, `Reflect`, `Zero`). The robustness iteration loop was incorrectly using augmented data indices instead of original data for query point evaluation.

# loess-rs 0.1.0

## Added

* Initial release.

For the full changelog, see:
<https://github.com/thisisamirv/loess-project/blob/main/CHANGELOG.md>
