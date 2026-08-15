<!-- markdownlint-disable MD024 MD046 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.9.1

### Fixed

**Monorepo:**

- Fixed `make r` (and other Python-dependent targets) failing on Ubuntu 24.04 and other modern Linux distributions where `python` is not in `PATH` by changing the default from `PYTHON ?= python` to `PYTHON ?= python3`. The interpreter can still be overridden via `make PYTHON=...`.
- Fixed `ModuleNotFoundError: No module named 'tomli_w'` on systems with an externally-managed Python (Ubuntu 24.04+, Debian 12+, PEP 668) by adding a `--user` fallback to the automatic `tomli`/`tomli_w` install step in the Makefile.
- Documented in `docs/contributing.md` that `python3` with `tomli` and `tomli_w` is required to run the R binding's vendoring scripts, and why Python is needed for an R build.
- Removed `dev/isolate_cargo.py` and the `ISOLATE` Makefile variable. All per-component targets (`loess-rs`, `fastLoess`, `python`, `r`, `julia`, `nodejs`, `wasm`, `cpp`) now call their `_*_impl` sub-target directly; Cargo's `-p <package>` flag already scopes each build to the relevant crate without workspace mutation.
- Removed `dev/check_root_cargo.py`. This script guarded against `Cargo.toml` being left in an isolated state by `isolate_cargo.py`; since workspace isolation no longer happens, the guard is unnecessary.
- Removed `dev/fix_doc_snippets.py`. All 11 documentation code snippets that previously required runtime transformation (missing R/Julia imports and data preambles, Node.js variable injection) now carry their boilerplate directly in the Markdown source.
- Removed `dev/prepare_cargo.py`. The two actions it provided — (1) stripping `[workspace]`/`[patch.crates-io]` before vendoring and (2) appending them back afterward — are now performed inline in the Makefiles with `sed` and `printf`. The `exclude`/`restore` actions it also defined were never called.
- Removed `dev/patch_vendor_crates.py`. The only real work the script did was strip the `version` field from the `loess-rs` path dep in `fastLoess/Cargo.toml` (no GPU deps, no workspace inheritance). This is now a single `sed -i.bak` call inline in the Makefiles. This also eliminates the `tomli`/`tomli_w` pip-install step from the R build.

**Python:**

- Fixed `ruff` linting errors in the Python binding: replaced `from typing import Sequence` with `from collections.abc import Sequence` in `_core.pyi` (UP035), removed redundant `...` literals from all property and method stub bodies — a docstring alone is the correct single-statement body in a `.pyi` file (PYI048, PIE790), and replaced bare `exit(1)` with `sys.exit(1)` in `tests/python/test_gil.py` (PLR1722).

## 0.9.0

### Added

**Monorepo:**

- Added Python, R, WASM, Node.js, C++, and Julia bindings.

**loess-rs and fastLoess:**

- Added the option to pass custom weights by the user to the algorithm.

### Changed

**Monorepo:**

- Implement monorepo structure.
- Converted all documentation tables to compact single-space format.
- Updated `.clang-tidy` to configure `lower_case` as the required naming convention for functions and member functions, matching the new snake_case public API.
- Moved `BENCHMARKS.md`, `CHANGELOG.md`, and `CONTRIBUTING.md` from the repository root into `docs/` and added them to the documentation site navigation.

**loess-rs and fastLoess:**

- Added `Loess<T>`, `StreamingLoess<T>`, and `OnlineLoess<T>` type aliases as the primary user-facing constructors (e.g. `StreamingLoess::new().chunk_size(50).build()`). Mode-specific builder methods (`chunk_size`, `overlap`, `window_capacity`, `min_points`, `update_mode`) are now called directly on the type alias rather than after `.adapter()`.
- Made `BatchLoessBuilder`, `StreamingLoessBuilder`, and `OnlineLoessBuilder` internal-only: all public setter methods have been removed from these types. All smoothing configuration now flows through `LoessBuilder<T, Mode>` (exposed via the type aliases above). This is a **breaking change** for any code that called setter methods on an adapter builder directly.
- Changed all enum-typed builder methods to accept strings instead: `weight_function`, `robustness_method`, `scaling_method`, `boundary_policy`, `zero_weight_fallback`, `merge_strategy`, and `update_mode` now take `impl IntoEnum<T>` (accepting both enum variants and strings such as `.weight_function("tricube")`) rather than requiring enum variants to be imported. This is a **breaking change** for any code passing enum variants directly.
- Added a `parse` module to both `loess` and `fastLoess` defining the `IntoEnum<E>` trait and its macro-generated impls for all enum-typed builder parameters. This allows builder methods to accept either a typed enum value (e.g. `.weight_function(WeightFunction::Tricube)`) or a string (e.g. `.weight_function("tricube")`) interchangeably.
- Replaced the `cross_validate(CVConfig)` builder method (which required importing `KFold` or `LOOCV` types) with a string-based cross-validation API: `.cv_method("kfold")` / `.cv_method("loocv")`, `.cv_k(n)`, `.cv_fractions(vec![...])`, and `.cv_seed(n)`. `KFold` and `LOOCV` are no longer exported from the prelude. This is a **breaking change** for any code using the old `cross_validate` API.
- Removed `smooth()`, `smooth_streaming()`, and `smooth_online()` convenience function stubs from `_core.pyi`.

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

**fastloess (Python):**

- Added the python binding for `fastLoess`.
