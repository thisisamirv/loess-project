<!-- markdownlint-disable MD024 MD046 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
