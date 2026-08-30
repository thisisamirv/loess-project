<!-- markdownlint-disable MD024 MD025 -->
# loess-rs (development version)

## Changed

* Consolidated every crate/binding README: merged the "Installation" and "Documentation" sections, replaced GitHub-only alert syntax with plain blockquotes, removed the redundant "API Reference" and "Changelog" sections (each now has its own docs-site page), and added a "Read more" link to the Concepts page. The top-level repository README is unchanged, since it's only ever viewed on GitHub.
* Renamed the batch adapter's "When to Use" heading to "When to Use Batch Adapter" across every binding/crate's API docs.
* Vendored the [doxygen-awesome-css](https://github.com/jothepro/doxygen-awesome-css) theme (v2.4.2) for a modern, sidebar-only cpp Doxygen site with automatic dark mode.
* Added `dev/update_changelogs.py`, which regenerates a per-binding/crate `NEWS.md`/`news.md` from the root `CHANGELOG.md`. Wired into every docs site's navigation, the Rust crates' rustdoc module tree, and every `Makefile` `dev` target.
* Replaced `kernels.md`'s "Choosing a Kernel" mermaid flowchart (every binding/crate) with an equivalent decision table, since Doxygen and rustdoc don't render mermaid and the deeply-nested diamond chain was hard to read even where it did render.
* Replaced `adapter-choice.md`/`adapters.md`'s "Overview" flowchart (mermaid in most bindings/crates, ASCII art in the C++ docs) with an equivalent decision table, unifying on a single rendering-agnostic format across every binding/crate.
* Consolidated `parameters.md`/the auto-generated `@autodocs` parameter reference across every binding and crate (C++, Julia, Node.js, Python, WASM, `fastLoess`, `loess-rs`): merged its unique content (fraction/iterations choice guidance, and inline `zero_weight_fallback`/`surface_mode` behavior tables) into each `api.md`'s builder/options tables (Julia: into the `Loess`/`StreamingLoess`/`OnlineLoess` docstrings), and removed `parameters.md` itself along with its docs-site navigation entries, `doc::parameters` rustdoc module, and cross-references (now pointing at `api.md`) — the parameter tables, kernel/robustness/boundary/scaling/degree/distance-metric option lists, and interval/custom-weights code examples it duplicated already live on their own dedicated pages.

## Fixed

* Fixed `docs.yml` triggering GitHub's "pages build and deployment" once per docs job; per-language jobs now upload artifacts, and a single final `deploy` job pushes to `gh-pages` once per run.
* Fixed `docs.yml`'s reliance on GitHub's legacy branch-based Pages deployment, which auto-triggers an unpinned, GitHub-managed "pages build and deployment" job on every `gh-pages` push (surfacing deprecation warnings, e.g. for Node.js 20, that aren't fixable from this repo). The former `deploy` job is now `build`, which still pushes the merged `_site` to `gh-pages` as a cache for future incremental runs, but publishing now goes through `actions/upload-pages-artifact` and a new `deploy` job using the official `actions/deploy-pages`, which this repo pins directly. Requires the repository's Pages source to be switched to "GitHub Actions" in settings.
* Fixed the "Handling Outliers" quickstart example (every binding and the `loess-rs`/`fastLoess` crates) printing nothing: with only 6 points and `fraction = 0.5`, the local window is small enough that tricube weighting drives the farthest neighbor's weight to ~0, leaving just 2 effectively-weighted points, which a degree-1 fit reproduces exactly (zero residual, no downweighting) — confirmed directly against the `lowess`/`loess` core, not binding-specific. Bumped to `fraction = 0.7`, which correctly downweights the injected outlier.
* Fixed the R `OnlineLoess()` roxygen example printing one line per point (48 lines for a 50-point loop); it now collects the smoothed values and prints only `head(smoothed, 5)`.
* Fixed the R `add_point()` roxygen example always printing `NULL`, since a single call never reaches the default `min_points = 3`; it now uses `min_points = 2L` and shows the second (non-`NULL`) call's result.
* Fixed the Julia `intervals.md` "Confidence Intervals" and "Standard Errors" examples each looping over all 100 points instead of a short sample; switched to `result.y[1:5]`/`result.confidence_lower[1:5]`/`result.standard_errors[1:5]`-style slicing, matching the already-concise Python version.
* Fixed inline/display LaTeX math rendering as literal text on docs.rs; added a `katex-header.html` that renders it client-side with KaTeX.
* Fixed every cross-reference link across the `loess-rs`/`fastLoess` crate docs (`quickstart.md`, `api.md`, `concepts.md`, and others) leading nowhere: these pages are embedded into rustdoc via `#![doc = include_str!(...)]`, and plain relative links like `[Concepts](concepts.md)` are rendered verbatim rather than resolved against the generated module tree. Converted every such link to a proper intra-doc link (e.g. `[Concepts](crate::doc::concepts)`), validated with `cargo doc --all-features` under `-D warnings` (zero broken-link warnings).

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
