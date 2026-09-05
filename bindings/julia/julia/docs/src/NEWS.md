<!-- markdownlint-disable MD024 MD025 -->
# FastLOESS.jl 2.0.0

## Added

* Added an "Ideas for Contribution" section to `CONTRIBUTING.md`, listing concrete Batch/Streaming/Online feature gaps (out-of-sample prediction, per-point local gradients, adaptive fraction selection, `cell`/`interpolation_vertices` tuning, bootstrap intervals, GPU backend, concurrent chunk processing, checkpointable streaming state, `OnlineOutput.standard_error`, distance-based window eviction, configurable warm-up).
* Added `dev/bump_version.py --version X.Y.Z` to bump every crate/binding version file, `CITATION.cff`, the Spack recipe, and `CONTRIBUTING.md`'s example version in one pass (supports `--dry-run`).
* Added `dev/check_pinned_versions.py` and a weekly `check-versions.yml` to catch hardcoded version pins Dependabot can't see.
* Added `.github/dependabot.yml`, covering every dependency ecosystem in one weekly PR per directory.
* Added an optional `commit` input to release workflows' `workflow_dispatch` trigger, to pin the built commit for manual runs.
* Added `dev/check_links.py` to validate Markdown cross-reference links across all docs.
* Added `return_sorted` and `missing` options to `Loess`, `StreamingLoess`, and `OnlineLoess`.

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
* Removed the same fields as Python from `StreamingLoess`/`OnlineLoess` keyword arguments, including `parallel` from `OnlineLoess`. Breaking change.
* `weighted_metric_weights` now requires `distance_metric = "weighted"` explicitly, matching Python. Breaking change.
* `StreamingLoess`'s `overlap` default changed from a fixed `500` to a dynamic `chunk_size / 10`, like every other binding. Breaking change.
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
* Fixed the Documenter homepage being a stale, separately-maintained `index.md`; now regenerated from `README.md` on every build.
* Fixed `release-julia-register.yml` pulling release notes from the full changelog instead of the Julia-filtered `NEWS.md`.
* Fixed `make julia-dev` resolving an outdated `fastloess_jll`, Windows mojibake in `dev/runners/julia.py`, inconsistent tab indentation, and stale `lowess-project` links.
* Ported a missing custom-weights test case from `fastlowess`'s Julia suite.
* Fixed `cell`/`interpolation_vertices`/`boundary_degree_fallback`/`cv_seed` being silently non-functional due to no-op FFI setters, and `jl_streaming_loess_new` wrapping negative `dimensions` instead of clamping to 1.
* Simplified redundant null-pointer comparisons in `FastLOESS.jl`.

# FastLOESS.jl 1.1.0

## Added

* Added a GitHub Pages landing page at the repository root, built from `README.md` via pandoc and deployed by `docs.yml`.
* Added a GitHub workflow for running validation scripts.
* `release-julia-register.yml` now automatically extracts the matching changelog section and appends it as release notes in the JuliaRegistrator comment, enabling auto-merge on major version bumps.

## Changed

* Split the monolithic `.github/workflows/ci.yml` into seven per-language workflow files: `ci-rust.yml`, `ci-python.yml`, `ci-julia.yml`, `ci-nodejs.yml`, `ci-wasm.yml`, `ci-cpp.yml`, and `ci-r.yml`. Each file carries the relevant `ci` (multi-OS matrix), `asan`, and `gpu` jobs for its language.
* Each crate/binding sub-Makefile now runs `dev/verify_snippets.py --lang <lang>` for its own language as the final step of `make default`. The root `docs-test` target remains as a convenience to run all languages at once.
* Split every sub-Makefile `default:` into `default:` (build and system install) and `dev:` (full quality-check workflow). Both root Makefiles gain `<name>-dev` targets for each binding and crate, and an `all-dev` aggregate target.
* Split `dev/verify_snippets.py` into a lean orchestrator and a `dev/runners/` package. Each language has its own module (`python.py`, `julia.py`, `nodejs.py`, `r.py`, `wasm.py`, `rust.py`, `cpp.py`) containing its `run_<lang>()` function and a `skip_reason()` predicate. Shared types (`Snippet`, `RunResult`) and utilities live in `runners/base.py`; the registry (`RUNNERS`, `SKIP_CHECKS`) is exported from `runners/__init__.py`.
* Moved CHANGELOG and CONTRIBUTING guides to project root.
* Updated README files to be binding/crate specific instead of one generic README for all bindings/crates.
* Moved Julia documentation from ReadTheDocs to GitHub Pages, served by Documenter.jl at <https://thisisamirv.github.io/loess-project/julia/stable/>. The ReadTheDocs site no longer includes Julia-specific content. Code blocks use Documenter.jl `@example` sections, which execute and embed output automatically during the docs build.
* `make julia` (`default:`) now builds the Rust library and installs the Julia package via `Pkg.develop`. The full dev workflow moves to `make julia-dev`.

## Fixed

* Fixed `.cargo/config.toml` hardcoding absolute `c:/rtools45/...` paths for the `x86_64-pc-windows-gnu` linker and ar tool. Replaced with bare tool names resolved via `PATH`, matching the existing fix in `bindings/r/src/cargo-config.toml`.
* Fixed `fit(l::Loess, x::Matrix{Float64}, y)` not validating that `size(x, 2) == l.dimensions` before flattening the matrix. If the column count differed from the configured dimensions, the library either silently used wrong data or produced a confusing C-level error. The `Loess` struct now stores `dimensions` as a field, and the matrix overload checks `size(x, 2) != l.dimensions` upfront with a clear message naming the parameter to fix.
* Fixed `FastLOESS.jl` never actually loading the prebuilt `fastloess_jll` binary: `find_library()` only checked the `FASTLOESS_LIB` env var and local dev-mode paths, so the package installed from the registry had no working native library for end users. Added the `fastloess_jll` dependency (`Project.toml`), a JLL-loading branch in `find_library()`, and switched from an eager `const libfastloess = find_library()` (resolved once at precompile time) to a lazy `current_library()` accessor re-resolved in `__init__()`.

# FastLOESS.jl 1.0.0

## Fixed

* Fixed `LoessResult.iterations_used` returning the raw FFI sentinel `-1` instead of `nothing` when robustness iterations were not applicable.

## Changed

* Removed `dev/isolate_cargo.py`, `dev/check_root_cargo.py`, `dev/fix_doc_snippets.py`, and `check_js_licenses.js` — workspace isolation, doc-snippet transformation, and license checks are no longer needed.
* Split the monolithic root `Makefile` into per-crate/binding sub-Makefiles (e.g. `crates/loess-rs/Makefile`, `bindings/r/Makefile`), each invokable directly via `make -f path/Makefile`. The root `Makefile` now only aggregates (`docs`, `check-msrv`, `all*`).
* Moved Rust and binding tests into their respective crate/binding directories (e.g. `tests/loess-rs/` → `crates/loess-rs/tests/loess-rs/`, `tests/cpp/` → `bindings/cpp/tests/`). Removed the standalone `tests/` workspace packages.
* Renamed `OnlineOutput`'s `smoothed` and `std_error` fields to `y` and `standard_error`. This is a **breaking change**.
* Removed `dev/format_julia.jl`; formatting is now inlined in `bindings/julia/Makefile`.

# FastLOESS.jl 0.9.0

## Added

* Added Python, R, WASM, Node.js, C++, and Julia bindings.

## Changed

* Implement monorepo structure.
* Converted all documentation tables to compact single-space format.
* Updated `.clang-tidy` to configure `lower_case` as the required naming convention for functions and member functions, matching the new snake_case public API.
* Moved `BENCHMARKS.md`, `CHANGELOG.md`, and `CONTRIBUTING.md` from the repository root into `docs/` and added them to the documentation site navigation.

For the full changelog, see:
<https://github.com/thisisamirv/loess-project/blob/main/CHANGELOG.md>
