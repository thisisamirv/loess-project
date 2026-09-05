<!-- markdownlint-disable MD024 MD046 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 1.2.0

### Added

**Monorepo:**

- Added `dev/bump_version.py --version X.Y.Z` to bump every crate/binding's version files, `CITATION.cff`, and the Spack recipe in one pass (supports `--dry-run`).
- Added `dev/check_pinned_versions.py` and a weekly `check-versions.yml` to catch hardcoded version pins that Dependabot can't see.
- Added `.github/dependabot.yml`, covering every dependency ecosystem, grouped per directory into a single weekly PR.
- Added an optional `commit` input to every release workflow's `workflow_dispatch` trigger, to pin the built commit for manual runs.
- Added `dev/check_links.py` to validate every Markdown cross-reference link across all docs.

**loess-rs:**

- Added `.return_sorted()` to the batch builder, to return results sorted ascending by `x` instead of input order. Default `false`.
- Added `release-rust.yml` to publish to crates.io on release.

**fastLoess:**

- Added `return_sorted` to `BuilderOptionSet`/`TypedBuilderOptionSet` and the `Loess` (Batch) builder.
- Now published via `release-rust.yml`, 3 minutes after `loess-rs` to let the crates.io index catch up.

**Python:**

- Added a `return_sorted` option to `Loess`.

**R:**

- Added a `return_sorted` option to `Loess()`.

**Julia:**

- Added a `return_sorted` option to `Loess`.

**Go:**

- Added a new Go binding (`bindings/go`): `cgo`-based `fastloess` package with `Loess`/`StreamingLoess`/`OnlineLoess` types, a Hugo docs site, CI/release workflows, and full doc-snippet/test coverage.
- Added a `ReturnSorted` option to `Options`.

**Java:**

- Added a new Java binding (`bindings/java`): JNI-based `fastloess` Maven package with `Loess`/`StreamingLoess`/`OnlineLoess` classes (including LOESS-specific options like `degree`, `dimensions`, `distanceMetric`, `surfaceMode`, plus hat-matrix stats via `Result.hatMatrix()`), an Antora docs site, CI/release workflows, and full doc-snippet/test coverage.
- Added a `returnSorted` option to `Options`.

**Node.js:**

- Added `aarch64-unknown-linux-musl` and `armv7-unknown-linux-gnueabihf` prebuilt targets with matching optional npm subpackages.
- Added a `return_sorted` option to `Loess`'s `SmoothOptions`.

**WASM:**

- Added a `return_sorted` option to `Loess`'s `SmoothOptions`.

**C++:**

- Added CMake package-config support so consumers can `find_package(fastloess)`.
- Added CI coverage for `clang-cl` (Windows), `clang` (Linux), MinGW-w64, and Intel oneAPI.
- Added ARM64 release binaries for Linux/Windows/macOS; also fixed the macOS x64 job silently shipping a mislabeled arm64 binary.
- Renamed `cpp_loess_fit`/`cpp_streaming_process`'s `x`/`y` params to `x_values`/`y_values`, avoiding a collision with `CppLoessResult`'s own `x`/`y` fields.
- Added a `return_sorted` option to `LoessOptions`.

### Changed

**Monorepo:**

- Added a `large` benchmark category (exact-fit, high-iteration, high-fraction) to the R/Rust benchmarks.
- Merged the standalone `dev/add-{cpp,rust,nodejs,wasm}-outputs` scripts into `dev/verify_snippets.py --update-outputs`.
- Replaced Unicode super/subscript stand-ins (`R²`, `xᵢ`, etc.) with plain ASCII throughout docs, READMEs, and Rust comments, also catching some leftover mojibake.
- Added `dev/add-readme-to-docs.py` to auto-embed `README.md` as the docs homepage (Starlight/Sphinx-aware); not yet wired into Python's `Makefile`.

**docs:**

- Harmonized the docs-site directory structure across every binding/crate, and fixed doc-tooling scripts that missed snippets in the newly-nested pages.
- Consolidated every README: merged Installation/Documentation sections, dropped GitHub-only alert syntax, and removed sections now covered by dedicated docs-site pages.
- Renamed "When to Use" to "When to Use Batch Adapter" across every binding's API docs.
- Vendored doxygen-awesome-css v2.4.2 for a modern cpp Doxygen theme.
- Added `dev/update_changelogs.py` to regenerate each binding/crate's `NEWS.md`/`news.md` from the root changelog.
- Replaced the `kernels.md`/`adapter-choice.md` mermaid flowcharts with rendering-agnostic tables (Doxygen/rustdoc don't render mermaid).
- Consolidated `parameters.md`/`@autodocs` into each `api.md`'s option tables, removing `parameters.md`.

**C++:**

- Documented `x_values`/`y_values` params, fixing a Doxygen warning.
- Restructured Doxygen nav from ~20 flat pages into 5 hub pages, mirroring other bindings.
- Added a Spack recipe, auto-updated by `release-cpp.yml` on release.
- Bumped the vendored Corrosion CMake module to v0.6.1.
- Removed the dead `confidence_intervals`, `prediction_intervals`, `return_diagnostics`, `return_residuals`, and `return_se` fields from `OnlineOptions` (a standalone struct, not inherited from `LoessOptions` as the docs previously and incorrectly claimed). Breaking change for `OnlineOptions` callers. Also removed `parallel` from `OnlineOptions` — unlike the fields above, it gated real internal KD-tree/interval-pass dispatch, but was dropped for consistency with `fastLowess`'s `OnlineLowess`; Online now always runs sequentially. `StreamingOptions`/`LoessOptions` keep `parallel` unaffected.
- `StreamingOptions` (which inherits `LoessOptions` and still has `confidence_intervals`/`prediction_intervals`/`return_se` structurally) no longer forwards them to the native constructor, since Streaming never computed them.
- `StreamingOptions::overlap`'s default changed from a fixed `500` to `-1` (a sentinel meaning "use the library default"), so it now resolves dynamically to `chunk_size / 10` like every other language binding, instead of always passing a concrete `500` to the native constructor regardless of `chunk_size`. Breaking change for callers relying on the previous flat default.
- `weighted_metric_weights` no longer auto-selects the `"weighted"` distance metric: previously, providing weights silently overrode any explicit `distance_metric` via `resolve_distance_metric_for_builder`; now `distance_metric = "weighted"` must be set explicitly, matching Python/R/Node.js/WASM, and omitting it while providing weights raises an error. Breaking change.

**R:**

- Removed the redundant `rfastloess-package` pkgdown topic and the internal `Nullable()` helper.
- Fixed `_pkgdown.yml` mislabeling the S3-based interface as "R6 classes".
- Merged `parameters.Rmd`/`batch.Rmd`/`streaming.Rmd`/`online.Rmd` into the constructors' roxygen docs, removing the now-redundant vignettes.
- Removed `confidence_intervals`, `prediction_intervals`, and `return_se` from `StreamingLoess()`'s constructor, and those plus `return_diagnostics`/`return_residuals` from `OnlineLoess()`'s constructor — none of these were ever computed by either adapter. Breaking change. Also removed `parallel` from `OnlineLoess()`'s constructor for consistency with `lowess`/`fastLowess` — it gated real internal dispatch there, unlike the fields above, but Online now always runs sequentially; `parallel` remains real and unaffected for `StreamingLoess()`.

**Node.js:**

- Updated `oxlint`, `napi`/`napi-derive`/`@napi-rs/cli`/`napi-build`, and `typedoc-plugin-markdown`.
- `make nodejs-dev` now runs `npm update` after `npm install`.
- `StreamingLoess`/`OnlineLoess` no longer accept `SmoothOptions` (the Batch options type); each now has its own dedicated `StreamingSmoothOptions`/`OnlineSmoothOptions` type that only declares the fields it actually supports, so `confidence_intervals`, `prediction_intervals`, `return_se`, `cv_fractions`, `cv_method`, `cv_k`, and `cv_seed` (Batch-only) are gone from both, and `return_diagnostics`/`return_residuals`/`parallel` are additionally gone from `OnlineSmoothOptions` (previously accepted and silently ignored/gated at runtime). Breaking change for TypeScript consumers relying on the old shared type; `Loess`'s `SmoothOptions` is unaffected.

**WASM:**

- Updated `oxlint` and `typedoc-plugin-markdown`.
- `make wasm-dev` now runs `npm update` after `npm install`.
- Same `StreamingSmoothOptions`/`OnlineSmoothOptions` split as Node.js, applied to the TypeScript interfaces and `options_to_builder` equivalents.

**loess-rs:**

- Updated `wide` to v1.7.
- Removed the dead `compute_residuals`/`backend` fields from `OnlineLoessBuilder` (residual is always computed regardless of the flag; `backend` was never read for Online). `StreamingLoessBuilder` lost its unused `backend` field too — `Backend` currently has only a `CPU` variant and is read only by the Batch adapter, as a placeholder for future GPU support.
- `Streaming::convert()` (used by the `StreamingLoess::new()...build()` type-alias API) no longer resolves `overlap` to a flat `500` when unset; it now resolves dynamically to `chunk_size / 10` (clamped to `[1, chunk_size - 10]`) via the new `adapters::defaults::default_overlap()`, matching every language binding's `build_streaming()` helper. Removed the now-superseded `DEFAULT_STREAMING_OVERLAP` constant. Breaking change for callers relying on the previous flat default with a customized `chunk_size`.

**fastLoess:**

- Same dead-field removal as loess-rs, mirrored in `OnlineLoessBuilder`/`StreamingLoessBuilder`.
- Removed `.confidenceIntervals()`, `.predictionIntervals()`, and `.returnSe()` from the `StreamingLoess`/`OnlineLoess` entry-point wrapper structs — these leaked in via the shared builder macro and were silently ignored (neither adapter computes confidence/prediction intervals or standard errors). Breaking change for direct Rust consumers; `Loess`'s own methods are unaffected. `parallel` was initially kept as real for `OnlineLoess` here (unlike `lowess`/`fastLowess`), since it gated internal KD-tree/interval-pass dispatch, but was subsequently also removed from `OnlineLoess`/`ParallelOnlineLoessBuilder` for cross-project consistency — Online now always runs sequentially. `Loess`/`StreamingLoess` keep `.parallel()` unaffected.
- Fixed a misleading comment on `binding_support::default_overlap()` claiming only cpp/julia/python/r used this formula while wasm/nodejs used a flat `500`; every binding actually computes `chunk_size / 10` via the same `build_streaming()` helper. No behavior changed there, only the comment.

**Python:**

- Removed `confidence_intervals`, `prediction_intervals`, and `return_se` from `StreamingLoess`'s constructor — never computed for Streaming. Breaking change.
- Removed `confidence_intervals`, `prediction_intervals`, `return_se`, `return_diagnostics`, and `return_residuals` from `OnlineLoess`'s constructor, for the same reason (plus `OnlineOutput` has no diagnostics field and always includes a residual regardless of the flag). Breaking change. Also removed `parallel` from `OnlineLoess`'s constructor — it gated real internal KD-tree/interval-pass dispatch (unlike the fields above), but was dropped for consistency with `fastlowess`'s `OnlineLowess`; Online now always runs sequentially. `Loess`/`StreamingLoess` are unaffected.

**Julia:**

- Removed the same fields as Python, from `StreamingLoess`/`OnlineLoess` keyword arguments. Breaking change. `parallel` was subsequently also removed from `OnlineLoess`'s keyword arguments — it gated real internal KD-tree/interval-pass dispatch, but was dropped for consistency with `fastlowess`'s `OnlineLowess`; Online now always runs sequentially.
- `weighted_metric_weights` no longer auto-selects the `"weighted"` distance metric for `Loess`/`StreamingLoess`/`OnlineLoess`; `distance_metric = "weighted"` must now be set explicitly, matching Python, and omitting it while providing weights raises an error. Breaking change.
- `StreamingLoess`'s `overlap` keyword argument default changed from a fixed `500` to `-1` (a sentinel meaning "use the library default"), so it now resolves dynamically to `chunk_size / 10` like every other language binding, instead of always passing a concrete `500` regardless of `chunk_size`. Breaking change for callers relying on the previous flat default.

**Go:**

- `StreamingOptions` and `OnlineOptions` no longer embed the shared `Options` struct (they now declare only the fields they actually support). `StreamingOptions` lost `ConfidenceIntervals`/`PredictionIntervals`/`ReturnSE`/`CVFractions`/`CVMethod`/`CVK`/`CVSeed`; `OnlineOptions` additionally lost `ReturnDiagnostics`/`ReturnResiduals`. Breaking change. `Options`'s own fields are unaffected; `Parallel` remains real for `StreamingOptions` but was subsequently also removed from `OnlineOptions` for consistency with `fastlowess`'s `OnlineOptions` — Online now always runs sequentially.
- `WeightedMetricWeights` no longer auto-selects the `"weighted"` distance metric for `Options`/`StreamingOptions`/`OnlineOptions`; `DistanceMetric = "weighted"` must now be set explicitly, matching Python, and omitting it while providing weights raises an error. Breaking change.

**Java:**

- Removed `confidenceIntervals`/`predictionIntervals`/`returnSe` builder methods from `StreamingOptions.Builder`, and those plus `returnDiagnostics`/`returnResiduals` from `OnlineOptions.Builder`. Breaking change. `Options.Builder`'s own methods are unaffected; `parallel` remains real for `StreamingOptions.Builder` but was subsequently also removed from `OnlineOptions.Builder` for consistency with `fastlowess`'s `OnlineOptions.Builder` — Online now always runs sequentially.
- `weightedMetricWeights` no longer auto-selects the `"weighted"` distance metric for `Options.Builder`/`StreamingOptions.Builder`/`OnlineOptions.Builder`; `distanceMetric("weighted")` must now be set explicitly, matching Python, and omitting it while providing weights raises an error. Breaking change.

**docs:**

- Mirrored the Python API docs' structure (complete field tables in canonical order, a `## Options` subsection per field, `## Result Structure` moved to the end) to every remaining crate/binding: `loess-rs`, `fastLoess`, Julia (docstrings), Node.js, WASM, C++, Go, and Java. Along the way, corrected several real accuracy/consistency issues found by auditing docs against source:
  - `weighted_metric_weights` previously auto-selected the `"weighted"` distance metric on C++/Go/Java/Julia (via `resolve_distance_metric_for_builder`), silently overriding any explicit `distance_metric`, while Python/R/Node.js/WASM required `distance_metric = "weighted"` to be set explicitly. Rather than just documenting the discrepancy, unified the behavior: C++/Go/Java/Julia now also require `distance_metric = "weighted"` explicitly and raise an error if weights are missing (see each binding's own changelog entry above).
  - Fixed C++'s `StreamingOptions` hardcoding `overlap` to a fixed `500` default (bypassing the Rust-side dynamic resolution since it always passed a concrete, non-negative value); its default is now `-1` (matching Go/Java's "negative means use the library default" convention), so it resolves dynamically like every other binding.
  - Docs previously showed a flat `500` for `overlap` for several bindings that actually use the dynamic default (Node.js, WASM, Go, Java, R).
  - Fixed Java's `api-online.adoc` disclaimer omitting `returnDiagnostics`/`returnResiduals` from the list of Batch/Streaming-only settings.
  - Fixed Julia's `Loess()`/`StreamingLoess()`/`OnlineLoess()` docstrings missing several real, accepted keyword arguments entirely (`weighted_metric_weights`, `cell`, `interpolation_vertices`, `boundary_degree_fallback`, `cv_seed` for `Loess()`).

**R:**

- Fixed `bindings/r/R/StreamingLoess.R`, which was corrupted to contain a duplicate (and outdated) copy of `OnlineLoess.R`'s content with a mangled fragment of the real `StreamingLoess()` body appended — meaning `StreamingLoess()` was not callable from R at all. Reconstructed from `man/StreamingLoess.Rd`, `utils.R`'s `streaming_params`, and the surrounding constructors' conventions; verified via `roxygen2::roxygenise()` and the full `testthat` suite (187 passed). Also fixed a stale `test-extendr-wrappers.R` fixture with 3 extra positional arguments left over from before `cell`/`interpolation_vertices`/`boundary_degree_fallback` were added to `RStreamingLoess$new`.

### Fixed

**Monorepo:**

- Fixed the R benchmark script calling `fit` as a field instead of the S3 generic `fit(model, x, y)`.
- Fixed `release-conda.yml`'s version-line `sed` pattern to match any indentation.
- Fixed benchmark vendoring nulling every crate's checksum instead of just the two local path crates.
- Fixed the benchmark README's inaccurate "Iterations" scenario count.
- Fixed `docs.yml`'s Pages deployment: merged per-language jobs into one artifact upload/deploy job, using `upload/deploy-pages` actions instead of legacy branch-based deployment.
- Fixed 51 broken doc cross-reference links left over from the docs-site restructure (found via new `dev/check_links.py`).
- Fixed several `OnlineLoess`/`StreamingLoess` defaults silently diverging from every binding's docs and actual behavior: `min_points` (was 3, now 2), `update_mode` (was `"full"`, now `"incremental"`), and the internal robustness-iteration defaults (streaming 2→3, online 1→3) that only the Rust crates' bare prelude API could ever see.
- Fixed every binding's/crate's docs and doc-comments describing `LoessResult.x` (and equivalents) as "Sorted x values"; it's actually returned in the same order as the input `x` (the algorithm sorts internally, then maps every output field back to the original order). Also strengthened Python's `test_unsorted_input` to assert this instead of only checking output length.

**docs:**

- Fixed the "Handling Outliers" quickstart example printing nothing with only 6 points at `fraction = 0.5`; bumped to `0.7` so the outlier is actually downweighted.
- Fixed two R roxygen examples: `OnlineLoess()` printing 48 lines instead of a `head(smoothed, 5)` sample, and `add_point()` always printing `NULL` due to the default `min_points`.
- Fixed Julia's `intervals.md` examples looping over all 100 points instead of a short sample, matching the concise Python version.

**C++:**

- Fixed several Doxygen rendering bugs (wrong homepage, broken blockquotes/math/admonitions); `README.md` is now the native Doxygen homepage.
- Fixed `ci-cpp.yml`'s untrusted Homebrew tap warning and a broken Windows `cppcheck` install.
- Fixed `Doxyfile`'s wrong `PROJECT_NAME` and a malformed `FILE_PATTERNS` glob.

**Julia:**

- Fixed the Documenter homepage being a stale, separately-maintained `index.md`; now regenerated from `README.md` on every build.
- Fixed `release-julia-register.yml` pulling release notes from the full changelog instead of the Julia-filtered `NEWS.md`.
- Fixed `make julia-dev` resolving an outdated `fastloess_jll`, Windows mojibake in `dev/runners/julia.py`, inconsistent tab indentation in `make.jl`/`FastLOESS.jl`, and stale `lowess-project` links in README/index.
- Ported a missing custom-weights test case from `fastlowess`'s Julia suite.
- Fixed `cell`/`interpolation_vertices`/`boundary_degree_fallback`/`cv_seed` being silently non-functional due to no-op FFI setters, and `jl_streaming_loess_new` wrapping negative `dimensions` instead of clamping to 1.
- Simplified redundant null-pointer comparisons in `FastLOESS.jl`.

**R:**

- Reformatted `configure` to tabs and fixed `.Rbuildignore` missing exclusions.
- Removed the empty `R/params.R` stub and simplified `plot.LoessResult()` to return `NULL` invisibly.
- Inlined the `.make_*` constructor helpers, and consolidated `utils.R`'s parameter validators into two generic helpers.

**Node.js:**

- Fixed the docs homepage never showing README content.
- Fixed an `@astrojs/sitemap` warning, TypeDoc/Starlight "API Reference" 404s, and an `astro build` failure from a missing dependency.

**WASM:**

- Same docs-homepage/sitemap/API-404/astro-build fixes as Node.js.
- Fixed `concepts.md` figures not rendering and LaTeX math rendering as literal text.
- Fixed generated `.d.ts` doc comments showing literal backslashes instead of quotes.

**Python:**

- Fixed the empty "API Reference" page (stale toctree references).
- Fixed noisy pip version-check output in `release-pypi.yml` and a Pyright false-positive warning.
- Converted 3 plain comments to doc comments and ported 2 missing custom-weights test cases from `fastlowess`.

**loess-rs:**

- Fixed LaTeX math rendering as literal text on docs.rs.
- Fixed cross-reference links not resolving against the rustdoc module tree.

**fastLoess:**

- Fixed cross-reference links not resolving against the rustdoc module tree.
- Fixed LaTeX math rendering as literal text on docs.rs.
- Added `#[allow(clippy::excessive_precision)]` to kernel constants.
- Fixed `fastLoess`'s `build_streaming`/`build_online` hardcoding `chunk_size`/`window_capacity`/`min_points` fallback defaults as bare numeric literals instead of referencing `loess-rs`'s named `DEFAULT_*` constants, unlike every other default in the same file; a future change to those constants would have silently drifted from what the FFI layer actually applies.

## 1.1.0

### Added

**Monorepo:**

- Added a GitHub Pages landing page at the repository root, built from `README.md` via pandoc and deployed by `docs.yml`.
- Added a GitHub workflow for running validation scripts.

**Julia:**

- `release-julia-register.yml` now automatically extracts the matching changelog section and appends it as release notes in the JuliaRegistrator comment, enabling auto-merge on major version bumps.

**R:**

- Added `lenght` gaurds for extra arguments.

**Node.js:**

- Added `npm run lint` to the `Lint` step in `ci-nodejs.yml`, so JavaScript source and test files are linted via `oxlint` on every CI run.

**WASM:**

- Added `npm run lint` to the `Lint` step in `ci-wasm.yml`, so JavaScript source and test files are linted via `oxlint` on every CI run.

**C++:**

- Added clang-tidy and cppcheck installation to Makefile.

### Changed

**Monorepo:**

- Split the monolithic `.github/workflows/ci.yml` into seven per-language workflow files: `ci-rust.yml`, `ci-python.yml`, `ci-julia.yml`, `ci-nodejs.yml`, `ci-wasm.yml`, `ci-cpp.yml`, and `ci-r.yml`. Each file carries the relevant `ci` (multi-OS matrix), `asan`, and `gpu` jobs for its language.
- Each crate/binding sub-Makefile now runs `dev/verify_snippets.py --lang <lang>` for its own language as the final step of `make default`. The root `docs-test` target remains as a convenience to run all languages at once.
- Split every sub-Makefile `default:` into `default:` (build and system install) and `dev:` (full quality-check workflow). Both root Makefiles gain `<name>-dev` targets for each binding and crate, and an `all-dev` aggregate target.
- Split `dev/verify_snippets.py` into a lean orchestrator and a `dev/runners/` package. Each language has its own module (`python.py`, `julia.py`, `nodejs.py`, `r.py`, `wasm.py`, `rust.py`, `cpp.py`) containing its `run_<lang>()` function and a `skip_reason()` predicate. Shared types (`Snippet`, `RunResult`) and utilities live in `runners/base.py`; the registry (`RUNNERS`, `SKIP_CHECKS`) is exported from `runners/__init__.py`.

**docs:**

- Moved CHANGELOG and CONTRIBUTING guides to project root.
- Updated README files to be binding/crate specific instead of one generic README for all bindings/crates.

**R:**

- Moved R documentation from ReadTheDocs to GitHub Pages, served by pkgdown at <https://thisisamirv.github.io/loess-project/r/>. The ReadTheDocs site no longer includes R-specific content.
- Simplified `bindings/r/Makefile`: replaced `Cargo.toml.orig` save/restore vendoring with `src/vendor-update.sh`; made `[workspace]` permanent in `src/Cargo.toml`; removed Bioconductor dependencies, redundant `cargo fmt --check`, `NAMESPACE` indentation post-processing, and `pkgdown::build_site` from the dev workflow.
- Changed R version dependency to 4.4.0 due to issues with installing Bioconducter packages on R < 4.4.0.
- Replaced the multi-step `install.packages` / `BiocManager::install` package installation logic in `bindings/r/Makefile` with a single [`pak`](https://pak.r-lib.org/)-based block. `pak` handles RSPM binary vs source selection automatically (including Linux), skips already-installed packages, and installs CRAN, Bioconductor (`bioc::` prefix), and R-universe packages in one call.
- `make r` (`default:`) now runs `R CMD INSTALL $(R_DIR)` directly; R's `configure` script handles Rust compilation from the committed `vendor.tar.xz`. The full dev workflow moves to `make r-dev`.

**Python:**

- Migrated Python documentation from MkDocs to Sphinx (with MyST-Parser and jupyter-sphinx). Code blocks now execute and embed output automatically via `jupyter-sphinx`.
- `make python` (`default:`) now installs to the user Python environment via `pip install --user`. The full dev workflow (venv setup, formatting, linting, testing, doc-snippet verification) moves to `make python-dev`.

**Julia:**

- Moved Julia documentation from ReadTheDocs to GitHub Pages, served by Documenter.jl at <https://thisisamirv.github.io/loess-project/julia/stable/>. The ReadTheDocs site no longer includes Julia-specific content. Code blocks use Documenter.jl `@example` sections, which execute and embed output automatically during the docs build.
- `make julia` (`default:`) now builds the Rust library and installs the Julia package via `Pkg.develop`. The full dev workflow moves to `make julia-dev`.

**Node.js:**

- Moved Node.js documentation from ReadTheDocs to GitHub Pages, served by Starlight at <https://thisisamirv.github.io/loess-project/nodejs/>. The ReadTheDocs site no longer includes Node.js-specific content. `dev/add-nodejs-outputs.js` runs as part of `make nodejs-dev`, executing each JavaScript code block in the docs and injecting its output back into the Markdown source.
- `make nodejs` (`default:`) now builds the native addon and links it globally via `npm link`. The full dev workflow moves to `make nodejs-dev`.
- Updated `oxlint` dependency to 1.80.

**WASM:**

- Moved WASM documentation from ReadTheDocs to GitHub Pages, served by Starlight at <https://thisisamirv.github.io/loess-project/wasm/>. The ReadTheDocs site no longer includes WASM-specific content. `dev/add-wasm-outputs.js` runs as part of `make wasm-dev`, executing each JavaScript code block in the docs and injecting its output back into the Markdown source.
- `make wasm` (`default:`) now builds both the Node.js and web WASM targets and links the Node.js package globally via `npm link`. The full dev workflow moves to `make wasm-dev`.
- Updated `oxlint` dependency to 1.80.
- Replace the outdated `jetli/wasm-pack-action` workflow with `taiki-e/install-action`.

**C++:**

- Moved C++ documentation from ReadTheDocs to GitHub Pages, served by Doxygen at <https://thisisamirv.github.io/loess-project/cpp/>. The ReadTheDocs site no longer includes C++-specific content.
- `make cpp` (`default:`) now only runs `cargo build`. The full dev workflow (formatting, linting, cbindgen idempotency, symbol export verification, cmake tests, valgrind, doc-snippet verification) moves to `make cpp-dev`.

**fastLoess:**

- Moved crate documentation from ReadTheDocs to <https://docs.rs/fastLoess>.
- `make fastLoess` (`default:`) now only runs `cargo build`. The full dev workflow moves to `make fastLoess-dev`.

**loess-rs:**

- Moved crate documentation from ReadTheDocs to <https://docs.rs/loess-rs>.
- `make loess-rs` (`default:`) now only runs `cargo build`. The full dev workflow moves to `make loess-rs-dev`.

### Fixed

**Monorepo:**

- Fixed `.cargo/config.toml` hardcoding absolute `c:/rtools45/...` paths for the `x86_64-pc-windows-gnu` linker and ar tool. Replaced with bare tool names resolved via `PATH`, matching the existing fix in `bindings/r/src/cargo-config.toml`.

**loess-rs:**

- Improved `MismatchedInputs` error: added a `dimensions` field and updated the message to show the expected x length (`y_len × dimensions`), making it self-explanatory for both 1-D and multi-dimensional mismatches.

**C++:**

- Fixed `make cpp` Windows CI failure (`cannot find -lgcc_eh`): the C++ binding's Makefile detected MinGW via `gcc -dumpmachine` and selected the GNU target, which then used the Rtools cross-compiler from the workspace `.cargo/config.toml`; that compiler delegated to `C:\mingw64\bin\ld.exe`, which lacks `lgcc_eh`. Fixed by always targeting `x86_64-pc-windows-msvc` on Windows, removing the MinGW detection branch entirely.

**R:**

- Fixed Windows arm64 (R-Universe) build: `ar x` without a member name correctly resolves long-name archive entries (>16 chars stored as `/<offset>`); named extraction silently fails for such entries. Used `objcopy --remove-section=.idata$4` on each extracted `.dll` stub to strip the invalid relocations that lld 19 rejects, then `ar r` to re-insert.
- Fixed `ld.lld` crashing or dropping symbols (`WakeByAddressSingle`, `WaitOnAddress`) on Windows arm64: `--whole-archive` pulls every crate's raw-dylib stub for a given DLL into the link, but different crates' stubs cover different, non-overlapping symbols of that DLL — `--allow-multiple-definition` works on x86_64 but crashes lld's arm64pe backend. Fixed by dropping `--whole-archive` on `gnullvm`; normal archive resolution applies and nothing is lost since `entrypoint.c` already references the extendr init symbol directly.
- Fixed CRAN Windows build (`error: linker not found`): `cargo-config.toml` hardcoded linker/ar as `c:/rtools45/...` absolute paths, which break when Rtools is installed on a different drive. Fixed by using bare tool names resolved via `PATH`.
- Fixed CRAN Windows build (`cannot find -lgcc_eh`): the Rtools gcc lib directory is not writable on CRAN's server, and config-file `rustflags` does not reach build-script linker invocations. `Makevars.win` creates an empty stub via `touch` in `$(TARGET_DIR)/libgcc_mock/` and passes `LIBRARY_PATH` inline on `cargo build`. The path is resolved to an absolute path via `$(pwd)` at shell execution time — a relative path silently fails because Cargo invokes GCC to link build scripts from its own temp directory, not from `src/`.
- Fixed `Loess(fraction = 0.3, 4)` incorrectly succeeding: `reject_extra_positional_args()` counted unnamed arguments but did not check their position, so a single unnamed arg in any non-first slot passed validation. The check now rejects any unnamed argument that is not in position 1.
- Fixed `fit()` and `process_chunk()` silently flattening a matrix `x` and producing a confusing Rust-level length-mismatch error when `dimensions` was not set to match `ncol(x)`. Both methods now raise an informative error at the R level, naming the `dimensions` parameter to fix.

**Julia:**

- Fixed `fit(l::Loess, x::Matrix{Float64}, y)` not validating that `size(x, 2) == l.dimensions` before flattening the matrix. If the column count differed from the configured dimensions, the library either silently used wrong data or produced a confusing C-level error. The `Loess` struct now stores `dimensions` as a field, and the matrix overload checks `size(x, 2) != l.dimensions` upfront with a clear message naming the parameter to fix.
- Fixed `FastLOESS.jl` never actually loading the prebuilt `fastloess_jll` binary: `find_library()` only checked the `FASTLOESS_LIB` env var and local dev-mode paths, so the package installed from the registry had no working native library for end users. Added the `fastloess_jll` dependency (`Project.toml`), a JLL-loading branch in `find_library()`, and switched from an eager `const libfastloess = find_library()` (resolved once at precompile time) to a lazy `current_library()` accessor re-resolved in `__init__()`.

**Python:**

- Enforced keyword-only arguments beyond the first positional allowance in `Loess`, `StreamingLoess`, and `OnlineLoess`, matching R's behaviour: `Loess(fraction, *, ...)`, `StreamingLoess(fraction, chunk_size, *, ...)`, `OnlineLoess(fraction, window_capacity, min_points, *, ...)`. The `.pyi` stubs were updated with the same `*` separator.

## 1.0.0

### Added

**R:**

- Introduced S3 generics `fit()`, `process_chunk()`, `finalize()`, and `add_point()`, replacing the previous list-closure API.
- `bindings/r/Makefile` now auto-installs [Air](https://posit-dev.github.io/air/) if missing, before running `air format`.
- Added a `reject_extra_positional_args()` helper to reject extra unnamed arguments.

### Fixed

**R:**

- Fixed incorrect URLs in R binding docs.

**Julia:**

- Fixed `LoessResult.iterations_used` returning the raw FFI sentinel `-1` instead of `nothing` when robustness iterations were not applicable.

**WASM:**

- Fixed `OnlineLoess.add_point()` returning `undefined` instead of `null` when the sliding window has not yet accumulated enough points.

### Changed

**Monorepo:**

- Removed `dev/isolate_cargo.py`, `dev/check_root_cargo.py`, `dev/fix_doc_snippets.py`, and `check_js_licenses.js` — workspace isolation, doc-snippet transformation, and license checks are no longer needed.
- Split the monolithic root `Makefile` into per-crate/binding sub-Makefiles (e.g. `crates/loess-rs/Makefile`, `bindings/r/Makefile`), each invokable directly via `make -f path/Makefile`. The root `Makefile` now only aggregates (`docs`, `check-msrv`, `all*`).
- Moved Rust and binding tests into their respective crate/binding directories (e.g. `tests/loess-rs/` → `crates/loess-rs/tests/loess-rs/`, `tests/cpp/` → `bindings/cpp/tests/`). Removed the standalone `tests/` workspace packages.

**loess-rs:**

- Renamed `OnlineOutput`'s `smoothed` and `std_error` fields to `y` and `standard_error`, matching `LoessResult`. This is a **breaking change**.
- Updated `wide` to v1.6.

**Python:**

- Renamed `OnlineOutput`'s `smoothed` and `std_error` properties to `y` and `standard_error`. This is a **breaking change**.

**R:**

- Renamed the `smoothed` and `std_error` fields returned by `OnlineLoess`'s `add_point()` to `y` and `standard_error`. This is a **breaking change**.
- Replaced `dev/style_pkg.R` with [Air](https://posit-dev.github.io/air/) for formatting.
- Removed `dev/fix_rd_style.R`, `dev/prepare_cargo.py`, `dev/patch_vendor_crates.py`, `dev/clean_checksums.py`, and `dev/prepare_cran.sh` — their logic is now inlined directly in `bindings/r/Makefile`, so the R build no longer requires any Python scripts.
- Added `...` to `Loess()`, `StreamingLoess()`, and `OnlineLoess()` to force named arguments for optional parameters.
- Added `Depends: R (>= 4.6)` to `DESCRIPTION` and a matching CI matrix entry.
- Expanded roxygen2 `@param` docs and added a `See Also` section linking to <https://loess.readthedocs.io/>.
- Expanded `rfastloess-intro.Rmd` vignettes.

**Julia:**

- Renamed `OnlineOutput`'s `smoothed` and `std_error` fields to `y` and `standard_error`. This is a **breaking change**.
- Removed `dev/format_julia.jl`; formatting is now inlined in `bindings/julia/Makefile`.

**Node.js:**

- Renamed `OnlineOutput`'s `smoothed` and `std_error` fields to `y` and `standard_error`. This is a **breaking change**.
- Updated `@napi-rs/cli` to v3.8 and `oxlint` to v1.79.

**WASM:**

- Renamed `OnlineOutput`'s `smoothed` and `std_error` getters to `y` and `standard_error`. This is a **breaking change**.
- Updated `oxlint` to v1.79.

**C++:**

- Renamed `OnlineOutput`'s `smoothed()` and `std_error()` methods to `y()` and `standard_error()`. This is a **breaking change**.

**Docs:**

- Split `StreamingLoess`/`OnlineLoess` content out of each binding's main API reference page into dedicated `{lang}-streaming.md`/`{lang}-online.md` files.
- Moved the `tutorials/` pages into a new `user-guide/use-cases/` section.
- Standardized `docs/api/` code examples across every binding, with expected output comments.
- `dev/verify_snippets.py` now also runs the R code chunks in vignettes.

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
