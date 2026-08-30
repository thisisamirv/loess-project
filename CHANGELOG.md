<!-- markdownlint-disable MD024 MD046 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

**docs:**

- Consolidated every crate/binding README: merged the "Installation" and "Documentation" sections, replaced GitHub-only alert syntax with plain blockquotes, removed the redundant "API Reference" and "Changelog" sections (each now has its own docs-site page), and added a "Read more" link to the Concepts page. The top-level repository README is unchanged, since it's only ever viewed on GitHub.
- Renamed the batch adapter's "When to Use" heading to "When to Use Batch Adapter" across every binding/crate's API docs.
- Vendored the [doxygen-awesome-css](https://github.com/jothepro/doxygen-awesome-css) theme (v2.4.2) for a modern, sidebar-only cpp Doxygen site with automatic dark mode.
- Added `dev/update_changelogs.py`, which regenerates a per-binding/crate `NEWS.md`/`news.md` from the root `CHANGELOG.md`. Wired into every docs site's navigation, the Rust crates' rustdoc module tree, and every `Makefile` `dev` target.
- Replaced `kernels.md`'s "Choosing a Kernel" mermaid flowchart (every binding/crate) with an equivalent decision table, since Doxygen and rustdoc don't render mermaid and the deeply-nested diamond chain was hard to read even where it did render.
- Replaced `adapter-choice.md`/`adapters.md`'s "Overview" flowchart (mermaid in most bindings/crates, ASCII art in the C++ docs) with an equivalent decision table, unifying on a single rendering-agnostic format across every binding/crate.
- Consolidated `parameters.md`/the auto-generated `@autodocs` parameter reference across every binding and crate (C++, Julia, Node.js, Python, WASM, `fastLoess`, `loess-rs`): merged its unique content (fraction/iterations choice guidance, and inline `zero_weight_fallback`/`surface_mode` behavior tables) into each `api.md`'s builder/options tables (Julia: into the `Loess`/`StreamingLoess`/`OnlineLoess` docstrings), and removed `parameters.md` itself along with its docs-site navigation entries, `doc::parameters` rustdoc module, and cross-references (now pointing at `api.md`) — the parameter tables, kernel/robustness/boundary/scaling/degree/distance-metric option lists, and interval/custom-weights code examples it duplicated already live on their own dedicated pages.

**R:**

- Removed the `rfastloess-package` pkgdown topic, which duplicated the adapter class list, and unexported the internal `Nullable()` helper.
- Fixed `_pkgdown.yml` describing the core interface as "R6 classes" when the package actually uses S3 classes.
- Merged `vignettes/parameters.Rmd`'s parameter reference (ranges, defaults, and fraction-choice guidance) into the `@param`/`@details` roxygen docs of `Loess()`, `StreamingLoess()`, and `OnlineLoess()`, and removed the now-redundant vignette.
- Merged `vignettes/batch.Rmd`, `streaming.Rmd`, and `online.Rmd`'s unique content (When to Use guidance, merge strategy comparison) into the `@description`/`@details` roxygen docs of `Loess()`, `StreamingLoess()`, and `OnlineLoess()`, and removed the now-redundant vignettes and their orphaned `gap_handling.svg`/`online_comparison.svg` diagrams.

**Node.js:**

- Updated `typedoc-plugin-markdown` to v4.13.
- `make nodejs-dev` now runs `npm update` after `npm install`, so dependencies are kept current.

**WASM:**

- Updated `typedoc-plugin-markdown` to v4.13.
- `make wasm-dev` now runs `npm update` after `npm install`, so dependencies are kept current.

### Fixed

**Monorepo:**

- Fixed `docs.yml` triggering GitHub's "pages build and deployment" once per docs job; per-language jobs now upload artifacts, and a single final `deploy` job pushes to `gh-pages` once per run.
- Fixed `docs.yml`'s reliance on GitHub's legacy branch-based Pages deployment, which auto-triggers an unpinned, GitHub-managed "pages build and deployment" job on every `gh-pages` push (surfacing deprecation warnings, e.g. for Node.js 20, that aren't fixable from this repo). The former `deploy` job is now `build`, which still pushes the merged `_site` to `gh-pages` as a cache for future incremental runs, but publishing now goes through `actions/upload-pages-artifact` and a new `deploy` job using the official `actions/deploy-pages`, which this repo pins directly. Requires the repository's Pages source to be switched to "GitHub Actions" in settings.

**docs:**

- Fixed the "Handling Outliers" quickstart example (every binding and the `loess-rs`/`fastLoess` crates) printing nothing: with only 6 points and `fraction = 0.5`, the local window is small enough that tricube weighting drives the farthest neighbor's weight to ~0, leaving just 2 effectively-weighted points, which a degree-1 fit reproduces exactly (zero residual, no downweighting) — confirmed directly against the `lowess`/`loess` core, not binding-specific. Bumped to `fraction = 0.7`, which correctly downweights the injected outlier.
- Fixed the R `OnlineLoess()` roxygen example printing one line per point (48 lines for a 50-point loop); it now collects the smoothed values and prints only `head(smoothed, 5)`.
- Fixed the R `add_point()` roxygen example always printing `NULL`, since a single call never reaches the default `min_points = 3`; it now uses `min_points = 2L` and shows the second (non-`NULL`) call's result.
- Fixed the Julia `intervals.md` "Confidence Intervals" and "Standard Errors" examples each looping over all 100 points instead of a short sample; switched to `result.y[1:5]`/`result.confidence_lower[1:5]`/`result.standard_errors[1:5]`-style slicing, matching the already-concise Python version.

**C++:**

- Fixed several Doxygen rendering bugs: the homepage showed `concepts.md` instead of `README.md`; blockquotes, heading+codespan combinations, MkDocs-only admonitions, inline/display math, and `---` after a blockquote all rendered as literal or broken text. `README.md` is now the Doxygen main page, and the affected docs use Doxygen-native syntax (`\f$...\f$`/`\f[...\f]` math, blockquote admonitions, explicit `<hr>`).
- Fixed `ci-cpp.yml`'s macOS job warning that the pre-installed `aws/tap` Homebrew tap is untrusted; `brew untap aws/tap` now runs before `brew install llvm cppcheck`, since that tap isn't needed for this build.
- Fixed `ci-cpp.yml`'s Windows job installing `cppcheck` via Chocolatey, whose package is missing its `cfg/std.cfg` library files, causing `make cpp-dev`'s static analysis pass to be silently skipped; it now installs `cppcheck` via `winget` instead (matching the already-working `install-tools` target), with its install directory added to `$GITHUB_PATH`.
- Fixed `Doxyfile`'s `PROJECT_NAME` showing `"fastLoess"` (the separate Rust crate's name) instead of the actual CMake project/library name; changed to `"fastloess-cpp"`.
- Fixed `Doxyfile`'s `FILE_PATTERNS` missing a space (`*.hpp*.h`), which Doxygen parses as a single malformed glob instead of two separate `*.hpp`/`*.h` patterns; changed to `*.hpp *.h *.md`.

**Julia:**

- Fixed the Documenter homepage: it was a stale, separately maintained `index.md` instead of the README, and the README's centered badge/logo HTML and markdownlint comment rendered as literal text. `make.jl` now regenerates `index.md` from `README.md` on every build.
- Fixed `release-julia-register.yml` extracting the matching version section from the root `CHANGELOG.md`, which includes every binding/crate's entries; it now extracts from the already Julia-filtered `bindings/julia/julia/docs/src/NEWS.md` instead, so the JuliaRegistrator release notes only cover Julia-relevant changes.
- Fixed `make julia-dev` failing with "empty intersection between `fastloess_jll@X.Y.Z` and project compatibility ..." whenever a locally cached `Manifest.toml` still pinned an older `fastloess_jll` version after a new one was published: `Pkg.resolve()` treats an already-pinned manifest entry as fixed and won't search the registry for an upgrade, even when the relaxed compat bound requires one. The `dev` target now runs `Pkg.update("fastloess_jll")` before `Pkg.resolve()` to actively pick up the newly published version.

**Node.js:**

- Fixed the docs homepage never showing the README content ("Get Started" jumped straight to Installation): a new `dev/add-readme-to-docs.js` script embeds `README.md` below the hero (stripping its redundant `# LOESS Project` H1, since the hero already shows the title), wired into `npm run docs` and `make nodejs-dev`.
- Fixed the docs build always emitting a `[@astrojs/sitemap] The Sitemap integration requires the site astro.config option` warning when the `SITE` environment variable isn't set (e.g. local builds); `astro.config.mjs` now falls back to the production GitHub Pages URL.
- Fixed every link on the "API Reference" page 404ing: TypeDoc's markdown output preserves the original TypeScript symbol casing (e.g. `classes/Loess.md`) and Astro renders those relative links verbatim, but Starlight always lowercases content-collection route slugs (e.g. `classes/loess/`) and never actually strips the `.md` extension automatically. A new `dev/lowercase-typedoc-refs.js` script lowercases every generated reference file name and rewrites their internal links (stripped of `.md`, also lowercased) after `typedoc` runs and before `astro build`, wired into `npm run docs`.

**WASM:**

- Same fix as Node.js: `README.md` is now embedded via `dev/add-readme-to-docs.js`, wired into `npm run docs` and `make wasm-dev`.
- Fixed `concepts.md` figures (MkDocs-only `<figure>`/attr_list syntax) not rendering; converted to plain images with italicized captions.
- Fixed inline/display LaTeX math rendering as literal text; wired `remark-math`/`rehype-katex` into `astro.config.mjs`.
- Fixed the same `@astrojs/sitemap` warning as Node.js, with the same fallback in `astro.config.mjs`.
- Fixed the same "API Reference" 404s as Node.js, via the same `dev/lowercase-typedoc-refs.js` script.

**Python:**

- Fixed the "API Reference" page rendering empty: its `api/index.md` toctree still referenced the pre-rename `python`/`python-streaming`/`python-online` document names; updated to `api`/`api-streaming`/`api-online`, matching the files' current names. Sphinx toctree entries omit the `.md` extension, so this was missed by the earlier rename's link verification.

**Rust:**

- Fixed inline/display LaTeX math rendering as literal text on docs.rs; added a `katex-header.html` that renders it client-side with KaTeX.
- Fixed every cross-reference link across the `loess-rs`/`fastLoess` crate docs (`quickstart.md`, `api.md`, `concepts.md`, and others) leading nowhere: these pages are embedded into rustdoc via `#![doc = include_str!(...)]`, and plain relative links like `[Concepts](concepts.md)` are rendered verbatim rather than resolved against the generated module tree. Converted every such link to a proper intra-doc link (e.g. `[Concepts](crate::doc::concepts)`), validated with `cargo doc --all-features` under `-D warnings` (zero broken-link warnings).

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
- Fixed `FastLOESS.jl` never actually loading the prebuilt `fastloess_jll` binary: `find_library()` only checked the `FASTLOESS_LIB` env var and local dev-mode paths, so the package installed from the registry had no working native library for end users. Added the `fastloess_jll` dependency (`Project.toml`), a JLL-loading branch in `find_library()`, and switched from an eager `const libfastloess = find_library()` (resolved once at precompile time) to a lazy `current_library()` accessor re-resolved in `__init__()`, matching `FastLOWESS.jl`.

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
