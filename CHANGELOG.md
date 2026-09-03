<!-- markdownlint-disable MD024 MD046 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

**Monorepo:**

- Added `dev/bump_version.py --version X.Y.Z`, which updates the version across every crate/binding's version files, `CITATION.cff`, and the Spack recipe in one pass. Supports `--dry-run`.
- Added `dev/check_pinned_versions.py` and a weekly `.github/workflows/check-versions.yml`, which check hardcoded tool/library version pins that Dependabot can't see and fail CI if any are outdated.
- Added `.github/dependabot.yml`, covering every dependency ecosystem in the repo, grouped per directory into a single weekly PR.
- Added an optional `commit` input to every release workflow's `workflow_dispatch` trigger, so a manual run can pin the checked-out/built commit instead of always building from the triggering ref.
- Added `dev/check_links.py`, which validates every Markdown cross-reference link across every binding/crate's docs, failing on missing targets.

**Node.js:**

- Added two prebuilt targets, `aarch64-unknown-linux-musl` (via cargo-zigbuild) and `armv7-unknown-linux-gnueabihf` (via an apt cross toolchain), with matching optional npm subpackages.

**C++:**

- Added CMake package-config support (`fastloessConfig.cmake`/`fastloessConfigVersion.cmake`), so consumers can `find_package(fastloess)` and link against `fastloess::fastloess` regardless of their own compiler/build setup.
- Added CI coverage for four additional compiler/toolchain combinations in `ci-cpp.yml`: `clang-cl` on Windows, `clang` on Linux, native MinGW-w64, and Intel oneAPI (last two non-blocking).
- Added ARM64 release binaries to `release-cpp.yml` for Linux, Windows, and macOS. Also re-pinned the macOS x64 job to `macos-13`, since `macos-latest` had been silently shipping an arm64 binary mislabeled as x64.
- Set `no_includes = true` in `cbindgen.toml` and renamed `cpp_loess_fit`/`cpp_streaming_process`'s `x`/`y` parameters to `x_values`/`y_values`, avoiding a naming collision with `CppLoessResult`'s own `x`/`y` output fields.

### Changed

**Monorepo:**

- Added a `large` benchmark category to `benchmarks/rfastloess.R`/`stats_loess.R` stressing exact-fit, high-iteration, and high-fraction scenarios at scale, and documented it in `benchmarks/README.md` with measured `stats::loess` vs. fastLoess median times.
- Merged `dev/add-cpp-outputs.py`, `dev/add-rust-outputs.py`, `dev/add-nodejs-outputs.js`, and `dev/add-wasm-outputs.js` into `dev/verify_snippets.py` as a new `--update-outputs` flag, removing the four standalone scripts.
- Replaced Unicode superscript/subscript characters used as ASCII-alphanumeric stand-ins (`R²` → `R2`, `O(n²)` → `O(n^2)`, `xᵢ` → `x_i`, etc.) with plain ASCII throughout every doc page, README, and Rust doc-comment/test-comment, including the `Diagnostics` `Display` impl (now prints `R2`). Regenerated the affected ```output blocks, which also caught a handful of pre-existing `RÂ²` mojibake blocks left over from before the Windows encoding fix above.
- Added `dev/add-readme-to-docs.py`, which auto-detects the docs-site flavor (Starlight for Node.js/WASM, Sphinx for Python) and embeds `README.md` as the home page accordingly. Not wired into Python's `Makefile` yet, since the Sphinx branch currently embeds the raw GitHub README verbatim rather than anything tailored to the Sphinx site.

**docs:**

- Harmonized the docs-site directory structure across every binding and crate (`introduction/`, `guide/`, `weighting/`, `advanced/`, `use-case/`, `api/`, each grouped under a hub page). Also fixed the doc-tooling scripts, which enumerated files via a non-recursive glob and would have silently stopped finding snippets in the newly-nested pages.
- Consolidated every crate/binding README: merged the "Installation" and "Documentation" sections, replaced GitHub-only alert syntax with plain blockquotes, removed the redundant "API Reference" and "Changelog" sections (each now has its own docs-site page), and added a "Read more" link to the Concepts page.
- Renamed the batch adapter's "When to Use" heading to "When to Use Batch Adapter" across every binding/crate's API docs.
- Vendored the [doxygen-awesome-css](https://github.com/jothepro/doxygen-awesome-css) theme (v2.4.2) for a modern, sidebar-only cpp Doxygen site with automatic dark mode.
- Added `dev/update_changelogs.py`, which regenerates a per-binding/crate `NEWS.md`/`news.md` from the root `CHANGELOG.md`.
- Replaced `kernels.md`'s "Choosing a Kernel" mermaid flowchart (every binding/crate) with an equivalent decision table, since Doxygen and rustdoc don't render mermaid.
- Replaced `adapter-choice.md`/`adapters.md`'s "Overview" flowchart with an equivalent decision table, unifying on a single rendering-agnostic format across every binding/crate.
- Consolidated `parameters.md`/the auto-generated `@autodocs` parameter reference into each `api.md`'s builder/options tables, and removed `parameters.md` itself.

**C++:**

- Documented the previously-undocumented `x_values`/`y_values` parameters of `Loess::fit()`, fixing a Doxygen "parameters are not documented" warning.
- Restructured the Doxygen site's navigation, previously ~20 flat pages, into five nested hub pages (`Getting Started`, `User Guide`, `Customization`, `Advanced`, `Use Cases`) via `\subpage`, mirroring the R/Node.js/WASM sidebar grouping.
- Added a Spack recipe (`bindings/cpp/spack/package.py`); `release-cpp.yml` now updates its `version()`/`sha256` and opens a PR to `spack/spack-packages` on every release, so `fastloess-cpp` stays installable via `spack install`.
- Bumped the vendored Corrosion CMake module from `v0.5.1` to `v0.6.1`.

**R:**

- Removed the `rfastloess-package` pkgdown topic, which duplicated the adapter class list, and unexported the internal `Nullable()` helper.
- Fixed `_pkgdown.yml` describing the core interface as "R6 classes" when the package actually uses S3 classes.
- Merged `vignettes/parameters.Rmd`'s parameter reference into the `@param`/`@details` roxygen docs of `Loess()`, `StreamingLoess()`, and `OnlineLoess()`, and removed the now-redundant vignette.
- Merged `vignettes/batch.Rmd`, `streaming.Rmd`, and `online.Rmd`'s unique content into the `@description`/`@details` roxygen docs of `Loess()`, `StreamingLoess()`, and `OnlineLoess()`, and removed the now-redundant vignettes and their orphaned diagrams.

**Node.js:**

- Updated `oxlint` to v1.81, `napi` to v3.12, `napi-derive` to v3.6, `napi-build` to v2.4, and `typedoc-plugin-markdown` to v4.13.
- `make nodejs-dev` now runs `npm update` after `npm install`, so dependencies are kept current.

**WASM:**

- Updated `oxlint` to v1.81 and `typedoc-plugin-markdown` to v4.13.
- `make wasm-dev` now runs `npm update` after `npm install`, so dependencies are kept current.

**loess-rs:**

- Updated `wide` to v1.7.

### Fixed

**Monorepo:**

- Fixed every benchmark category in `benchmarks/rfastloess.R` failing with `attempt to apply non-function`: it called `model$fit(x, y)`, but `fit` is an S3 generic (`fit(model, x, y)`), not a field on the `Loess` object.
- Fixed `release-conda.yml`'s `sed` pattern for `recipe.yaml`'s version, which only matched an exactly 2-space-indented `version:` line; now matches any `version: "X.Y.Z"` line by semver shape.
- Fixed `benchmarks/Makefile`'s vendoring step nulling out the original package checksum for every crates.io dependency, not just the two local path crates; now uses `jq '.files = {}'` to reset only the `files` map.
- Fixed `benchmarks/README.md`'s "Iterations" scenario row claiming "0 – 10 (6 levels)"; `stats::loess` has no meaningful `iterations = 0` setting, so it only ever ran 5 levels.
- Fixed `docs.yml`'s Pages deployment: consolidated per-language jobs into a single artifact upload/deploy job, then switched from legacy branch-based deployment to `actions/upload-pages-artifact`/`actions/deploy-pages`, eliminating GitHub's automatic "pages build and deployment" trigger. Requires the repository's Pages source to be switched to "GitHub Actions" in settings.
- Fixed 51 broken relative cross-reference links across C++, Julia, Node.js, WASM, and both Rust crates' docs, left over from the earlier docs-site restructuring into hub-grouped subdirectories. Found via the new `dev/check_links.py`.
- Fixed `OnlineLoess`'s `min_points` default being `3` instead of `2`, and `update_mode` defaulting to `"full"` instead of `"incremental"` — both diverging from every binding's docs. Originated in the Rust core and `fastLoess`'s shared `build_online` fallback, so every binding that duplicated the defaults had inherited the bug: C++, Julia, Python, and R. Updated the "Default" column in every binding's `api-online.md` to match and fixed a Rust core test that asserted the old threshold.

**docs:**

- Fixed the "Handling Outliers" quickstart example (every binding and both Rust crates) printing nothing: with only 6 points and `fraction = 0.5`, the local window was small enough that a degree-1 fit reproduced the outlier exactly (zero residual). Bumped to `fraction = 0.7`, which correctly downweights it.
- Fixed the R `OnlineLoess()` roxygen example printing one line per point (48 lines for a 50-point loop); it now collects the smoothed values and prints only `head(smoothed, 5)`.
- Fixed the R `add_point()` roxygen example always printing `NULL`, since a single call never reaches the default `min_points = 3`; it now uses `min_points = 2L` and shows the second (non-`NULL`) call's result.
- Fixed the Julia `intervals.md` "Confidence Intervals" and "Standard Errors" examples each looping over all 100 points instead of a short sample; switched to `result.y[1:5]`/`result.confidence_lower[1:5]`/`result.standard_errors[1:5]`-style slicing, matching the already-concise Python version.

**C++:**

- Fixed several Doxygen rendering bugs: wrong homepage, and blockquotes/math/admonitions rendering as literal or broken text. `README.md` is now the Doxygen main page, using Doxygen-native syntax.
- Fixed `ci-cpp.yml`'s macOS job warning about the untrusted `aws/tap` Homebrew tap, and its Windows job installing a broken `cppcheck` via Chocolatey (missing `cfg/std.cfg`); `brew untap aws/tap` now runs first on macOS, and `cppcheck` now installs via `winget` on Windows.
- Fixed `Doxyfile`'s `PROJECT_NAME` showing `"fastLoess"` (the Rust crate's name) instead of `"fastloess-cpp"`, and its `FILE_PATTERNS` missing a space (`*.hpp*.h`, parsed as one malformed glob); changed to `*.hpp *.h *.md`.

**Julia:**

- Fixed the Documenter homepage: it was a stale, separately maintained `index.md`; `make.jl` now regenerates it from `README.md` on every build.
- Fixed `release-julia-register.yml` pulling release notes from the root `CHANGELOG.md` (every binding's entries); it now extracts from the Julia-filtered `NEWS.md`.
- Fixed `make julia-dev` failing to resolve an already-pinned but outdated `fastloess_jll`; the `dev` target now runs `Pkg.update("fastloess_jll")` before `Pkg.resolve()`.

**Node.js:**

- Fixed the docs homepage never showing the README content: `dev/add-readme-to-docs.py` now embeds `README.md` below the hero, wired into `npm run docs` and `make nodejs-dev`.
- Fixed the docs build emitting an `@astrojs/sitemap` warning when `SITE` isn't set locally; `astro.config.mjs` now falls back to the production URL.
- Fixed every "API Reference" link 404ing due to a TypeDoc/Starlight casing mismatch; a new `dev/lowercase-typedoc-refs.js` script normalizes generated file names and links.
- Fixed `astro build` failing since Astro 7 no longer bundles `@astrojs/markdown-remark`; added it as an explicit devDependency.

**WASM:**

- Same fix as Node.js: `README.md` is now embedded via `dev/add-readme-to-docs.py`, wired into `npm run docs` and `make wasm-dev`.
- Fixed `concepts.md` figures (MkDocs-only `<figure>`/attr_list syntax) not rendering; converted to plain images with italicized captions.
- Fixed inline/display LaTeX math rendering as literal text; wired `remark-math`/`rehype-katex` into `astro.config.mjs`.
- Fixed the same `@astrojs/sitemap` warning, "API Reference" 404s, and `astro build`/`@astrojs/markdown-remark` failure as Node.js, via the same fixes.

**Python:**

- Fixed the "API Reference" page rendering empty: its `api/index.md` toctree still referenced the pre-rename document names; updated to match the files' current names.
- Fixed `release-pypi.yml`'s macOS/Windows jobs printing a pip version-check notice on every run; added `PIP_DISABLE_PIP_VERSION_CHECK: "1"`.

**Rust:**

- Fixed inline/display LaTeX math rendering as literal text on docs.rs; added a `katex-header.html` that renders it client-side.
- Fixed every cross-reference link across the Rust crate docs leading nowhere, since plain relative links aren't resolved against the rustdoc module tree; converted them to proper intra-doc links (e.g. `crate::doc::concepts`).

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
