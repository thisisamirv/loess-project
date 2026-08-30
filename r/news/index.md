# Changelog

## rfastloess (development version)

### Changed

- Consolidated every crate/binding README: merged the “Installation” and
  “Documentation” sections, replaced GitHub-only alert syntax with plain
  blockquotes, removed the redundant “API Reference” and “Changelog”
  sections (each now has its own docs-site page), and added a “Read
  more” link to the Concepts page. The top-level repository README is
  unchanged, since it’s only ever viewed on GitHub.
- Renamed the batch adapter’s “When to Use” heading to “When to Use
  Batch Adapter” across every binding/crate’s API docs.
- Vendored the
  [doxygen-awesome-css](https://github.com/jothepro/doxygen-awesome-css)
  theme (v2.4.2) for a modern, sidebar-only cpp Doxygen site with
  automatic dark mode.
- Added `dev/update_changelogs.py`, which regenerates a
  per-binding/crate `NEWS.md`/`news.md` from the root `CHANGELOG.md`.
  Wired into every docs site’s navigation, the Rust crates’ rustdoc
  module tree, and every `Makefile` `dev` target.
- Replaced `kernels.md`’s “Choosing a Kernel” mermaid flowchart (every
  binding/crate) with an equivalent decision table, since Doxygen and
  rustdoc don’t render mermaid and the deeply-nested diamond chain was
  hard to read even where it did render.
- Replaced `adapter-choice.md`/`adapters.md`’s “Overview” flowchart
  (mermaid in most bindings/crates, ASCII art in the C++ docs) with an
  equivalent decision table, unifying on a single rendering-agnostic
  format across every binding/crate.
- Consolidated `parameters.md`/the auto-generated `@autodocs` parameter
  reference across every binding and crate (C++, Julia, Node.js, Python,
  WASM, `fastLoess`, `loess-rs`): merged its unique content
  (fraction/iterations choice guidance, and inline
  `zero_weight_fallback`/`surface_mode` behavior tables) into each
  `api.md`’s builder/options tables (Julia: into the
  `Loess`/`StreamingLoess`/`OnlineLoess` docstrings), and removed
  `parameters.md` itself along with its docs-site navigation entries,
  `doc::parameters` rustdoc module, and cross-references (now pointing
  at `api.md`) — the parameter tables,
  kernel/robustness/boundary/scaling/degree/distance-metric option
  lists, and interval/custom-weights code examples it duplicated already
  live on their own dedicated pages.
- Removed the `rfastloess-package` pkgdown topic, which duplicated the
  adapter class list, and unexported the internal `Nullable()` helper.
- Fixed `_pkgdown.yml` describing the core interface as “R6 classes”
  when the package actually uses S3 classes.
- Merged `vignettes/parameters.Rmd`’s parameter reference (ranges,
  defaults, and fraction-choice guidance) into the `@param`/`@details`
  roxygen docs of
  [`Loess()`](https://thisisamirv.github.io/loess-project/r/reference/Loess.md),
  [`StreamingLoess()`](https://thisisamirv.github.io/loess-project/r/reference/StreamingLoess.md),
  and
  [`OnlineLoess()`](https://thisisamirv.github.io/loess-project/r/reference/OnlineLoess.md),
  and removed the now-redundant vignette.
- Merged `vignettes/batch.Rmd`, `streaming.Rmd`, and `online.Rmd`’s
  unique content (When to Use guidance, merge strategy comparison) into
  the `@description`/`@details` roxygen docs of
  [`Loess()`](https://thisisamirv.github.io/loess-project/r/reference/Loess.md),
  [`StreamingLoess()`](https://thisisamirv.github.io/loess-project/r/reference/StreamingLoess.md),
  and
  [`OnlineLoess()`](https://thisisamirv.github.io/loess-project/r/reference/OnlineLoess.md),
  and removed the now-redundant vignettes and their orphaned
  `gap_handling.svg`/`online_comparison.svg` diagrams.

### Fixed

- Fixed `docs.yml` triggering GitHub’s “pages build and deployment” once
  per docs job; per-language jobs now upload artifacts, and a single
  final `deploy` job pushes to `gh-pages` once per run.
- Fixed `docs.yml`’s reliance on GitHub’s legacy branch-based Pages
  deployment, which auto-triggers an unpinned, GitHub-managed “pages
  build and deployment” job on every `gh-pages` push (surfacing
  deprecation warnings, e.g. for Node.js 20, that aren’t fixable from
  this repo). The former `deploy` job is now `build`, which still pushes
  the merged `_site` to `gh-pages` as a cache for future incremental
  runs, but publishing now goes through `actions/upload-pages-artifact`
  and a new `deploy` job using the official `actions/deploy-pages`,
  which this repo pins directly. Requires the repository’s Pages source
  to be switched to “GitHub Actions” in settings.
- Fixed the “Handling Outliers” quickstart example (every binding and
  the `loess-rs`/`fastLoess` crates) printing nothing: with only 6
  points and `fraction = 0.5`, the local window is small enough that
  tricube weighting drives the farthest neighbor’s weight to ~0, leaving
  just 2 effectively-weighted points, which a degree-1 fit reproduces
  exactly (zero residual, no downweighting) — confirmed directly against
  the `lowess`/`loess` core, not binding-specific. Bumped to
  `fraction = 0.7`, which correctly downweights the injected outlier.
- Fixed the R
  [`OnlineLoess()`](https://thisisamirv.github.io/loess-project/r/reference/OnlineLoess.md)
  roxygen example printing one line per point (48 lines for a 50-point
  loop); it now collects the smoothed values and prints only
  `head(smoothed, 5)`.
- Fixed the R
  [`add_point()`](https://thisisamirv.github.io/loess-project/r/reference/add_point.md)
  roxygen example always printing `NULL`, since a single call never
  reaches the default `min_points = 3`; it now uses `min_points = 2L`
  and shows the second (non-`NULL`) call’s result.
- Fixed the Julia `intervals.md` “Confidence Intervals” and “Standard
  Errors” examples each looping over all 100 points instead of a short
  sample; switched to
  `result.y[1:5]`/`result.confidence_lower[1:5]`/`result.standard_errors[1:5]`-style
  slicing, matching the already-concise Python version.

## rfastloess 1.1.0

### Added

- Added a GitHub Pages landing page at the repository root, built from
  `README.md` via pandoc and deployed by `docs.yml`.
- Added a GitHub workflow for running validation scripts.
- Added `lenght` gaurds for extra arguments.

### Changed

- Split the monolithic `.github/workflows/ci.yml` into seven
  per-language workflow files: `ci-rust.yml`, `ci-python.yml`,
  `ci-julia.yml`, `ci-nodejs.yml`, `ci-wasm.yml`, `ci-cpp.yml`, and
  `ci-r.yml`. Each file carries the relevant `ci` (multi-OS matrix),
  `asan`, and `gpu` jobs for its language.
- Each crate/binding sub-Makefile now runs
  `dev/verify_snippets.py --lang <lang>` for its own language as the
  final step of `make default`. The root `docs-test` target remains as a
  convenience to run all languages at once.
- Split every sub-Makefile `default:` into `default:` (build and system
  install) and `dev:` (full quality-check workflow). Both root Makefiles
  gain `<name>-dev` targets for each binding and crate, and an `all-dev`
  aggregate target.
- Split `dev/verify_snippets.py` into a lean orchestrator and a
  `dev/runners/` package. Each language has its own module (`python.py`,
  `julia.py`, `nodejs.py`, `r.py`, `wasm.py`, `rust.py`, `cpp.py`)
  containing its `run_<lang>()` function and a `skip_reason()`
  predicate. Shared types (`Snippet`, `RunResult`) and utilities live in
  `runners/base.py`; the registry (`RUNNERS`, `SKIP_CHECKS`) is exported
  from `runners/__init__.py`.
- Moved CHANGELOG and CONTRIBUTING guides to project root.
- Updated README files to be binding/crate specific instead of one
  generic README for all bindings/crates.
- Moved R documentation from ReadTheDocs to GitHub Pages, served by
  pkgdown at <https://thisisamirv.github.io/loess-project/r/>. The
  ReadTheDocs site no longer includes R-specific content.
- Simplified `bindings/r/Makefile`: replaced `Cargo.toml.orig`
  save/restore vendoring with `src/vendor-update.sh`; made `[workspace]`
  permanent in `src/Cargo.toml`; removed Bioconductor dependencies,
  redundant `cargo fmt --check`, `NAMESPACE` indentation
  post-processing, and
  [`pkgdown::build_site`](https://pkgdown.r-lib.org/reference/build_site.html)
  from the dev workflow.
- Changed R version dependency to 4.4.0 due to issues with installing
  Bioconducter packages on R \< 4.4.0.
- Replaced the multi-step `install.packages` /
  [`BiocManager::install`](https://bioconductor.github.io/BiocManager/reference/install.html)
  package installation logic in `bindings/r/Makefile` with a single
  [`pak`](https://pak.r-lib.org/)-based block. `pak` handles RSPM binary
  vs source selection automatically (including Linux), skips
  already-installed packages, and installs CRAN, Bioconductor (`bioc::`
  prefix), and R-universe packages in one call.
- `make r` (`default:`) now runs `R CMD INSTALL $(R_DIR)` directly; R’s
  `configure` script handles Rust compilation from the committed
  `vendor.tar.xz`. The full dev workflow moves to `make r-dev`.

### Fixed

- Fixed `.cargo/config.toml` hardcoding absolute `c:/rtools45/...` paths
  for the `x86_64-pc-windows-gnu` linker and ar tool. Replaced with bare
  tool names resolved via `PATH`, matching the existing fix in
  `bindings/r/src/cargo-config.toml`.
- Fixed Windows arm64 (R-Universe) build: `ar x` without a member name
  correctly resolves long-name archive entries (\>16 chars stored as
  `/<offset>`); named extraction silently fails for such entries. Used
  `objcopy --remove-section=.idata$4` on each extracted `.dll` stub to
  strip the invalid relocations that lld 19 rejects, then `ar r` to
  re-insert.
- Fixed `ld.lld` crashing or dropping symbols (`WakeByAddressSingle`,
  `WaitOnAddress`) on Windows arm64: `--whole-archive` pulls every
  crate’s raw-dylib stub for a given DLL into the link, but different
  crates’ stubs cover different, non-overlapping symbols of that DLL —
  `--allow-multiple-definition` works on x86_64 but crashes lld’s
  arm64pe backend. Fixed by dropping `--whole-archive` on `gnullvm`;
  normal archive resolution applies and nothing is lost since
  `entrypoint.c` already references the extendr init symbol directly.
- Fixed CRAN Windows build (`error: linker not found`):
  `cargo-config.toml` hardcoded linker/ar as `c:/rtools45/...` absolute
  paths, which break when Rtools is installed on a different drive.
  Fixed by using bare tool names resolved via `PATH`.
- Fixed CRAN Windows build (`cannot find -lgcc_eh`): the Rtools gcc lib
  directory is not writable on CRAN’s server, and config-file
  `rustflags` does not reach build-script linker invocations.
  `Makevars.win` creates an empty stub via `touch` in
  `$(TARGET_DIR)/libgcc_mock/` and passes `LIBRARY_PATH` inline on
  `cargo build`. The path is resolved to an absolute path via `$(pwd)`
  at shell execution time — a relative path silently fails because Cargo
  invokes GCC to link build scripts from its own temp directory, not
  from `src/`.
- Fixed `Loess(fraction = 0.3, 4)` incorrectly succeeding:
  `reject_extra_positional_args()` counted unnamed arguments but did not
  check their position, so a single unnamed arg in any non-first slot
  passed validation. The check now rejects any unnamed argument that is
  not in position 1.
- Fixed
  [`fit()`](https://thisisamirv.github.io/loess-project/r/reference/fit.md)
  and
  [`process_chunk()`](https://thisisamirv.github.io/loess-project/r/reference/process_chunk.md)
  silently flattening a matrix `x` and producing a confusing Rust-level
  length-mismatch error when `dimensions` was not set to match
  `ncol(x)`. Both methods now raise an informative error at the R level,
  naming the `dimensions` parameter to fix.

## rfastloess 1.0.0

### Added

- Introduced S3 generics
  [`fit()`](https://thisisamirv.github.io/loess-project/r/reference/fit.md),
  [`process_chunk()`](https://thisisamirv.github.io/loess-project/r/reference/process_chunk.md),
  [`finalize()`](https://thisisamirv.github.io/loess-project/r/reference/finalize.md),
  and
  [`add_point()`](https://thisisamirv.github.io/loess-project/r/reference/add_point.md),
  replacing the previous list-closure API.
- `bindings/r/Makefile` now auto-installs
  [Air](https://posit-dev.github.io/air/) if missing, before running
  `air format`.
- Added a `reject_extra_positional_args()` helper to reject extra
  unnamed arguments.

### Fixed

- Fixed incorrect URLs in R binding docs.

### Changed

- Removed `dev/isolate_cargo.py`, `dev/check_root_cargo.py`,
  `dev/fix_doc_snippets.py`, and `check_js_licenses.js` — workspace
  isolation, doc-snippet transformation, and license checks are no
  longer needed.
- Split the monolithic root `Makefile` into per-crate/binding
  sub-Makefiles (e.g. `crates/loess-rs/Makefile`,
  `bindings/r/Makefile`), each invokable directly via
  `make -f path/Makefile`. The root `Makefile` now only aggregates
  (`docs`, `check-msrv`, `all*`).
- Moved Rust and binding tests into their respective crate/binding
  directories (e.g. `tests/loess-rs/` →
  `crates/loess-rs/tests/loess-rs/`, `tests/cpp/` →
  `bindings/cpp/tests/`). Removed the standalone `tests/` workspace
  packages.
- Renamed the `smoothed` and `std_error` fields returned by
  `OnlineLoess`’s
  [`add_point()`](https://thisisamirv.github.io/loess-project/r/reference/add_point.md)
  to `y` and `standard_error`. This is a **breaking change**.
- Replaced `dev/style_pkg.R` with
  [Air](https://posit-dev.github.io/air/) for formatting.
- Removed `dev/fix_rd_style.R`, `dev/prepare_cargo.py`,
  `dev/patch_vendor_crates.py`, `dev/clean_checksums.py`, and
  `dev/prepare_cran.sh` — their logic is now inlined directly in
  `bindings/r/Makefile`, so the R build no longer requires any Python
  scripts.
- Added `...` to
  [`Loess()`](https://thisisamirv.github.io/loess-project/r/reference/Loess.md),
  [`StreamingLoess()`](https://thisisamirv.github.io/loess-project/r/reference/StreamingLoess.md),
  and
  [`OnlineLoess()`](https://thisisamirv.github.io/loess-project/r/reference/OnlineLoess.md)
  to force named arguments for optional parameters.
- Added `Depends: R (>= 4.6)` to `DESCRIPTION` and a matching CI matrix
  entry.
- Expanded roxygen2 `@param` docs and added a `See Also` section linking
  to <https://loess.readthedocs.io/>.
- Expanded `rfastloess-intro.Rmd` vignettes.

## rfastloess 0.9.0

### Added

- Added Python, R, WASM, Node.js, C++, and Julia bindings.

### Changed

- Implement monorepo structure.
- Converted all documentation tables to compact single-space format.
- Updated `.clang-tidy` to configure `lower_case` as the required naming
  convention for functions and member functions, matching the new
  snake_case public API.
- Moved `BENCHMARKS.md`, `CHANGELOG.md`, and `CONTRIBUTING.md` from the
  repository root into `docs/` and added them to the documentation site
  navigation.

For the full changelog, see:
<https://github.com/thisisamirv/loess-project/blob/main/CHANGELOG.md>
