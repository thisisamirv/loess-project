# Changelog

## rfastloess 2.0.0

### Added

- Added an “Ideas for Contribution” section to `CONTRIBUTING.md`,
  listing concrete Batch/Streaming/Online feature gaps (out-of-sample
  prediction, per-point local gradients, adaptive fraction selection,
  `cell`/`interpolation_vertices` tuning, bootstrap intervals, GPU
  backend, concurrent chunk processing, checkpointable streaming state,
  `OnlineOutput.standard_error`, distance-based window eviction,
  configurable warm-up).
- Added `dev/bump_version.py --version X.Y.Z` to bump every
  crate/binding version file, `CITATION.cff`, the Spack recipe, and
  `CONTRIBUTING.md`’s example version in one pass (supports
  `--dry-run`).
- Added `dev/check_pinned_versions.py` and a weekly `check-versions.yml`
  to catch hardcoded version pins Dependabot can’t see.
- Added `.github/dependabot.yml`, covering every dependency ecosystem in
  one weekly PR per directory.
- Added an optional `commit` input to release workflows’
  `workflow_dispatch` trigger, to pin the built commit for manual runs.
- Added `dev/check_links.py` to validate Markdown cross-reference links
  across all docs.
- Added `return_sorted` and `missing` options to
  [`Loess()`](https://thisisamirv.github.io/loess-project/r/reference/Loess.md),
  [`StreamingLoess()`](https://thisisamirv.github.io/loess-project/r/reference/StreamingLoess.md),
  and
  [`OnlineLoess()`](https://thisisamirv.github.io/loess-project/r/reference/OnlineLoess.md).

### Changed

- Added a `large` benchmark category (exact-fit, high-iteration,
  high-fraction) to the R/Rust benchmarks.
- Merged the standalone `dev/add-{cpp,rust,nodejs,wasm}-outputs` scripts
  into `dev/verify_snippets.py --update-outputs`.
- Replaced Unicode super/subscript stand-ins (`R²`, `xᵢ`, etc.) with
  plain ASCII throughout docs and comments, catching some leftover
  mojibake.
- Added `dev/add-readme-to-docs.py` to auto-embed `README.md` as the
  docs homepage (Starlight/Sphinx-aware); not yet wired into Python’s
  `Makefile`.
- Harmonized the docs-site directory structure across every
  binding/crate, and fixed doc-tooling scripts that missed snippets in
  the newly-nested pages.
- Consolidated every README (merged Installation/Documentation sections,
  dropped GitHub-only alert syntax, removed sections now covered by
  docs-site pages) and renamed “When to Use” to “When to Use Batch
  Adapter” everywhere.
- Vendored doxygen-awesome-css v2.4.2 for a modern C++ Doxygen theme.
- Added `dev/update_changelogs.py` to regenerate each binding/crate’s
  `NEWS.md`/`news.md` from the root changelog.
- Replaced the `kernels.md`/`adapter-choice.md` mermaid flowcharts with
  tables (Doxygen/rustdoc don’t render mermaid).
- Consolidated `parameters.md`/`@autodocs` into each `api.md`’s option
  tables, removing `parameters.md`.
- Removed the redundant `rfastloess-package` pkgdown topic and the
  internal
  [`Nullable()`](https://rdrr.io/pkg/rfastloess/man/Nullable.html)
  helper.
- Fixed `_pkgdown.yml` mislabeling the S3-based interface as “R6
  classes”.
- Merged `parameters.Rmd`/`batch.Rmd`/`streaming.Rmd`/`online.Rmd` into
  the constructors’ roxygen docs, removing the now-redundant vignettes.
- Removed `confidence_intervals`/`prediction_intervals`/`return_se` from
  [`StreamingLoess()`](https://thisisamirv.github.io/loess-project/r/reference/StreamingLoess.md),
  and those plus `return_diagnostics`/`return_residuals`/`parallel` from
  [`OnlineLoess()`](https://thisisamirv.github.io/loess-project/r/reference/OnlineLoess.md)
  — none were ever computed by either adapter, and Online now always
  runs sequentially. Breaking change;
  [`StreamingLoess()`](https://thisisamirv.github.io/loess-project/r/reference/StreamingLoess.md)’s
  `parallel` is unaffected.
- Mirrored the Python API docs’ structure (field tables, `## Options`
  per field, `## Result Structure` at the end) to every remaining
  crate/binding. Along the way, unified `weighted_metric_weights` to
  require explicit `distance_metric = "weighted"` on C++/Go/Java/Julia
  (previously auto-selected); fixed C++‘s `StreamingOptions.overlap`
  hardcoded `500` default; corrected several bindings’ docs showing a
  flat `500` for `overlap` when they actually use the dynamic default
  (Node.js, WASM, Go, Java, R); fixed Java’s `api-online.adoc`
  disclaimer and Julia’s constructor docstrings missing several accepted
  keyword arguments.
- Fixed `bindings/r/R/StreamingLoess.R`, which was corrupted (a
  duplicate of `OnlineLoess.R` with a mangled fragment appended), making
  [`StreamingLoess()`](https://thisisamirv.github.io/loess-project/r/reference/StreamingLoess.md)
  uncallable. Reconstructed from `man/StreamingLoess.Rd`/`utils.R`,
  verified via
  [`roxygen2::roxygenise()`](https://roxygen2.r-lib.org/reference/roxygenize.html)
  and the full `testthat` suite (187 passed). Also fixed a stale
  `test-extendr-wrappers.R` fixture with 3 extra positional args.

### Fixed

- Fixed `CONTRIBUTING.md`’s stale Go prerequisite (`1.21+` → `1.23+`),
  `air` auto-install target (`make r` → `make r-dev`), and example crate
  version (`0.9.0` → `1.2.0`).
- Fixed the R benchmark script calling `fit` as a field instead of the
  S3 generic `fit(model, x, y)`.
- Fixed `release-conda.yml`’s version-line `sed` pattern to match any
  indentation.
- Fixed benchmark vendoring nulling every crate’s checksum instead of
  just the two local path crates.
- Fixed the benchmark README’s inaccurate “Iterations” scenario count.
- Fixed `docs.yml`’s Pages deployment: merged per-language jobs into one
  artifact upload/deploy job using `upload/deploy-pages` actions instead
  of legacy branch-based deployment.
- Fixed 51 broken doc cross-reference links left over from the docs-site
  restructure (found via `dev/check_links.py`).
- Fixed `OnlineLoess`/`StreamingLoess` defaults silently diverging from
  docs: `min_points` (3→2), `update_mode` (`"full"`→`"incremental"`),
  and internal robustness-iteration defaults (streaming 2→3, online
  1→3).
- Fixed every binding’s docs describing `LoessResult.x` as “sorted”;
  it’s actually returned in input order (sorted internally, then mapped
  back). Strengthened Python’s `test_unsorted_input` to assert this.
- Fixed the “Handling Outliers” quickstart example printing nothing at
  `fraction = 0.5` with only 6 points; bumped to `0.7`.
- Fixed two R roxygen examples printing too much/nothing
  ([`OnlineLoess()`](https://thisisamirv.github.io/loess-project/r/reference/OnlineLoess.md),
  [`add_point()`](https://thisisamirv.github.io/loess-project/r/reference/add_point.md)).
- Fixed Julia’s `intervals.md` examples looping over all 100 points
  instead of a short sample.
- Reformatted `configure` to tabs and fixed `.Rbuildignore` missing
  exclusions.
- Removed the empty `R/params.R` stub and simplified
  [`plot.LoessResult()`](https://thisisamirv.github.io/loess-project/r/reference/plot.LoessResult.md)
  to return `NULL` invisibly.
- Inlined the `.make_*` constructor helpers, and consolidated
  `utils.R`’s parameter validators into two generic helpers.
- Fixed `utils.R`’s internal `validate_min_points()` guard hardcoding a
  stricter minimum of 3 points, diverging from the Rust core’s actual
  minimum of 2; lowered to 2.

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
