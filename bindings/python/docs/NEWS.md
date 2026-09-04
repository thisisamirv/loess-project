<!-- markdownlint-disable MD024 MD025 -->
# fastloess (Python) 1.2.0

## Added

* Added `dev/bump_version.py --version X.Y.Z` to bump every crate/binding's version files, `CITATION.cff`, and the Spack recipe in one pass (supports `--dry-run`).
* Added `dev/check_pinned_versions.py` and a weekly `check-versions.yml` to catch hardcoded version pins that Dependabot can't see.
* Added `.github/dependabot.yml`, covering every dependency ecosystem, grouped per directory into a single weekly PR.
* Added an optional `commit` input to every release workflow's `workflow_dispatch` trigger, to pin the built commit for manual runs.
* Added `dev/check_links.py` to validate every Markdown cross-reference link across all docs.

## Changed

* Added a `large` benchmark category (exact-fit, high-iteration, high-fraction) to the R/Rust benchmarks.
* Merged the standalone `dev/add-{cpp,rust,nodejs,wasm}-outputs` scripts into `dev/verify_snippets.py --update-outputs`.
* Replaced Unicode super/subscript stand-ins (`R²`, `xᵢ`, etc.) with plain ASCII throughout docs, READMEs, and Rust comments, also catching some leftover mojibake.
* Added `dev/add-readme-to-docs.py` to auto-embed `README.md` as the docs homepage (Starlight/Sphinx-aware); not yet wired into Python's `Makefile`.
* Harmonized the docs-site directory structure across every binding/crate, and fixed doc-tooling scripts that missed snippets in the newly-nested pages.
* Consolidated every README: merged Installation/Documentation sections, dropped GitHub-only alert syntax, and removed sections now covered by dedicated docs-site pages.
* Renamed "When to Use" to "When to Use Batch Adapter" across every binding's API docs.
* Vendored doxygen-awesome-css v2.4.2 for a modern cpp Doxygen theme.
* Added `dev/update_changelogs.py` to regenerate each binding/crate's `NEWS.md`/`news.md` from the root changelog.
* Replaced the `kernels.md`/`adapter-choice.md` mermaid flowcharts with rendering-agnostic tables (Doxygen/rustdoc don't render mermaid).
* Consolidated `parameters.md`/`@autodocs` into each `api.md`'s option tables, removing `parameters.md`.

## Fixed

* Fixed the R benchmark script calling `fit` as a field instead of the S3 generic `fit(model, x, y)`.
* Fixed `release-conda.yml`'s version-line `sed` pattern to match any indentation.
* Fixed benchmark vendoring nulling every crate's checksum instead of just the two local path crates.
* Fixed the benchmark README's inaccurate "Iterations" scenario count.
* Fixed `docs.yml`'s Pages deployment: merged per-language jobs into one artifact upload/deploy job, using `upload/deploy-pages` actions instead of legacy branch-based deployment.
* Fixed 51 broken doc cross-reference links left over from the docs-site restructure (found via new `dev/check_links.py`).
* Fixed several `OnlineLoess`/`StreamingLoess` defaults silently diverging from every binding's docs and actual behavior: `min_points` (was 3, now 2), `update_mode` (was `"full"`, now `"incremental"`), and the internal robustness-iteration defaults (streaming 2→3, online 1→3) that only the Rust crates' bare prelude API could ever see.
* Fixed every binding's/crate's docs and doc-comments describing `LoessResult.x` (and equivalents) as "Sorted x values"; it's actually returned in the same order as the input `x` (the algorithm sorts internally, then maps every output field back to the original order). Also strengthened Python's `test_unsorted_input` to assert this instead of only checking output length.
* Fixed the "Handling Outliers" quickstart example printing nothing with only 6 points at `fraction = 0.5`; bumped to `0.7` so the outlier is actually downweighted.
* Fixed two R roxygen examples: `OnlineLoess()` printing 48 lines instead of a `head(smoothed, 5)` sample, and `add_point()` always printing `NULL` due to the default `min_points`.
* Fixed Julia's `intervals.md` examples looping over all 100 points instead of a short sample, matching the concise Python version.
* Fixed the empty "API Reference" page (stale toctree references).
* Fixed noisy pip version-check output in `release-pypi.yml` and a Pyright false-positive warning.
* Converted 3 plain comments to doc comments and ported 2 missing custom-weights test cases from `fastlowess`.

# fastloess (Python) 1.1.0

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
* Migrated Python documentation from MkDocs to Sphinx (with MyST-Parser and jupyter-sphinx). Code blocks now execute and embed output automatically via `jupyter-sphinx`.
* `make python` (`default:`) now installs to the user Python environment via `pip install --user`. The full dev workflow (venv setup, formatting, linting, testing, doc-snippet verification) moves to `make python-dev`.

## Fixed

* Fixed `.cargo/config.toml` hardcoding absolute `c:/rtools45/...` paths for the `x86_64-pc-windows-gnu` linker and ar tool. Replaced with bare tool names resolved via `PATH`, matching the existing fix in `bindings/r/src/cargo-config.toml`.
* Enforced keyword-only arguments beyond the first positional allowance in `Loess`, `StreamingLoess`, and `OnlineLoess`, matching R's behaviour: `Loess(fraction, *, ...)`, `StreamingLoess(fraction, chunk_size, *, ...)`, `OnlineLoess(fraction, window_capacity, min_points, *, ...)`. The `.pyi` stubs were updated with the same `*` separator.

# fastloess (Python) 1.0.0

## Changed

* Removed `dev/isolate_cargo.py`, `dev/check_root_cargo.py`, `dev/fix_doc_snippets.py`, and `check_js_licenses.js` — workspace isolation, doc-snippet transformation, and license checks are no longer needed.
* Split the monolithic root `Makefile` into per-crate/binding sub-Makefiles (e.g. `crates/loess-rs/Makefile`, `bindings/r/Makefile`), each invokable directly via `make -f path/Makefile`. The root `Makefile` now only aggregates (`docs`, `check-msrv`, `all*`).
* Moved Rust and binding tests into their respective crate/binding directories (e.g. `tests/loess-rs/` → `crates/loess-rs/tests/loess-rs/`, `tests/cpp/` → `bindings/cpp/tests/`). Removed the standalone `tests/` workspace packages.
* Renamed `OnlineOutput`'s `smoothed` and `std_error` properties to `y` and `standard_error`. This is a **breaking change**.

# fastloess (Python) 0.9.0

## Added

* Added Python, R, WASM, Node.js, C++, and Julia bindings.

## Changed

* Implement monorepo structure.
* Converted all documentation tables to compact single-space format.
* Updated `.clang-tidy` to configure `lower_case` as the required naming convention for functions and member functions, matching the new snake_case public API.
* Moved `BENCHMARKS.md`, `CHANGELOG.md`, and `CONTRIBUTING.md` from the repository root into `docs/` and added them to the documentation site navigation.

For the full changelog, see:
<https://github.com/thisisamirv/loess-project/blob/main/CHANGELOG.md>
