<!-- markdownlint-disable MD024 MD025 -->
# fastLoess (development version)

## Added

* Added `dev/bump_version.py --version X.Y.Z`, which updates the version across every Rust crate/binding `Cargo.toml`, the internal `fastLoess`/`loess-rs` path-dependency requirements, each binding's own version file (`package.json` + npm subpackages, `pyproject`-adjacent `__version__.py`, `DESCRIPTION`, `Project.toml`, `CMakeLists.txt`), `CITATION.cff`, and the Spack recipe's example `url`, in one pass. Supports `--dry-run`.
* Added `dev/check_pinned_versions.py` and a weekly `.github/workflows/check-versions.yml`, which check hardcoded tool/library version pins that Dependabot can't see (Corrosion's CMake `FetchContent` tag, the vendored doxygen-awesome-css theme, R's `Config/rextendr/version`/`Config/roxygen2/version`, and the KaTeX CDN version in both `crates/loess-rs/katex-header.html` and `crates/fastLoess/katex-header.html`) against their latest GitHub release and fail CI if any are outdated. Read-only: it never opens PRs or edits files itself.
* Added `.github/dependabot.yml`, covering every dependency ecosystem in the repo (`github-actions`, `cargo` (root workspace + the standalone `bindings/r/src` crate), `npm` (Node.js + WASM), `pip` (Python)). Each directory is grouped so all its updates, including majors, land in a single weekly PR.
* Added an optional `commit` input to `release-cpp.yml`, `release-node.yml`, `release-pypi.yml`, `release-wasm.yml`, and `release-julia-jll.yml`'s `workflow_dispatch` triggers, so a manual run can pin the checked-out/built commit instead of always building from the triggering ref (`release-julia-register.yml` already had an equivalent `commit_sha` input; `release-conda.yml` never checks out this repo's own source, so it was left unchanged). Standardized the input's description text ("defaults to the workflow's own ref") to match lowess-project's wording.

## Changed

* Added a `large` benchmark category to `benchmarks/rfastloess.R` and `benchmarks/stats_loess.R` with 4 scenarios stressing different parameters at scale (n = 50000 unless noted): `large_direct` (`surface = "direct"`, disabling `loess`'s k-d tree interpolation shortcut for an exact fit), `large_interp` (same workload with the default interpolating surface, showing the shortcut's speedup), `large_high_iter` (n = 15000, `family = "symmetric"` and 10 robustness iterations instead of 3 — `family = "symmetric"` is required for `stats::loess` to actually perform robustness reweighting; the default `family = "gaussian"` ignores `loess.control`'s `iterations` entirely), and `large_high_fraction` (`span = 0.67`, since a wide window is costly even with the interpolation shortcut active). `benchmarks/compare.py`'s plot grid grew from 5x2 to 7x2 rows to fit the new categories. Documented the new category in `benchmarks/README.md` (a "Large Scale" scenarios row plus a dedicated "Large Scale Benchmarks" section with measured `stats::loess` vs. serial/parallel fastLoess median times), matching lowess-project's write-up of its analogous `delta`-based `large` category — this had been missed when the benchmark code itself was added.
* Harmonized the docs-site directory structure across every binding and crate to match lowess-project's layout (`introduction/`, `guide/`, `weighting/`, `advanced/`, `use-case/`, `api/`, each grouped under a hub page): `loess-rs`/`fastLoess` (nested `#[cfg(doc)]` rustdoc modules), C++ (Doxygen `\page`/`\subpage` hub files, `RECURSIVE` now `YES`), Julia (`Documenter.jl` `pages=[...]`, `make.jl`'s leading-comment-stripping pass now uses `walkdir` instead of a non-recursive `readdir` so nested pages aren't silently skipped), Node.js/WASM (Starlight `sidebar`), and Python (Sphinx toctree; split `user-guide/` into `guide/`/`weighting/`/`advanced/`, renamed `getting-started/` to `introduction/`, dropped the dead, already-broken `mkdocs.yml`). `degree.md`/`dimensions.md` (LOESS-specific, absent from LOWESS) were folded into each binding's `advanced/` hub, mirroring how `gpu-backend.md` sits there in lowess-project. R's `vignettes/` stays flat (CRAN/pkgdown requirement); only `_pkgdown.yml`'s `articles:` grouping changed. Also fixed `dev/verify_snippets.py`, `dev/add-cpp-outputs.py`, `dev/add-rust-outputs.py`, `dev/add-nodejs-outputs.js`, and `dev/add-wasm-outputs.js`, which all enumerated doc files via a non-recursive glob/`readdir` and would have silently stopped finding snippets in the newly-nested pages. Validated with a live `cargo doc`, `doxygen`, `astro build` (Node.js and WASM), and `sphinx-build` for every restructured target.
* Consolidated every crate/binding README: merged the "Installation" and "Documentation" sections, replaced GitHub-only alert syntax with plain blockquotes, removed the redundant "API Reference" and "Changelog" sections (each now has its own docs-site page), and added a "Read more" link to the Concepts page. The top-level repository README is unchanged, since it's only ever viewed on GitHub.
* Renamed the batch adapter's "When to Use" heading to "When to Use Batch Adapter" across every binding/crate's API docs.
* Vendored the [doxygen-awesome-css](https://github.com/jothepro/doxygen-awesome-css) theme (v2.4.2) for a modern, sidebar-only cpp Doxygen site with automatic dark mode.
* Added `dev/update_changelogs.py`, which regenerates a per-binding/crate `NEWS.md`/`news.md` from the root `CHANGELOG.md`. Wired into every docs site's navigation, the Rust crates' rustdoc module tree, and every `Makefile` `dev` target.
* Replaced `kernels.md`'s "Choosing a Kernel" mermaid flowchart (every binding/crate) with an equivalent decision table, since Doxygen and rustdoc don't render mermaid and the deeply-nested diamond chain was hard to read even where it did render.
* Replaced `adapter-choice.md`/`adapters.md`'s "Overview" flowchart (mermaid in most bindings/crates, ASCII art in the C++ docs) with an equivalent decision table, unifying on a single rendering-agnostic format across every binding/crate.
* Consolidated `parameters.md`/the auto-generated `@autodocs` parameter reference across every binding and crate (C++, Julia, Node.js, Python, WASM, `fastLoess`, `loess-rs`): merged its unique content (fraction/iterations choice guidance, and inline `zero_weight_fallback`/`surface_mode` behavior tables) into each `api.md`'s builder/options tables (Julia: into the `Loess`/`StreamingLoess`/`OnlineLoess` docstrings), and removed `parameters.md` itself along with its docs-site navigation entries, `doc::parameters` rustdoc module, and cross-references (now pointing at `api.md`) — the parameter tables, kernel/robustness/boundary/scaling/degree/distance-metric option lists, and interval/custom-weights code examples it duplicated already live on their own dedicated pages.

## Fixed

* Fixed every benchmark category in `benchmarks/rfastloess.R` failing with `attempt to apply non-function`: it called the R6-style `model$fit(x, y)`, but `fit` is an S3 generic (`fit(model, x, y)`), not a field on the `Loess` object.
* Fixed `release-conda.yml`'s `sed` pattern for updating `recipe.yaml`'s version, which only matched a `version:` line at exactly 2-space indentation (`^  version: ".*"`) and would silently no-op if the feedstock's recipe formatting ever shifted; now matches any `version: "X.Y.Z"` line by semver shape, matching lowess-project's more robust pattern. Also narrowed the post-update debug output from dumping the entire `recipe.yaml` to just the `context`/`source`/`build` blocks.
* Fixed `benchmarks/Makefile`'s vendoring step blanket-writing `{"files":{},"package":null}` into every vendored crate's `.cargo-checksum.json`, including crates.io-sourced dependencies (not just the two local `loess-rs`/`fastLoess` path crates, for which `"package":null` is correct). This nulled out the original package checksum `cargo vendor` had recorded for every crates.io dependency, which `bindings/r/src/vendor-update.sh` has always preserved instead; now uses `jq '.files = {}'` to reset only the `files` map, matching lowess-project's and `vendor-update.sh`'s approach.
* Fixed `benchmarks/README.md`'s "Iterations" scenario row claiming "0 – 10 (6 levels)"; `benchmarks/rfastloess.R` and `benchmarks/stats_loess.R` have never tested `iterations = 0` (only `benchmarks/rfastlowess.R`/`stats_lowess.R` in lowess-project do, since `stats::lowess`'s `iter = 0` is a valid no-robustness setting, unlike `stats::loess`), so it only ever ran 5 levels (`1, 2, 3, 5, 10`).
* Fixed `docs.yml` triggering GitHub's "pages build and deployment" once per docs job; per-language jobs now upload artifacts, and a single final `deploy` job pushes to `gh-pages` once per run.
* Fixed `docs.yml`'s reliance on GitHub's legacy branch-based Pages deployment, which auto-triggers an unpinned, GitHub-managed "pages build and deployment" job on every `gh-pages` push (surfacing deprecation warnings, e.g. for Node.js 20, that aren't fixable from this repo). The former `deploy` job is now `build`, which still pushes the merged `_site` to `gh-pages` as a cache for future incremental runs, but publishing now goes through `actions/upload-pages-artifact` and a new `deploy` job using the official `actions/deploy-pages`, which this repo pins directly. Requires the repository's Pages source to be switched to "GitHub Actions" in settings.
* Fixed `OnlineLoess`'s `min_points` default being `3` instead of `2`, and its `update_mode` default falling back to `"full"` instead of `"incremental"` — both diverging from what every binding's docs already claimed and, for `min_points`, sitting right at the edge of the validator's own `>= 2` floor for no documented reason. Unlike prior divergences of this kind, this one originated in the Rust core itself (`loess_rs::adapters::defaults::DEFAULT_ONLINE_MIN_POINTS`) and in `fastLoess`'s shared `binding_support::build_online` fallback (used directly by Node.js and WASM), so every binding that duplicates these defaults independently had inherited the same bug: C++ (`fastloess.hpp`'s `OnlineOptions`), Julia (`FastLOESS.jl`'s `OnlineLoess` keyword defaults), Python (`lib.rs`'s `#[pyo3(signature = ...)]` and the `_core.pyi` stub), and R (`OnlineLoess.R`, its generated `OnlineLoess.Rd`, and the R binding's vendored `loess-rs`/`fastLoess` copies, re-synced via `vendor-update.sh`). Updated the `min_points` "Default" column in C++/Julia/Node.js/WASM/Python's `api-online.md` reference tables (including the two Rust crates' own `api-online.md` "Builder Options" tables, initially missed in the first pass) and Python's `guide/adapters.md` to match, added an explicit `min_points=3` to Python's `api-online.md` "Methods" example (previously relying on the now-different default, leaving its `# None` comments stale), and fixed a Rust core test (`test_online_propagates_options`) that asserted the old 3-point threshold.
* Fixed the "Handling Outliers" quickstart example (every binding and the `loess-rs`/`fastLoess` crates) printing nothing: with only 6 points and `fraction = 0.5`, the local window is small enough that tricube weighting drives the farthest neighbor's weight to ~0, leaving just 2 effectively-weighted points, which a degree-1 fit reproduces exactly (zero residual, no downweighting) — confirmed directly against the `lowess`/`loess` core, not binding-specific. Bumped to `fraction = 0.7`, which correctly downweights the injected outlier.
* Fixed the R `OnlineLoess()` roxygen example printing one line per point (48 lines for a 50-point loop); it now collects the smoothed values and prints only `head(smoothed, 5)`.
* Fixed the R `add_point()` roxygen example always printing `NULL`, since a single call never reaches the default `min_points = 3`; it now uses `min_points = 2L` and shows the second (non-`NULL`) call's result.
* Fixed the Julia `intervals.md` "Confidence Intervals" and "Standard Errors" examples each looping over all 100 points instead of a short sample; switched to `result.y[1:5]`/`result.confidence_lower[1:5]`/`result.standard_errors[1:5]`-style slicing, matching the already-concise Python version.
* Fixed inline/display LaTeX math rendering as literal text on docs.rs; added a `katex-header.html` that renders it client-side with KaTeX.
* Fixed every cross-reference link across the `loess-rs`/`fastLoess` crate docs (`quickstart.md`, `api.md`, `concepts.md`, and others) leading nowhere: these pages are embedded into rustdoc via `#![doc = include_str!(...)]`, and plain relative links like `[Concepts](concepts.md)` are rendered verbatim rather than resolved against the generated module tree. Converted every such link to a proper intra-doc link (e.g. `[Concepts](crate::doc::concepts)`), validated with `cargo doc --all-features` under `-D warnings` (zero broken-link warnings).

# fastLoess 1.1.0

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
* Moved crate documentation from ReadTheDocs to <https://docs.rs/fastLoess>.
* `make fastLoess` (`default:`) now only runs `cargo build`. The full dev workflow moves to `make fastLoess-dev`.

## Fixed

* Fixed `.cargo/config.toml` hardcoding absolute `c:/rtools45/...` paths for the `x86_64-pc-windows-gnu` linker and ar tool. Replaced with bare tool names resolved via `PATH`, matching the existing fix in `bindings/r/src/cargo-config.toml`.

# fastLoess 1.0.0

## Changed

* Removed `dev/isolate_cargo.py`, `dev/check_root_cargo.py`, `dev/fix_doc_snippets.py`, and `check_js_licenses.js` — workspace isolation, doc-snippet transformation, and license checks are no longer needed.
* Split the monolithic root `Makefile` into per-crate/binding sub-Makefiles (e.g. `crates/loess-rs/Makefile`, `bindings/r/Makefile`), each invokable directly via `make -f path/Makefile`. The root `Makefile` now only aggregates (`docs`, `check-msrv`, `all*`).
* Moved Rust and binding tests into their respective crate/binding directories (e.g. `tests/loess-rs/` → `crates/loess-rs/tests/loess-rs/`, `tests/cpp/` → `bindings/cpp/tests/`). Removed the standalone `tests/` workspace packages.

# fastLoess 0.9.0

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

# fastLoess 0.1.0

## Added

* Initial release with parallel execution support.

For the full changelog, see:
<https://github.com/thisisamirv/loess-project/blob/main/CHANGELOG.md>
