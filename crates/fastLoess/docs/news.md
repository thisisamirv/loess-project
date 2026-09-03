<!-- markdownlint-disable MD024 MD025 -->
# fastLoess (development version)

## Added

* Added `dev/bump_version.py --version X.Y.Z`, which updates the version across every crate/binding's version files, `CITATION.cff`, and the Spack recipe in one pass. Supports `--dry-run`.
* Added `dev/check_pinned_versions.py` and a weekly `.github/workflows/check-versions.yml`, which check hardcoded tool/library version pins that Dependabot can't see and fail CI if any are outdated.
* Added `.github/dependabot.yml`, covering every dependency ecosystem in the repo, grouped per directory into a single weekly PR.
* Added an optional `commit` input to every release workflow's `workflow_dispatch` trigger, so a manual run can pin the checked-out/built commit instead of always building from the triggering ref.
* Added `dev/check_links.py`, which validates every Markdown cross-reference link across every binding/crate's docs, failing on missing targets.

## Changed

* Added a `large` benchmark category to `benchmarks/rfastloess.R`/`stats_loess.R` stressing exact-fit, high-iteration, and high-fraction scenarios at scale, and documented it in `benchmarks/README.md` with measured `stats::loess` vs. fastLoess median times.
* Merged `dev/add-cpp-outputs.py`, `dev/add-rust-outputs.py`, `dev/add-nodejs-outputs.js`, and `dev/add-wasm-outputs.js` into `dev/verify_snippets.py` as a new `--update-outputs` flag, removing the four standalone scripts.
* Replaced Unicode superscript/subscript characters used as ASCII-alphanumeric stand-ins (`R²` → `R2`, `O(n²)` → `O(n^2)`, `xᵢ` → `x_i`, etc.) with plain ASCII throughout every doc page, README, and Rust doc-comment/test-comment, including the `Diagnostics` `Display` impl (now prints `R2`). Regenerated the affected ```output blocks, which also caught a handful of pre-existing `RÂ²` mojibake blocks left over from before the Windows encoding fix above.
* Added `dev/add-readme-to-docs.py`, which auto-detects the docs-site flavor (Starlight for Node.js/WASM, Sphinx for Python) and embeds `README.md` as the home page accordingly. Not wired into Python's `Makefile` yet, since the Sphinx branch currently embeds the raw GitHub README verbatim rather than anything tailored to the Sphinx site.
* Harmonized the docs-site directory structure across every binding and crate (`introduction/`, `guide/`, `weighting/`, `advanced/`, `use-case/`, `api/`, each grouped under a hub page). Also fixed the doc-tooling scripts, which enumerated files via a non-recursive glob and would have silently stopped finding snippets in the newly-nested pages.
* Consolidated every crate/binding README: merged the "Installation" and "Documentation" sections, replaced GitHub-only alert syntax with plain blockquotes, removed the redundant "API Reference" and "Changelog" sections (each now has its own docs-site page), and added a "Read more" link to the Concepts page.
* Renamed the batch adapter's "When to Use" heading to "When to Use Batch Adapter" across every binding/crate's API docs.
* Vendored the [doxygen-awesome-css](https://github.com/jothepro/doxygen-awesome-css) theme (v2.4.2) for a modern, sidebar-only cpp Doxygen site with automatic dark mode.
* Added `dev/update_changelogs.py`, which regenerates a per-binding/crate `NEWS.md`/`news.md` from the root `CHANGELOG.md`.
* Replaced `kernels.md`'s "Choosing a Kernel" mermaid flowchart (every binding/crate) with an equivalent decision table, since Doxygen and rustdoc don't render mermaid.
* Replaced `adapter-choice.md`/`adapters.md`'s "Overview" flowchart with an equivalent decision table, unifying on a single rendering-agnostic format across every binding/crate.
* Consolidated `parameters.md`/the auto-generated `@autodocs` parameter reference into each `api.md`'s builder/options tables, and removed `parameters.md` itself.

## Fixed

* Fixed every benchmark category in `benchmarks/rfastloess.R` failing with `attempt to apply non-function`: it called `model$fit(x, y)`, but `fit` is an S3 generic (`fit(model, x, y)`), not a field on the `Loess` object.
* Fixed `release-conda.yml`'s `sed` pattern for `recipe.yaml`'s version, which only matched an exactly 2-space-indented `version:` line; now matches any `version: "X.Y.Z"` line by semver shape.
* Fixed `benchmarks/Makefile`'s vendoring step nulling out the original package checksum for every crates.io dependency, not just the two local path crates; now uses `jq '.files = {}'` to reset only the `files` map.
* Fixed `benchmarks/README.md`'s "Iterations" scenario row claiming "0 – 10 (6 levels)"; `stats::loess` has no meaningful `iterations = 0` setting, so it only ever ran 5 levels.
* Fixed `docs.yml`'s Pages deployment: consolidated per-language jobs into a single artifact upload/deploy job, then switched from legacy branch-based deployment to `actions/upload-pages-artifact`/`actions/deploy-pages`, eliminating GitHub's automatic "pages build and deployment" trigger. Requires the repository's Pages source to be switched to "GitHub Actions" in settings.
* Fixed 51 broken relative cross-reference links across C++, Julia, Node.js, WASM, and both Rust crates' docs, left over from the earlier docs-site restructuring into hub-grouped subdirectories. Found via the new `dev/check_links.py`.
* Fixed `OnlineLoess`'s `min_points` default being `3` instead of `2`, and `update_mode` defaulting to `"full"` instead of `"incremental"` — both diverging from every binding's docs. Originated in the Rust core and `fastLoess`'s shared `build_online` fallback, so every binding that duplicated the defaults had inherited the bug: C++, Julia, Python, and R. Updated the "Default" column in every binding's `api-online.md` to match and fixed a Rust core test that asserted the old threshold.
* Fixed the "Handling Outliers" quickstart example (every binding and both Rust crates) printing nothing: with only 6 points and `fraction = 0.5`, the local window was small enough that a degree-1 fit reproduced the outlier exactly (zero residual). Bumped to `fraction = 0.7`, which correctly downweights it.
* Fixed the R `OnlineLoess()` roxygen example printing one line per point (48 lines for a 50-point loop); it now collects the smoothed values and prints only `head(smoothed, 5)`.
* Fixed the R `add_point()` roxygen example always printing `NULL`, since a single call never reaches the default `min_points = 3`; it now uses `min_points = 2L` and shows the second (non-`NULL`) call's result.
* Fixed the Julia `intervals.md` "Confidence Intervals" and "Standard Errors" examples each looping over all 100 points instead of a short sample; switched to `result.y[1:5]`/`result.confidence_lower[1:5]`/`result.standard_errors[1:5]`-style slicing, matching the already-concise Python version.
* Fixed inline/display LaTeX math rendering as literal text on docs.rs; added a `katex-header.html` that renders it client-side.
* Fixed every cross-reference link across the Rust crate docs leading nowhere, since plain relative links aren't resolved against the rustdoc module tree; converted them to proper intra-doc links (e.g. `crate::doc::concepts`).
* Added the missing `dev/check_links.py --lang rust` step to `crates/loess-rs/Makefile`'s and `crates/fastLoess/Makefile`'s dev targets.
* Added `#[allow(clippy::excessive_precision)]` to `math/kernel.rs`'s `SQRT_2PI`/`SQRT_PI` constants, matching `fastlowess`.

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
