<!-- markdownlint-disable MD024 MD025 -->
# FastLOESS.jl (development version)

## Fixed

* Fixed the Documenter site homepage being a separately maintained `docs/src/index.md` instead of the top-level `README.md`. `make.jl` now regenerates `index.md` from `README.md` before every build, and the stale static copy was removed.
* Fixed the README's raw `<p align="center">` badge/logo HTML blocks rendering as literal text on the Documenter site (unlike GitHub/pkgdown/Starlight/Doxygen); `make.jl` now converts them to plain Markdown image/link syntax before writing `index.md`.
* Fixed the README's `<!-- markdownlint-disable ... -->` comment rendering as literal text on the Documenter site; `make.jl` now strips HTML comments before writing `index.md`.

# FastLOESS.jl 1.1.0

## Added

* `release-julia-register.yml` now automatically extracts the matching changelog section and appends it as release notes in the JuliaRegistrator comment, enabling auto-merge on major version bumps.

## Changed

* Moved Julia documentation from ReadTheDocs to GitHub Pages, served by Documenter.jl at <https://thisisamirv.github.io/loess-project/julia/stable/>. The ReadTheDocs site no longer includes Julia-specific content. Code blocks use Documenter.jl `@example` sections, which execute and embed output automatically during the docs build.
* `make julia` (`default:`) now builds the Rust library and installs the Julia package via `Pkg.develop`. The full dev workflow moves to `make julia-dev`.

## Fixed

* Fixed `fit(l::Loess, x::Matrix{Float64}, y)` not validating that `size(x, 2) == l.dimensions` before flattening the matrix. If the column count differed from the configured dimensions, the library either silently used wrong data or produced a confusing C-level error. The `Loess` struct now stores `dimensions` as a field, and the matrix overload checks `size(x, 2) != l.dimensions` upfront with a clear message naming the parameter to fix.
* Fixed `FastLOESS.jl` never actually loading the prebuilt `fastloess_jll` binary: `find_library()` only checked the `FASTLOESS_LIB` env var and local dev-mode paths, so the package installed from the registry had no working native library for end users. Added the `fastloess_jll` dependency (`Project.toml`), a JLL-loading branch in `find_library()`, and switched from an eager `const libfastloess = find_library()` (resolved once at precompile time) to a lazy `current_library()` accessor re-resolved in `__init__()`, matching `FastLOWESS.jl`.

# FastLOESS.jl 1.0.0

## Fixed

* Fixed `LoessResult.iterations_used` returning the raw FFI sentinel `-1` instead of `nothing` when robustness iterations were not applicable.

## Changed

* Renamed `OnlineOutput`'s `smoothed` and `std_error` fields to `y` and `standard_error`. This is a **breaking change**.
* Removed `dev/format_julia.jl`; formatting is now inlined in `bindings/julia/Makefile`.

For the full changelog, see:
<https://github.com/thisisamirv/loess-project/blob/main/CHANGELOG.md>
