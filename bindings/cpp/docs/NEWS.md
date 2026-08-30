<!-- markdownlint-disable MD024 MD025 -->
# fastloess (C++) (development version)

## Fixed

* Fixed the Doxygen site homepage showing `docs/concepts.md` instead of `README.md`. `Doxyfile` now includes `README.md` in `INPUT` and sets it as `USE_MDFILE_AS_MAINPAGE`.
* Fixed Doxygen rendering the "View the full documentation" blockquote as raw `<a>` tag text instead of a styled link, by dropping the markdown heading (`###`) nested inside the blockquote.
* Fixed Doxygen rendering headings that mixed a heading level with an inline code span (e.g. `` ### `fastloess::Loess` ``) as literal `<tt>...</tt>` tag text in `api.md`, `api-streaming.md`, and `api-online.md`; the backticks were dropped from those headings.
* Fixed Doxygen rendering MkDocs-only `!!! note/warning/tip "title"` admonitions as literal `!!! ...` text across the cpp docs; converted every occurrence to a plain `> **Title:** ...` blockquote.
* Fixed Doxygen rendering inline/display LaTeX math (`$...$`/`$$...$$`) as literal text across the cpp docs; converted every occurrence to Doxygen's `\f$...\f$`/`\f[...\f]` syntax.
* Fixed Doxygen leaking a stray `</blockquote>` tag when a `---` thematic break immediately followed a `>` blockquote; replaced those specific separators with an explicit `<hr>` tag across the cpp docs.

# fastloess (C++) 1.1.0

## Added

* Added clang-tidy and cppcheck installation to Makefile.

## Changed

* Moved C++ documentation from ReadTheDocs to GitHub Pages, served by Doxygen at <https://thisisamirv.github.io/loess-project/cpp/>. The ReadTheDocs site no longer includes C++-specific content.
* `make cpp` (`default:`) now only runs `cargo build`. The full dev workflow (formatting, linting, cbindgen idempotency, symbol export verification, cmake tests, valgrind, doc-snippet verification) moves to `make cpp-dev`.

## Fixed

* Fixed `make cpp` Windows CI failure (`cannot find -lgcc_eh`): the C++ binding's Makefile detected MinGW via `gcc -dumpmachine` and selected the GNU target, which then used the Rtools cross-compiler from the workspace `.cargo/config.toml`; that compiler delegated to `C:\mingw64\bin\ld.exe`, which lacks `lgcc_eh`. Fixed by always targeting `x86_64-pc-windows-msvc` on Windows, removing the MinGW detection branch entirely.

# fastloess (C++) 1.0.0

## Changed

* Renamed `OnlineOutput`'s `smoothed()` and `std_error()` methods to `y()` and `standard_error()`. This is a **breaking change**.

For the full changelog, see:
<https://github.com/thisisamirv/loess-project/blob/main/CHANGELOG.md>
