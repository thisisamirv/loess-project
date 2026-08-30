---
title: News
---
<!-- markdownlint-disable MD024 MD025 -->
# fastloess-wasm (development version)

## Changed

* Updated `typedoc-plugin-markdown` to v4.13.
* `make wasm-dev` now runs `npm update` after `npm install`, so dependencies are kept current.

## Fixed

* Same fix as Node.js: `README.md` is now embedded on the Starlight homepage via `dev/add-readme-to-docs.js`, wired into `npm run docs` and `make wasm-dev`.
* Fixed `concepts.md` figures not rendering: the MkDocs-only `<figure markdown="span">`/attr_list (`{ width="..." }`) syntax isn't supported by Starlight's Markdown renderer, so the image markdown inside was left as raw unprocessed text. Converted all 4 figures to plain `![alt](src)` images with an italicized caption below.
* Fixed inline/display LaTeX math (`$...$`/`$$...$$`) rendering as literal text on the Node.js/WASM docs sites; wired `remark-math`/`rehype-katex` into `astro.config.mjs` and added a KaTeX stylesheet, so the existing math syntax now renders properly.

# fastloess-wasm 1.1.0

## Added

* Added `npm run lint` to the `Lint` step in `ci-wasm.yml`, so JavaScript source and test files are linted via `oxlint` on every CI run.

## Changed

* Moved WASM documentation from ReadTheDocs to GitHub Pages, served by Starlight at <https://thisisamirv.github.io/loess-project/wasm/>. The ReadTheDocs site no longer includes WASM-specific content. `dev/add-wasm-outputs.js` runs as part of `make wasm-dev`, executing each JavaScript code block in the docs and injecting its output back into the Markdown source.
* `make wasm` (`default:`) now builds both the Node.js and web WASM targets and links the Node.js package globally via `npm link`. The full dev workflow moves to `make wasm-dev`.
* Updated `oxlint` dependency to 1.80.
* Replace the outdated `jetli/wasm-pack-action` workflow with `taiki-e/install-action`.

# fastloess-wasm 1.0.0

## Fixed

* Fixed `OnlineLoess.add_point()` returning `undefined` instead of `null` when the sliding window has not yet accumulated enough points.

## Changed

* Renamed `OnlineOutput`'s `smoothed` and `std_error` getters to `y` and `standard_error`. This is a **breaking change**.
* Updated `oxlint` to v1.79.

For the full changelog, see:
<https://github.com/thisisamirv/loess-project/blob/main/CHANGELOG.md>
