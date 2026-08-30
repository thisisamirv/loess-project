---
title: News
---
<!-- markdownlint-disable MD024 MD025 -->
# fastloess (Node.js) (development version)

## Changed

* Updated `typedoc-plugin-markdown` to v4.13.
* `make nodejs-dev` now runs `npm update` after `npm install`, so dependencies are kept current.

## Fixed

* Fixed the docs homepage's "Get Started" button jumping straight to the Installation page without ever showing the README content. `README.md` is now embedded on the Starlight homepage below the hero via a new `dev/add-readme-to-docs.js` script, wired into both `npm run docs` and `make nodejs-dev`.

# fastloess (Node.js) 1.1.0

## Added

* Added `npm run lint` to the `Lint` step in `ci-nodejs.yml`, so JavaScript source and test files are linted via `oxlint` on every CI run.

## Changed

* Moved Node.js documentation from ReadTheDocs to GitHub Pages, served by Starlight at <https://thisisamirv.github.io/loess-project/nodejs/>. The ReadTheDocs site no longer includes Node.js-specific content. `dev/add-nodejs-outputs.js` runs as part of `make nodejs-dev`, executing each JavaScript code block in the docs and injecting its output back into the Markdown source.
* `make nodejs` (`default:`) now builds the native addon and links it globally via `npm link`. The full dev workflow moves to `make nodejs-dev`.
* Updated `oxlint` dependency to 1.80.

# fastloess (Node.js) 1.0.0

## Changed

* Renamed `OnlineOutput`'s `smoothed` and `std_error` fields to `y` and `standard_error`. This is a **breaking change**.
* Updated `@napi-rs/cli` to v3.8 and `oxlint` to v1.79.

For the full changelog, see:
<https://github.com/thisisamirv/loess-project/blob/main/CHANGELOG.md>
