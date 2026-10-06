# Changelog

## rfastloess (development version)

### Added

- Added case weights to Streaming chunks and Online points, plus
  current-window Online diagnostics and prediction.
- [`OnlineLoess()`](https://thisisamirv.github.io/loess-project/r/reference/OnlineLoess.md)
  now accepts multivariate coordinate vectors in
  [`add_point()`](https://thisisamirv.github.io/loess-project/r/reference/add_point.md)
  when `dimensions` is greater than one.
- Added an Alternative Software vignette with runnable Gaussian and
  robust comparisons to
  [`stats::loess()`](https://rdrr.io/r/stats/loess.html) and a guide to
  LOESS-specific defaults.
- Added
  [`cv_opts()`](https://thisisamirv.github.io/loess-project/r/reference/cv_opts.md)
  and the `cv` argument on
  [`Loess()`](https://thisisamirv.github.io/loess-project/r/reference/Loess.md)
  for grouped Batch cross-validation.
- Added `outputs` to
  [`Loess()`](https://thisisamirv.github.io/loess-project/r/reference/Loess.md),
  [`StreamingLoess()`](https://thisisamirv.github.io/loess-project/r/reference/StreamingLoess.md),
  [`OnlineLoess()`](https://thisisamirv.github.io/loess-project/r/reference/OnlineLoess.md),
  and
  [`predict.Loess()`](https://thisisamirv.github.io/loess-project/r/reference/predict.Loess.md)
  for grouped optional results with mode-specific name validation.
- Added `retain_model` and a
  [`predict.Loess()`](https://thisisamirv.github.io/loess-project/r/reference/predict.Loess.md)
  S3 method for out-of-sample prediction.
- Added `return_gradient` to
  [`Loess()`](https://thisisamirv.github.io/loess-project/r/reference/Loess.md),
  [`StreamingLoess()`](https://thisisamirv.github.io/loess-project/r/reference/StreamingLoess.md),
  and
  [`OnlineLoess()`](https://thisisamirv.github.io/loess-project/r/reference/OnlineLoess.md).
- Added `confidence_intervals`/`prediction_intervals`/`return_se` to
  [`StreamingLoess()`](https://thisisamirv.github.io/loess-project/r/reference/StreamingLoess.md)
  and
  [`OnlineLoess()`](https://thisisamirv.github.io/loess-project/r/reference/OnlineLoess.md).
  [`OnlineLoess()`](https://thisisamirv.github.io/loess-project/r/reference/OnlineLoess.md)
  requires `update_mode = "full"` or errors. New bound fields on
  [`add_point()`](https://thisisamirv.github.io/loess-project/r/reference/add_point.md)’s
  result.

### Changed

- Clarified Batch `residual_sd` as `1.4826 * MAD`; Streaming reports the
  cumulative sample standard deviation of emitted residuals.
- Breaking change: replaced individual `return_*` output arguments with
  `outputs = c(...)` for Batch, Streaming, Online, and prediction.
- Breaking change: replaced flat interval arguments and prediction
  levels with
  `intervals = intervals_opts(confidence = ..., prediction = ...)`; CV
  uses only `cv = cv_opts(...)` with an outer `seed`, replacing flat CV
  arguments and `cv_opts(seed = ...)`. Removed named constructor
  arguments now error instead of being silently ignored.
- Unavailable diagnostics are now represented as R `NA` rather than
  generic `NaN` values.

### Fixed

- Correct tied-coordinate interpolation splits and normalized cell
  geometry, add two-dimensional neighboring-edge blending, and match R’s
  neighborhood rounding near integer span boundaries.
- Preserve zero interpolation subdivision thresholds to match R’s
  small-span cell trees, retaining terminal observation vertices and
  correcting interpolated fits on small datasets.
- Normalize local case weights by their neighborhood maximum so common
  scaling cannot turn valid positive weights into an epsilon-triggered
  unweighted fallback.
- Prevented overflow in even medians, mean/bisquare scales,
  Batch/Streaming diagnostics and AIC, and local/all-tied weight sums
  for large finite inputs.
- Reject fractional/overflowing integer and count options before FFI
  coercion; validate grouped CV/interval lists and numeric vector/matrix
  shapes; accept integer custom weights.
- Validate case-weight lengths and values before dropping missing
  observations, so invalid weights on dropped rows are not silently
  ignored.
- Preserve case weights through sorted CV training subsets and
  multidimensional predictions. Serial and parallel CV now agree on
  seeded folds and held-out LOOCV predictions; K-fold counts above the
  retained observation count are rejected.
- Reject non-positive or non-finite Streaming/Online auto-convergence
  tolerances. Online auto-convergence requires full updates with
  robustness iterations.
- Include all observations for Gaussian smoothing and prediction while
  preserving the k-th-neighbor bandwidth; use the true exponential
  without an artificial tail floor.
- Correct case-weighted standard errors for direct, unpadded
  one-dimensional linear fits and retained prediction, with matching
  serial/parallel local moments. Span-one fits no longer substitute a
  kernel-free global OLS formula.
- Honor configured zero-weight fallback policies in constant-degree,
  zero-bandwidth, insufficient-neighbor, and coefficient-fit paths.
- Match R LOESS’s even-sample bisquare MAR scale arithmetic, including
  extremely small residuals, while preserving the centered-MAD fallback.
- Aligned `OnlineLoess` defaults across the Rust core and bindings:
  `iterations` is now `0` with the default
  `update_mode = "incremental"`; positive robustness iterations require
  `update_mode = "full"`.
- Fixed `cv_seed` silently accepting negative values and reinterpreting
  them as a huge unsigned seed instead of raising an error. Now
  validated before the cast.
- Fixed multivariate Batch/Streaming matrix inputs being flattened in
  R’s column-major order; preserve row-wise coordinates. CV seeds now
  honor the documented full exact-integer range through 2^53, matrix
  [`predict_window()`](https://thisisamirv.github.io/loess-project/r/reference/predict_window.md)
  queries are rejected instead of being misinterpreted, and plotting
  multivariate results now returns a clear unsupported-mode error.

## rfastloess 2.0.0

### Added

### Changed

- Updated the R README to be binding-specific instead of using a generic
  README shared across bindings.
- Moved R documentation from ReadTheDocs to GitHub Pages, served by
  pkgdown at <https://thisisamirv.github.io/loess-project/r/>. The
  ReadTheDocs site no longer includes R-specific content.
- Changed R version dependency to 4.4.0 due to issues with installing
  Bioconducter packages on R \< 4.4.0.

### Fixed

- Fixed `.cargo/config.toml` hardcoding absolute `c:/rtools45/...` paths
  for the `x86_64-pc-windows-gnu` linker and ar tool; it now uses bare
  tool names resolved via `PATH`.
- Fixed Windows ARM64 and CRAN source builds that failed while linking
  the native R library.
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
- Added a `reject_extra_positional_args()` helper to reject extra
  unnamed arguments.

### Fixed

- Fixed incorrect URLs in R binding docs.

### Changed

- Moved the `tutorials/` pages into a new `user-guide/use-cases/`
  section.
- Split Streaming and Online API reference material into dedicated
  user-guide pages and added runnable vignette examples.
- Breaking: Renamed the `smoothed` and `std_error` fields returned by
  `OnlineLoess`’s
  [`add_point()`](https://thisisamirv.github.io/loess-project/r/reference/add_point.md)
  to `y` and `standard_error`.
- Added `...` to
  [`Loess()`](https://thisisamirv.github.io/loess-project/r/reference/Loess.md),
  [`StreamingLoess()`](https://thisisamirv.github.io/loess-project/r/reference/StreamingLoess.md),
  and
  [`OnlineLoess()`](https://thisisamirv.github.io/loess-project/r/reference/OnlineLoess.md)
  to force named arguments for optional parameters.
- Requires R 4.6 or later.
- Expanded roxygen2 `@param` docs and added a `See Also` section linking
  to <https://loess.readthedocs.io/>.
- Expanded `rfastloess-intro.Rmd` vignettes.

## rfastloess 0.9.0

### Added

- Added the R LOESS package with S3 fitting and result methods.
