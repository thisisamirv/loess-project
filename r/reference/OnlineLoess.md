# LOESS Online Smoothing

Create a stateful LOESS model for real-time online data. Maintains a
sliding window and processes each incoming point immediately via
[`add_point`](https://thisisamirv.github.io/loess-project/r/reference/add_point.md).

## Usage

``` r
OnlineLoess(
    fraction = 0.67,
    window_capacity = 1000L,
    min_points = 2L,
    ...,
    iterations = 3L,
    weight_function = "tricube",
    robustness_method = "bisquare",
    scaling_method = "mad",
    boundary_policy = "extend",
    zero_weight_fallback = "use_local_mean",
    update_mode = "incremental",
    auto_converge = NULL,
    return_robustness_weights = FALSE,
    degree = "linear",
    dimensions = 1L,
    distance_metric = "normalized",
    surface_mode = "interpolation",
    weighted_metric_weights = NULL,
    cell = NULL,
    interpolation_vertices = NULL,
    boundary_degree_fallback = NULL,
    missing = "error"
)
```

## Arguments

- fraction:

  Smoothing fraction, greater than 0 and up to 1. Default: 0.67. See
  Details for guidance on choosing a value.

- window_capacity:

  Maximum number of points kept in the sliding window, at least 3.
  Default: 1000.

- min_points:

  Minimum number of points required before smoothing begins, between 2
  and `window_capacity`. Default: 2.

- ...:

  Not used; forces all subsequent arguments to be named.

- iterations:

  Number of robustness iterations, between 0 and 1000 (inclusive).
  Default: 3.

- weight_function:

  Kernel weight function. One of `"tricube"` (default), `"gaussian"`,
  `"uniform"` (alias: `"boxcar"`), `"cosine"`, `"epanechnikov"`,
  `"biweight"` (alias: `"bisquare"`), or `"triangle"` (alias:
  `"triangular"`).

- robustness_method:

  Outlier downweighting method: `"bisquare"` (default; alias:
  `"biweight"`), `"huber"`, or `"talwar"`.

- scaling_method:

  Residual scale estimation for robustness weights: `"mad"` (default;
  alias: `"median_absolute_deviation"`), `"mar"` (alias:
  `"median_absolute_residual"`), or `"mean"` (alias:
  `"mean_absolute_residual"`).

- boundary_policy:

  Boundary handling strategy: `"extend"` (default; alias: `"pad"`),
  `"reflect"` (alias: `"mirror"`), `"zero"`, or `"noboundary"` (alias:
  `"none"`).

- zero_weight_fallback:

  Fallback policy when all robustness weights drop to zero:
  `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`),
  `"return_original"` (alias: `"original"`), or `"return_none"` (alias:
  `"none"`).

- update_mode:

  Window update strategy: `"incremental"` (default; alias: `"single"`)
  updates only the newest point; `"full"` (alias: `"resmooth"`)
  re-smooths all window points after each addition.

- auto_converge:

  Convergence tolerance for early stopping of robustness iterations.
  `NULL` (default) disables early stopping.

- return_robustness_weights:

  Logical; if `TRUE`, return per-point robustness weights. Default:
  `FALSE`.

- degree:

  Local polynomial degree: `"constant"`, `"linear"` (default),
  `"quadratic"`, `"cubic"`, or `"quartic"`.

- dimensions:

  Number of predictor dimensions. Default: 1.

- distance_metric:

  Distance metric for neighbourhood computation: `"normalized"`
  (default), `"euclidean"`, `"manhattan"`, `"chebyshev"`, `"minkowski"`,
  or `"weighted"`. Use `"minkowski:p"` to set a custom *p* value.

- surface_mode:

  Surface evaluation mode: `"interpolation"` (default) or `"direct"`.

- weighted_metric_weights:

  Numeric vector of per-dimension weights. Length must equal
  `dimensions`. Only used when `distance_metric = "weighted"`; setting
  `distance_metric = "weighted"` without providing this raises an error.
  `NULL` (default) has no effect unless `distance_metric = "weighted"`
  is set.

- cell:

  Cell size tuning parameter for the interpolation grid. `NULL`
  (default) uses the library default.

- interpolation_vertices:

  Number of vertices in the interpolation grid. `NULL` (default) uses
  the library default.

- boundary_degree_fallback:

  Logical; if `TRUE`, fall back to lower polynomial degree at boundaries
  when fitting at the requested degree fails. `NULL` (default) uses the
  library default.

- missing:

  Policy for a non-finite (NaN/Inf) `x` or `y` value passed to
  [`add_point`](https://thisisamirv.github.io/loess-project/r/reference/add_point.md):
  `"error"` (default) raises an error, `"drop"` silently ignores the
  point (returns `NULL`) instead of adding it to the window.

## Value

An OnlineLoess object.

## Details

Best suited when data arrives incrementally (e.g. sensors or streams),
real-time smoothed values are needed, or memory is fixed. For datasets
that fit in memory, see
[`Loess`](https://thisisamirv.github.io/loess-project/r/reference/Loess.md);
for large batches processed in chunks, see
[`StreamingLoess`](https://thisisamirv.github.io/loess-project/r/reference/StreamingLoess.md).

## Examples

``` r
model <- OnlineLoess(fraction = 0.2, window_capacity = 20)
x <- 1:50
y <- sin(x * 0.1) + rnorm(50, 0, 0.1)
smoothed <- numeric(0)
for (i in seq_along(x)) {
    result <- add_point(model, x[i], y[i])
    if (!is.null(result)) smoothed <- c(smoothed, result$y)
}
head(smoothed, 5)
#> [1] 0.1201261 0.1898465 0.3098642 0.3037980 0.4955887
```
