# LOESS Online Smoothing

Create a stateful LOESS model for real-time online data.

## Usage

``` r
OnlineLoess(
    fraction = 0.67,
    window_capacity = 1000L,
    min_points = 3L,
    ...,
    iterations = 3L,
    weight_function = "tricube",
    robustness_method = "bisquare",
    scaling_method = "mad",
    boundary_policy = "extend",
    update_mode = "full",
    auto_converge = NULL,
    return_robustness_weights = FALSE,
    return_diagnostics = FALSE,
    return_residuals = FALSE,
    zero_weight_fallback = "use_local_mean",
    parallel = FALSE,
    degree = "linear",
    dimensions = 1L,
    distance_metric = "normalized",
    surface_mode = "interpolation",
    return_se = FALSE,
    confidence_intervals = NULL,
    prediction_intervals = NULL,
    weighted_metric_weights = NULL,
    cell = NULL,
    interpolation_vertices = NULL,
    boundary_degree_fallback = NULL
)
```

## Arguments

- fraction:

  Smoothing fraction (between 0 and 1).

- window_capacity:

  Maximum number of points kept in the sliding window.

- min_points:

  Minimum number of points required before smoothing begins.

- ...:

  Not used; forces all subsequent arguments to be named.

- iterations:

  Number of robustness iterations (non-negative integer). Default: 3.

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

- update_mode:

  Window update strategy: `"full"` (default; alias: `"resmooth"`)
  re-smooths all window points after each addition; `"incremental"`
  (alias: `"single"`) updates only the newest point.

- auto_converge:

  Convergence tolerance for early stopping of robustness iterations.
  `NULL` (default) disables early stopping.

- return_robustness_weights:

  Logical; if `TRUE`, return per-point robustness weights. Default:
  `FALSE`.

- return_diagnostics:

  Logical; if `TRUE`, return fit-quality metrics (RMSE, MAE, R-squared,
  AIC, etc.). Default: `FALSE`.

- return_residuals:

  Logical; if `TRUE`, return residuals in the result. Default: `FALSE`.

- zero_weight_fallback:

  Fallback policy when all robustness weights drop to zero:
  `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`),
  `"return_original"` (alias: `"original"`), or `"return_none"` (alias:
  `"none"`).

- parallel:

  Logical; enable parallel processing. Default: `TRUE`.

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

- return_se:

  Logical; if `TRUE`, compute hat-matrix statistics (effective degrees
  of freedom, leverage, standard errors). Default: `FALSE`.

- confidence_intervals:

  Confidence level for confidence intervals (e.g., 0.95). `NULL`
  (default) disables confidence intervals.

- prediction_intervals:

  Confidence level for prediction intervals (e.g., 0.95). `NULL`
  (default) disables prediction intervals.

- weighted_metric_weights:

  Numeric vector of per-dimension weights used when
  `distance_metric = "weighted"`. Length must equal `dimensions`. `NULL`
  (default) uses equal weights.

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

## Value

An OnlineLoess object.

## See also

<https://loess.readthedocs.io/> for full documentation.

## Examples

``` r
model <- OnlineLoess(fraction = 0.2, window_capacity = 20)
x <- 1:50
y <- sin(x * 0.1) + rnorm(50, 0, 0.1)
for (i in seq_along(x)) {
    result <- add_point(model, x[i], y[i])
    if (!is.null(result)) cat("smoothed:", result$y, "\n")
}
#> smoothed: 0.1898465 
#> smoothed: 0.3098642 
#> smoothed: 0.303798 
#> smoothed: 0.4955887 
#> smoothed: 0.5883635 
#> smoothed: 0.6636898 
#> smoothed: 0.8060396 
#> smoothed: 0.9393165 
#> smoothed: 0.8703191 
#> smoothed: 0.792098 
#> smoothed: 0.9894119 
#> smoothed: 0.9412698 
#> smoothed: 1.054355 
#> smoothed: 1.212259 
#> smoothed: 1.034151 
#> smoothed: 0.8054195 
#> smoothed: 0.9712403 
#> smoothed: 1.003511 
#> smoothed: 1.052556 
#> smoothed: 0.9094165 
#> smoothed: 0.8772958 
#> smoothed: 0.7652905 
#> smoothed: 0.6407956 
#> smoothed: 0.5443473 
#> smoothed: 0.4962978 
#> smoothed: 0.3606822 
#> smoothed: 0.2222168 
#> smoothed: 0.1676355 
#> smoothed: -0.005188761 
#> smoothed: 0.02624432 
#> smoothed: -0.1029366 
#> smoothed: -0.3134436 
#> smoothed: -0.4298295 
#> smoothed: -0.4104698 
#> smoothed: -0.5583784 
#> smoothed: -0.6339216 
#> smoothed: -0.6103144 
#> smoothed: -0.7088237 
#> smoothed: -0.8096968 
#> smoothed: -0.9167906 
#> smoothed: -0.7285154 
#> smoothed: -0.8749581 
#> smoothed: -0.8782427 
#> smoothed: -0.8666714 
#> smoothed: -1.043442 
#> smoothed: -0.9644613 
#> smoothed: -1.085386 
#> smoothed: -0.8056501 
```
