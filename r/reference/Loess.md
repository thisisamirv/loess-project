# LOESS Batch Smoothing

Create a stateful LOESS model for batch smoothing. This is the default
mode: it processes the entire dataset at once and supports every feature
(confidence/prediction intervals and cross-validation).

## Usage

``` r
Loess(
    fraction = 0.67,
    ...,
    iterations = 3L,
    weight_function = "tricube",
    robustness_method = "bisquare",
    scaling_method = "mad",
    boundary_policy = "extend",
    outputs = NULL,
    intervals = NULL,
    zero_weight_fallback = "use_local_mean",
    auto_converge = NULL,
    parallel = TRUE,
    degree = "linear",
    dimensions = 1L,
    distance_metric = "normalized",
    surface_mode = "interpolation",
    weighted_metric_weights = NULL,
    cell = NULL,
    interpolation_vertices = NULL,
    boundary_degree_fallback = NULL,
    seed = NULL,
    missing = "error",
    retain_model = FALSE,
    cv = NULL
)
```

## Arguments

- fraction:

  Smoothing fraction, greater than 0 and up to 1. Default: 0.67. See
  Details for guidance on choosing a value.

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

- outputs:

  Optional character vector selecting `"diagnostics"`, `"residuals"`,
  `"weights"`, `"gradient"` (or `"derivative"`), `"se"`, and `"sorted"`.
  `NULL` (default) selects no optional components.

- intervals:

  Grouped coverage levels from
  [`intervals_opts`](https://thisisamirv.github.io/loess-project/r/reference/intervals_opts.md).

- zero_weight_fallback:

  Fallback policy when all robustness weights drop to zero:
  `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`),
  `"return_original"` (alias: `"original"`), or `"return_none"` (alias:
  `"none"`).

- auto_converge:

  Convergence tolerance for early stopping of robustness iterations.
  `NULL` (default) disables early stopping.

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

  Logical; if `TRUE`, interpolation vertices lying outside the range of
  the data are fitted with a linear model rather than the requested
  degree, avoiding unstable extrapolation. It has no effect below
  quadratic degree, and `FALSE` reproduces
  [`stats::loess()`](https://rdrr.io/r/stats/loess.html). `NULL`
  (default) uses the library default.

- seed:

  Seed for reproducible CV folds, or `NULL`.

- missing:

  Policy for non-finite (NaN/Inf) values in the input data: `"error"`
  (default) raises an error, `"drop"` silently removes observations
  (rows) where any x dimension or y is non-finite (and the matching
  `custom_weights` entry) before fitting. A length mismatch between `x`
  and `y` always raises an error, even under `"drop"`.

- retain_model:

  Logical; if `TRUE`, retain the fitted model's training data, enabling
  [`predict.Loess`](https://thisisamirv.github.io/loess-project/r/reference/predict.Loess.md)
  for out-of-sample prediction. Default: `FALSE`.

- cv:

  Grouped cross-validation settings from
  [`cv_opts`](https://thisisamirv.github.io/loess-project/r/reference/cv_opts.md).

## Value

A Loess object.

## Details

Best suited when the dataset fits in memory and you need intervals,
cross-validation, or diagnostics. For datasets that don't fit in memory
or arrive in chunks, see
[`StreamingLoess`](https://thisisamirv.github.io/loess-project/r/reference/StreamingLoess.md);
for point-by-point real-time data, see
[`OnlineLoess`](https://thisisamirv.github.io/loess-project/r/reference/OnlineLoess.md).

`fraction` is the most important parameter: it controls the size of the
local neighbourhood used at each point.

When `outputs` includes `"diagnostics"`, Batch `residual_sd` is the
robust residual scale estimate `1.4826 * MAD`.

|         |                 |                          |
|---------|-----------------|--------------------------|
| Range   | Effect          | Use case                 |
| 0.1-0.3 | Fine detail     | Rapidly changing signals |
| 0.3-0.5 | Balanced        | General purpose          |
| 0.5-0.7 | Heavy smoothing | Noisy data               |
| 0.7-1.0 | Very smooth     | Trend extraction         |

## Examples

``` r
x <- seq(0, 10, length.out = 100)
y <- sin(x) + rnorm(100, 0, 0.1)
model <- Loess(fraction = 0.2)
result <- fit(model, x, y)
plot(x, y)
lines(x, result$y, col = "red")
```
