# Predict from a fitted LOESS model at out-of-sample points

Predict from a fitted LOESS model at out-of-sample points

## Usage

``` r
# S3 method for class 'Loess'
predict(
    object,
    new_x,
    return_se = FALSE,
    confidence_level = NULL,
    prediction_level = NULL,
    return_derivative = FALSE,
    extrapolation = "clamp",
    max_extrapolation_distance = NULL,
    max_neighbor_distance = NULL,
    outputs = NULL,
    ...
)
```

## Arguments

- object:

  A `Loess` object, fitted (via
  [`fit`](https://thisisamirv.github.io/loess-project/r/reference/fit.md))
  with `retain_model = TRUE` passed to
  [`Loess`](https://thisisamirv.github.io/loess-project/r/reference/Loess.md).

- new_x:

  Numeric vector of out-of-sample query points (flattened, `dimensions`
  values per point).

- return_se:

  Logical; include standard errors in the output. Default: `FALSE`.

- confidence_level:

  Confidence interval coverage level (e.g. 0.95). `NULL` (default)
  disables it.

- prediction_level:

  Prediction interval coverage level (e.g. 0.95). `NULL` (default)
  disables it.

- return_derivative:

  Logical; include the local fit's gradient in the output. Default:
  `FALSE`.

- extrapolation:

  Behavior for query points outside the training range: `"clamp"`
  (default), `"linear"`, or `"error"`.

- max_extrapolation_distance:

  Under `"linear"` extrapolation, the maximum allowed distance beyond
  the training boundary before `predict` errors instead of returning an
  unbounded value. `NULL` (default) disables the cap.

- max_neighbor_distance:

  Maximum allowed distance to the farthest point in a query's neighbor
  window before `predict` errors, catching in-range-but-sparse query
  points. `NULL` (default) disables the cap.

- outputs:

  Optional character vector selecting `"se"`, `"gradient"`, or
  `"derivative"`; combined with the corresponding individual flags.

- ...:

  Must be empty.

## Value

A list with a `y` element (predicted values) and optional
`standard_errors`/`confidence_lower`/`confidence_upper`/
`prediction_lower`/`prediction_upper`/`derivative` elements.

## Examples

``` r
x <- seq(0, 10, length.out = 100)
y <- sin(x) + rnorm(100, 0, 0.1)
model <- Loess(fraction = 0.2, retain_model = TRUE)
fit(model, x, y)
#> <LoessResult>
#>   Points:            100 
#>   Fraction Used:     0.2 
#>   Iterations Used:   3 
predict(model, c(2.5, 7.5))
#> $y
#> [1] 0.5095815 0.8260879
#> 
```
