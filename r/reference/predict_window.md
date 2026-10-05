# Predict from the current Online window

Predict from the current Online window

## Usage

``` r
predict_window(model, ...)

# S3 method for class 'OnlineLoess'
predict_window(
  model,
  new_x,
  outputs = NULL,
  intervals = NULL,
  extrapolation = "clamp",
  max_extrapolation_distance = NULL,
  max_neighbor_distance = NULL,
  ...
)
```

## Arguments

- model:

  An OnlineLoess object.

- ...:

  Must be empty.

- new_x:

  Numeric query points (flattened, one coordinate per dimension).

- outputs:

  Optional character vector selecting `"se"` and/or `"gradient"` (alias
  `"derivative"`).

- intervals:

  Grouped coverage levels from
  [`intervals_opts`](https://thisisamirv.github.io/loess-project/r/reference/intervals_opts.md).

- extrapolation:

  Behavior outside the window's predictor bounds: `"clamp"` (default),
  `"linear"`, or `"error"`.

- max_extrapolation_distance:

  Optional cap for linear extrapolation.

- max_neighbor_distance:

  Optional cap for sparse-neighborhood predictions.

## Value

A list containing predicted values and requested optional outputs.

## Examples

``` r
model <- OnlineLoess(
    fraction = 1, window_capacity = 20L, surface_mode = "direct"
)
for (x in 1:10) {
    invisible(add_point(model, x, 2 * x + 1))
}
predict_window(model, c(4.5, 5.5), outputs = "gradient")
#> $y
#> [1] 10 12
#> 
#> $derivative
#> [1] 2 2
#> 
```
