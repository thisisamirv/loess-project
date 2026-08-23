# Fit a LOESS model to data

Fit a LOESS model to data

## Usage

``` r
fit(model, ...)

# S3 method for class 'Loess'
fit(model, x, y, custom_weights = NULL, ...)
```

## Arguments

- model:

  A `Loess` object.

- ...:

  Not used.

- x:

  Numeric vector of predictor values.

- y:

  Numeric vector of response values.

- custom_weights:

  Optional numeric vector of non-negative per-observation weights.
  `NULL` (default) applies no custom weighting.

## Value

A `LoessResult` object.

## Examples

``` r
x <- seq(0, 10, length.out = 100)
y <- sin(x) + rnorm(100, 0, 0.1)
model <- Loess(fraction = 0.2)
result <- fit(model, x, y)
```
