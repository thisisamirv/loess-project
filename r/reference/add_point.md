# Add a single point to an online LOESS model

Add a single point to an online LOESS model

## Usage

``` r
add_point(model, ...)

# S3 method for class 'OnlineLoess'
add_point(model, x, y, weight = 1, ...)
```

## Arguments

- model:

  An `OnlineLoess` object.

- ...:

  Must be empty.

- x:

  A numeric coordinate vector with one value per configured dimension.
  For one-dimensional models, a scalar is also accepted.

- y:

  A single numeric y value.

- weight:

  Finite non-negative case weight for this observation; defaults to 1.

## Value

A list with `status` (`"filling"`, `"ready"`, or `"dropped"`) and
`point` (an online-output list when ready; otherwise `NULL`).

## Examples

``` r
model <- OnlineLoess(fraction = 0.2, window_capacity = 20L, min_points = 2L)
invisible(add_point(model, 1.0, 0.5))
add_point(model, 2.0, 0.6)
#> $status
#> [1] "ready"
#> 
#> $point
#> $point$y
#> [1] 0.6
#> 
#> $point$residual
#> [1] 0
#> 
#> $point$robustness_weight
#> [1] 1
#> 
#> 
```
