# Compute diagnostics for the current Online window

Compute diagnostics for the current Online window

## Usage

``` r
window_diagnostics(model, ...)
```

## Arguments

- model:

  An OnlineLoess object.

- ...:

  Must be empty.

## Value

A list of goodness-of-fit metrics, or `NULL` until the window reaches
`min_points`.

## Examples

``` r
model <- OnlineLoess(fraction = 1, window_capacity = 20L)
for (x in 1:10) {
    invisible(add_point(model, x, 2 * x + 1))
}
window_diagnostics(model)
#> $rmse
#> [1] 0.7620002
#> 
#> $mae
#> [1] 0.5111417
#> 
#> $r_squared
#> [1] 0.9824047
#> 
#> $aic
#> [1] NA
#> 
#> $aicc
#> [1] NA
#> 
#> $effective_df
#> [1] NA
#> 
#> $residual_sd
#> [1] 0.3659058
#> 
```
