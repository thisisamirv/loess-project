# Cross-validation options for [`Loess`](https://thisisamirv.github.io/loess-project/r/reference/Loess.md)

Cross-validation options for
[`Loess`](https://thisisamirv.github.io/loess-project/r/reference/Loess.md)

## Usage

``` r
cv_opts(fractions, method = "kfold", k = 5L, seed = NULL)
```

## Arguments

- fractions:

  Numeric vector of candidate smoothing fractions.

- method:

  Cross-validation method: `"kfold"` or `"loocv"`.

- k:

  Number of folds for k-fold cross-validation. Default: 5.

- seed:

  Seed for reproducible fold assignment, or `NULL`.

## Value

A `cv_opts` list for `Loess(cv = ...)`.

## Examples

``` r
model <- Loess(cv = cv_opts(fractions = c(0.2, 0.3, 0.5)))
```
