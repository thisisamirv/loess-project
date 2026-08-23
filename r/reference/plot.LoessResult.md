# Plot Loess Result

Plot Loess Result

## Usage

``` r
# S3 method for class 'LoessResult'
plot(x, main = "LOESS Fit", ...)
```

## Arguments

- x:

  A LoessResult object.

- main:

  Plot title.

- ...:

  Additional arguments passed to plot() and lines().

## Value

The input object `x`, invisibly.

## Examples

``` r
x <- seq(0, 10, length.out = 100)
y <- sin(x) + rnorm(100, 0, 0.1)
model <- Loess(fraction = 0.2)
res <- fit(model, x, y)
plot(res)
```
