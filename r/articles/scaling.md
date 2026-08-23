# Scaling Methods

## Overview

When `iterations > 0`, LOESS computes robustness weights by comparing
each residual to the current residual scale estimate $`\hat{\sigma}`$.
The `scaling_method` parameter controls how $`\hat{\sigma}`$ is
estimated.

``` math
w_i = B\!\left(\frac{|r_i|}{6 \cdot \hat{\sigma}}\right)
```

| Method   | Formula                                     | Robustness  | Speed    |
|----------|---------------------------------------------|-------------|----------|
| `"mad"`  | Median of \|residuals − median(residuals)\| | Very robust | Moderate |
| `"mar"`  | Median of \|residuals\|                     | Robust      | Fast     |
| `"mean"` | Mean of \|residuals\|                       | Less robust | Fastest  |

![Scaling method
comparison](../reference/figures/scaling_comparison.svg)

Scaling method comparison

------------------------------------------------------------------------

## MAD — Median Absolute Deviation (Default)

``` math
\hat{\sigma} = \text{median}(|r_i - \text{median}(r_i)|)
```

First centers residuals at their median, then takes the median of the
absolute deviations. Double use of the median makes it highly resistant
to extreme outliers.

**Use when**: Data may contain outliers (default for most applications).

``` r

library(rfastloess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

model <- Loess(iterations = 3, scaling_method = "mad")
result <- fit(model, x, y)
```

------------------------------------------------------------------------

## MAR — Median Absolute Residual

``` math
\hat{\sigma} = \text{median}(|r_i|)
```

Uses the uncentered median — unlike MAD it does not subtract the
residual median first. Still robust (median-based), faster (one partial
sort instead of two).

**Use when**: Speed matters and data have minimal systematic bias in
residuals.

``` r

model <- Loess(iterations = 3, scaling_method = "mar")
result <- fit(model, x, y)
```

------------------------------------------------------------------------

## Mean — Mean Absolute Residual

``` math
\hat{\sigma} = \frac{1}{n}\sum_i |r_i|
```

Arithmetic mean of absolute residuals. Non-robust: a single extreme
outlier inflates $`\hat{\sigma}`$, causing the bisquare weight function
to under-downweight other outliers.

**Use when**: Clean data with no outliers; maximum computation speed
required.

``` r

model <- Loess(iterations = 3, scaling_method = "mean")
result <- fit(model, x, y)
```

------------------------------------------------------------------------

## Comparing Scaling Methods

``` r

library(rfastloess)
set.seed(42)
x <- 1:100
y <- sin(x / 10) + rnorm(100, sd = 0.3)
y[c(20, 50, 80)] <- y[c(20, 50, 80)] + 5  # outliers

methods <- c("mad", "mar", "mean")
colors  <- c("blue", "red", "green")

plot(x, y, pch = 16, col = "gray",
     main = "Scaling Method Comparison (with outliers)")

for (i in seq_along(methods)) {
    model  <- Loess(iterations = 3, scaling_method = methods[i])
    result <- fit(model, x, y)
    lines(result$x, result$y, col = colors[i], lwd = 2)
}

legend("topright", methods, col = colors, lwd = 2)
```
