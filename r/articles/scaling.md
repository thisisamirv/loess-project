# Scaling Methods

## Overview

The `scaling_method` parameter controls how the residual scale
$`\hat{\sigma}`$ is estimated during robustness iterations.

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

## MAD (Default)

``` math
\hat{\sigma} = \text{median}(|r_i - \text{median}(r_i)|)
```

``` r

library(rfastloess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

model <- Loess(iterations = 3, scaling_method = "mad")
result <- fit(model, x, y)
```

------------------------------------------------------------------------

## MAR

``` math
\hat{\sigma} = \text{median}(|r_i|)
```

``` r

model <- Loess(iterations = 3, scaling_method = "mar")
result <- fit(model, x, y)
```

------------------------------------------------------------------------

## Mean

``` math
\hat{\sigma} = \frac{1}{n}\sum_i |r_i|
```

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
