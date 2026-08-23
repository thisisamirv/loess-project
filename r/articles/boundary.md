# Boundary Handling

## Overview

![Boundary policy
comparison](../reference/figures/boundary_comparison.svg)

Boundary policy comparison

The `boundary_policy` parameter controls how data is padded at the edges
to reduce bias from asymmetric neighbourhoods.

| Policy | Padding Strategy | Best For |
|----|----|----|
| `"extend"` | Repeat first / last value | Most datasets (default) |
| `"reflect"` | Mirror data at boundaries | Periodic or symmetric data |
| `"zero"` | Pad with zeros | Data known to approach zero |
| `"noboundary"` | No padding (Cleveland original) | Reproducing reference behaviour |

------------------------------------------------------------------------

## Extend (Default)

``` r

library(rfastloess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

model <- Loess(boundary_policy = "extend")
result <- fit(model, x, y)
```

------------------------------------------------------------------------

## Reflect

``` r

model <- Loess(boundary_policy = "reflect")
result <- fit(model, x, y)
```

------------------------------------------------------------------------

## Zero

``` r

model <- Loess(boundary_policy = "zero")
result <- fit(model, x, y)
```

------------------------------------------------------------------------

## No Boundary Padding

``` r

model <- Loess(boundary_policy = "noboundary")
result <- fit(model, x, y)
```

------------------------------------------------------------------------

## Comparing Policies

``` r

library(rfastloess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

policies <- c("extend", "reflect", "zero", "noboundary")
colors   <- c("blue", "red", "green", "purple")

plot(x, y, pch = 16, col = "gray",
     main = "Boundary Policy Comparison")

for (i in seq_along(policies)) {
    model  <- Loess(boundary_policy = policies[i])
    result <- fit(model, x, y)
    lines(result$x, result$y, col = colors[i], lwd = 2)
}

legend("topright", policies, col = colors, lwd = 2)
```
