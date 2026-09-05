# Multivariate LOESS

## Overview

Standard LOESS operates on a single predictor `x`. Setting
`dimensions > 1` extends the neighbourhood search and local polynomial
fit into an **n-dimensional predictor space**, enabling surface
smoothing over spatial grids, time–altitude combinations, and similar
multi-predictor datasets.

![Multivariate LOESS](../reference/figures/multivariate_loess.svg)

Multivariate LOESS

| Dimensions | Use Case                           | `x` Input Shape |
|------------|------------------------------------|-----------------|
| `1`        | Time series, 1D signal (default)   | Numeric vector  |
| `2`        | Spatial surface, 2-predictor model | `n × 2` matrix  |
| `3+`       | High-dimensional regression        | `n × d` matrix  |

> **Note:** Neighbourhood search scales with the number of dimensions.
> For `d ≥ 3` keep `fraction` small and consider the `"interpolation"`
> surface mode (the default) to reduce computation.

------------------------------------------------------------------------

## 1D — Standard (Default)

Single predictor. No configuration required.

``` r

library(rfastloess)
set.seed(42)
x <- seq(0, 10, length.out = 200)
y <- sin(x) + rnorm(200, sd = 0.2)

model <- Loess(fraction = 0.3)
result <- fit(model, x, y)
cat("First 6 smoothed values (1D LOESS):\n")
#> First 6 smoothed values (1D LOESS):
print(head(result$y))
#> [1] 0.3808654 0.3912167 0.4028709 0.4154835 0.4286045 0.4422857
```

------------------------------------------------------------------------

## 2D — Spatial Surface

Two predictors (e.g., latitude/longitude, time/altitude). Pass an
`n × 2` matrix as `x`.

``` r

set.seed(42)
n <- 100
lat <- runif(n, -pi, pi)
lon <- runif(n, -pi, pi)
z   <- sin(lat) + cos(lon) + rnorm(n, sd = 0.1)

x2d <- cbind(lat, lon)   # n × 2 matrix

model <- Loess(fraction = 0.3, dimensions = 2L)
result <- fit(model, x2d, z)
cat("First 6 smoothed values (2D LOESS, lat/lon):\n")
#> First 6 smoothed values (2D LOESS, lat/lon):
print(head(result$y))
#> [1]  0.17983519  0.02191686 -0.01849028  0.22186652  0.09969107 -0.22507546
```

------------------------------------------------------------------------

## 3D and Higher

Pass an `n × d` matrix as `x`. Higher dimensions require a larger
`fraction` to keep the local neighbourhood populated.

``` r

set.seed(42)
n <- 200
x1 <- runif(n)
x2 <- runif(n)
x3 <- runif(n)
y  <- x1 + sin(2 * pi * x2) + x3^2 + rnorm(n, sd = 0.1)

x3d <- cbind(x1, x2, x3)   # n × 3 matrix

model <- Loess(fraction = 0.5, dimensions = 3L)
result <- fit(model, x3d, y)
cat("First 6 smoothed values (3D LOESS):\n")
#> First 6 smoothed values (3D LOESS):
print(head(result$y))
#> [1] 0.8070023 0.9764349 0.8780069 1.0434288 0.9838051 0.8222218
```

------------------------------------------------------------------------

## Distance Metrics for Multivariate Data

When `dimensions > 1` you can also control how inter-point distances are
computed.

| Metric | Description | When to Use |
|----|----|----|
| `"normalized"` | Scaled by dimension range (default) | Mixed-scale |
| `"euclidean"` | Raw Euclidean distance | Same-scale predictors |
| `"minkowski:p"` | Minkowski ($`L_p`$) norm | Custom geometry |
| `"weighted"` | Per-dimension via `weighted_metric_weights` | Anisotropic |

``` r

# Reuse the 2D spatial data from above; upweight the lat dimension
model <- Loess(
    fraction = 0.3,
    dimensions = 2L,
    distance_metric = "weighted",
    weighted_metric_weights = c(2.0, 1.0)  # lat counts twice
)
result <- fit(model, x2d, z)
cat("First 6 smoothed values (2D LOESS, weighted distance):\n")
#> First 6 smoothed values (2D LOESS, weighted distance):
print(head(result$y))
#> [1]  0.17709387  0.08359416 -0.05142187  0.14223791  0.12033868 -0.13208634
```

------------------------------------------------------------------------

## Surface Mode for Multivariate Data

| Mode              | Behaviour                                 | Speed         |
|-------------------|-------------------------------------------|---------------|
| `"interpolation"` | Grid-fit, interpolate elsewhere (default) | Fast          |
| `"direct"`        | Fit directly at every query point         | Exact, slower |

``` r

# Exact fit at every point — slower but no interpolation artefacts
model <- Loess(surface_mode = "direct", fraction = 0.3, dimensions = 2L)
result <- fit(model, x2d, z)
cat("First 6 smoothed values (2D LOESS, direct surface):\n")
#> First 6 smoothed values (2D LOESS, direct surface):
print(head(result$y))
#> [1]  0.19156485  0.02608782  0.01152671  0.22715513  0.13701873 -0.24090868
```

For large 2D or 3D datasets use `"interpolation"` (default) and tune
`cell` and `interpolation_vertices` to control grid resolution.

``` r

model <- Loess(
    surface_mode = "interpolation",
    cell = 0.2,
    fraction = 0.3,
    dimensions = 2L
)
result <- fit(model, x2d, z)
cat("First 6 smoothed values (2D LOESS, interpolation surface):\n")
#> First 6 smoothed values (2D LOESS, interpolation surface):
print(head(result$y))
#> [1]  0.17983519  0.02191686 -0.01849028  0.22186652  0.09969107 -0.22507546
```

``` r

sessionInfo()
#> R version 4.6.1 (2026-06-24)
#> Platform: x86_64-pc-linux-gnu
#> Running under: Ubuntu 24.04.4 LTS
#> 
#> Matrix products: default
#> BLAS:   /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3 
#> LAPACK: /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblasp-r0.3.26.so;  LAPACK version 3.12.0
#> 
#> locale:
#>  [1] LC_CTYPE=C.UTF-8       LC_NUMERIC=C           LC_TIME=C.UTF-8       
#>  [4] LC_COLLATE=C.UTF-8     LC_MONETARY=C.UTF-8    LC_MESSAGES=C.UTF-8   
#>  [7] LC_PAPER=C.UTF-8       LC_NAME=C              LC_ADDRESS=C          
#> [10] LC_TELEPHONE=C         LC_MEASUREMENT=C.UTF-8 LC_IDENTIFICATION=C   
#> 
#> time zone: UTC
#> tzcode source: system (glibc)
#> 
#> attached base packages:
#> [1] stats     graphics  grDevices utils     datasets  methods   base     
#> 
#> other attached packages:
#> [1] rfastloess_2.0.0
#> 
#> loaded via a namespace (and not attached):
#>  [1] digest_0.6.39     desc_1.4.3        R6_2.6.1          fastmap_1.2.0    
#>  [5] xfun_0.60         cachem_1.1.0      knitr_1.51        htmltools_0.5.9  
#>  [9] rmarkdown_2.32    lifecycle_1.0.5   cli_3.6.6         sass_0.4.10      
#> [13] pkgdown_2.2.1     textshaping_1.0.5 jquerylib_0.1.4   systemfonts_1.3.2
#> [17] compiler_4.6.1    tools_4.6.1       ragg_1.5.2        bslib_0.12.0     
#> [21] evaluate_1.0.5    yaml_2.3.12       otel_0.2.0        jsonlite_2.0.0   
#> [25] rlang_1.3.0       fs_2.1.0          htmlwidgets_1.6.4
```
