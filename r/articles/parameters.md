# Parameters Reference

## All Parameters

| Parameter | Default | Range/Options | Description | Mode |
|----|----|----|----|----|
| **fraction** | 0.67 | (0, 1\] | Smoothing span | All |
| **iterations** | 3 | \[0, 1000\] | Robustness iterations | All |
| **degree** | 1 | 0–4 | Polynomial degree | All |
| **surface_mode** | `"interpolation"` | 2 options | Fit vs interpolate | All |
| **weight_function** | `"tricube"` | 7 options | Distance kernel | All |
| **robustness_method** | `"bisquare"` | 3 options | Outlier weighting | All |
| **zero_weight_fallback** | `"use_local_mean"` | 3 | Zero-weight | All |
| **boundary_policy** | `"extend"` | 4 options | Edge handling | All |
| **scaling_method** | `"mad"` | 3 options | Scale estimation | All |
| **auto_converge** | `NULL` | tolerance | Early stopping | All |
| **parallel** | `TRUE` | `TRUE`/`FALSE` | Multi-threaded execution | Batch |
| **custom_weights** | `NULL` | positive | Per-observation weights | Batch |
| **return_residuals** | `FALSE` | logical | Include residuals | All |
| **return_robustness_weights** | `FALSE` | logical | Include weights | All |
| **return_diagnostics** | `FALSE` | logical | Include metrics | All |
| **confidence_intervals** | `NULL` | (0, 1) | CI level | Batch |
| **prediction_intervals** | `NULL` | (0, 1) | PI level | Batch |
| **distance_metric** | `"normalized"` | string | Distance metric | All |
| **weighted_metric_weights** | `NULL` | numeric | Per-dim weights | All |
| **cell** | `NULL` | (0, ∞) | Interpolation cell size | All |
| **interpolation_vertices** | `NULL` | integer | Grid vertices | All |
| **boundary_degree_fallback** | `FALSE` | logical | Deg fallback | All |
| **cv_method** | `NULL` | method | Auto-select fraction | Batch |
| **cv_k** | 5 | \[2, ∞) | K-fold count | Batch |
| **cv_fractions** | `NULL` | numeric | Fractions to evaluate | Batch |
| **cv_seed** | `NULL` | integer | CV fold randomisation seed | Batch |
| **chunk_size** | 5000 | \[10, ∞) | Points per chunk | Streaming |
| **overlap** | 500 | \[0, chunk) | Overlap between chunks | Streaming |
| **merge_strategy** | `"weighted_average"` | 4 | Chunk merge | Streaming |
| **window_capacity** | 1000 | \[3, ∞) | Max window size | Online |
| **min_points** | 2 | \[2, window\] | Min before output | Online |
| **update_mode** | `"incremental"` | 2 options | Update strategy | Online |

------------------------------------------------------------------------

## Parameter Options

- **degree**: `0`/`"constant"`, `1`/`"linear"` (default),
  `2`/`"quadratic"`, `3`/`"cubic"`, `4`/`"quartic"`
- **surface_mode**: `"interpolation"` (default), `"direct"`
- **weight_function**: `"tricube"` (default), `"epanechnikov"`,
  `"gaussian"`, `"biweight"`, `"cosine"`, `"triangle"`, `"uniform"`
- **robustness_method**: `"bisquare"` (default), `"huber"`, `"talwar"`
- **zero_weight_fallback**: `"use_local_mean"` (default),
  `"return_original"`, `"return_none"`
- **boundary_policy**: `"extend"` (default), `"reflect"`, `"zero"`,
  `"noboundary"`
- **scaling_method**: `"mad"` (default), `"mar"`, `"mean"`
- **distance_metric**: `"normalized"` (default), `"euclidean"`,
  `"weighted"`
- **merge_strategy**: `"weighted_average"` (default), `"average"`,
  `"take_first"`, `"take_last"`
- **update_mode**: `"incremental"` (default), `"full"`

------------------------------------------------------------------------

## Core Parameters

### fraction

The proportion of data used for each local fit. **Most important
parameter.**

| Value   | Effect          | Use Case                 |
|---------|-----------------|--------------------------|
| 0.1–0.3 | Fine detail     | Rapidly changing signals |
| 0.3–0.5 | Balanced        | General purpose          |
| 0.5–0.7 | Heavy smoothing | Noisy data               |
| 0.7–1.0 | Very smooth     | Trend extraction         |

``` r

library(rfastloess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

model <- Loess(fraction = 0.3)
result <- fit(model, x, y)
cat("First 6 smoothed values (fraction=0.3):\n")
#> First 6 smoothed values (fraction=0.3):
print(head(result$y))
#> [1] 0.4263272 0.4353748 0.4470767 0.4624238 0.4806954 0.5011710
```

### iterations

Number of robustness iterations. Each iteration downweights
high-residual points.

``` r

model <- Loess(iterations = 3L)
result <- fit(model, x, y)
cat("First 6 smoothed values (iterations=3):\n")
#> First 6 smoothed values (iterations=3):
print(head(result$y))
#> [1] 0.4897657 0.4961584 0.5029665 0.5102219 0.5179564 0.5262019
```

### degree

Polynomial degree for local fits. See the [Polynomial
Degree](https://thisisamirv.github.io/loess-project/r/articles/degree.md)
vignette.

``` r

model <- Loess(degree = 2L)  # quadratic
result <- fit(model, x, y)
cat("First 6 smoothed values (degree=2, quadratic):\n")
#> First 6 smoothed values (degree=2, quadratic):
print(head(result$y))
#> [1] 0.4577829 0.4682789 0.4793020 0.4908341 0.5028573 0.5153538
```

### surface_mode

Controls whether LOESS fits directly at every query point (`"direct"`)
or fits at a coarse grid and interpolates (`"interpolation"`, default).
Interpolation is faster for large datasets.

``` r

model <- Loess(surface_mode = "direct")
result <- fit(model, x, y)
cat("First 6 smoothed values (direct surface mode):\n")
#> First 6 smoothed values (direct surface mode):
print(head(result$y))
#> [1] 0.4839495 0.4916503 0.4997001 0.5080695 0.5166623 0.5253365
```

### distance_metric

Metric used when computing neighbour distances in multivariate mode. Use
`"weighted"` with `weighted_metric_weights` to scale each dimension
separately.

``` r

# 2D fit with per-dimension scaling
x2d <- cbind(seq(0, 1, length.out = 100), seq(0, 2, length.out = 100))
model <- Loess(
    dimensions = 2L,
    distance_metric = "weighted",
    weighted_metric_weights = c(1.0, 0.5)
)
result <- fit(model, x2d, y)
cat("First 6 smoothed values (2D LOESS, weighted distance):\n")
#> First 6 smoothed values (2D LOESS, weighted distance):
print(head(result$y))
#> [1] 0.6347210 0.6556046 0.6881319 0.7223324 0.7409650 0.7287990
```

### parallel

Enable parallel CPU execution (multiple cores).

``` r

model <- Loess(parallel = TRUE)
result <- fit(model, x, y)
cat("First 6 smoothed values (parallel=TRUE):\n")
#> First 6 smoothed values (parallel=TRUE):
print(head(result$y))
#> [1] 0.4897657 0.4961584 0.5029665 0.5102219 0.5179564 0.5262019
```

### return_diagnostics

Return goodness-of-fit diagnostics (R², residuals, etc.).

``` r

model <- Loess(return_diagnostics = TRUE)
result <- fit(model, x, y)
cat("R²:", result$diagnostics$r_squared, "\n")
#> R²: 0.7227642
```

------------------------------------------------------------------------

## Streaming Parameters

### chunk_size and overlap

``` r

chunk_size <- 5000L
overlap    <- 500L
model <- StreamingLoess(
    chunk_size = chunk_size,
    overlap = overlap,
    merge_strategy = "weighted_average"
)
cat(sprintf("overlap = %d points (%.0f%% of chunk)\n",
            overlap, 100 * overlap / chunk_size))
#> overlap = 500 points (10% of chunk)
```

------------------------------------------------------------------------

## Online Parameters

### window_capacity and min_points

``` r

model <- OnlineLoess(
    window_capacity = 50L,
    min_points = 5L,
    update_mode = "incremental"
)
cat("OnlineLoess model created.\n")
#> OnlineLoess model created.
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
#> [1] rfastloess_1.0.0
#> 
#> loaded via a namespace (and not attached):
#>  [1] digest_0.6.39     desc_1.4.3        R6_2.6.1          fastmap_1.2.0    
#>  [5] xfun_0.60         cachem_1.1.0      knitr_1.51        htmltools_0.5.9  
#>  [9] rmarkdown_2.31    lifecycle_1.0.5   cli_3.6.6         sass_0.4.10      
#> [13] pkgdown_2.2.1     textshaping_1.0.5 jquerylib_0.1.4   systemfonts_1.3.2
#> [17] compiler_4.6.1    tools_4.6.1       ragg_1.5.2        bslib_0.12.0     
#> [21] evaluate_1.0.5    yaml_2.3.12       otel_0.2.0        jsonlite_2.0.0   
#> [25] rlang_1.3.0       fs_2.1.0          htmlwidgets_1.6.4
```
