# Use Case: Time Series Analysis

## Overview

LOESS provides flexible trend extraction from time series without
parametric assumptions.

------------------------------------------------------------------------

## Basic Trend Extraction

`fraction = 0.1` sizes the neighbourhood as 10% of the data at each
evaluation point — narrow enough to follow a slowly varying trend
without smearing periodic variation. Three robustness `iterations`
down-weight noise spikes so they cannot bias the fitted curve; this is
especially important when the signal-to-noise ratio is low or when
occasional outliers are expected.

``` r

library(rfastloess)

set.seed(42)
t <- seq(0, 100, length.out = 500)
trend <- 10 + 0.5 * t + 3 * sin(t / 10)
noise <- rnorm(500, sd = 3)
y <- trend + noise

model <- Loess(fraction = 0.1, iterations = 3)
result <- fit(model, t, y)

plot(t, y, col = "gray", pch = ".",
    xlab = "Time", ylab = "Value", main = "Trend Extraction")
lines(result$x, result$y, col = "blue", lwd = 2)
legend("topleft", c("Observed", "Trend (LOESS)"),
        pch = c(1, NA), lty = c(NA, 1), col = c("gray", "blue"))
```

![](use-case-time-series_files/figure-html/use_case_time_series_1-1.png)

------------------------------------------------------------------------

## Seasonal Decomposition

Setting `return_residuals = TRUE` stores `observed − smoothed` alongside
the smooth. A slightly wider `fraction = 0.4` produces a smoother
baseline trend, so short-duration oscillations end up in the residuals
rather than being absorbed into the trend component. The residual series
is then ready for spectral analysis, seasonality detection, or
change-point methods.

``` r

library(rfastloess)
set.seed(42)

t        <- 1:120
trend    <- 100 + 0.5 * t
seasonal <- 15 * sin(2 * pi * t / 12)
noise    <- rnorm(120, sd = 5)
y        <- trend + seasonal + noise

model    <- Loess(fraction = 0.4, iterations = 2)
result   <- fit(model, t, y)
residual <- y - result$y

par(mfrow = c(2, 1), mar = c(4, 4, 2, 0.5))
plot(t, y, type = "l", col = "gray",
    main = "Original + LOESS Trend", xlab = "Month", ylab = "Value")
lines(result$x, result$y, col = "blue", lwd = 2)

plot(t, residual, type = "l", col = "darkgreen",
    main = "Detrended Residual (Seasonal Component)",
    xlab = "Month", ylab = "Residual")
abline(h = 0, lty = 2, col = "red")
```

![](use-case-time-series_files/figure-html/use_case_time_series_2-1.png)

``` r

par(mfrow = c(1, 1))
```

------------------------------------------------------------------------

## Irregular Time Grids

``` r

library(rfastloess)
set.seed(42)

t_irregular <- c(sort(runif(200, 0, 50)), sort(runif(50, 50, 100)))
y_irregular <- sin(t_irregular / 10) + rnorm(length(t_irregular), sd = 0.5)

model  <- Loess(fraction = 0.2, iterations = 2)
result <- fit(model, t_irregular, y_irregular)

plot(t_irregular, y_irregular, pch = 16, cex = 0.4, col = "gray",
    xlab = "Time", ylab = "Value", main = "Irregularly Sampled Time Series")
lines(result$x, result$y, col = "blue", lwd = 2)
```

![](use-case-time-series_files/figure-html/use_case_time_series_3-1.png)

------------------------------------------------------------------------

## Uncertainty Bands

``` r

library(rfastloess)
set.seed(42)
t <- seq(0, 100, length.out = 500)
y <- 10 + 0.3 * t + sin(t / 5) + rnorm(500, sd = 2)

model <- Loess(
    fraction = 0.2,
    iterations = 3,
    confidence_intervals = 0.95
)
result <- fit(model, t, y)

plot(t, y, pch = ".", col = "gray",
    xlab = "Time", ylab = "Value",
    main = "Trend with 95% Confidence Interval")
polygon(c(result$x, rev(result$x)),
        c(result$confidence_upper, rev(result$confidence_lower)),
        col = rgb(0, 0, 1, 0.15), border = NA)
lines(result$x, result$y, col = "blue", lwd = 2)
```

![](use-case-time-series_files/figure-html/use_case_time_series_4-1.png)

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
#>  [1] digest_0.6.39       desc_1.4.3          R6_2.6.1           
#>  [4] fastmap_1.2.0       xfun_0.60           cachem_1.1.0       
#>  [7] knitr_1.51          BiocGenerics_0.58.1 htmltools_0.5.9    
#> [10] generics_0.1.4      rmarkdown_2.31      lifecycle_1.0.5    
#> [13] cli_3.6.6           sass_0.4.10         pkgdown_2.2.1      
#> [16] textshaping_1.0.5   jquerylib_0.1.4     systemfonts_1.3.2  
#> [19] compiler_4.6.1      tools_4.6.1         ragg_1.5.2         
#> [22] bslib_0.12.0        evaluate_1.0.5      yaml_2.3.12        
#> [25] otel_0.2.0          jsonlite_2.0.0      rlang_1.3.0        
#> [28] fs_2.1.0            htmlwidgets_1.6.4
```
