# Benchmarks

## CPU Benchmarks

Speedup relative to R’s
[`stats::loess`](https://rdrr.io/r/stats/loess.html) (higher is better):

| Category                    | R (stats) | rfastloess Serial | rfastloess Parallel |
|-----------------------------|-----------|-------------------|---------------------|
| **Clustered**               | 1×        | 19.7×             | **27.1×**           |
| **Constant Y**              | 1×        | 15.5×             | **19.1×**           |
| **Extreme Outliers**        | 1×        | **6.9×**          | 5.4×                |
| **Financial** (500–5K)      | 1×        | **3.7×**          | 3.4×                |
| **Fraction** (0.05–0.67)    | 1×        | 17.3×             | **25.4×**           |
| **Genomic** (1K–5K)         | 1×        | **4.9×**          | 2.9×                |
| **Genomic** (100K)          | 1×        | 129.6×            | **131.9×**          |
| **High Noise**              | 1×        | 17.8×             | **20.7×**           |
| **Iterations** (1–10)       | 1×        | 9.9×              | **15.0×**           |
| **Large** (Direct)          | 1×        | 0.8×              | **2.4×**            |
| **Large** (High Fraction)   | 1×        | **612.5×**        | 550.6×              |
| **Large** (High Iterations) | 1×        | 5.3×              | **7.3×**            |
| **Large** (Interpolate)     | 1×        | 57.6×             | **90.7×**           |
| **Scale** (1K–10K)          | 1×        | 7.5×              | **9.2×**            |
| **Scientific** (500–5K)     | 1×        | 3.6×              | **4.0×**            |

*Mean per-scenario speedup within each category, rounded to one
decimal.*

``` r

sessionInfo()
#> R version 4.6.1 (2026-06-24)
#> Platform: x86_64-pc-linux-gnu
#> Running under: Ubuntu 24.04.5 LTS
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
#> loaded via a namespace (and not attached):
#>  [1] digest_0.6.39     desc_1.4.3        R6_2.6.1          fastmap_1.2.0    
#>  [5] xfun_0.61         cachem_1.1.0      knitr_1.52        htmltools_0.5.9  
#>  [9] rmarkdown_2.32    lifecycle_1.0.5   cli_3.6.6         sass_0.4.10      
#> [13] pkgdown_2.2.1     textshaping_1.0.5 jquerylib_0.1.4   systemfonts_1.3.2
#> [17] compiler_4.6.1    tools_4.6.1       ragg_1.5.2        bslib_0.12.0     
#> [21] evaluate_1.0.5    yaml_2.3.12       otel_0.2.0        jsonlite_2.0.0   
#> [25] rlang_1.3.0       fs_2.1.0          htmlwidgets_1.6.4
```
