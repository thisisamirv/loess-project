# Merge Strategies

## Overview

Streaming LOESS processes data in fixed-size chunks with a configurable
overlap. Points inside the overlap zone receive estimates from both
adjacent chunks; the merge strategy decides how those two estimates are
reconciled.

``` text
Chunk A:   [=========|=====]
Chunk B:            [=====|=========]
Overlap:            [=====]
```

| Strategy             | Method                        | Robustness | Speed    |
|----------------------|-------------------------------|------------|----------|
| `"average"`          | Simple mean of both estimates | Low        | Fastest  |
| `"take_first"`       | Left-chunk estimate only      | Low        | Fastest  |
| `"take_last"`        | Right-chunk estimate only     | Low        | Fastest  |
| `"weighted_average"` | Distance-weighted mean        | High       | Moderate |

![Merge strategy comparison](../reference/figures/merge_comparison.svg)

Merge strategy comparison

------------------------------------------------------------------------

## Average

Takes the arithmetic mean of the left-chunk and right-chunk estimates in
the overlap region. Fast and sufficient when both chunks have similar
smoothing quality.

**Use when**: Chunks are large and the overlap region has uniform data
density.

``` r

library(rfastloess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

model <- StreamingLoess(
    merge_strategy = "average",
    chunk_size = 60,
    overlap = 20
)
invisible(process_chunk(model, x[1:60], y[1:60]))
# The second chunk's overlap region (its first 20 points) is where
# merge_strategy actually blends the two chunks' estimates.
result <- process_chunk(model, x[61:100], y[61:100])
cat("Merged value in overlap region (average):\n")
#> Merged value in overlap region (average):
print(result$y[6])
#> [1] 0.2286069
```

------------------------------------------------------------------------

## Take First

Keeps only the left-chunk estimate in the overlap zone and discards the
right-chunk estimate. Produces a left-flush output.

**Use when**: You need final output values immediately after each chunk
(no look-ahead revision); left-chunk context dominates.

``` r

model <- StreamingLoess(
    merge_strategy = "take_first",
    chunk_size = 60,
    overlap = 20
)
invisible(process_chunk(model, x[1:60], y[1:60]))
result <- process_chunk(model, x[61:100], y[61:100])
cat("Merged value in overlap region (take_first):\n")
#> Merged value in overlap region (take_first):
print(result$y[6])
#> [1] 0.236299
```

------------------------------------------------------------------------

## Take Last

Keeps only the right-chunk estimate in the overlap zone. The right chunk
sees more of the surrounding data, so its estimates can be more accurate
in the overlap region.

**Use when**: Right-chunk context improves overlap quality; you are
post-processing complete data rather than streaming in real time.

``` r

model <- StreamingLoess(
    merge_strategy = "take_last",
    chunk_size = 60,
    overlap = 20
)
invisible(process_chunk(model, x[1:60], y[1:60]))
result <- process_chunk(model, x[61:100], y[61:100])
cat("Merged value in overlap region (take_last):\n")
#> Merged value in overlap region (take_last):
print(result$y[6])
#> [1] 0.2209148
```

------------------------------------------------------------------------

## Weighted Average

Assigns each overlap point a weight proportional to its proximity to the
centre of its respective chunk: points near the left-chunk centre favour
the left estimate; points near the right-chunk centre favour the right.
Minimises boundary artefacts.

**Use when**: Minimising boundary artefacts is more important than
speed; moderate overlap (10–20% of chunk size) is used.

``` r

model <- StreamingLoess(
    merge_strategy = "weighted_average",
    chunk_size = 60,
    overlap = 20
)
invisible(process_chunk(model, x[1:60], y[1:60]))
result <- process_chunk(model, x[61:100], y[61:100])
cat("Merged value in overlap region (weighted_average):\n")
#> Merged value in overlap region (weighted_average):
print(result$y[6])
#> [1] 0.232453
```

------------------------------------------------------------------------

## Choosing a Strategy

| Situation                             | Recommended Strategy |
|---------------------------------------|----------------------|
| General purpose                       | `"weighted_average"` |
| Maximum throughput                    | `"average"`          |
| Immediate finalised output            | `"take_first"`       |
| Post-processing, right context better | `"take_last"`        |
| Minimising boundary artefacts         | `"weighted_average"` |

> **Overlap size matters:** A larger overlap gives the merge strategy
> more room to blend, reducing boundary artefacts regardless of the
> strategy chosen. A good starting point is 10% of `chunk_size`.

------------------------------------------------------------------------

## Choosing Chunk Size and Overlap

``` r

# Rule of thumb: overlap = fraction * chunk_size
fraction  <- 0.3
chunk_size <- 5000
overlap    <- ceiling(fraction * chunk_size)  # 1500

model <- StreamingLoess(
    fraction       = fraction,
    chunk_size     = chunk_size,
    overlap        = overlap,
    merge_strategy = "weighted_average"
)
cat(sprintf("overlap = %d points (%.0f%% of chunk)\n",
            overlap, 100 * overlap / chunk_size))
#> overlap = 1500 points (30% of chunk)
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
