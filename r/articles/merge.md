# Merge Strategies

## Overview

Streaming LOESS processes data in fixed-size chunks with overlap. The
`merge_strategy` decides how overlapping estimates from adjacent chunks
are combined.

| Strategy | Method | Best For |
|----|----|----|
| `"average"` | Simple mean | Uniform data density |
| `"take_first"` | Left-chunk estimate only | Left chunk is more accurate |
| `"take_last"` | Right-chunk estimate only | Right chunk is more accurate |
| `"weighted_average"` | Distance-weighted mean (default) | Most situations |

![Merge strategy comparison](../reference/figures/merge_comparison.svg)

Merge strategy comparison

------------------------------------------------------------------------

## Weighted Average (Default)

``` r

library(rfastloess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)
x_chunk <- x[seq_len(50)]
y_chunk <- y[seq_len(50)]

model <- StreamingLoess(
    merge_strategy = "weighted_average",
    chunk_size = 5000,
    overlap = 500
)
result <- process_chunk(model, x_chunk, y_chunk)
```

------------------------------------------------------------------------

## Average

``` r

model <- StreamingLoess(
    merge_strategy = "average",
    chunk_size = 5000,
    overlap = 500
)
result <- process_chunk(model, x_chunk, y_chunk)
```

------------------------------------------------------------------------

## Take First

``` r

model <- StreamingLoess(
    merge_strategy = "take_first",
    chunk_size = 5000,
    overlap = 500
)
result <- process_chunk(model, x_chunk, y_chunk)
```

------------------------------------------------------------------------

## Take Last

``` r

model <- StreamingLoess(
    merge_strategy = "take_last",
    chunk_size = 5000,
    overlap = 500
)
result <- process_chunk(model, x_chunk, y_chunk)
```

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
```
