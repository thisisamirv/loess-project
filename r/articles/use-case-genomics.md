# Use Case: Genomic Data Smoothing

## Overview

LOESS is well-suited for genomic data such as DNA methylation profiles,
ChIP-seq signals, and expression arrays.

------------------------------------------------------------------------

## Methylation Profile Smoothing

### The Challenge

DNA methylation data (from bisulfite sequencing or arrays) shows
position-dependent patterns that can be obscured by measurement noise.

### Solution

A small `fraction = 0.1` lets LOESS follow fine-scale spatial structure
without smearing the transitions between methylated and unmethylated
regions. `confidence_intervals = 0.95` produces uncertainty bands that
naturally widen at positions with sparser CpG coverage, making
low-confidence segments immediately apparent in the plot.

``` r

library(rfastloess)

set.seed(42)
n <- 1000
positions <- sort(runif(n, 0, 1e6))

true_meth <- 0.5 + 0.3 * sin(positions / 1e5)
observed  <- true_meth + rnorm(n, sd = 0.15)
observed  <- pmax(0, pmin(1, observed))

model <- Loess(
    fraction = 0.1,
    iterations = 3,
    confidence_intervals = 0.95
)
result <- fit(model, positions, observed)

plot(positions, observed, pch = ".", col = "gray",
     xlab = "Genomic Position (bp)", ylab = "Methylation Level",
     main = "Methylation Profile Smoothing")
lines(result$x, result$y, col = "blue", lwd = 2)
lines(result$x, result$confidence_lower, col = "blue", lty = 2)
lines(result$x, result$confidence_upper, col = "blue", lty = 2)
legend("topright", c("Observed", "Smoothed", "95% CI"),
       pch = c(1, NA, NA), lty = c(NA, 1, 2),
       col = c("gray", "blue", "blue"))
```

------------------------------------------------------------------------

## ChIP-seq Signal Smoothing

``` r

library(rfastloess)
set.seed(42)

n <- 500
positions    <- seq(1, 50000, length.out = n)
signal       <- 10 + 5 * dnorm(positions, mean = 25000, sd = 5000) * 1000
noise        <- rpois(n, lambda = 2)
signal_noisy <- signal + noise
signal_noisy[c(100, 200, 300)] <- signal_noisy[c(100, 200, 300)] * 5

model  <- Loess(fraction = 0.1, iterations = 3)
result <- fit(model, positions, signal_noisy)

plot(positions, signal_noisy, pch = 16, cex = 0.4, col = "gray",
     xlab = "Genomic Position", ylab = "Read Count",
     main = "ChIP-seq Signal Smoothing")
lines(result$x, result$y, col = "blue", lwd = 2)
```

------------------------------------------------------------------------

## Large Genomic Datasets (Streaming)

``` r

library(rfastloess)

model <- StreamingLoess(
    fraction   = 0.05,
    iterations = 2,
    chunk_size = 1000,
    overlap    = 100,
    merge_strategy = "weighted_average"
)

set.seed(42)
for (chunk_i in 1:5) {
    pos_chunk <- seq((chunk_i - 1) * 1e5 + 1, chunk_i * 1e5, length.out = 1000)
    val_chunk <- sin(pos_chunk / 1e4) + rnorm(1000, sd = 0.2)
    result    <- process_chunk(model, pos_chunk, val_chunk)
}
final <- finalize(model)
```
