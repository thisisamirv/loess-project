# Benchmarks

## CPU Benchmarks

Speedup relative to R’s
[`stats::loess`](https://rdrr.io/r/stats/loess.html) (higher is better):

| Category                 | R baseline | rfastloess Serial | rfastloess Parallel |
|--------------------------|------------|-------------------|---------------------|
| **Clustered**            | 2.34 ms    | 2.0×              | **2.5×**            |
| **Constant Y**           | 1.81 ms    | 1.7×              | **3.2×**            |
| **Extreme Outliers**     | 5.81 ms    | 1.5×              | **2.6×**            |
| **Financial** (500–5K)   | 0.65 ms    | **2.0×**          | 1.4×                |
| **Fraction** (0.05–0.67) | 3.8 ms     | 1.6×              | **3.2×**            |
| **Genomic** (1K–100K)    | 11.2 ms    | 2.2×              | **2.4×**            |
| **High Noise**           | 7.08 ms    | 1.5×              | **3.6×**            |
| **Iterations** (0–10)    | 3.0 ms     | 1.9×              | **2.7×**            |
| **Scale** (1K–10K)       | 1.6 ms     | 1.5×              | **1.6×**            |
| **Scientific** (500–5K)  | 0.9 ms     | 1.4×              | 1.4×                |

------------------------------------------------------------------------

## GPU Backend

| Scenario                 | CPU-Parallel | GPU     | Speedup  |
|--------------------------|--------------|---------|----------|
| n = 50K, fraction = 0.5  | ~200 ms      | ~180 ms | ~1.1×    |
| n = 100K, fraction = 0.5 | ~800 ms      | ~350 ms | ~2.3×    |
| n = 1M, fraction = 0.5   | 1.24 s       | 187 ms  | **6.6×** |

------------------------------------------------------------------------

## Reproducing Benchmarks

``` r

# install.packages("microbenchmark")

library(rfastloess)
library(microbenchmark)

set.seed(42)
n <- 5000
x <- seq(0, 10, length.out = n)
y <- sin(x) + rnorm(n, sd = 0.3)

mb <- microbenchmark(
    stats_loess = stats::loess(y ~ x, span = 0.67),
    rfastloess_serial = {
        m <- Loess(fraction = 0.67)
        fit(m, x, y)
    },
    rfastloess_parallel = {
        m <- Loess(fraction = 0.67, parallel = TRUE)
        fit(m, x, y)
    },
    times = 50
)
print(mb)
```
