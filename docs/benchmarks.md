# Benchmarks

Speedup relative to R's `stats::loess` (higher is better):

| Category | R (stats) | Serial | Parallel |
| --- | --- | --- | --- |
| **Clustered** | 1× | 18× | **21×** |
| **Constant Y** | 1× | 15× | **21×** |
| **Extreme Outliers** | 1× | 7× | **8×** |
| **Financial** (500–5K) | 1× | 5× | **5×** |
| **Fraction** (0.05–0.67) | 1× | **22×** | 18× |
| **Genomic** (1K–5K) | 1× | 6× | **7×** |
| **Genomic** (100K) | 1× | 137× | **201×** |
| **High Noise** | 1× | 22× | **25×** |
| **Iterations** (1–10) | 1× | 13× | **16×** |
| **Scale** (1K–10K) | 1× | **8×** | 8× |
| **Scientific** (500–5K) | 1× | 4× | **5×** |

*Averages across all sizes within each category.*
