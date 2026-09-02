# Benchmarks

## CPU Benchmarks

Speedup relative to R's `stats::loess` (higher is better):

| Category | R (stats) | Serial |
| --- | --- | --- |
| **Clustered** | 1× | 18× |
| **Constant Y** | 1× | 15× |
| **Extreme Outliers** | 1× | 7× |
| **Financial** (500–5K) | 1× | 5× |
| **Fraction** (0.05–0.67) | 1× | 22× |
| **Genomic** (1K–5K) | 1× | 6× |
| **Genomic** (100K) | 1× | 137× |
| **High Noise** | 1× | 22× |
| **Iterations** (1–10) | 1× | 13× |
| **Large** (Direct) | 1× | 0.9× |
| **Large** (High Fraction) | 1× | 577× |
| **Large** (High Iterations) | 1× | 6× |
| **Large** (Interpolate) | 1× | 60× |
| **Scale** (1K–10K) | 1× | 8× |
| **Scientific** (500–5K) | 1× | 4× |

*Averages across all sizes within each category. The `loess-rs` crate has no `parallel` or `gpu` feature — for CPU-parallel and GPU-accelerated numbers, see the `fastLoess` crate's benchmarks.*
