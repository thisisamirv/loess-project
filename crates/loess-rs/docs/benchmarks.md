# Benchmarks

## CPU Benchmarks

Speedup relative to R's `stats::loess` (higher is better):

| Category | R (stats) | Serial |
| --- | --- | --- |
| **Clustered** | 1× | 19.7× |
| **Constant Y** | 1× | 15.5× |
| **Extreme Outliers** | 1× | 6.9× |
| **Financial** (500–5K) | 1× | 3.7× |
| **Fraction** (0.05–0.67) | 1× | 17.3× |
| **Genomic** (1K–5K) | 1× | 4.9× |
| **Genomic** (100K) | 1× | 129.6× |
| **High Noise** | 1× | 17.8× |
| **Iterations** (1–10) | 1× | 9.9× |
| **Large** (Direct) | 1× | 0.8× |
| **Large** (High Fraction) | 1× | 612.5× |
| **Large** (High Iterations) | 1× | 5.3× |
| **Large** (Interpolate) | 1× | 57.6× |
| **Scale** (1K–10K) | 1× | 7.5× |
| **Scientific** (500–5K) | 1× | 3.6× |

*Mean per-scenario speedup within each category, rounded to one decimal. The `loess-rs` crate has no `parallel` or `gpu` feature — for CPU-parallel and GPU-accelerated numbers, see the `fastLoess` crate's benchmarks.*
