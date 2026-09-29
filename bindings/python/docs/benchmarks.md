# Benchmarks

Speedup relative to R's `stats::loess` (higher is better):

| Category | R (stats) | Serial | Parallel |
| --- | --- | --- | --- |
| **Clustered** | 1× | 19.7× | **27.1×** |
| **Constant Y** | 1× | 15.5× | **19.1×** |
| **Extreme Outliers** | 1× | **6.9×** | 5.4× |
| **Financial** (500–5K) | 1× | **3.7×** | 3.4× |
| **Fraction** (0.05–0.67) | 1× | 17.3× | **25.4×** |
| **Genomic** (1K–5K) | 1× | **4.9×** | 2.9× |
| **Genomic** (100K) | 1× | 129.6× | **131.9×** |
| **High Noise** | 1× | 17.8× | **20.7×** |
| **Iterations** (1–10) | 1× | 9.9× | **15.0×** |
| **Large** (Direct) | 1× | 0.8× | **2.4×** |
| **Large** (High Fraction) | 1× | **612.5×** | 550.6× |
| **Large** (High Iterations) | 1× | 5.3× | **7.3×** |
| **Large** (Interpolate) | 1× | 57.6× | **90.7×** |
| **Scale** (1K–10K) | 1× | 7.5× | **9.2×** |
| **Scientific** (500–5K) | 1× | 3.6× | **4.0×** |

*Mean per-scenario speedup within each category, rounded to one decimal.*
