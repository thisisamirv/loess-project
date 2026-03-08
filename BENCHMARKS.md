# Benchmarks

Speedup relative to R's `stats::loess` (higher is better):

| Category                        | R (stats) | Serial | Parallel | GPU     |
|---------------------------------|-----------|--------|----------|---------|
| **Clustered**                   | 1×        | 2.4×   | **5.2×** | 0.4×    |
| **Constant Y**                  | 1×        | 2.3×   | **4.5×** | 0.2×    |
| **Extreme Outliers**            | 1×        | 1.9×   | **3.7×** | 0.3×    |
| **Financial** (500–10K)         | 1×        | 2.4×   | **2.8×** | 0.1×    |
| **Fraction** (0.05–0.67)        | 1×        | 2.2×   | **3.8×** | 0.2×    |
| **Genomic** (1K–50K)            | 1×        | 1.3×   | 2.9×     | **13×** |
| **High Noise**                  | 1×        | 1.0×   | **2.8×** | 0.2×    |
| **Iterations** (0–10)           | 1×        | 1.9×   | **3.3×** | 0.1×    |
| **Scale** (1K–50K)              | 1×        | 1.8×   | **2.2×** | 0.4×    |
| **Scientific** (500–10K)        | 1×        | 1.9×   | **2.9×** | 0.1×    |

*The numbers are the average across a range of scenarios for each category (port from lowess-project adapted for fastLoess).*
