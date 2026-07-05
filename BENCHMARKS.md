# Benchmarks

Speedup relative to R's `stats::loess` (higher is better):

| Category | R (stats) | Serial | Parallel |
| --- | --- | --- | --- |
| **Clustered** | 1× | 2.4× | **5.2×** |
| **Constant Y** | 1× | 2.3× | **4.5×** |
| **Extreme Outliers** | 1× | 1.9× | **3.7×** |
| **Financial** (500–10K) | 1× | 2.4× | **2.8×** |
| **Fraction** (0.05–0.67) | 1× | 2.2× | **3.8×** |
| **Genomic** (1K–50K) | 1× | 1.3× | **2.9×** |
| **High Noise** | 1× | 1.0× | **2.8×** |
| **Iterations** (0–10) | 1× | 1.9× | **3.3×** |
| **Scale** (1K–50K) | 1× | 1.8× | **2.2×** |
| **Scientific** (500–10K) | 1× | 1.9× | **2.9×** |

*The numbers are the average across a range of scenarios for each category (e.g., Fraction from 0.05 to 0.67).*
