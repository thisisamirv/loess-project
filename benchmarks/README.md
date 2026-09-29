# Benchmarks

Compares `stats::loess` (base R) against `rfastloess` (this package) across a set of representative scenarios.

## Scenarios

| Category | Variants | Description |
| --- | --- | --- |
| **Scalability** | n = 1 000 / 5 000 / 10 000 | Sine wave, fraction 0.1, 3 robustness iterations |
| **Fraction** | 0.05 – 0.67 (6 levels) | Effect of smoothing span, n = 5 000 |
| **Iterations** | 1 – 10 (5 levels) | Effect of robustness iterations on outlier data, n = 5 000 |
| **Financial** | n = 500 / 1 000 / 5 000 | Cumulative-return time series, fraction 0.1 |
| **Scientific** | n = 500 / 1 000 / 5 000 | Damped-oscillator signal, fraction 0.15 |
| **Genomic** | n = 1 000 / 5 000 / 100 000 | Step-function expression data, fraction 0.1 |
| **Pathological** | clustered, high-noise | Edge cases: clustered x-values and high-noise signal |
| **Large Scale** | n = 15 000 / 50 000 | Stress tests at scale: exact fit (`surface = "direct"`), interpolation shortcut, high iteration count, high fraction |

## Large Scale Benchmarks

Every other scenario above completes in well under 100ms, which doesn't stress-test performance differences at scale. The `large` category forces `surface = "direct"` (disabling `stats::loess`'s k-d tree interpolation shortcut, which approximates the fit surface at a fixed grid of vertices and interpolates between them) for an exact, apples-to-apples comparison, across four variants:

| Variant | Size | Description |
| --- | --- | --- |
| `large_direct` | 50 000 | Exact fit baseline: fraction 0.1, 3 iterations, `surface = "direct"` |
| `large_interp` | 50 000 | Same workload with the default `surface = "interpolate"`, showing the k-d tree shortcut's speedup |
| `large_high_iter` | 15 000 | 10 robustness iterations instead of 3, `family = "symmetric"` (still `surface = "direct"`) |
| `large_high_fraction` | 50 000 | Fraction 0.67 (wider local window), `surface = "interpolate"` |

Mean times from the latest comparison run, and fastLoess's speedup over `stats::loess`:

| Variant | `stats::loess` | fastLoess (serial) | fastLoess (parallel) | Speedup (serial) | Speedup (parallel) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `large_direct` | 7.86 s | 9.55 s | 3.27 s | 0.8× | 2.4× |
| `large_interp` | 1.21 s | 20.96 ms | 13.30 ms | 57.6× | 90.7× |
| `large_high_iter` | 6.75 s | 1.27 s | 0.92 s | 5.3× | 7.3× |
| `large_high_fraction` | 8.71 s | 14.22 ms | 15.82 ms | 612.5× | 550.6× |

`large_direct` remains the case where `stats::loess`'s `surface = "direct"` Fortran routine edges out fastLoess's serial build (0.8×); parallel execution is 2.4× faster than R. `large_interp` and `large_high_fraction` show the largest gains with interpolation enabled, reaching 90.7× and 612.5× respectively. For `large_high_fraction`, serial is faster than parallel in this run.

## Running

```sh
# Build and install rfastloess to system R (required before benchmarking)
make install

# Run benchmarks
make bench-r                    # stats::loess only
make bench-rfastloess-serial
make bench-rfastloess-parallel

# Generate comparison plot (output/benchmark_comparison.svg)
make compare
```

Output JSON files are written to `output/`.
