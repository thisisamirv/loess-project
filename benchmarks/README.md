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

Median times, and fastLoess's speedup over `stats::loess`:

| Variant | `stats::loess` | fastLoess (serial) | fastLoess (parallel) | Speedup (serial) | Speedup (parallel) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `large_direct` | 9.27 s | 10.1 s | 3.33 s | 0.9× | 2.8× |
| `large_interp` | 1.40 s | 23.3 ms | 16.7 ms | 60× | 84× |
| `large_high_iter` | 8.46 s | 1.35 s | 0.90 s | 6.3× | 9.4× |
| `large_high_fraction` | 9.99 s | 17.1 ms | 14.8 ms | 585× | 676× |

`large_direct` is the one case where `stats::loess`'s `surface = "direct"` Fortran routine is fast enough to edge out fastLoess's serial build (0.9×); parallel execution is needed to regain a lead (2.8×). `large_interp` and `large_high_fraction` show the largest gains once both implementations' k-d tree interpolation shortcuts are active, reaching 84× and 676× respectively.

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
