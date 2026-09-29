---
title: "Benchmarks"
weight: 90
---

The Go binding calls the shared `fastLoess` Rust core through `cgo`. The figures below are reference measurements of that core from the repository's R comparison harness; they do not measure Go FFI overhead or Go-specific workloads.

## CPU Benchmarks

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

## Large-Scale Timings

Mean elapsed times from the latest comparison run:

| Variant | `stats::loess` | Serial core | Parallel core | Speedup (serial) | Speedup (parallel) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `large_direct` | 7.86 s | 9.55 s | 3.27 s | 0.8× | 2.4× |
| `large_interp` | 1.21 s | 20.96 ms | 13.30 ms | 57.6× | 90.7× |
| `large_high_iter` | 6.75 s | 1.27 s | 0.92 s | 5.3× | 7.3× |
| `large_high_fraction` | 8.71 s | 14.22 ms | 15.82 ms | 612.5× | 550.6× |

Parallel execution is not faster for every workload: the serial core is faster in the `large_high_fraction` run. A dedicated Go `testing.B` benchmark can measure end-to-end Go overhead and compare Go-specific option configurations with `go test -bench=.`.
