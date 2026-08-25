---
title: Benchmarks
---
<!-- markdownlint-disable MD024 MD046 -->

## CPU Benchmarks

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

---

## Reproducing Benchmarks

Use Node.js `perf_hooks` to time serial vs parallel runs:

```javascript
const { performance } = require('perf_hooks');
const { Loess } = require('fastloess');

function benchMs(fn, reps = 10) {
    fn(); // warm-up
    const t0 = performance.now();
    for (let i = 0; i < reps; i++) fn();
    return (performance.now() - t0) / reps;
}

const n = 5000;
const x = Float64Array.from({ length: n }, (_, i) => (i / (n - 1)) * 10);
const y = Float64Array.from(x, (xi, i) =>
    Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6
);

const serialMs   = benchMs(() => new Loess({ fraction: 0.67, parallel: false }).fit(x, y));
const parallelMs = benchMs(() => new Loess({ fraction: 0.67, parallel: true  }).fit(x, y));

console.log(`Serial:   ${serialMs.toFixed(2)} ms`);
console.log(`Parallel: ${parallelMs.toFixed(2)} ms`);
console.log(`Speedup:  ${(serialMs / parallelMs).toFixed(2)}×`);
```

```output
Serial:   2.18 ms
Parallel: 2.96 ms
Speedup:  0.74×
```
