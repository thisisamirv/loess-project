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

:::note
The WebAssembly build runs single-threaded (no `parallel` option). The figures
above reflect the native Node.js binding with worker-thread parallelism. WASM
serial performance is comparable to the Serial column.
:::

---

## Reproducing Benchmarks

Use `performance.now()` to time serial WASM runs:

```javascript
const { Loess } = require('fastloess-wasm');

function benchMs(fn, reps = 10) {
    fn(); // warm-up
    const { performance } = require('perf_hooks');
    const t0 = performance.now();
    for (let i = 0; i < reps; i++) fn();
    return (performance.now() - t0) / reps;
}

const n = 5000;
const x = Float64Array.from({ length: n }, (_, i) => (i / (n - 1)) * 10);
const y = Float64Array.from(x, (xi, i) =>
    Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6
);

const ms = benchMs(() => new Loess({ fraction: 0.67 }).fit(x, y));
console.log(`WASM: ${ms.toFixed(2)} ms`);
```

```output
WASM: 3.73 ms
```
