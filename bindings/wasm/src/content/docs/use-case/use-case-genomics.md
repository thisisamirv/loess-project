---
title: Genomic Data Smoothing
---
<!-- markdownlint-disable MD033 -->
LOESS for methylation profiles, ChIP-seq signals, and other genomic data.

## Overview

Genomic data often contains noise from sequencing depth variation, PCR artifacts, or biological heterogeneity. LOESS smoothing helps reveal underlying patterns.

---

## Methylation Profile Smoothing

### The Challenge

DNA methylation data (from bisulfite sequencing or arrays) shows position-dependent patterns that can be obscured by measurement noise.

### Solution

A small `fraction = 0.1` lets LOESS follow fine-scale spatial structure without smearing the transitions between methylated and unmethylated regions. `confidence_intervals = 0.95` produces uncertainty bands that naturally widen at positions with sparser CpG coverage, making low-confidence segments immediately apparent in the plot.

```javascript
const { Loess } = require('fastloess-wasm');

const n = 100;
const positions = Float64Array.from({ length: n }, (_, i) => i * 100.0);
const observed = Float64Array.from(positions, p => 50 + Math.sin(p / 100) * 20 + ((p * 7 % 17) / 17 - 0.5) * 5);

// positions and observed are your methylation data (Float64Array)
const model = new Loess({
    fraction: 0.1,
    iterations: 3,
    confidence_intervals: 0.95
});
const result = model.fit(positions, observed);
console.log("CI lower[0]:", result.confidence_lower[0].toFixed(4));
```

```output
CI lower[0]: 36.2283
```

---

## ChIP-seq Signal Smoothing

### Application

ChIP-seq experiments produce sparse, noisy coverage data. LOESS can help identify binding regions.

`fraction = 0.05` provides high spatial resolution — important for resolving narrow binding peaks that would otherwise be smeared into the background. The larger `iterations = 5` is deliberate: Poisson-distributed read counts produce tall, isolated spikes, and extra robustness iterations progressively down-weight them so the estimated background level is not inflated by a handful of extreme counts.

```javascript
const { Loess } = require('fastloess-wasm');

const n = 100;
const positions = Float64Array.from({ length: n }, (_, i) => i * 100.0);
const observed = Float64Array.from(positions, p => 50 + Math.sin(p / 100) * 20 + ((p * 7 % 17) / 17 - 0.5) * 5);

const model = new Loess({
    fraction: 0.05,
    iterations: 5
});
const result = model.fit(positions, observed);

// Find peaks
const smoothed = result.y;
const peaks = positions.filter((p, i) => smoothed[i] > 25.0);
console.log("Peak count:", peaks.length);
```

```output
Peak count: 100
```

---

## Large Genome Coverage (Streaming)

For whole-genome data that doesn't fit in memory:

```javascript
const { StreamingLoess } = require('fastloess-wasm');

const xChunk = Float64Array.from({ length: 1001 }, (_, i) => i * 10.0);
const yChunk = Float64Array.from(xChunk, p => 50 + Math.sin(p / 100) * 20 + 5.0);

const processor = new StreamingLoess(
    { fraction: 0.05, iterations: 3 },
    { chunk_size: 100, overlap: 10 }
);

processor.process_chunk(xChunk, yChunk);
const result = processor.finalize();
console.log("y[0]:", result.y[0].toFixed(4));
```

```output
y[0]: 41.5626
```

---

## Best Practices for Genomic Data

| Consideration | Recommendation |
| --- | --- |
| **Fraction** | 0.05–0.15 (preserve local features) |
| **Iterations** | 3–5 (handle sequencing outliers) |
| **Large data** | Use streaming mode |
| **Sparse regions** | Use `boundary_policy="extend"` |
| **Multiple chromosomes** | Process separately or ensure sorted |

---

## See Also

- [Concepts](../introduction/concepts.md) — How LOESS works
- [API Reference](../api/api.md) — All options
- [Robustness](../weighting/robustness.md) — Outlier downweighting in depth
- [Merge Strategies](../advanced/merge.md) — Streaming chunk reconciliation
- [Boundary Handling](../advanced/boundary.md) — Edge handling for sparse regions
- [Real-Time Processing](use-case-real-time.md) — For sequencing runs
