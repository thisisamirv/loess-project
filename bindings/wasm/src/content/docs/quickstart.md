---
title: Quick Start
---
<!-- markdownlint-disable MD024 MD046 -->
Get up and running with LOESS in minutes.

## Basic Smoothing

Smooth a noisy sine wave — the kind of signal where LOESS shines. Each example recovers the underlying trend from 100 points of Gaussian noise.

```javascript
const { Loess } = require('fastloess-wasm');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

const model = new Loess({ fraction: 0.3, iterations: 3 });
const result = model.fit(x, y);

console.log(`First smoothed: ${result.y[0].toFixed(4)}`);
```

```output
First smoothed: 0.0281
```

---

## With Confidence Intervals

```javascript
const { Loess } = require('fastloess-wasm');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

const model = new Loess({
    fraction: 0.5,
    iterations: 3,
    confidence_intervals: 0.95,
    prediction_intervals: 0.95,
    return_diagnostics: true
});
const result = model.fit(x, y);

console.log("Smoothed (first 5):", [...result.y.slice(0, 5)].map(v => v.toFixed(4)));
console.log("CI lower (first 5):", [...result.confidence_lower.slice(0, 5)].map(v => v.toFixed(4)));
console.log("CI upper (first 5):", [...result.confidence_upper.slice(0, 5)].map(v => v.toFixed(4)));
console.log("R²:", result.diagnostics.r_squared.toFixed(4));
```

```output
Smoothed (first 5): [ '0.1118', '0.1389', '0.1702', '0.2064', '0.2445' ]
CI lower (first 5): [ '-0.0078', '0.0201', '0.0521', '0.0890', '0.1279' ]
CI upper (first 5): [ '0.2313', '0.2577', '0.2882', '0.3237', '0.3611' ]
R²: 0.8860
```

---

## Handling Outliers

LOESS can robustly handle outliers through iterative reweighting:

```javascript
const { Loess } = require('fastloess-wasm');

// Data with an outlier at position 3
const x = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
const yWithOutlier = new Float64Array([2.0, 4.0, 6.0, 50.0, 10.0, 12.0]);

const model = new Loess({
    fraction: 0.5,
    iterations: 5,
    robustness_method: "bisquare",
    return_robustness_weights: true
});
const result = model.fit(x, yWithOutlier);

// Outliers will have low robustness weights
result.robustness_weights.forEach((w, i) => {
    if (w < 0.5) {
        console.log(`Point ${i} is likely an outlier (weight: ${w.toFixed(3)})`);
    }
});
```

---

## Streaming Mode

For datasets too large to fit in memory, stream them in fixed-size chunks with overlap.

```javascript
const { StreamingLoess } = require('fastloess-wasm');

const n = 5000;
const x = Float64Array.from({ length: n }, (_, i) => i * 10 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) =>
    Math.sin(xi / Math.PI) * Math.exp(-xi / 30) +
    (((i * 7 + 3) % 17) / 17 - 0.5) * 0.3
);

const model = new StreamingLoess(
    { fraction: 0.2 },
    { chunk_size: 1000, overlap: 100, merge_strategy: 'weighted_average' }
);

const chunk_size = 1000;
for (let start = 0; start <= 4000; start += chunk_size) {
    const end = Math.min(start + chunk_size, n);
    model.process_chunk(x.slice(start, end), y.slice(start, end));
}
const result = model.finalize();
console.log(`Smoothed ${result.y.length} points`);
```

```output
Smoothed 100 points
```

---

## Next Steps

| Topic | Link |
| --- | --- |
| How LOESS works | [Concepts](concepts.md) |
| All parameters explained | [API Reference](api.md) |
| Batch vs Streaming vs Online | [Execution Modes](adapter-choice.md) |
| Polynomial degree choices | [Degree](degree.md) |
| Multivariate smoothing | [Dimensions](dimensions.md) |
| Edge handling | [Boundary](boundary.md) |
| Outlier handling in depth | [Robustness](robustness.md) |
| Full API per language | [API Reference](api.md) |
