---
title: Custom Weights
---
Per-observation weights that encode data quality directly into the LOESS fit.

## How Custom Weights Work

Standard LOESS assigns equal prior trust to all observations. Custom weights
let you override this assumption point by point — before any distance or
robustness weighting is applied.

The effective weight of observation $j$ in a local fit centred at $x_i$ is:

$$w_{ij} = \text{custom\_weights}[j] \times K\!\left(\frac{d_{ij}}{h_i}\right) \times r_j$$

where $K$ is the distance kernel, $h_i$ is the local bandwidth, and $r_j$ is
the robustness weight from the current iteration.

:::note[Batch adapter only]
`custom_weights` applies in **Batch** mode. It is silently ignored in
Streaming and Online adapters.
:::

---

## When to Use Custom Weights

| Situation | Recommended weight |
| --- | --- |
| Point known to be erroneous | `0.0` — fully excluded |
| Unreliable sensor / low precision | `0.1 – 0.5` |
| Standard observation | `1.0` (default) |
| Carefully calibrated measurement | `> 1.0` |
| Measurement uncertainty $\sigma_i$ | $1 / \sigma_i^2$ |

### Custom Weights vs. Robustness Iterations

Both mechanisms handle unreliable data, but they serve different purposes:

| | Custom Weights | Robustness Iterations |
| --- | --- | --- |
| **When known** | Before fitting | Computed from residuals |
| **Knowledge required** | Prior knowledge of quality | None — data-driven |
| **Effect** | Fixed throughout fit | Adapts each iteration |
| **Use case** | Known bad sensors, calibration | Unknown outlier contamination |

They compose: you can use both simultaneously. Custom weights suppress
*a priori* bad points; robustness iterations then handle any *residual*
outliers that remain.

---

## Basic Usage

### Suppress a Known Outlier

Set the weight to `0` at the bad point — it is excluded from every local fit
that would otherwise include it.

```javascript
const { Loess } = require('fastloess-wasm');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

const weights = new Array(y.length).fill(1);
weights[4] = 0; // Exclude 5th point
const model = new Loess({ custom_weights: weights });
const result = model.fit(x, y);
console.log("y[0]:", result.y[0].toFixed(4));
```

```output
y[0]: 0.1663
```

---

## Validation Rules

| Rule | Effect |
| --- | --- |
| Length must equal `n` | Error at fit time if mismatched |
| All values must be ≥ 0 | Negative weights are rejected |
| All-zero weight vector | Error: no points remain for any local fit |
| Uniform weights (`1.0` everywhere) | Identical result to omitting weights |

:::caution[Zero-weight windows]
If a local neighbourhood contains only zero-weight points, the fit at
that centre point falls back to the behaviour specified by
`zero_weight_fallback` (default: `"use_local_mean"`).
:::

---

## See Also

- [Robustness](robustness.md) — adaptive outlier downweighting via IRLS
- [Parameters](parameters.md#custom_weights) — full parameter reference
