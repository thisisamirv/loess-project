<!-- markdownlint-disable MD033 -->
# Custom Weights

Per-observation weights that encode data quality directly into the LOESS fit.

## How Custom Weights Work

Standard LOESS assigns equal prior trust to all observations. Custom weights
let you override this assumption point by point — before any distance or
robustness weighting is applied.

The effective weight of observation $j$ in a local fit centred at $x_i$ is:

$$w_{ij} = \text{custom\_weights}_j \times K\!\left(\frac{d_{ij}}{h_i}\right) \times r_j$$

where $K$ is the distance kernel, $h_i$ is the local bandwidth, and $r_j$ is
the robustness weight from the current iteration.

!!! note "Batch adapter only"
    `custom_weights` applies in **Batch** mode. It is silently ignored in
    Streaming and Online adapters.

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

```rust
use fastLoess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let mut weights = vec![1.0f64; y.len()];
    weights[4] = 0.0; // Exclude 5th point
    let model = Loess::new()
        .custom_weights(weights)
        .build()?;
    let result = model.fit(&x, &y)?;

    println!("First smoothed value (custom weights): {}", result.y[0]);
    Ok(())
}
```

```output
First smoothed value (custom weights): 0.38960937887234354
```

---

## Validation Rules

| Rule | Effect |
| --- | --- |
| Length must equal `n` | Error at fit time if mismatched |
| All values must be ≥ 0 | Negative weights are rejected |
| All-zero weight vector | Error: no points remain for any local fit |
| Uniform weights (`1.0` everywhere) | Identical result to omitting weights |

!!! warning "Zero-weight windows"
    If a local neighbourhood contains only zero-weight points, the fit at
    that centre point falls back to the behaviour specified by
    `zero_weight_fallback` (default: `"use_local_mean"`).

---

## See Also

- [Robustness](crate::doc::robustness) — adaptive outlier downweighting via IRLS
- [API Reference](crate::doc::api) — full parameter reference
