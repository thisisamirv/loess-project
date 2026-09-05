<!-- markdownlint-disable MD024 MD033 -->
# Multivariate LOESS

Smoothing over multiple predictor dimensions simultaneously.

## Overview

Standard LOESS operates on a single predictor $x$. Setting `dimensions` `> 1` extends the neighbourhood search and local polynomial fit into an $n$-dimensional predictor space, enabling surface smoothing over spatial grids, time–altitude combinations, and similar multi-predictor datasets. `x` is passed as a flat, row-major slice of length `y.len() * dimensions`.

![Multivariate LOESS](https://raw.githubusercontent.com/thisisamirv/loess-project/main/crates/loess-rs/assets/diagrams/multivariate_loess.svg)

| Dimensions | Use Case | Input Shape |
| --- | --- | --- |
| `1` | Time series, 1D signal (default) | `x`: 1-D array |
| `2` | Spatial surface, 2-predictor model | `x`: flat array of length `n*2`, row-major |
| `3+` | High-dimensional regression | `x`: flat array of length `n*d`, row-major |

!!! warning "Computational cost"
    Neighbourhood search scales with $d$ dimensions. For `dimensions ≥ 3` keep `fraction` small.

---

## 1D — Standard (Default)

Single predictor. No configuration required.

```rust
use loess_rs::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Loess::new()
        .fraction(0.3)
        .build()?;
    let result = model.fit(&x, &y)?;

    println!("First smoothed value (fraction=0.3): {}", result.y[0]);
    Ok(())
}
```

```output
First smoothed value (fraction=0.3): 0.24731032286452273
```

---

## 2D — Spatial Surface

Two predictors (e.g., latitude/longitude, time/altitude). Pass a flat, row-major array (slice) of length `n*2` as `x`.

```rust
use loess_rs::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let lat: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let lon: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let z: Vec<f64> = (0..n).map(|i| lat[i].sin() + lon[i].cos() + 0.05).collect();
    let x2d: Vec<f64> = (0..n).flat_map(|i| [lat[i], lon[i]]).collect();

    let model = Loess::new()
        .dimensions(2)
        .fraction(0.3)
        .build()?;
    let result = model.fit(&x2d, &z)?;

    println!("First smoothed value (2D): {}", result.y[0]);
    Ok(())
}
```

```output
First smoothed value (2D): 1.1932872258359153
```

---

## 3D and Higher

Three or more predictors. The neighbourhood radius grows in each additional dimension, so a larger `fraction` (or smaller dataset) is typically needed.

```rust
use loess_rs::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x1: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let x2: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let x3: Vec<f64> = (0..n).map(|i| 1.0 - i as f64 / (n - 1) as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| x1[i].sin() + x2[i] - x3[i] + 0.05).collect();
    let x3d: Vec<f64> = (0..n).flat_map(|i| [x1[i], x2[i], x3[i]]).collect();

    let model = Loess::new()
        .dimensions(3)
        .fraction(0.5)
        .build()?;
    let result = model.fit(&x3d, &y)?;

    println!("First smoothed value (3D): {}", result.y[0]);
    Ok(())
}
```

```output
First smoothed value (3D): -0.6707908018247327
```

---

## Distance Metrics for Multivariate Data

When `dimensions > 1` you can also control how inter-point distances are computed.

| Metric | Description | When to Use |
| --- | --- | --- |
| `"normalized"` | Each dimension scaled to unit range (default) | Predictors on different scales |
| `"euclidean"` | Raw Euclidean distance | Predictors already on same scale |
| `"minkowski:p"` | Generalised Minkowski ($L_p$) norm | Custom distance geometry |
| `"weighted"` | Per-dimension weighted Euclidean | Domain-specific importance |

See [API Reference](../api/api.md#distance_metric) for the full list of options per language.
