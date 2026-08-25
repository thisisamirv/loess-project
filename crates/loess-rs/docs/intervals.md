<!-- markdownlint-disable MD024 MD033 -->
# Intervals

Confidence and prediction intervals for uncertainty quantification.

## Overview

![Confidence and Prediction Intervals](https://raw.githubusercontent.com/thisisamirv/loess-project/main/crates/loess-rs/assets/diagrams/intervals_comparison.svg)

!!! note "Adapter support"
    Confidence and prediction intervals are available in **all three adapters**: Batch, Streaming, and Online.

| Type | Represents | Width | Use |
| --- | --- | --- | --- |
| **Confidence** | Uncertainty in mean curve | Narrow | Where is the true trend? |
| **Prediction** | Uncertainty for new points | Wide | Where will new data fall? |

---

## Confidence Intervals

Estimate uncertainty in the smoothed curve itself.

```rust
use loess_rs::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Loess::new()
        .fraction(0.5)
        .confidence_intervals(0.95)  // 95% CI
        .build()?;

    let result = model.fit(&x, &y)?;

    // Access intervals
    if let (Some(lower), Some(upper)) = (&result.confidence_lower, &result.confidence_upper) {
        for i in 0..3 {
            println!("x={:.2}: y={:.2} [{:.2}, {:.2}]",
                result.x[i], result.y[i], lower[i], upper[i]);
        }
    }

    Ok(())
}
```

```output
x=0.00: y=0.33 [0.30, 0.35]
x=0.06: y=0.35 [0.33, 0.38]
x=0.13: y=0.38 [0.35, 0.40]
```

---

## Prediction Intervals

Estimate where new observations might fall.

```rust
use loess_rs::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Loess::new()
        .fraction(0.5)
        .prediction_intervals(0.95)  // 95% PI
        .build()?;

    let result = model.fit(&x, &y)?;

    if let (Some(lower), Some(upper)) = (&result.prediction_lower, &result.prediction_upper) {
        println!("Prediction bounds: [{:.2}, {:.2}]", lower[0], upper[0]);
    }

    Ok(())
}
```

```output
Prediction bounds: [-0.03, 0.69]
```

---

## Both Intervals

Request both types simultaneously:

```rust
use loess_rs::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Loess::new()
        .fraction(0.5)
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .build()?;
    let result = model.fit(&x, &y)?;

    if let (Some(lo), Some(hi)) = (&result.confidence_lower, &result.confidence_upper) {
        println!("First point 95% CI: [{}, {}]", lo[0], hi[0]);
    }
    Ok(())
}
```

```output
First point 95% CI: [0.3017142421484391, 0.3530368379935054]
```

---

## Confidence Levels

Common levels and their z-values:

| Level | z-value | Interpretation |
| --- | --- | --- |
| 0.90 | 1.645 | 90% of intervals contain true value |
| 0.95 | 1.960 | 95% of intervals contain true value |
| 0.99 | 2.576 | 99% of intervals contain true value |

```rust
use loess_rs::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    // 99% confidence interval
    let model = Loess::new()
        .confidence_intervals(0.99)
        .build()?;
    let result = model.fit(&x, &y)?;

    if let Some(lo) = &result.confidence_lower {
        println!("First lower CI bound (99%): {}", lo[0]);
    }
    Ok(())
}
```

```output
First lower CI bound (99%): 0.32028604658278614
```

---

## Standard Errors

Access standard errors directly (available when intervals are computed):

```rust
use loess_rs::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Loess::new()
        .confidence_intervals(0.95)
        .build()?;
    let result = model.fit(&x, &y)?;

    if let Some(se) = &result.standard_errors {
        for (i, &se_val) in se.iter().enumerate().take(3) {
            println!("Point {}: SE = {:.4}", i, se_val);
        }
    }

    Ok(())
}
```

```output
Point 0: SE = 0.0252
Point 1: SE = 0.0252
Point 2: SE = 0.0252
```

---

## Availability

!!! warning "Batch Mode Only"
    Confidence and prediction intervals are only available in **Batch** mode. Streaming and Online modes do not support intervals.

| Feature | Batch | Streaming | Online |
| --- | --- | --- | --- |
| Confidence intervals | ✓ | ✗ | ✗ |
| Prediction intervals | ✓ | ✗ | ✗ |
| Standard errors | ✓ | ✗ | ✗ |
