<!-- markdownlint-disable MD024 MD033 -->
# Intervals

Confidence and prediction intervals for uncertainty quantification.

## Overview

![Confidence and Prediction Intervals](https://raw.githubusercontent.com/thisisamirv/loess-project/main/crates/loess-rs/assets/diagrams/intervals_comparison.svg)

!!! note "Adapter support"
    Confidence and prediction intervals are available in Batch, Streaming, full-update Online, and retained-model prediction. Online intervals and bootstrap require `update_mode("full")`.

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
        .intervals(IntervalsBuilder::new().confidence(0.95))  // 95% CI
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
        .intervals(IntervalsBuilder::new().prediction(0.95))  // 95% PI
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
        .intervals(IntervalsBuilder::new().confidence(0.95).prediction(0.95))
        .build()?;
    let result = model.fit(&x, &y)?;

    if let (Some(lo), Some(hi)) = (&result.confidence_lower, &result.confidence_upper) {
        println!("First point 95% CI: [{}, {}]", lo[0], hi[0]);
    }
    Ok(())
}
```

```output
First point 95% CI: [0.301714242148439, 0.35303683799350516]
```

---

## Residual Bootstrap

Add `.intervals(IntervalsBuilder::new().bootstrap(n_boot))` to replace analytic standard errors and bounds with residual-bootstrap estimates. At least two replicates are required. `.seed(seed)` makes sampling reproducible; omitting it uses a fixed default seed. The seed also configures Batch cross-validation. Without bootstrap, existing analytic interval behavior is unchanged.

Each replicate samples centered training residuals with replacement, adds them to the fitted response, and refits using the same degree, dimensions, metric, case weights, robustness settings, and selected fraction. Refits are processed in batches of at most 256. Standard errors are sample standard deviations; bounds use linearly interpolated percentiles. Prediction bounds include a fresh residual draw, not a normal-error assumption.

```rust
use loess_rs::prelude::*;

fn main() -> Result<(), LoessError> {
    let x: Vec<f64> = (0..30).map(|index| index as f64 / 29.0).collect();
    let y: Vec<f64> = x.iter().enumerate()
        .map(|(index, value)| value.sin() + (index % 3) as f64 * 0.05)
        .collect();
    let result = Loess::new()
        .fraction(0.5)
        .intervals(IntervalsBuilder::new().confidence(0.95).prediction(0.95).bootstrap(64))
        .seed(42)
        .retain_model(true)
        .build()?
        .fit(&x, &y)?;
    let prediction = Predict::new()
        .intervals(IntervalsBuilder::new().confidence(0.90).prediction(0.95).bootstrap(64))
        .seed(42)
        .build()?
        .call(&result, &[0.25, 0.75])?;
    assert_eq!(prediction.standard_errors.as_ref().unwrap().len(), 2);
    Ok(())
}
```

Bootstrap alone returns standard errors without interval bounds. Streaming resamples each combined overlap-and-chunk window before applying its usual merge policy. Full-update Online resamples each sliding window and returns the latest point. Prediction refits the original observations before evaluating query points, preserving interpolation and extrapolation settings. Retaining a model also retains its bootstrap refit context; sampling is performed only when requested.

`IntervalsBuilder` and the outer `.seed(...)` are also available through `fastLoess`. Language-binding option structs remain unchanged.

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
        .intervals(IntervalsBuilder::new().confidence(0.99))
        .build()?;
    let result = model.fit(&x, &y)?;

    if let Some(lo) = &result.confidence_lower {
        println!("First lower CI bound (99%): {}", lo[0]);
    }
    Ok(())
}
```

```output
First lower CI bound (99%): 0.3202860465827859
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
        .intervals(IntervalsBuilder::new().confidence(0.95))
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
    Confidence and prediction intervals are available in Batch and Streaming. Online computes them only in `UpdateMode::Full`; incremental Online updates do not compute standard errors.

| Feature | Batch | Streaming | Online |
| --- | --- | --- | --- |
| Confidence intervals | ✓ | ✓ (per chunk, overlap-merged) | ✓ (`Full` updates only) |
| Prediction intervals | ✓ | ✓ (per chunk, overlap-merged) | ✓ (`Full` updates only) |
| Standard errors | ✓ | ✓ | ✓ (`Full` updates only) |
