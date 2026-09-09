<!-- markdownlint-disable MD024 MD033 -->
# Out-of-Sample Prediction

Evaluate a fitted Batch model at query points that were not in the training set.

## Overview

!!! note "Adapter support"
    Out-of-sample prediction is available in **Batch** mode only. Streaming and Online modes do not support it.

`Predict::new()...build()?.call(&result, new_x)` evaluates the fit at arbitrary query points, like R's `predict(model, newdata)`. Query points are flattened, `dimensions` values per point.

It reuses `fit()`'s own interpolation surface (when built under the default `SurfaceMode::Interpolation`) for in-range query points — so predicting at a training point always exactly matches that point's `fit()` output, regardless of `surface_mode()`. A fresh local regression is only run when `return_derivative`, an out-of-range extrapolation, or `max_neighbor_distance` needs the actual gradient/leverage at the query point.

Requires `.retain_model(true)` on the builder before `fit()`, otherwise `.call(...)` returns `LoessError::PredictionUnavailable`.

---

## Builder Configuration

| Method | Argument Type | Default | Description |
| --- | --- | --- | --- |
| `return_se()` | `bool` | `false` | Include standard errors in the output |
| `confidence_intervals(T)` | `T: Float` | disabled | Confidence interval coverage level (e.g. `0.95`) |
| `prediction_intervals(T)` | `T: Float` | disabled | Prediction interval coverage level (e.g. `0.95`) |
| `return_derivative()` | `bool` | `false` | Include the local fit's gradient (`dimensions` values per point, flattened) |
| `extrapolation(...)` | `&str` | `"clamp"` | Behavior for query points outside the training range, on any dimension |
| `max_extrapolation_distance(T)` | `T: Float` | disabled | Under `"linear"` extrapolation, the max allowed per-dimension distance beyond the training boundary before erroring |
| `max_neighbor_distance(T)` | `T: Float` | disabled | Max allowed distance to the farthest point in a query's k-nearest-neighbor window before erroring |

## Options

### return_se

Computes standard errors for each query point, using the retained model's residual scale and per-point leverage. Required for `confidence_intervals`/`prediction_intervals` to be populated. `false` by default.

### confidence_intervals

Confidence level for the confidence interval around the mean response at each query point (e.g. `0.95`). Uses the same z-score convention as `fit()`'s own confidence intervals. Disabled by default.

### prediction_intervals

Confidence level for the prediction interval for a new observation at each query point (e.g. `0.95`). Widens using the same residual scale `fit()` used for its own intervals (`sqrt(RSS / delta1)`, populated when `.return_se()` plus an interval method was set under `.surface_mode("direct")`), otherwise falling back to a MAD-based estimate. Disabled by default.

### return_derivative

Includes the local fit's gradient (`dimensions` values per query point, flattened) in the output. `false` by default.

### extrapolation

Behavior for query points outside the training range, on any dimension:

| Policy | Behavior |
| --- | --- |
| `"clamp"` (default) | Clamps each out-of-range dimension to the nearest training boundary |
| `"linear"` | Linearly extrapolates from the boundary point's local fit and gradient (first-order Taylor expansion) |
| `"error"` | Fails the whole call with `LoessError::PredictOutOfRange` |

### max_extrapolation_distance

Under `"linear"` extrapolation, the maximum allowed per-dimension distance beyond the training boundary before `.call(...)` errors with `LoessError::ExtrapolationTooFar`, instead of returning an unbounded value. Disabled by default (uncapped).

### max_neighbor_distance

Maximum allowed distance to the farthest point in a query's k-nearest-neighbor window before `.call(...)` errors with `LoessError::SparseNeighborhood`. Guards against an "empty corner" blind spot: a query point can sit inside every dimension's min/max bounding box yet still be far from any real training data. Disabled by default (uncapped); measured as a plain (raw-coordinate) Euclidean distance, independent of `distance_metric`.

## Example

### Basic Usage

```rust
use loess_rs::prelude::*;

fn main() -> Result<(), LoessError> {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.1_f64, 4.0, 6.2, 8.0, 10.1];

    let model = Loess::new().fraction(0.7).retain_model(true).build()?;
    let result = model.fit(&x, &y)?;

    let new_x = vec![1.5_f64, 4.5];
    let prediction = Predict::new().build()?.call(&result, &new_x)?;
    println!("Predicted y: {:?}", prediction.y);

    Ok(())
}
```

```output
Predicted y: [3.05, 9.05]
```

### Standard Errors and Derivative

```rust
use loess_rs::prelude::*;

fn main() -> Result<(), LoessError> {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.1_f64, 4.0, 6.2, 8.0, 10.1];

    let model = Loess::new().fraction(0.7).retain_model(true).build()?;
    let result = model.fit(&x, &y)?;

    let options = Predict::new().return_se().return_derivative().build()?;
    let prediction = options.call(&result, &[2.5_f64])?;

    println!("y: {:?}", prediction.y);
    println!("SE: {:?}", prediction.standard_errors);
    println!("Derivative: {:?}", prediction.derivative);

    Ok(())
}
```

```output
y: [5.1]
SE: Some([0.0])
Derivative: Some([2.2])
```

### Linear Extrapolation

```rust
use loess_rs::prelude::*;

fn main() -> Result<(), LoessError> {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.1_f64, 4.0, 6.2, 8.0, 10.1];

    let model = Loess::new().fraction(0.7).retain_model(true).build()?;
    let result = model.fit(&x, &y)?;

    let options = Predict::new().extrapolation("linear").build()?;
    let prediction = options.call(&result, &[10.0_f64])?;
    println!("Extrapolated y: {:?}", prediction.y);

    Ok(())
}
```

```output
Extrapolated y: [10.1]
```
