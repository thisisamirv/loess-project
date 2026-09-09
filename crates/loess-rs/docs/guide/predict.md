<!-- markdownlint-disable MD024 MD033 -->
# Out-of-Sample Prediction

Evaluate a fitted Batch model at query points that were not in the training set.

## Overview

!!! note "Adapter support"
    Out-of-sample prediction is available in **Batch** mode only. Streaming and Online modes do not support it.

`Predict::new()...build()?.call(&result, new_x)` evaluates the local polynomial fit at arbitrary query points, similar to R's `predict(model, newdata)`. It supports the full nD / polynomial-degree / distance-metric generality of the Batch adapter, reusing the same `RegressionContext` and `KDTree` neighbor search used during fitting. Query points are flattened, `dimensions` values per point — the same layout `fit()` uses for multivariate `x`.

It always fits an exact local regression at each query point, unlike `fit()` under the default `SurfaceMode::Interpolation` (which only fits exactly at a coarser vertex grid and interpolates the rest) — so predicting at a point already in the training set may not exactly reproduce that point's `fit()` output unless `.surface_mode("direct")` was used.

Enable it by calling `.retain_model(true)` on the builder before `fit()`; this retains the fitted model's (boundary-padded) training data, final robustness weights, residual SD, and normalization scales. Calling `.call(...)` without `.retain_model(true)` returns `LoessError::PredictionUnavailable`.

## Basic Usage

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

---

## Options

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `return_se` | `bool` | `false` | Include standard errors in the output |
| `confidence_intervals` | `Option<T>` | `None` | Confidence interval coverage level (e.g. `Some(0.95)`) |
| `prediction_intervals` | `Option<T>` | `None` | Prediction interval coverage level (e.g. `Some(0.95)`) |
| `return_derivative` | `bool` | `false` | Include the local fit's gradient (`dimensions` values per point, flattened) |
| `extrapolation` | `&str` | `"clamp"` | Behavior for query points outside the training range, on any dimension |
| `max_extrapolation_distance` | `T` | none | Under `"linear"` extrapolation, the max allowed per-dimension distance beyond the training boundary before erroring |
| `max_neighbor_distance` | `T` | none | Max allowed distance to the farthest point in a query's k-nearest-neighbor window before erroring |

`Predict` is configured the same way as the `Loess` builder itself: `Predict::new()` (or `::default()`), chained setter methods, and a mandatory `.build()?` to validate (e.g. an invalid `.extrapolation("bogus")` string) and obtain the ready-to-call configuration. Standard errors/intervals use the same z-score convention as `fit()`'s existing intervals, using per-point leverage scaled by the global MAD-based residual SD.

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

---

## Extrapolation Policies

Behavior for query points outside the training range on any dimension:

| Policy | Behavior |
| --- | --- |
| `"clamp"` (default) | Clamps each out-of-range dimension to the nearest training boundary |
| `"linear"` | Linearly extrapolates from the boundary point's local fit and gradient (first-order Taylor expansion) |
| `"error"` | Fails the whole call with `LoessError::PredictOutOfRange` |

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

Under `"linear"`, `.max_extrapolation_distance(...)` caps how far beyond the boundary (per dimension) the extrapolation may extend before `.call(...)` fails with `LoessError::ExtrapolationTooFar`, instead of returning an unbounded value.

`.max_neighbor_distance(...)` guards a separate blind spot: a query point can sit inside every dimension's min/max range yet fall in an empty region far from any real training data (e.g. an empty "corner" of non-rectangularly distributed data). Setting it makes `.call(...)` fail with `LoessError::SparseNeighborhood` instead of silently extrapolating there. It applies regardless of `extrapolation`, and is measured as a plain (raw-coordinate) Euclidean distance, independent of `distance_metric`.

---

## Availability

!!! warning "Batch Mode Only"
    Out-of-sample prediction is only available in **Batch** mode. Streaming and Online modes do not support it.

| Feature | Batch | Streaming | Online |
| --- | --- | --- | --- |
| `retain_model` | ✓ | ✗ | ✗ |
| `Predict::call()` | ✓ | ✗ | ✗ |
