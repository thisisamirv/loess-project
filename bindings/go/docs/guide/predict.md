---
title: "Out-of-Sample Prediction"
weight: 40
---

Evaluate a fitted Batch model at query points that were not in the training set.

## Overview

> Out-of-sample prediction is available in **Batch** mode only. Streaming and Online modes do not support it.

`(*PredictModel) Predict(newX, opts)` evaluates the fit at arbitrary query points, like R's `predict(model, newdata)`. Query points are flattened, `Dimensions` values per point.

It always fits exactly, unlike `Fit`'s default `SurfaceMode = "interpolation"` — so predicting at a training point may not exactly match `Fit`'s output unless `SurfaceMode = "direct"` was used.

`Result.PredictModel` is non-nil only when `Options.RetainModel` was set to `true`. Call `Close()` on it when done (or let its finalizer run).

---

## Options

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `Outputs` | `[]string` | `nil` | Select `se` and/or `derivative`/`gradient`. |
| `Intervals` | `*IntervalsOptions` | `nil` | Grouped confidence and prediction coverage levels. |
| `Extrapolation` | `string` | `"clamp"` | Behavior for query points outside the training range, on any dimension |
| `MaxExtrapolationDistance` | `*float64` | `nil` | Under `"linear"` extrapolation, the max allowed per-dimension distance beyond the training boundary before erroring |
| `MaxNeighborDistance` | `*float64` | `nil` | Max allowed distance to the farthest point in a query's k-nearest-neighbor window before erroring |

`Outputs` is the only prediction output selector.

### outputs: se

Computes standard errors for each query point, using the retained model's residual scale and per-point leverage. Required for `Intervals.Confidence`/`Intervals.Prediction` to be populated. Omitted by default.

### intervals.confidence

Confidence level for the confidence interval around the mean response at each query point (e.g. `0.95`). Uses the same z-score convention as `Fit`'s own confidence intervals. `nil` (default) disables it.

### intervals.prediction

Confidence level for the prediction interval for a new observation at each query point (e.g. `0.95`). Widens using the same residual scale `Fit` used for its own intervals when available, otherwise falling back to a MAD-based estimate. `nil` (default) disables it.

### outputs: derivative

Includes the local fit's gradient (`Dimensions` values per query point, flattened) in the output. Omitted by default.

### Extrapolation

Behavior for query points outside the training range, on any dimension:

| Policy | Behavior |
| --- | --- |
| `"clamp"` (default) | Clamps each out-of-range dimension to the nearest training boundary |
| `"linear"` | Linearly extrapolates from the boundary point's local fit and gradient (first-order Taylor expansion) |
| `"error"` | Fails the whole call with an error |

### MaxExtrapolationDistance

Under `"linear"` extrapolation, the maximum allowed per-dimension distance beyond the training boundary before `Predict` errors, instead of returning an unbounded value. `nil` (default, uncapped).

### MaxNeighborDistance

Maximum allowed distance to the farthest point in a query's k-nearest-neighbor window before `Predict` errors. Guards against an "empty corner" blind spot: a query point can sit inside every dimension's min/max bounding box yet still be far from any real training data. `nil` (default, uncapped); measured as a plain (raw-coordinate) Euclidean distance, independent of `DistanceMetric`.

## Example

### Basic Usage

```go
opts := fastloess.DefaultOptions()
opts.Fraction = 0.7
opts.RetainModel = true

model, _ := fastloess.NewLoess(opts)
defer model.Close()

x := []float64{1, 2, 3, 4, 5}
y := []float64{2.1, 4.0, 6.2, 8.0, 10.1}
result, _ := model.Fit(x, y)
defer result.PredictModel.Close()

prediction, _ := result.PredictModel.Predict([]float64{1.5, 4.5}, fastloess.PredictOptions{})
fmt.Println(prediction.Y)
```

```output
[3.05 9.05]
```

### Standard Errors and Derivative

```go
prediction, _ := result.PredictModel.Predict([]float64{2.5}, fastloess.PredictOptions{
 Outputs: []string{"se", "derivative"},
})
fmt.Println(prediction.Y, prediction.StandardErrors, prediction.Derivative)
```

```output
[5.1] [0] [2.2]
```

### Linear Extrapolation

```go
prediction, _ := result.PredictModel.Predict([]float64{10.0}, fastloess.PredictOptions{
 Extrapolation: "linear",
})
fmt.Println(prediction.Y)
```

```output
[10.1]
```
