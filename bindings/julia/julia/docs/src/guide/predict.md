# Out-of-Sample Prediction

Evaluate a fitted Batch model at query points that were not in the training set.

## Overview

Retained-model prediction is available in **Batch** mode. Online also provides `predict_window` for a fresh fit of its current bounded window; Streaming does not retain a complete training model for query-time prediction.

`predict(model::PredictModel, new_x; kwargs...)` evaluates the fit at arbitrary query points, like R's `predict(model, newdata)`. Query points are flattened, `dimensions` values per point.

It always fits exactly, unlike `fit`'s default `surface_mode="interpolation"` — so predicting at a training point may not exactly match `fit`'s output unless `surface_mode="direct"` was used.

`model` comes from `result.predict_model`, populated only when `retain_model=true` was passed to `Loess`.

---

## Options

| Keyword Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `outputs` | `Vector{String}` | `String[]` | Select `"se"` and/or `"gradient"` (alias `"derivative"`). |
| `intervals` | `NamedTuple` | `nothing` | Grouped confidence and prediction coverage levels. |
| `extrapolation` | `String` | `"clamp"` | Behavior for query points outside the training range, on any dimension |
| `max_extrapolation_distance` | `Union{Float64, Nothing}` | `nothing` | Under `"linear"` extrapolation, the max allowed per-dimension distance beyond the training boundary before erroring |
| `max_neighbor_distance` | `Union{Float64, Nothing}` | `nothing` | Max allowed distance to the farthest point in a query's k-nearest-neighbor window before erroring |

### outputs: se

Computes standard errors for each query point, using the retained model's residual scale and per-point leverage. Required for `intervals.confidence`/`intervals.prediction` to be populated. Omitted by default.

### intervals.confidence

Confidence level for the confidence interval around the mean response at each query point (e.g. `0.95`). Uses the same z-score convention as `fit`'s own confidence intervals. `nothing` (default) disables it.

### intervals.prediction

Confidence level for the prediction interval for a new observation at each query point (e.g. `0.95`). Widens using the same residual scale `fit` used for its own intervals when available, otherwise falling back to a MAD-based estimate. `nothing` (default) disables it.

### outputs: derivative

Includes the local fit's gradient (`dimensions` values per query point, flattened) in the output. Omitted by default.

### extrapolation

Behavior for query points outside the training range, on any dimension:

| Policy | Behavior |
| --- | --- |
| `"clamp"` (default) | Clamps each out-of-range dimension to the nearest training boundary |
| `"linear"` | Linearly extrapolates from the boundary point's local fit and gradient (first-order Taylor expansion) |
| `"error"` | Fails the whole call with an error |

### max_extrapolation_distance

Under `"linear"` extrapolation, the maximum allowed per-dimension distance beyond the training boundary before `predict` errors, instead of returning an unbounded value. `nothing` (default, uncapped).

### max_neighbor_distance

Maximum allowed distance to the farthest point in a query's k-nearest-neighbor window before `predict` errors. Guards against an "empty corner" blind spot: a query point can sit inside every dimension's min/max bounding box yet still be far from any real training data. `nothing` (default, uncapped); measured as a plain (raw-coordinate) Euclidean distance, independent of `distance_metric`.

## Example

### Basic Usage

```@example predict
using FastLOESS

x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [2.1, 4.0, 6.2, 8.0, 10.1]

model = Loess(; fraction=0.7, retain_model=true)
result = fit(model, x, y)

prediction = predict(result.predict_model, [1.5, 4.5])
println("Predicted y: ", prediction.y)
```

### Standard Errors and Derivative

```@example predict
prediction = predict(result.predict_model, [2.5]; outputs = ["se", "derivative"])
println("y: ", prediction.y)
println("SE: ", prediction.standard_errors)
println("Derivative: ", prediction.derivative)
```

### Linear Extrapolation

```@example predict
prediction = predict(result.predict_model, [10.0]; extrapolation="linear")
println("Extrapolated y: ", prediction.y)
```
