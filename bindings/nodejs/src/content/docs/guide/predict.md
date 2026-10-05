---
title: fastLoess Node.js Out-of-Sample Prediction
---
Evaluate a fitted Batch model at query points that were not in the training set.

## Overview

> Retained-model prediction is available in **Batch** mode. Online also provides `predict_window()` for a fresh fit of its current bounded window; Streaming does not retain a complete training model for query-time prediction.

`result.predict(newX, options)` evaluates the fit at arbitrary query points, like R's `predict(model, newdata)`. Query points are flattened, `dimensions` values per point.

It always fits exactly, unlike `fit()`'s default `surface_mode: "interpolation"` — so predicting at a training point may not exactly match `fit()`'s output unless `surface_mode: "direct"` was used.

Requires `retain_model: true` on the constructor before `fit()`, otherwise `predict()` throws.

---

## Options

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `outputs` | `string[]` | `[]` | Optional fields: `"se"`, `"gradient"` (or `"derivative"`) |
| `intervals` | `{ confidence?: number; prediction?: number }` | `disabled` | Grouped confidence and prediction coverage levels. |
| `extrapolation` | `string` | `"clamp"` | Behavior for query points outside the training range, on any dimension |
| `max_extrapolation_distance` | `number` | disabled | Under `"linear"` extrapolation, the max allowed per-dimension distance beyond the training boundary before erroring |
| `max_neighbor_distance` | `number` | disabled | Max allowed distance to the farthest point in a query's k-nearest-neighbor window before erroring |

### outputs

Select `"se"` to compute standard errors for each query point using the retained model's residual scale and per-point leverage. Interval levels also enable standard errors. Select `"gradient"` (or `"derivative"`) to include the local fit's gradient (`dimensions` values per query point, flattened). Optional outputs are omitted by default.

### intervals.confidence

Confidence level for the confidence interval around the mean response at each query point (e.g. `0.95`). Uses the same z-score convention as `fit()`'s own confidence intervals. Disabled by default.

### intervals.prediction

Confidence level for the prediction interval for a new observation at each query point (e.g. `0.95`). Widens using the same residual scale `fit()` used for its own intervals when available, otherwise falling back to a MAD-based estimate. Disabled by default.

### extrapolation

Behavior for query points outside the training range, on any dimension:

| Policy | Behavior |
| --- | --- |
| `"clamp"` (default) | Clamps each out-of-range dimension to the nearest training boundary |
| `"linear"` | Linearly extrapolates from the boundary point's local fit and gradient (first-order Taylor expansion) |
| `"error"` | Fails the whole call with an error |

### max_extrapolation_distance

Under `"linear"` extrapolation, the maximum allowed per-dimension distance beyond the training boundary before `predict()` throws, instead of returning an unbounded value. Disabled by default (uncapped).

### max_neighbor_distance

Maximum allowed distance to the farthest point in a query's k-nearest-neighbor window before `predict()` throws. Guards against an "empty corner" blind spot: a query point can sit inside every dimension's min/max bounding box yet still be far from any real training data. Disabled by default (uncapped); measured as a plain (raw-coordinate) Euclidean distance, independent of `distance_metric`.

## Example

### Basic Usage

```javascript
const { Loess } = require('fastloess');

const x = new Float64Array([1, 2, 3, 4, 5]);
const y = new Float64Array([2.1, 4.0, 6.2, 8.0, 10.1]);

const model = new Loess({ fraction: 0.7, retain_model: true });
const result = model.fit(x, y);

const prediction = result.predict(new Float64Array([1.5, 4.5]));
console.log("Predicted y:", prediction.y);
```

```output
Predicted y: Float64Array(2) [ 3.05, 9.05 ]
```

### Standard Errors and Derivative

```javascript
const { Loess } = require('fastloess');

const x = new Float64Array([1, 2, 3, 4, 5]);
const y = new Float64Array([2.1, 4.0, 6.2, 8.0, 10.1]);

const model = new Loess({ fraction: 0.7, retain_model: true });
const result = model.fit(x, y);

const prediction = result.predict(new Float64Array([2.5]), {
    outputs: ["se"],
    outputs: ["derivative"],
});
console.log(prediction.y, prediction.standard_errors, prediction.derivative);
```

```output
Float64Array(1) [ 5.1 ] null Float64Array(1) [ 2.2000000000000006 ]
```

### Linear Extrapolation

```javascript
const { Loess } = require('fastloess');

const x = new Float64Array([1, 2, 3, 4, 5]);
const y = new Float64Array([2.1, 4.0, 6.2, 8.0, 10.1]);

const model = new Loess({ fraction: 0.7, retain_model: true });
const result = model.fit(x, y);

const prediction = result.predict(new Float64Array([10.0]), {
    extrapolation: "linear",
});
console.log("Extrapolated y:", prediction.y);
```

```output
Extrapolated y: Float64Array(1) [ 10.1 ]
```
