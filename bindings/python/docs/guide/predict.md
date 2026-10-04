# Out-of-Sample Prediction

Evaluate a fitted Batch model at query points that were not in the training set.

## Overview

:::{note} Adapter support
Out-of-sample prediction is available in **Batch** mode only. Streaming and Online modes do not support it.
:::

`LoessResult.predict(new_x, ...)` evaluates the fit at arbitrary query points, like R's `predict(model, newdata)`. Query points are flattened, `dimensions` values per point.

It always fits exactly, unlike `fit()`'s default `surface_mode="interpolation"` — so predicting at a training point may not exactly match `fit()`'s output unless `surface_mode="direct"` was used.

Requires `retain_model=True` on the constructor before `fit()`, otherwise `predict()` raises `LoessError`.

---

## Options

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `outputs` | `Sequence[str] \| None` | `None` | Select `"se"` and/or `"gradient"` (alias: `"derivative"`) |
| `confidence_level` | `float \| None` | `None` | Confidence interval coverage level (e.g. `0.95`) |
| `prediction_level` | `float \| None` | `None` | Prediction interval coverage level (e.g. `0.95`) |
| `extrapolation` | `str` | `"clamp"` | Behavior for query points outside the training range, on any dimension |
| `max_extrapolation_distance` | `float \| None` | `None` | Under `"linear"` extrapolation, the max allowed per-dimension distance beyond the training boundary before erroring |
| `max_neighbor_distance` | `float \| None` | `None` | Max allowed distance to the farthest point in a query's k-nearest-neighbor window before erroring |

### outputs: se

Select `"se"` to compute standard errors for each query point, using the retained model's residual scale and per-point leverage. Required for `confidence_level`/`prediction_level` to be populated.

### confidence_level

Confidence level for the confidence interval around the mean response at each query point (e.g. `0.95`). Uses the same z-score convention as `fit()`'s own confidence intervals. `None` (default) disables it.

### prediction_level

Confidence level for the prediction interval for a new observation at each query point (e.g. `0.95`). Widens using the same residual scale `fit()` used for its own intervals when available, otherwise falling back to a MAD-based estimate. `None` (default) disables it.

### outputs: gradient

Select `"gradient"` (or `"derivative"`) to include the local fit's gradient (`dimensions` values per query point, flattened) in the output.

### extrapolation

Behavior for query points outside the training range, on any dimension:

| Policy | Behavior |
| --- | --- |
| `"clamp"` (default) | Clamps each out-of-range dimension to the nearest training boundary |
| `"linear"` | Linearly extrapolates from the boundary point's local fit and gradient (first-order Taylor expansion) |
| `"error"` | Fails the whole call with `LoessError` |

### max_extrapolation_distance

Under `"linear"` extrapolation, the maximum allowed per-dimension distance beyond the training boundary before `predict()` raises, instead of returning an unbounded value. `None` (default, uncapped).

### max_neighbor_distance

Maximum allowed distance to the farthest point in a query's k-nearest-neighbor window before `predict()` raises. Guards against an "empty corner" blind spot: a query point can sit inside every dimension's min/max bounding box yet still be far from any real training data. `None` (default, uncapped); measured as a plain (raw-coordinate) Euclidean distance, independent of `distance_metric`.

## Example

### Basic Usage

:::{jupyter-execute}
import fastloess as fl
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.1, 4.0, 6.2, 8.0, 10.1])

model = fl.Loess(fraction=0.7, retain_model=True)
result = model.fit(x, y)

new_x = np.array([1.5, 4.5])
prediction = result.predict(new_x)
print("Predicted y:", prediction.y)
:::

### Standard Errors and Derivative

:::{jupyter-execute}
import fastloess as fl
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.1, 4.0, 6.2, 8.0, 10.1])

model = fl.Loess(fraction=0.7, retain_model=True)
result = model.fit(x, y)

prediction = result.predict(np.array([2.5]), outputs=["se", "gradient"])
print("y:", prediction.y)
print("SE:", prediction.standard_errors)
print("Derivative:", prediction.derivative)
:::

### Linear Extrapolation

:::{jupyter-execute}
import fastloess as fl
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.1, 4.0, 6.2, 8.0, 10.1])

model = fl.Loess(fraction=0.7, retain_model=True)
result = model.fit(x, y)

prediction = result.predict(np.array([10.0]), extrapolation="linear")
print("Extrapolated y:", prediction.y)
:::
