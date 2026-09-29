\page guide_predict Out-of-Sample Prediction

# Out-of-Sample Prediction

Evaluate a fitted Batch model at query points that were not in the training set.

## Overview

Out-of-sample prediction is available in **Batch** mode only. Streaming and Online modes do not support it.

`fastloess::PredictModel::predict(new_x, options)` evaluates the fit at arbitrary query points, like R's `predict(model, newdata)`. Query points are flattened, `dimensions` values per point.

It always fits exactly, unlike `fit()`'s default `surface_mode = "interpolation"` — so predicting at a training point may not exactly match `fit()`'s output unless `surface_mode = "direct"` was used.

Requires `retain_model = true` on `LoessOptions` before `fit()`; obtain the `PredictModel` via `LoessResult::predict_model()` (moves the retained state out — only valid once, check `PredictModel::valid()`).

---

## Options

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `outputs` | `std::vector<std::string>` | `{}` | Optional prediction fields: `se`, `gradient`/`derivative` |
| `confidence_level` | `double` | `NaN` | Confidence interval coverage level (e.g. `0.95`; NaN to disable) |
| `prediction_level` | `double` | `NaN` | Prediction interval coverage level (e.g. `0.95`; NaN to disable) |
| `extrapolation` | `std::string` | `"clamp"` | Behavior for query points outside the training range, on any dimension |
| `max_extrapolation_distance` | `double` | `NaN` | Under `"linear"` extrapolation, the max allowed per-dimension distance beyond the training boundary before erroring |
| `max_neighbor_distance` | `double` | `NaN` | Max allowed distance to the farthest point in a query's k-nearest-neighbor window before erroring |

### outputs

Select optional prediction fields by name. Use `"se"` to compute standard errors for each query point, using the retained model's residual scale and per-point leverage; standard errors are also computed when `confidence_level` or `prediction_level` is set. Use `"gradient"` or `"derivative"` to include the local fit gradient (`dimensions` values per query point, flattened). An empty vector (default) requests neither field.

### confidence_level

Confidence level for the confidence interval around the mean response at each query point (e.g. `0.95`). Uses the same z-score convention as `fit()`'s own confidence intervals. `NaN` (default) disables it.

### prediction_level

Confidence level for the prediction interval for a new observation at each query point (e.g. `0.95`). Widens using the same residual scale `fit()` used for its own intervals when available, otherwise falling back to a MAD-based estimate. `NaN` (default) disables it.

### extrapolation

Behavior for query points outside the training range, on any dimension:

| Policy | Behavior |
| --- | --- |
| `"clamp"` (default) | Clamps each out-of-range dimension to the nearest training boundary |
| `"linear"` | Linearly extrapolates from the boundary point's local fit and gradient (first-order Taylor expansion) |
| `"error"` | Fails the whole call with an error |

### max_extrapolation_distance

Under `"linear"` extrapolation, the maximum allowed per-dimension distance beyond the training boundary before `predict()` errors, instead of returning an unbounded value. `NaN` (default, uncapped).

### max_neighbor_distance

Maximum allowed distance to the farthest point in a query's k-nearest-neighbor window before `predict()` errors. Guards against an "empty corner" blind spot: a query point can sit inside every dimension's min/max bounding box yet still be far from any real training data. `NaN` (default, uncapped); measured as a plain (raw-coordinate) Euclidean distance, independent of `distance_metric`.

## Example

### Basic Usage

```cpp
#include <fastloess.hpp>
#include <iostream>
#include <vector>

int main() {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2.1, 4.0, 6.2, 8.0, 10.1};

    fastloess::LoessOptions opts;
    opts.fraction = 0.7;
    opts.retain_model = true;
    fastloess::Loess model(opts);
    auto result = model.fit(x, y).value();

    auto predict_model = result.predict_model();
    auto prediction = predict_model.predict({1.5, 4.5});
    for (double v : prediction.y()) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
    return 0;
}
```

```output
3.05 9.05
```

### Standard Errors and Derivative

```cpp
#include <fastloess.hpp>
#include <iostream>
#include <vector>

int main() {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2.1, 4.0, 6.2, 8.0, 10.1};

    fastloess::LoessOptions opts;
    opts.fraction = 0.7;
    opts.retain_model = true;
    fastloess::Loess model(opts);
    auto result = model.fit(x, y).value();

    auto predict_model = result.predict_model();
    fastloess::PredictOptions popts;
    popts.outputs = {"se", "derivative"};
    auto prediction = predict_model.predict({2.5}, popts);
    return 0;
}
```

### Linear Extrapolation

```cpp
#include <fastloess.hpp>
#include <iostream>
#include <vector>

int main() {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2.1, 4.0, 6.2, 8.0, 10.1};

    fastloess::LoessOptions opts;
    opts.fraction = 0.7;
    opts.retain_model = true;
    fastloess::Loess model(opts);
    auto result = model.fit(x, y).value();

    auto predict_model = result.predict_model();
    fastloess::PredictOptions popts;
    popts.extrapolation = "linear";
    auto prediction = predict_model.predict({10.0}, popts);
    return 0;
}
```
