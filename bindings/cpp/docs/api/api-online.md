\page api_online OnlineLoess API

# OnlineLoess API

See also: [fastLoess](api.md)

## When to Use

- Data arrives incrementally (sensors, streams)
- Need real-time smoothed values
- Fixed memory budget

![Online Adapter](online_comparison.svg)

## Class

### fastloess::OnlineLoess

The `OnlineLoess` class updates the model incrementally with new data points.

**Constructor:**

```cpp
#include <fastloess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = i * 2 * M_PI / (n - 1);
        y[i] = std::sin(x[i]) + 0.1;
    }

    fastloess::OnlineOptions opts;
    opts.fraction = 0.5;
    opts.window_capacity = 50;
    opts.min_points = 3;
    fastloess::OnlineLoess model(opts);

    for (size_t i = 0; i < x.size(); ++i) {
        auto out = model.add_point(x[i], y[i]).value();
        if (out.has_value()) {
            std::cout << "y[0]: " << out.y() << "\n";
            break;
        }
    }
    return 0;
}
```

```output
y[0]: 0.226592
```

- `options`: An `OnlineOptions` struct with `window_capacity`, `min_points`, and `update_mode`.

#### `add_point(x, y)`

Adds a single point to the sliding window. Returns `Expected<OnlineOutput>` — check `result.has_value()` to see whether the window is ready.

```cpp
#include <fastloess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = i * 2 * M_PI / (n - 1);
        y[i] = std::sin(x[i]) + 0.1;
    }

    fastloess::OnlineOptions opts;
    opts.fraction = 0.5;
    opts.window_capacity = 50;
    opts.min_points = 3;
    fastloess::OnlineLoess model(opts);

    // Returns OnlineOutput with has_value() == false until min_points (3) are reached
    auto r1 = model.add_point(x[0], y[0]).value();  // r1.has_value() == false
    auto r2 = model.add_point(x[1], y[1]).value();  // r2.has_value() == false

    // Returns OnlineOutput with has_value() == true once enough points are available
    auto r3 = model.add_point(x[2], y[2]).value();
    if (r3.has_value()) {
        std::cout << r3.y() << std::endl;  // 0.22659245357374927
    }

    return 0;
}
```

```output
0.226592
```

## Options Structure

### OnlineOptions

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `double` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `int` | `3` | Number of robustifying iterations |
| `weight_function` | `std::string` | `"tricube"` | Weight function name |
| `robustness_method` | `std::string` | `"bisquare"` | Robustness method name |
| `scaling_method` | `std::string` | `"mad"` | Residual scaling method |
| `boundary_policy` | `std::string` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback` | `std::string` | `"use_local_mean"` | Zero-weight handling strategy |
| `auto_converge` | `double` | `NaN` | Auto-convergence tolerance (NaN to disable) |
| `return_robustness_weights` | `bool` | `false` | Include robustness weight in result |
| `degree` | `std::string` | `"linear"` | Polynomial degree of local fit |
| `dimensions` | `int` | `1` | Number of predictor dimensions |
| `distance_metric` | `std::string` | `"normalized"` | Distance metric; use `"minkowski:p"` for custom p |
| `weighted_metric_weights` | `std::vector<double>` | `{}` | Per-dimension weights (used when `distance_metric = "weighted"`) |
| `surface_mode` | `std::string` | `"interpolation"` | Surface computation mode |
| `cell` | `double` | `NaN` | Cell size for interpolation grid (smaller → more vertices, higher accuracy) |
| `interpolation_vertices` | `int` | `0` | Number of interpolation vertices (0 for default) |
| `boundary_degree_fallback` | `int` | `-1` | Fall back to lower polynomial degree at boundaries (-1 = unset/library default, 0 = false, 1 = true) |
| `window_capacity` | `int` | `1000` | Max points in sliding window |
| `min_points` | `int` | `2` | Min points before smoothing starts |
| `update_mode` | `std::string` | `"incremental"` | Update mode (`"full"` or `"incremental"`) |

Confidence/prediction intervals, standard errors, cross-validation, `return_diagnostics`, `return_residuals`, and `parallel` are Batch-only (or Batch/Streaming-only) and not available here; see [fastLoess](api.md) for those. Online always runs sequentially.

## Options

### fraction

`fraction` is the most important parameter: it controls the size of the local neighbourhood used at each point.

| Range | Effect | Use case |
| --- | --- | --- |
| 0.1-0.3 | Fine detail | Rapidly changing signals |
| 0.3-0.5 | Balanced | General purpose |
| 0.5-0.7 | Heavy smoothing | Noisy data |
| 0.7-1.0 | Very smooth | Trend extraction |

### iterations

`iterations` controls robustness to outliers, at the cost of speed.

| Value | Effect | Performance |
| --- | --- | --- |
| 0 | No robustness | Fastest |
| 1-3 | Moderate | Recommended |
| 4-6 | Strong | Contaminated data |
| 7+ | Very strong | Heavy outliers |

### weight_function

*See: [Weight Functions](../weighting/kernels.md)*

- `"tricube"` (default)
- `"epanechnikov"`
- `"gaussian"`
- `"uniform"` (alias: `"boxcar"`)
- `"biweight"` (alias: `"bisquare"`)
- `"triangle"` (alias: `"triangular"`)
- `"cosine"`

### robustness_method

*See: [Robustness](../weighting/robustness.md)*

- `"bisquare"` (default; alias: `"biweight"`)
- `"huber"`
- `"talwar"`

### scaling_method

*See: [Scaling Methods](../weighting/scaling.md)*

- `"mad"` (default; alias: `"median_absolute_deviation"`)
- `"mar"` (alias: `"median_absolute_residual"`)
- `"mean"` (alias: `"mean_absolute_residual"`)

### boundary_policy

*See: [Boundary Handling](../advanced/boundary.md)*

- `"extend"` (default; alias: `"pad"`)
- `"reflect"` (alias: `"mirror"`)
- `"zero"`
- `"noboundary"` (alias: `"none"`)

### zero_weight_fallback

Behavior when all neighborhood weights are zero:

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`) | Use the mean of the neighborhood |
| `"return_original"` (alias: `"original"`) | Return the original y value |
| `"return_none"` (alias: `"none"`) | Return `NaN` |

### auto_converge

*See: [Robustness](../weighting/robustness.md#auto-convergence)*

Convergence tolerance for early stopping of robustness iterations. `NaN` (default) disables early stopping.

### return_robustness_weights

Include the robustness weight for the latest point (from the last robustness iteration) in the result.

- `false` (default) — leaves `robustness_weight()` as NaN
- `true` — populates `robustness_weight()`

### degree

*See: [Polynomial Degree](../advanced/degree.md)*

- `"constant"` or `"0"` (degree 0)
- `"linear"` or `"1"` (default, degree 1)
- `"quadratic"` or `"2"` (degree 2)
- `"cubic"` or `"3"` (degree 3)
- `"quartic"` or `"4"` (degree 4)

### dimensions

*See: [Multivariate LOESS](../advanced/dimensions.md)*

Number of predictor dimensions. Set to match the number of columns in a multivariate `x` array.

- Any integer `>= 1`; `1` (default) is univariate

### distance_metric

*See: [Multivariate LOESS](../advanced/dimensions.md)*

- `"normalized"` (default — scales each dimension by its range; alias: `"norm"`)
- `"euclidean"` (alias: `"euclid"`)
- `"manhattan"` (alias: `"l1"`)
- `"chebyshev"` (alias: `"linf"`)
- `"minkowski"` (use `"minkowski:p"` for custom p, e.g. `"minkowski:3"`)
- `"weighted"` plus `weighted_metric_weights` for per-dimension scaling (alias: `"weighted_euclidean"`)

### weighted_metric_weights

*See: [Multivariate LOESS](../advanced/dimensions.md)*

Per-dimension weights, one per dimension declared in `dimensions`. Only used when `distance_metric = "weighted"`; setting `distance_metric = "weighted"` without providing this raises an error.

- `{}` (default, empty vector) — has no effect unless `distance_metric = "weighted"` is set
- A non-empty `std::vector<double>` of per-dimension weights, required when `distance_metric = "weighted"`

### surface_mode

*See: [Polynomial Degree](../advanced/degree.md#surface-mode)*

Controls whether the local polynomial is evaluated at every query point or at a sparser grid of anchor vertices with Hermite cubic interpolation in between.

| Mode | Behavior | Speed | Accuracy |
| --- | --- | --- | --- |
| `"interpolation"` (default) | Evaluate at vertices, interpolate between | Faster | Slight approximation |
| `"direct"` | Evaluate at every query point | Slower | Full precision |

### cell

Cell size for the interpolation grid, as a fraction of the data range. Smaller values place more vertices (denser grid), improving accuracy at the cost of speed. Only applies when `surface_mode = "interpolation"`.

- `NaN` (default) — uses the library default (`0.2`)
- Any value in `(0, 1]`

### interpolation_vertices

Caps the maximum number of interpolation vertices, overriding the count implied by `cell`. Only applies when `surface_mode = "interpolation"`.

- `0` (default) — uses the library default (no explicit cap)
- Any integer `>= 1`

### boundary_degree_fallback

Whether to reduce the polynomial degree at boundary vertices when the requested `degree` can't be fit there (e.g., not enough neighbours). Only applies when `surface_mode = "interpolation"`.

- `-1` (default) — uses the library default (enabled)
- `1` — falls back to a lower degree at boundaries
- `0` — raises an error instead of silently falling back

### window_capacity

Maximum number of most recent points kept in the sliding window; older points are evicted as new ones arrive. Each `add_point()` call costs O(`window_capacity`) rather than growing with total history.

### min_points

Minimum number of points required before `add_point()` starts returning smoothed output (`has_value() == true`).

### update_mode

*See: [Execution Modes](../guide/adapter-choice.md)*

| Mode | Alias | Behavior | Speed |
| --- | --- | --- | --- |
| `"incremental"` (default) | `"single"` | Update only affected fits | Faster |
| `"full"` | `"resmooth"` | Recompute entire window | More accurate |

## Result Structure

### fastloess::OnlineOutput

Returned (inside `Expected`) by `add_point()`. Check `has_value()` before reading fields.

| Method | Return Type | Description |
| --- | --- | --- |
| `has_value()` | `bool` | `false` while window fills; `true` when output is ready |
| `y()` | `double` | Smoothed value for the latest point |
| `standard_error()` | `double` | Always `NaN` — standard errors require confidence intervals, which are Batch-only |
| `residual()` | `double` | Residual y − smoothed; always present (there is no `return_residuals` option for Online) |
| `robustness_weight()` | `double` | Robustness weight, if `return_robustness_weights` was set (`NaN` otherwise) |
| `iterations_used()` | `int` | Robustness iterations performed (−1 if N/A) |

There is no `Diagnostics` object or `return_diagnostics` option for `OnlineLoess`: `OnlineOutput` carries no diagnostics field, since diagnostics like RMSE/R² need more than one point's worth of history to be meaningful.
