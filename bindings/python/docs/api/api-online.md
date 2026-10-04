# OnlineLoess API

See also: [fastLoess](api.md)

## When to Use

- Data arrives incrementally (sensors, streams)
- Need real-time smoothed values
- Fixed memory budget

![Online Adapter](../assets/diagrams/online_comparison.svg)

## Class

### `OnlineLoess`

The `OnlineLoess` class updates the model incrementally with new data points.

**Constructor:**

:::{jupyter-execute}
import fastloess as fl

online = fl.OnlineLoess(fraction=0.5, window_capacity=50)
:::

#### `add_point(x, y)`

Adds a single point to the sliding window and returns the smoothed value for that point, or `None` while the window is still filling up (fewer than `min_points` seen so far). Once the window reaches `window_capacity`, each new point evicts the oldest one, so memory stays bounded regardless of how much history has passed through. `update_mode` controls how much work each call does: `"incremental"` re-fits only the newest point, while `"full"` re-smooths the entire window for a more accurate but slower result.

:::{jupyter-execute}
import fastloess as fl
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + 0.1

online = fl.OnlineLoess(fraction=0.5, window_capacity=50, min_points=3)

result = online.add_point(x[0], y[0])  # None
result = online.add_point(x[1], y[1])  # None

result = online.add_point(x[2], y[2])
print(result)
:::

## Options Structure

### `OnlineOptions` (inherits `LoessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `float` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `int` | `0` | Number of robustifying iterations; positive values require `update_mode="full"` |
| `weight_function` | `str` | `"tricube"` | Weight function name |
| `robustness_method` | `str` | `"bisquare"` | Robustness method name |
| `degree` | `str` | `"linear"` | Polynomial degree of local fit |
| `dimensions` | `int` | `1` | Number of predictor dimensions |
| `distance_metric` | `str` | `"normalized"` | Distance metric; use `"minkowski:p"` for custom p |
| `weighted_metric_weights` | `list[float]` | `None` | Per-dimension weights (used when `distance_metric="weighted"`) |
| `surface_mode` | `str` | `"interpolation"` | Surface computation mode |
| `cell` | `float` | `None` | Cell size for interpolation grid (smaller → more vertices, higher accuracy) |
| `interpolation_vertices` | `int` | `None` | Number of interpolation vertices |
| `zero_weight_fallback` | `str` | `"use_local_mean"` | Zero-weight handling strategy |
| `boundary_policy` | `str` | `"extend"` | Boundary handling policy |
| `boundary_degree_fallback` | `bool \| None` | `None` | Fall back to lower polynomial degree at boundaries when higher degrees fail |
| `scaling_method` | `str` | `"mad"` | Residual scaling method |
| `auto_converge` | `float` | `None` | Auto-convergence tolerance |
| `missing` | `str` | `"error"` | Policy for non-finite (NaN/Inf) values in each point |
| `window_capacity` | `int` | `1000` | Max points in sliding window |
| `min_points` | `int` | `2` | Min points before smoothing starts |
| `update_mode` | `str` | `"incremental"` | Update mode (`"full"` or `"incremental"`) |
| `outputs` | `Sequence[str] \| None` | `None` | Select `weights`, `gradient` (or `derivative`), and/or `se`; `se` requires `update_mode="full"` |
| `intervals` | `dict` | `None` | Grouped confidence and prediction coverage levels. |

Cross-validation, the `"sorted"` output, and `parallel` are Batch-only; the `"diagnostics"` and `"residuals"` outputs are unavailable online. See [fastLoess](api.md) for those options.

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
- `"minkowski"` (use `"minkowski:p"` string for custom exponent, e.g. `"minkowski:3"`)
- `"weighted"` plus `weighted_metric_weights` for per-dimension scaling (alias: `"weighted_euclidean"`)

### weighted_metric_weights

*See: [Multivariate LOESS](../advanced/dimensions.md)*

Per-dimension weights, one per dimension declared in `dimensions`. Only used when `distance_metric="weighted"`; setting `distance_metric="weighted"` without providing this raises an error.

- `None` (default) — has no effect unless `distance_metric="weighted"` is set
- A `list[float]` of per-dimension weights, required when `distance_metric="weighted"`

### surface_mode

*See: [Polynomial Degree](../advanced/degree.md#surface-mode)*

Controls whether the local polynomial is evaluated at every query point or at a sparser grid of anchor vertices with Hermite cubic interpolation in between.

| Mode | Behavior | Speed | Accuracy |
| --- | --- | --- | --- |
| `"interpolation"` (default) | Evaluate at vertices, interpolate between | Faster | Slight approximation |
| `"direct"` | Evaluate at every query point | Slower | Full precision |

### cell

Cell size for the interpolation grid, as a fraction of the data range. Smaller values place more vertices (denser grid), improving accuracy at the cost of speed. Only applies when `surface_mode="interpolation"`.

- `None` (default) — uses the library default (`0.2`)
- Any float in `(0, 1]`

### interpolation_vertices

Caps the maximum number of interpolation vertices, overriding the count implied by `cell`. Only applies when `surface_mode="interpolation"`.

- `None` (default) — uses the library default (no explicit cap)
- Any integer `>= 1`

### zero_weight_fallback

Behavior when all neighborhood weights are zero:

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`) | Use the mean of the neighborhood |
| `"return_original"` (alias: `"original"`) | Return the original y value |
| `"return_none"` (alias: `"none"`) | Return `NaN` |

### boundary_policy

*See: [Boundary Handling](../advanced/boundary.md)*

- `"extend"` (default; alias: `"pad"`)
- `"reflect"` (alias: `"mirror"`)
- `"zero"`
- `"noboundary"` (alias: `"none"`)

### boundary_degree_fallback

Whether to reduce the polynomial degree at boundary vertices when the requested `degree` can't be fit there (e.g., not enough neighbours). Only applies when `surface_mode="interpolation"`.

- `None` (default) — uses the library default (enabled)
- `True` — falls back to a lower degree at boundaries
- `False` — raises an error instead of silently falling back

### scaling_method

*See: [Scaling Methods](../weighting/scaling.md)*

- `"mad"` (default; alias: `"median_absolute_deviation"`)
- `"mar"` (alias: `"median_absolute_residual"`)
- `"mean"` (alias: `"mean_absolute_residual"`)

### auto_converge

*See: [Robustness](../weighting/robustness.md#auto-convergence)*

Convergence tolerance for early stopping of robustness iterations. `None` (default) disables early stopping.

### missing

Policy for handling a non-finite (NaN/Inf) `x` or `y` value passed to `add_point`:

| Option | Behavior |
| --- | --- |
| `"error"` (default) | Raise an error |
| `"drop"` | Silently ignore the point — `add_point` returns `None` instead of adding it to the window |

### window_capacity

Maximum number of most recent points kept in the sliding window; older points are discarded as new ones arrive. Each `add_point()` call costs O(`window_capacity`) rather than growing with total history.

### min_points

Minimum number of points required before smoothing starts. `add_point()` returns `None` until the window reaches this size.

### update_mode

*See: [Execution Modes](../guide/adapter-choice.md)*

| Mode | Alias | Behavior | Speed |
| --- | --- | --- | --- |
| `"incremental"` (default) | `"single"` | Update only affected fits | Faster |
| `"full"` | `"resmooth"` | Recompute entire window | More accurate |

### outputs: se

Select `"se"` to include the standard error for the latest point in the result (`OnlineOutput.standard_error`). It requires `update_mode="full"`.

### outputs: weights

Select `"weights"` to include the robustness weight for the latest point (from the last robustness iteration) in the result.

### outputs: gradient

Select `"gradient"` to expose the latest point's gradient (`dimensions` values) in `OnlineOutput.gradient` at effectively no extra computation cost. Only supported when `surface_mode` is `"direct"`; requesting it under `"interpolation"` raises an error.

### intervals.confidence

*See: [Intervals](../guide/intervals.md)*

Confidence level for the confidence interval around the mean response (e.g. `0.95`). Only computed under `update_mode="full"` — raises a `ValueError` at construction time if set (or `outputs` selects `"se"`/`intervals.prediction` is set) while `update_mode` is left at its default `"incremental"`, since incremental updates never compute standard errors. `None` (default) disables confidence intervals.

### intervals.prediction

*See: [Intervals](../guide/intervals.md)*

Confidence level for the prediction interval for new observations (e.g. `0.95`). Same `update_mode="full"` requirement as `intervals.confidence`. `None` (default) disables prediction intervals.

## Result Structure

### `OnlineOutput`

Returned by `add_point()` once the window has enough points (`None` until then).

| Field | Type | Description |
| --- | --- | --- |
| `y` | `float` | Smoothed value for the latest point |
| `standard_error` | `float \| None` | Standard error, if `"se"`/`intervals.confidence`/`intervals.prediction` was requested and `update_mode="full"` |
| `residual` | `float \| None` | Residual y − smoothed; always present (there is no `"residuals"` output for Online) |
| `robustness_weight` | `float \| None` | Robustness weight, if `"weights"` was requested |
| `iterations_used` | `int \| None` | Robustness iterations performed |
| `confidence_lower` / `confidence_upper` | `float \| None` | Confidence interval bounds, if `intervals.confidence` was set and `update_mode="full"` |
| `prediction_lower` / `prediction_upper` | `float \| None` | Prediction interval bounds, if `intervals.prediction` was set and `update_mode="full"` |
| `gradient` | `ndarray \| None` | Local fit gradient (`dimensions` values) for the latest point, if `"gradient"` was requested |

There is no `Diagnostics` object or `"diagnostics"` output for `OnlineLoess`: `OnlineOutput` carries no diagnostics field, since diagnostics like RMSE/R² need more than one point's worth of history to be meaningful.
