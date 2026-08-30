# fastLoess

The Python bindings provide a high-performance interface to the core Rust library, mirroring the Rust API structure.

> **StreamingLoess** and **OnlineLoess** are documented separately: [Streaming Adapter](api-streaming.md), [Online Adapter](api-online.md)

## When to Use Batch Adapter

- Dataset fits in memory
- Need intervals, cross-validation, or diagnostics
- Processing complete files

## Classes

### `Loess`

The `Loess` class allows configuring the LOESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

:::{jupyter-execute}
import fastloess as fl

model = fl.Loess(fraction=0.5, iterations=3)
print(model)
:::

**Methods:**

:::{jupyter-execute}
import fastloess as fl
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + 0.1

model = fl.Loess(fraction=0.5)
result = model.fit(x, y)
print(result)
:::

- Fits the model to the provided `x` and `y` array-like objects.
- `custom_weights`: Optional array of per-observation weights. All values must be ≥ 0 and length must match `x`. Batch only.
- Returns a `LoessResult` object containing the smoothed values and optional diagnostics.

See [Streaming Adapter](api-streaming.md) for the `StreamingLoess` class.

See [Online Adapter](api-online.md) for the `OnlineLoess` class.

## Options Structures

### `LoessOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `float` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `int` | `3` | Number of robustifying iterations |
| `weight_function` | `str` | `"tricube"` | Kernel weight function |
| `robustness_method` | `str` | `"bisquare"` | Robustness method |
| `scaling_method` | `str` | `"mad"` | Residual scaling method |
| `boundary_policy` | `str` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback` | `str` | `"use_local_mean"` | Zero-weight handling |
| `auto_converge` | `float` | `None` | Auto-convergence tolerance |
| `custom_weights` | `list[float]` | `None` | Per-observation case weights — passed to `fit()`, not the constructor (Batch only; see [Custom Weights](../user-guide/custom-weights.md)) |
| `confidence_intervals` | `float` | `None` | Confidence level (e.g., 0.95) — see [Intervals](../user-guide/intervals.md) |
| `prediction_intervals` | `float` | `None` | Prediction level (e.g., 0.95) — see [Intervals](../user-guide/intervals.md) |
| `return_diagnostics` | `bool` | `False` | Compute RMSE, MAE, R², AIC |
| `return_residuals` | `bool` | `False` | Include residuals in result |
| `return_robustness_weights` | `bool` | `False` | Include robustness weights in result |
| `return_se` | `bool` | `False` | Compute hat-matrix statistics (enp, leverage …) |
| `parallel` | `bool` | `True` | Enable parallel execution |
| `degree` | `str` | `"linear"` | Polynomial degree of local fit — see [Polynomial Degree](../user-guide/degree.md) |
| `dimensions` | `int` | `1` | Number of predictor dimensions — see [Multivariate LOESS](../user-guide/dimensions.md) |
| `distance_metric` | `str` | `"normalized"` | Distance metric; use `"minkowski:p"` for custom p |
| `weighted_metric_weights` | `list[float]` | `None` | Per-dimension weights (used when `distance_metric="weighted"`) |
| `surface_mode` | `str` | `"interpolation"` | Surface computation mode |
| `cell` | `float` | `None` | Cell size for interpolation grid (smaller → more vertices, higher accuracy) |
| `interpolation_vertices` | `int` | `None` | Number of interpolation vertices |
| `boundary_degree_fallback` | `bool \| None` | `None` | Fall back to lower polynomial degree at boundaries when higher degrees fail |
| `cv_method` | `str` | `"kfold"` | CV method (`"kfold"` fast or `"loocv"` slow, exhaustive) (Batch only) |
| `cv_k` | `int` | `5` | Number of folds for k-fold CV (Batch only) |
| `cv_fractions` | `list[float]` | `None` | Fractions to test for cross-validation (Batch only) |
| `cv_seed` | `int` | `None` | Random seed for cross-validation shuffling (Batch only) |

`fraction` is the most important parameter: it controls the size of the local neighbourhood used at each point.

| Range | Effect | Use case |
| --- | --- | --- |
| 0.1-0.3 | Fine detail | Rapidly changing signals |
| 0.3-0.5 | Balanced | General purpose |
| 0.5-0.7 | Heavy smoothing | Noisy data |
| 0.7-1.0 | Very smooth | Trend extraction |

`iterations` controls robustness to outliers, at the cost of speed.

| Value | Effect | Performance |
| --- | --- | --- |
| 0 | No robustness | Fastest |
| 1-3 | Moderate | Recommended |
| 4-6 | Strong | Contaminated data |
| 7+ | Very strong | Heavy outliers |

See [Streaming Adapter](api-streaming.md) for `StreamingOptions`.

See [Online Adapter](api-online.md) for `OnlineOptions`.

## Result Structure

See [Online Adapter](api-online.md) for `OnlineOutput`.

### `LoessResult`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `ndarray` | Sorted x values |
| `y` | `ndarray` | Smoothed y values |
| `fraction_used` | `float` | Fraction used (set or selected by CV) |
| `iterations_used` | int \| None | Robustness iterations actually performed |
| `standard_errors` | ndarray \| None | Per-point SE (if `return_se`) |
| `confidence_lower` | ndarray \| None | Lower confidence bounds |
| `confidence_upper` | ndarray \| None | Upper confidence bounds |
| `prediction_lower` | ndarray \| None | Lower prediction bounds |
| `prediction_upper` | ndarray \| None | Upper prediction bounds |
| `residuals` | ndarray \| None | Residuals (if `return_residuals`) |
| `robustness_weights` | ndarray \| None | Robustness weights (if `return_robustness_weights`) |
| `cv_scores` | ndarray \| None | CV score per tested fraction |
| `diagnostics` | Diagnostics \| None | Fit metrics (if `return_diagnostics`) |
| `enp` | float \| None | Equivalent number of parameters (if `return_se`) |
| `trace_hat` | float \| None | Trace of hat matrix (if `return_se`) |
| `delta1` | float \| None | First delta statistic (if `return_se`) |
| `delta2` | float \| None | Second delta statistic (if `return_se`) |
| `residual_scale` | float \| None | Residual scale estimate (if `return_se`) |
| `leverage` | ndarray \| None | Per-point hat-matrix diagonal (if `return_se`) |
| `dimensions` | `int` | Number of predictor dimensions |

### `Diagnostics`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `float` | Root Mean Squared Error |
| `mae` | `float` | Mean Absolute Error |
| `r_squared` | `float` | R-squared |
| `residual_sd` | `float` | Residual standard deviation |
| `effective_df` | float \| None | Effective degrees of freedom |
| `aic` | float \| None | AIC |
| `aicc` | float \| None | AICc |

## Options

### weight_function

*See: [Weight Functions](../user-guide/kernels.md)*

- `"tricube"` (default)
- `"epanechnikov"`
- `"gaussian"`
- `"uniform"` (alias: `"boxcar"`)
- `"biweight"` (alias: `"bisquare"`)
- `"triangle"` (alias: `"triangular"`)
- `"cosine"`

### robustness_method

*See: [Robustness](../user-guide/robustness.md)*

- `"bisquare"` (default; alias: `"biweight"`)
- `"huber"`
- `"talwar"`

### boundary_policy

*See: [Boundary Handling](../user-guide/boundary.md)*

- `"extend"` (default; alias: `"pad"`)
- `"reflect"` (alias: `"mirror"`)
- `"zero"`
- `"noboundary"` (alias: `"none"`)

### scaling_method

*See: [Scaling Methods](../user-guide/scaling.md)*

- `"mad"` (default; alias: `"median_absolute_deviation"`)
- `"mar"` (alias: `"median_absolute_residual"`)
- `"mean"` (alias: `"mean_absolute_residual"`)

### zero_weight_fallback

Behavior when all neighborhood weights are zero:

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`) | Use the mean of the neighborhood |
| `"return_original"` (alias: `"original"`) | Return the original y value |
| `"return_none"` (alias: `"none"`) | Return `NaN` |

### degree

*See: [Polynomial Degree](../user-guide/degree.md)*

- `"constant"` or `"0"` (degree 0)
- `"linear"` or `"1"` (default, degree 1)
- `"quadratic"` or `"2"` (degree 2)
- `"cubic"` or `"3"` (degree 3)
- `"quartic"` or `"4"` (degree 4)

### distance_metric

*See: [Multivariate LOESS](../user-guide/dimensions.md)*

- `"normalized"` (default — scales each dimension by its range; alias: `"norm"`)
- `"euclidean"` (alias: `"euclid"`)
- `"manhattan"` (alias: `"l1"`)
- `"chebyshev"` (alias: `"linf"`)
- `"minkowski"` (use `"minkowski:p"` string for custom exponent, e.g. `"minkowski:3"`)
- `"weighted"` plus `weighted_metric_weights` for per-dimension scaling (alias: `"weighted_euclidean"`)

### surface_mode

*See: [Polynomial Degree](../user-guide/degree.md#surface-mode)*

Controls whether the local polynomial is evaluated at every query point or at a sparser grid of anchor vertices with Hermite cubic interpolation in between.

| Mode | Behavior | Speed | Accuracy |
| --- | --- | --- | --- |
| `"interpolation"` (default) | Evaluate at vertices, interpolate between | Faster | Slight approximation |
| `"direct"` | Evaluate at every query point | Slower | Full precision |

## Example

:::{jupyter-execute}
from fastloess import Loess
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + 0.1

## Configure model

model = Loess(fraction=0.5)

## Fit data

result = model.fit(x, y)

print(result)
:::
