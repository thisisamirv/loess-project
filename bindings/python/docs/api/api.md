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

#### `fit(x, y)`

Fits the model to the provided `x` and `y` array-like objects. `custom_weights`: Optional array of per-observation weights. All values must be ≥ 0 and length must match `x`. Returns a `LoessResult` object containing the smoothed values and optional diagnostics.

:::{jupyter-execute}
import fastloess as fl
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + 0.1

model = fl.Loess(fraction=0.5)
result = model.fit(x, y)
print(result)
:::

## Options Structures

### `LoessOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `float` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `int` | `3` | Number of robustifying iterations |
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
| `missing` | `str` | `"error"` | Policy for non-finite (NaN/Inf) values in input data |
| `parallel` | `bool` | `True` | Enable parallel execution |
| `outputs` | `Sequence[str] \| None` | `None` | Select `diagnostics`, `residuals`, `weights`, `gradient` (or `derivative`), `se`, and/or `sorted` |
| `intervals` | `dict` | `None` | Grouped confidence and prediction coverage levels. |
| `cv` | `dict \| None` | `None` | Group `fractions`, `method`, `k`, and `seed`; supplied keys override individual CV arguments |
| `seed` | `int` | `None` | Seed for reproducible CV folds. |
| `retain_model` | `bool` | `False` | Retain training data, enabling `predict()` on the result |
| `custom_weights` | `list[float]` | `None` | Per-observation case weights — passed to `fit()`, not the constructor |

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

Policy for handling non-finite (NaN/Inf) values in `x`/`y` (and `custom_weights`):

| Option | Behavior |
| --- | --- |
| `"error"` (default) | Raise an error if any value is non-finite |
| `"drop"` | Silently remove observations (rows) where any x dimension or y is non-finite before fitting |

**Note:** A length mismatch between `x` and `y` always errors, even under `"drop"`.

### parallel

Enable multi-threaded execution via Rayon.

- `True` (default) — parallelizes the local regression fits across CPU cores
- `False` — forces single-threaded execution (useful for benchmarking or deterministic profiling)

### outputs: se

*See: [Intervals](../guide/intervals.md#standard-errors)*

Select `"se"` to compute standard errors and hat-matrix statistics (effective degrees of freedom, leverage, delta1/delta2).

### outputs: diagnostics

*See: [`Diagnostics`](#diagnostics)*

Select `"diagnostics"` to include a `Diagnostics` object (RMSE, MAE, R², AIC/AICc, effective degrees of freedom) in the result. AIC/AICc/`effective_df` additionally require `"se"` (or confidence/prediction intervals) to be selected, since they depend on hat-matrix statistics.

### outputs: residuals

Select `"residuals"` to include per-point residuals (`y - fitted`) in the result.

### outputs: weights

Select `"weights"` to include the final per-point robustness weights (from the last robustness iteration) in the result.

### outputs: gradient

Select `"gradient"` to expose the per-point gradient (rate of change of the smoothed surface, `dimensions` values per point, flattened) in `LoessResult.gradient` at effectively no extra computation cost. Only supported when `surface_mode` is `"direct"` — `.fit()` raises an error if requested under the default `"interpolation"` mode.

### outputs: sorted

Select `"sorted"` to reorder every result field (residuals, intervals, etc.) by `x` in ascending order instead of preserving input order.
To get both orderings, sort the default result client-side (e.g. `np.argsort(result.x)`) instead of calling `fit()` twice.

### intervals.confidence

*See: [Intervals](../guide/intervals.md)*

Confidence level for the confidence interval around the mean response (e.g. `0.95`). `None` (default) disables confidence intervals.

### intervals.prediction

*See: [Intervals](../guide/intervals.md)*

Confidence level for the prediction interval for new observations (e.g. `0.95`). `None` (default) disables prediction intervals.

### CV Options

*See: [Cross-Validation](../guide/cross-validation.md)*

- `cv.method`: `"kfold"` (default) — fast, evaluates each candidate fraction over `cv.k` folds; `"loocv"` — slow, exhaustive leave-one-out cross-validation
- `cv.k`: Number of folds for k-fold CV. Ignored when `cv={"method": "loocv"}`.
- `cv.fractions`: Candidate fractions to evaluate. Cross-validation is disabled unless this is set.
- `seed`: Seed for reproducible k-fold shuffling. `None` (default) uses a random seed.

### retain_model

*See: [Predict](../guide/predict.md)*

Retains the fitted model's training data, enabling `LoessResult.predict(new_x, ...)` to evaluate the fit at out-of-sample query points not in the training set. `False` (default) — no extra memory/copy cost unless requested.

### custom_weights

*See: [Custom Weights](../weighting/custom-weights.md)*

Per-observation weights, passed to `fit()` rather than the constructor.

## Result Structure

### `LoessResult`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `ndarray` | x values (same order as input) |
| `y` | `ndarray` | Smoothed y values |
| `fraction_used` | `float` | Fraction used (set or selected by CV) |
| `iterations_used` | `int \| None` | Robustness iterations actually performed |
| `standard_errors` | `ndarray \| None` | Per-point standard errors |
| `confidence_lower` | `ndarray \| None` | Lower confidence bounds |
| `confidence_upper` | `ndarray \| None` | Upper confidence bounds |
| `prediction_lower` | `ndarray \| None` | Lower prediction bounds |
| `prediction_upper` | `ndarray \| None` | Upper prediction bounds |
| `residuals` | `ndarray \| None` | Residuals (if `"residuals"` was requested) |
| `robustness_weights` | `ndarray \| None` | Robustness weights (if `"weights"` was requested) |
| `cv_scores` | `ndarray \| None` | CV score per tested fraction |
| `diagnostics` | `Diagnostics \| None` | Fit metrics (if `"diagnostics"` was requested) |
| `enp` | `float \| None` | Equivalent number of parameters (if `"se"` was requested) |
| `trace_hat` | `float \| None` | Trace of hat matrix (if `"se"` was requested) |
| `delta1` | `float \| None` | First delta statistic (if `"se"` was requested) |
| `delta2` | `float \| None` | Second delta statistic (if `"se"` was requested) |
| `residual_scale` | `float \| None` | Residual scale estimate (if `"se"` was requested) |
| `leverage` | `ndarray \| None` | Per-point hat-matrix diagonal (if `"se"` was requested) |
| `gradient` | `ndarray \| None` | Per-point local fit gradient, flattened (if `"gradient"` was requested, `surface_mode="direct"` only) |
| `dimensions` | `int` | Number of predictor dimensions |

### `Diagnostics`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `float` | Root Mean Squared Error |
| `mae` | `float` | Mean Absolute Error |
| `r_squared` | `float` | R-squared |
| `residual_sd` | `float` | Robust residual scale estimate (`1.4826 * MAD`) |
| `effective_df` | `float \| None` | Effective degrees of freedom (`None` if not computed) |
| `aic` | `float \| None` | AIC (`None` if not computed) |
| `aicc` | `float \| None` | AICc (`None` if not computed) |

## Predict

### `LoessResult.predict(new_x, ...) -> PredictOutput`

Evaluates the fitted model at out-of-sample query points (flattened, `dimensions` values per point). Requires `retain_model=True` on the constructor before `fit()`, otherwise raises `LoessError`.

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
