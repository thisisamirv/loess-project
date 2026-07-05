# fastLoess Python API Reference

The Python bindings provide a high-performance interface to the core Rust library, mirroring the Rust API structure.

## Classes

### `Loess`

The `Loess` class allows configuring the LOESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

```python
model = fastloess.Loess(**kwargs)
```

* `kwargs`: Keyword arguments corresponding to `LoessOptions` fields.

**Methods:**

```python
result = model.fit(x, y)
```

* Fits the model to the provided `x` and `y` array-like objects.
* Returns a `LoessResult` object containing the smoothed values and optional diagnostics.

### `StreamingLoess`

The `StreamingLoess` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```python
stream = fastloess.StreamingLoess(**kwargs)
```

* `kwargs`: Keyword arguments corresponding to `LoessOptions` and `StreamingOptions` fields.

**Methods:**

```python
partial_result = stream.process_chunk(x, y)
```

* Processes a chunk of data. Returns partial results.

```python
final_result = stream.finalize()
```

* Finalizes the smoothing process and returns any remaining buffered results.

### `OnlineLoess`

The `OnlineLoess` class updates the model incrementally with new data points.

**Constructor:**

```python
online = fastloess.OnlineLoess(**kwargs)
```

* `kwargs`: Keyword arguments corresponding to `LoessOptions` and `OnlineOptions` fields.

**Methods:**

```python
result = online.add_points(x, y)
```

* Adds new points to the model and returns the smoothed values (retrospective or prospective depending on mode).

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
| `zero_weight_fallback` | `str` | `"use_local_mean"` | Zero-weight handling strategy |
| `auto_converge` | `float` | `None` | Auto-convergence tolerance |
| `custom_weights` | `list[float]` | `None` | Per-observation case weights (Batch only) |
| `confidence_intervals` | `float` | `None` | Confidence level (e.g., 0.95) |
| `prediction_intervals` | `float` | `None` | Prediction level (e.g., 0.95) |
| `return_diagnostics` | `bool` | `False` | Compute RMSE, MAE, R², AIC |
| `return_residuals` | `bool` | `False` | Include residuals in result |
| `return_robustness_weights` | `bool` | `False` | Include robustness weights in result |
| `return_se` | `bool` | `False` | Compute hat-matrix statistics (enp, leverage …) |
| `parallel` | `bool` | `True` | Enable parallel execution |
| `degree` | `str` | `"linear"` | Polynomial degree of local fit |
| `dimensions` | `int` | `1` | Number of predictor dimensions |
| `distance_metric` | `str` | `"normalized"` | Distance metric; use `"minkowski:p"` for custom p |
| `surface_mode` | `str` | `"interpolation"` | Surface computation mode |
| `minkowski_p` | `float` | `2.0` | Minkowski exponent (used when `distance_metric="minkowski"`) |
| `weighted_metric_weights` | `list[float]` | `None` | Per-dimension weights (used when `distance_metric="weighted"`) |
| `cell` | `float` | `None` | Cell size for interpolation grid (smaller → more vertices, higher accuracy) |
| `interpolation_vertices` | `int` | `None` | Number of interpolation vertices |
| `boundary_degree_fallback` | `bool` | `None` | Fall back to lower polynomial degree at boundaries when higher degrees fail |
| `cv_seed` | `int` | `None` | Random seed for cross-validation shuffling (Batch only) |
| `cv_fractions` | `list[float]` | `None` | Fractions to test for cross-validation |
| `cv_method` | `str` | `"kfold"` | CV method (`"kfold"` or `"loocv"`) |
| `cv_k` | `int` | `5` | Number of folds for k-fold CV |

### `StreamingOptions` (inherits `LoessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size` | `int` | `5000` | Data chunk size |
| `overlap` | `int` | auto (10% of chunk) | Overlap between chunks |
| `merge_strategy` | `str` | `"weighted_average"` | Strategy for blending overlap: see Merge Strategies |

### `OnlineOptions` (inherits `LoessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `window_capacity` | `int` | `100` | Max points in sliding window |
| `min_points` | `int` | `2` | Min points before smoothing starts |
| `update_mode` | `str` | `"full"` | Update mode (`"full"` or `"incremental"`) |

## Result Structure

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
| `effective_df` | `float` | Effective degrees of freedom |
| `aic` | `float` | AIC |
| `aicc` | `float` | AICc |

## String Options

### Weight Functions

* `"tricube"` (default)
* `"epanechnikov"`
* `"gaussian"`
* `"uniform"`
* `"biweight"`
* `"triangle"`
* `"cosine"`

### Robustness Methods

* `"bisquare"` (default)
* `"huber"`
* `"talwar"`

### Boundary Policies

* `"extend"` (default - linear extrapolation)
* `"reflect"`
* `"zero"`
* `"noboundary"`

### Scaling Methods

* `"mad"` (default - Median Absolute Deviation)
* `"mar"` (Median Absolute Residual)
* `"mean"` (Mean Absolute Residual)

### Zero Weight Fallback

* `"use_local_mean"` (default)
* `"return_original"`
* `"return_none"`

### Polynomial Degrees

* `"constant"` (degree 0)
* `"linear"` (default, degree 1)
* `"quadratic"` (degree 2)
* `"cubic"` (degree 3)
* `"quartic"` (degree 4)

### Distance Metrics

* `"normalized"` (default — scales each dimension by its range)
* `"euclidean"`
* `"manhattan"`
* `"chebyshev"`
* `"minkowski"` (set `minkowski_p` for custom exponent)
* `"weighted"` (set `weighted_metric_weights` for per-dimension scaling)

### Surface Modes

* `"interpolation"` (default — faster, uses a spatial grid)
* `"direct"` (fits every point exactly; slower but more accurate)

### Merge Strategies (Streaming)

* `"weighted_average"` (default — weighted blend of overlapping regions)
* `"average"` (simple mean of overlapping regions)
* `"take_first"` (keep values from the earlier chunk)
* `"take_last"` (keep values from the later chunk)

### Update Modes (Online)

* `"full"` (default — re-smooth entire window each update)
* `"incremental"` (faster, O(1) incremental update)

## Example

```python
from fastloess import Loess
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.2, 100)

# Configure model
model = Loess(fraction=0.5)

# Fit data
result = model.fit(x, y)

print("Smoothed Y:", result.y)
```
