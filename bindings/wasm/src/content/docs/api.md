---
title: fastLoess WebAssembly API Reference
---
The WebAssembly bindings provide a high-performance interface to the core Rust library, mirroring the Rust API structure.

> **StreamingLoess** and **OnlineLoess** are documented separately: [wasm-streaming.md](api-streaming.md), [wasm-online.md](api-online.md)

## Classes and Functions

### `Loess`

The `Loess` class is the main entry point for batch smoothing.

**Constructor:**

```javascript
const { Loess } = require('fastloess-wasm');

const model = new Loess({ fraction: 0.5 });
console.log("typeof fit:", typeof model.fit);
```

```output
typeof fit: function
```

* `options`: An object containing `LoessOptions` fields.

**Methods:**

```javascript
const { Loess } = require('fastloess-wasm');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, xi => Math.sin(xi) + 0.1);

const model = new Loess({ fraction: 0.5 });
const result = model.fit(x, y);
console.log("Fraction used:", result.fraction_used);
console.log("Iterations used:", result.iterations_used);
// or with per-observation weights:
const weights = new Float64Array(n).fill(1);
const resultWeighted = model.fit(x, y, weights);
```

```output
Fraction used: 0.5
Iterations used: 3
```

* `x`: `Float64Array` of input x values.
* `y`: `Float64Array` of input y values.
* Returns: A `LoessResult` object.

See [wasm-streaming.md](api-streaming.md) for the `StreamingLoess` class.

See [wasm-online.md](api-online.md) for the `OnlineLoess` class.

## Options Structures

### `LoessOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `number` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `number` | `3` | Number of robustifying iterations |
| `weight_function` | `string` | `"tricube"` | Kernel weight function |
| `robustness_method` | `string` | `"bisquare"` | Robustness method |
| `scaling_method` | `string` | `"mad"` | Residual scaling method |
| `boundary_policy` | `string` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback` | `string` | `"use_local_mean"` | Zero-weight handling |
| `auto_converge` | `number` | `null` | Auto-convergence tolerance |
| `custom_weights` | `number[]` | `null` | Per-observation case weights — passed to `fit()`, not the options object (Batch only) |
| `confidence_intervals` | `number` | `null` | Confidence level (e.g., 0.95) |
| `prediction_intervals` | `number` | `null` | Prediction level (e.g., 0.95) |
| `return_diagnostics` | `boolean` | `false` | Compute RMSE, MAE, R², AIC |
| `return_residuals` | `boolean` | `false` | Include residuals in result |
| `return_robustness_weights` | `boolean` | `false` | Include robustness weights in result |
| `return_se` | `boolean` | `false` | Compute hat-matrix statistics (enp, leverage …) |
| `parallel` | `boolean` | `true` | Enable parallel execution |
| `degree` | `string` | `"linear"` | Polynomial degree of local fit |
| `dimensions` | `number` | `1` | Number of predictor dimensions |
| `distance_metric` | `string` | `"normalized"` | Distance metric; use `"minkowski:p"` for custom p |
| `weighted_metric_weights` | `number[]` | `null` | Per-dimension weights (used when `distance_metric = "weighted"`) |
| `surface_mode` | `string` | `"interpolation"` | Surface computation mode |
| `cell` | `number` | `null` | Cell size for interpolation grid (smaller → more vertices, higher accuracy) |
| `interpolation_vertices` | `number` | `null` | Number of interpolation vertices |
| `boundary_degree_fallback` | `boolean` | `null` | Fall back to lower polynomial degree at boundaries when higher degrees fail |
| `cv_method` | `string` | `"kfold"` | CV method (`"kfold"` or `"loocv"`) (Batch only) |
| `cv_k` | `number` | `5` | Number of folds for k-fold CV (Batch only) |
| `cv_fractions` | `number[]` | `null` | Fractions to test for cross-validation (Batch only) |
| `cv_seed` | `number` | `null` | Random seed for cross-validation shuffling (Batch only) |

See [wasm-streaming.md](api-streaming.md) for `StreamingOptions`.

See [wasm-online.md](api-online.md) for `OnlineOptions`.

## Result Structure

See [wasm-online.md](api-online.md) for `OnlineOutput`.

### `LoessResult`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Float64Array` | Sorted x values |
| `y` | `Float64Array` | Smoothed y values |
| `fraction_used` | `number` | Fraction used (set or selected by CV) |
| `iterations_used` | `number` \| `undefined` | Robustness iterations actually performed |
| `standard_errors` | `Float64Array` \| `undefined` | Per-point SE (if `return_se`) |
| `confidence_lower` | `Float64Array` \| `undefined` | Lower confidence bounds |
| `confidence_upper` | `Float64Array` \| `undefined` | Upper confidence bounds |
| `prediction_lower` | `Float64Array` \| `undefined` | Lower prediction bounds |
| `prediction_upper` | `Float64Array` \| `undefined` | Upper prediction bounds |
| `residuals` | `Float64Array` \| `undefined` | Residuals (if `return_residuals`) |
| `robustness_weights` | `Float64Array` \| `undefined` | Robustness weights (if `return_robustness_weights`) |
| `cv_scores` | `Float64Array` \| `undefined` | CV score per tested fraction |
| `diagnostics` | `Diagnostics` \| `undefined` | Fit metrics (if `return_diagnostics`) |
| `enp` | `number` \| `undefined` | Equivalent number of parameters (if `return_se`) |
| `trace_hat` | `number` \| `undefined` | Trace of hat matrix (if `return_se`) |
| `delta1` | `number` \| `undefined` | First delta statistic (if `return_se`) |
| `delta2` | `number` \| `undefined` | Second delta statistic (if `return_se`) |
| `residual_scale` | `number` \| `undefined` | Residual scale estimate (if `return_se`) |
| `leverage` | `Float64Array` \| `undefined` | Per-point hat-matrix diagonal (if `return_se`) |
| `dimensions` | `number` | Number of predictor dimensions |

### `Diagnostics`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `number` | Root Mean Squared Error |
| `mae` | `number` | Mean Absolute Error |
| `r_squared` | `number` | R-squared |
| `residual_sd` | `number` | Residual standard deviation |
| `effective_df` | `number` \| `undefined` | Effective degrees of freedom |
| `aic` | `number` \| `undefined` | AIC |
| `aicc` | `number` \| `undefined` | AICc |

## Options

### weight_function

*See: [Weight Functions](kernels.md)*

* `"tricube"` (default)
* `"epanechnikov"`
* `"gaussian"`
* `"uniform"` (alias: `"boxcar"`)
* `"biweight"` (alias: `"bisquare"`)
* `"triangle"` (alias: `"triangular"`)
* `"cosine"`

### robustness_method

*See: [Robustness](robustness.md)*

* `"bisquare"` (default; alias: `"biweight"`)
* `"huber"`
* `"talwar"`

### boundary_policy

*See: [Boundary Handling](boundary.md)*

* `"extend"` (default; alias: `"pad"`)
* `"reflect"` (alias: `"mirror"`)
* `"zero"`
* `"noboundary"` (alias: `"none"`)

### scaling_method

*See: [Scaling Methods](scaling.md)*

* `"mad"` (default; alias: `"median_absolute_deviation"`)
* `"mar"` (alias: `"median_absolute_residual"`)
* `"mean"` (alias: `"mean_absolute_residual"`)

### zero_weight_fallback

*See: [Parameters](parameters.md)*

* `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`)
* `"return_original"` (alias: `"original"`)
* `"return_none"` (alias: `"none"`)

### degree

*See: [Polynomial Degree](degree.md)*

* `"constant"` or `"0"` (degree 0)
* `"linear"` or `"1"` (default, degree 1)
* `"quadratic"` or `"2"` (degree 2)
* `"cubic"` or `"3"` (degree 3)
* `"quartic"` or `"4"` (degree 4)

### distance_metric

*See: [Multivariate LOESS](dimensions.md)*

* `"normalized"` (default — scales each dimension by its range; alias: `"norm"`)
* `"euclidean"` (alias: `"euclid"`)
* `"manhattan"` (alias: `"l1"`)
* `"chebyshev"` (alias: `"linf"`)
* `"minkowski"` (Euclidean when no suffix; use `"minkowski:p"` for custom p, e.g. `"minkowski:3"`)
* `"weighted"` plus `weighted_metric_weights` for per-dimension scaling (alias: `"weighted_euclidean"`)

### surface_mode

*See: [Parameters](parameters.md)*

* `"interpolation"` (default — faster, uses a spatial grid)
* `"direct"` (fits every point exactly; slower but more accurate)

### merge_strategy

See [wasm-streaming.md](api-streaming.md).

### update_mode

See [wasm-online.md](api-online.md).

## Example

```javascript
const { Loess } = require('fastloess-wasm');

const x = new Float64Array([1, 2, 3, 4, 5]);
const y = new Float64Array([2.1, 4.0, 6.2, 8.0, 10.1]);

// Fit data
const model = new Loess({ fraction: 0.5 });
const result = model.fit(x, y);

console.log("Smoothed Y:", result.y);
```

```output
Smoothed Y: Float64Array(5) [ 2.1, 4, 6.2, 8, 10.1 ]
```
