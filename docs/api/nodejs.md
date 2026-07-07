# fastLoess Node.js API Reference

The Node.js bindings provide a high-performance interface to the core Rust library, mirroring the Rust API structure.

## Classes

### `Loess`

The `Loess` class allows configuring the LOESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

```javascript
const model = new Loess(options);
```

* `options`: An object containing `LoessOptions` fields.

**Methods:**

```javascript
const result = model.fit(x, y, custom_weights);
```

* Fits the model to the provided `x` and `y` typed arrays.
* Returns a `LoessResult` object containing the smoothed values and optional diagnostics.

```javascript
const result = await model.fitAsync(x, y, custom_weights);
```

* Async variant of `fit()`. Returns a `Promise` that resolves to a `LoessResult`.

### `StreamingLoess`

The `StreamingLoess` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```javascript
const stream = new StreamingLoess(options, streamingOptions);
```

* `options`: An object containing `LoessOptions` fields.
* `streamingOptions`: An object containing `StreamingOptions` fields.

**Methods:**

```javascript
const partialResult = stream.processChunk(x, y);
```

* Processes a chunk of data. Returns partial results.

```javascript
const finalResult = stream.finalize();
```

* Finalizes the smoothing process and returns any remaining buffered results.

### `OnlineLoess`

The `OnlineLoess` class updates the model incrementally with new data points.

**Constructor:**

```javascript
const online = new OnlineLoess(options, onlineOptions);
```

* `options`: An object containing `LoessOptions` fields.
* `onlineOptions`: An object containing `OnlineOptions` fields.

**Methods:**

```javascript
const result = online.add_point(x, y);  // returns OnlineOutput | null
```

* Adds a single point to the sliding window and returns an `OnlineOutput` once enough points are available, or `null` while the window is still filling.

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
| `zero_weight_fallback` | `string` | `"use_local_mean"` | Zero-weight handling strategy |
| `auto_converge` | `number` | `null` | Auto-convergence tolerance |
| `custom_weights` | `number[]` | `null` | Per-observation case weights — passed to `fit()`/`fitAsync()`, not the options object (Batch only) |
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
| `surface_mode` | `string` | `"interpolation"` | Surface computation mode |
| `weighted_metric_weights` | `number[]` | `null` | Per-dimension weights (used when `distance_metric = "weighted"`) |
| `cell` | `number` | `null` | Cell size for interpolation grid (smaller → more vertices, higher accuracy) |
| `interpolation_vertices` | `number` | `null` | Number of interpolation vertices |
| `boundary_degree_fallback` | `boolean` | `null` | Fall back to lower polynomial degree at boundaries when higher degrees fail |
| `cv_seed` | `number` | `null` | Random seed for cross-validation shuffling (Batch only) |
| `cv_fractions` | `number[]` | `null` | Fractions to test for cross-validation |
| `cv_method` | `string` | `"kfold"` | CV method (`"kfold"` or `"loocv"`) |
| `cv_k` | `number` | `5` | Number of folds for k-fold CV |

### `StreamingOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size` | `number` | `5000` | Data chunk size |
| `overlap` | `number` | `500` | Overlap between chunks |
| `merge_strategy` | `string` | `"weighted_average"` | Strategy for blending overlap: see Merge Strategies |

### `OnlineOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `window_capacity` | `number` | `100` | Max points in sliding window |
| `min_points` | `number` | `2` | Min points before smoothing starts |
| `update_mode` | `string` | `"full"` | Update mode (`"full"` or `"incremental"`) |
| `parallel` | `boolean` | `false` | Enable parallel execution (off by default; online LOESS fits one point at a time) |

## Result Structure

### `OnlineOutput`

Returned by `add_point()` once the window has enough points (`null` until then). Field names are camelCase per JavaScript conventions.

| Field | Type | Description |
| --- | --- | --- |
| `smoothed` | `number` | Smoothed value for the latest point |
| `stdError` | `number \| null` | Standard error (if requested) |
| `residual` | `number \| null` | Residual y − smoothed (if requested) |
| `robustnessWeight` | `number \| null` | Robustness weight (if requested) |
| `iterationsUsed` | `number \| null` | Robustness iterations performed |

### `LoessResult`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Float64Array` | Sorted x values |
| `y` | `Float64Array` | Smoothed y values |
| `fractionUsed` | `number` | Fraction used (set or selected by CV) |
| `iterationsUsed` | number \| null | Robustness iterations actually performed |
| `standardErrors` | Float64Array \| null | Per-point SE (if `return_se`) |
| `confidenceLower` | Float64Array \| null | Lower confidence bounds |
| `confidenceUpper` | Float64Array \| null | Upper confidence bounds |
| `predictionLower` | Float64Array \| null | Lower prediction bounds |
| `predictionUpper` | Float64Array \| null | Upper prediction bounds |
| `residuals` | Float64Array \| null | Residuals (if `return_residuals`) |
| `robustnessWeights` | Float64Array \| null | Robustness weights (if `return_robustness_weights`) |
| `cvScores` | Float64Array \| null | CV score per tested fraction |
| `diagnostics` | Diagnostics \| null | Fit metrics (if `return_diagnostics`) |
| `enp` | number \| null | Equivalent number of parameters (if `return_se`) |
| `traceHat` | number \| null | Trace of hat matrix (if `return_se`) |
| `delta1` | number \| null | First delta statistic (if `return_se`) |
| `delta2` | number \| null | Second delta statistic (if `return_se`) |
| `residualScale` | number \| null | Residual scale estimate (if `return_se`) |
| `leverage` | Float64Array \| null | Per-point hat-matrix diagonal (if `return_se`) |
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

* `"tricube"` (default)
* `"epanechnikov"`
* `"gaussian"`
* `"uniform"` (alias: `"boxcar"`)
* `"biweight"` (alias: `"bisquare"`)
* `"triangle"` (alias: `"triangular"`)
* `"cosine"`

### robustness_method

* `"bisquare"` (default; alias: `"biweight"`)
* `"huber"`
* `"talwar"`

### boundary_policy

* `"extend"` (default; alias: `"pad"`)
* `"reflect"` (alias: `"mirror"`)
* `"zero"`
* `"noboundary"` (alias: `"none"`)

### scaling_method

* `"mad"` (default; alias: `"median_absolute_deviation"`)
* `"mar"` (alias: `"median_absolute_residual"`)
* `"mean"` (alias: `"mean_absolute_residual"`)

### zero_weight_fallback

* `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`)
* `"return_original"` (alias: `"original"`)
* `"return_none"` (alias: `"none"`)

### degree

* `"constant"` or `"0"` (degree 0)
* `"linear"` or `"1"` (default, degree 1)
* `"quadratic"` or `"2"` (degree 2)
* `"cubic"` or `"3"` (degree 3)
* `"quartic"` or `"4"` (degree 4)

### distance_metric

* `"normalized"` (default — scales each dimension by its range; alias: `"norm"`)
* `"euclidean"` (alias: `"euclid"`)
* `"manhattan"` (alias: `"l1"`)
* `"chebyshev"` (alias: `"linf"`)
* `"minkowski"` (use `"minkowski:p"` string for custom exponent, e.g. `"minkowski:3"`)
* `"weighted"` plus `weighted_metric_weights` for per-dimension scaling (alias: `"weighted_euclidean"`)

### surface_mode

* `"interpolation"` (default — faster, uses a spatial grid)
* `"direct"` (fits every point exactly; slower but more accurate)

### merge_strategy

* `"weighted_average"` (default; alias: `"weighted"`)
* `"average"` (alias: `"mean"`)
* `"take_first"` (alias: `"first"`)
* `"take_last"` (alias: `"last"`)

### update_mode

* `"full"` (default; alias: `"resmooth"`)
* `"incremental"` (alias: `"single"`)

## Example

```javascript
const { Loess } = require('fastloess');

const x = new Float64Array([1, 2, 3, 4, 5]);
const y = new Float64Array([2.1, 4.0, 6.2, 8.0, 10.1]);

// Configure model
const model = new Loess({ fraction: 0.5 });

// Fit data
const result = model.fit(x, y);

console.log("Smoothed Y:", result.y);
```
