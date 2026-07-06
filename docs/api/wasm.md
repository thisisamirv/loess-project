# fastLoess WebAssembly API Reference

The WebAssembly bindings provide a high-performance interface to the core Rust library, mirroring the Rust API structure.

## Classes and Functions

### `Loess`

The `Loess` class is the main entry point for batch smoothing.

**Constructor:**

```javascript
const model = new Loess(options);
```

* `options`: An object containing `LoessOptions` fields.

**Methods:**

```javascript
const result = model.fit(x, y);
```

* `x`: `Float64Array` of input x values.
* `y`: `Float64Array` of input y values.
* Returns: A `LoessResult` object.

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
const smoothed = online.update(x, y);
```

* Adds a **single** point `(x, y)` to the sliding window.
* Returns `number | null` — the smoothed value for the current point, or `null` if the window has fewer than `min_points`.

> **Note:** For batch addition of multiple points, call `update()` in a loop. There is no `addPoints()` equivalent in the WASM binding.

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
| `custom_weights` | `number[]` | `null` | Per-observation case weights (Batch only) |
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
| `cvK` | `number` | `5` | Number of folds for k-fold CV |

### `StreamingOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size` | `number` | `5000` | Data chunk size |
| `overlap` | `number` | auto (10% of chunk) | Overlap between chunks |
| `merge_strategy` | `string` | `"weighted_average"` | Strategy for blending overlap: see Merge Strategies |

### `OnlineOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `window_capacity` | `number` | `100` | Max points in sliding window |
| `min_points` | `number` | `2` | Min points before smoothing starts |
| `update_mode` | `string` | `"full"` | Update mode (`"full"` or `"incremental"`) |

## Result Structure

### `LoessResult`

> **Note:** `LoessResult` exposes `x`, `y`, `fractionUsed`, `iterationsUsed`, `standardErrors`, `confidenceLower/Upper`, `predictionLower/Upper`, `residuals`, `robustnessWeights`, `cvScores`, and `diagnostics`.
> Hat-matrix statistics (`enp`, `traceHat`, `delta1`, `delta2`, `residualScale`, `leverage`) are not yet exposed by the WASM binding; use the Node.js or Rust binding for those.

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

### `Diagnostics`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `number` | Root Mean Squared Error |
| `mae` | `number` | Mean Absolute Error |
| `r_squared` | `number` | R-squared |
| `residual_sd` | `number` | Residual standard deviation |
| `effective_df` | `number` | Effective degrees of freedom |
| `aic` | `number` | AIC |
| `aicc` | `number` | AICc |

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
* `"minkowski"` (Euclidean when no suffix; use `"minkowski:p"` for custom p, e.g. `"minkowski:3"`)
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

```javascript
import init, { Loess } from 'fastloess-wasm';

await init();

const x = new Float64Array([1, 2, 3, 4, 5]);
const y = new Float64Array([2.1, 4.0, 6.2, 8.0, 10.1]);

// Fit data
const result = new Loess({ fraction: 0.5 }).fit(x, y);

console.log("Smoothed Y:", result.y);
```
