---
title: StreamingLoess — Node.js API Reference
---
See also: [fastLoess](api.md)

## When to Use

- Dataset >100,000 points
- Memory-constrained environments
- Batch processing pipelines

## Class

### `StreamingLoess`

The `StreamingLoess` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```javascript
const { StreamingLoess } = require('fastloess');

const stream = new StreamingLoess({ fraction: 0.5 }, { chunk_size: 50, overlap: 10 });
const x = Float64Array.from({ length: 10 }, (_, i) => i);
const y = Float64Array.from({ length: 10 }, (_, i) => i * 0.5);
stream.process_chunk(x, y);
const result = stream.finalize();
console.log("Smoothed", result.y.length, "points via streaming");
```

```output
Smoothed 10 points via streaming
```

- `options`: An object containing `StreamingSmoothOptions` fields (a subset of the Batch `LoessOptions` fields — see below).
- `streamingOptions`: An object containing `StreamingOptions` fields.

#### `process_chunk(x, y)`

Processes a chunk of data. Returns partial results.

```javascript
const { StreamingLoess } = require('fastloess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, xi => Math.sin(xi) + 0.1);

const stream = new StreamingLoess({ fraction: 0.5 }, { chunk_size: 50, overlap: 10 });
const partialResult = stream.process_chunk(x.slice(0, 50), y.slice(0, 50));
console.log("Fraction used:", partialResult.fraction_used);
```

```output
Fraction used: 0.5
```

#### `finalize()`

Finalizes the smoothing process and returns any remaining buffered results.

```javascript
const { StreamingLoess } = require('fastloess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, xi => Math.sin(xi) + 0.1);

const stream = new StreamingLoess({ fraction: 0.5 }, { chunk_size: 50, overlap: 10 });
stream.process_chunk(x.slice(0, 50), y.slice(0, 50));
stream.process_chunk(x.slice(50), y.slice(50));
const finalResult = stream.finalize();
console.log("Fraction used:", finalResult.fraction_used);
```

```output
Fraction used: 0.5
```

## Options Structures

### `StreamingSmoothOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `number` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `number` | `3` | Number of robustifying iterations |
| `weight_function` | `string` | `"tricube"` | Weight function name |
| `robustness_method` | `string` | `"bisquare"` | Robustness method name |
| `scaling_method` | `string` | `"mad"` | Residual scaling method |
| `boundary_policy` | `string` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback` | `string` | `"use_local_mean"` | Zero-weight handling strategy |
| `auto_converge` | `number` | `null` | Auto-convergence tolerance |
| `return_diagnostics` | `boolean` | `false` | Include diagnostics in result |
| `return_residuals` | `boolean` | `false` | Include residuals in result |
| `return_robustness_weights` | `boolean` | `false` | Include weights in result |
| `parallel` | `boolean` | `true` | Enable parallel execution |
| `degree` | `string` | `"linear"` | Polynomial degree of local fit |
| `dimensions` | `number` | `1` | Number of predictor dimensions |
| `distance_metric` | `string` | `"normalized"` | Distance metric; use `"minkowski:p"` for custom p |
| `weighted_metric_weights` | `number[]` | `null` | Per-dimension weights (used when `distance_metric = "weighted"`) |
| `surface_mode` | `string` | `"interpolation"` | Surface computation mode |
| `cell` | `number` | `null` | Cell size for interpolation grid (smaller → more vertices, higher accuracy) |
| `interpolation_vertices` | `number` | `null` | Number of interpolation vertices |
| `boundary_degree_fallback` | `boolean` | `null` | Fall back to lower polynomial degree at boundaries when higher degrees fail |

Confidence/prediction intervals, standard errors, and cross-validation are Batch-only and not available here; see [fastLoess](api.md) for those.

### `StreamingOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size` | `number` | `5000` | Data chunk size |
| `overlap` | `number` | `chunk_size / 10` | Overlap between chunks |
| `merge_strategy` | `string` | `"weighted_average"` | Strategy for blending overlap regions |

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

Convergence tolerance for early stopping of robustness iterations. `null` (default) disables early stopping.

### return_diagnostics

Include a `Diagnostics` object (RMSE, MAE, R², residual_sd) in the result. `effective_df`/`aic`/`aicc` require standard errors, which are Batch-only, so they're always `null` here.

- `false` (default) — leaves `result.diagnostics` as `null`
- `true` — populates `result.diagnostics`

### return_residuals

Include per-point residuals (`y - fitted`) in the result.

- `false` (default) — leaves `result.residuals` as `null`
- `true` — populates `result.residuals`

### return_robustness_weights

Include the final per-point robustness weights (from the last robustness iteration) in the result.

- `false` (default) — leaves `result.robustness_weights` as `null`
- `true` — populates `result.robustness_weights`

### parallel

Enable multi-threaded execution via Rayon.

- `true` (default) — parallelizes the local regression fits across CPU cores
- `false` — forces single-threaded execution

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

Per-dimension weights, one per dimension declared in `dimensions`. Only used when `distance_metric = "weighted"`; setting `distance_metric = "weighted"` without providing this raises an error.

- `null` (default) — has no effect unless `distance_metric = "weighted"` is set
- A `number[]` of per-dimension weights, required when `distance_metric = "weighted"`

### surface_mode

*See: [Polynomial Degree](../advanced/degree.md#surface-mode)*

Controls whether the local polynomial is evaluated at every query point or at a sparser grid of anchor vertices with Hermite cubic interpolation in between.

| Mode | Behavior | Speed | Accuracy |
| --- | --- | --- | --- |
| `"interpolation"` (default) | Evaluate at vertices, interpolate between | Faster | Slight approximation |
| `"direct"` | Evaluate at every query point | Slower | Full precision |

### cell

Cell size for the interpolation grid, as a fraction of the data range. Smaller values place more vertices (denser grid), improving accuracy at the cost of speed. Only applies when `surface_mode = "interpolation"`.

- `null` (default) — uses the library default (`0.2`)
- Any number in `(0, 1]`

### interpolation_vertices

Caps the maximum number of interpolation vertices, overriding the count implied by `cell`. Only applies when `surface_mode = "interpolation"`.

- `null` (default) — uses the library default (no explicit cap)
- Any integer `>= 1`

### boundary_degree_fallback

Whether to reduce the polynomial degree at boundary vertices when the requested `degree` can't be fit there (e.g., not enough neighbours). Only applies when `surface_mode = "interpolation"`.

- `null` (default) — uses the library default (enabled)
- `true` — falls back to a lower degree at boundaries
- `false` — raises an error instead of silently falling back

### chunk_size

Number of points processed per call to `process_chunk()`. Larger chunks reduce per-chunk overhead and give each local fit more surrounding context, at the cost of higher peak memory; smaller chunks bound memory tightly but increase the fraction of points that fall in overlap regions. A good starting point is balancing available memory against how much processing overhead per chunk is acceptable — match it to your file-read buffer or message-batch size to avoid unnecessary copying.

### overlap

Number of points retained from the previous chunk as context, so the neighbourhood at chunk boundaries isn't artificially truncated. Points inside the overlap zone are fitted twice (once by each chunk) and reconciled via `merge_strategy`. A good starting point is 10–20% of `chunk_size`: too little overlap causes visible boundary artefacts, while too much wastes computation refitting the same points twice.

- `null` (default) — computes `chunk_size / 10`, clamped to at least 1 and at most `chunk_size - 10`
- Any integer `>= 1` and `< chunk_size`

### merge_strategy

*See: [Merge Strategies](../advanced/merge.md)*

| Strategy | Alias | Behavior |
| --- | --- | --- |
| `"weighted_average"` (default) | `"weighted"` | Distance-weighted blend |
| `"average"` | `"mean"` | Average overlapping values |
| `"take_first"` | `"first"` | Keep left chunk values |
| `"take_last"` | `"last"` | Keep right chunk values |

![Merge Strategies](../../assets/diagrams/merge_comparison.svg)

## Result Structure

### `LoessResult`

Returned by `process_chunk()` and `finalize()`.

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Float64Array` | x values (same order as input) |
| `y` | `Float64Array` | Smoothed y values |
| `fraction_used` | `number` | Fraction used |
| `iterations_used` | `number \| null` | Robustness iterations actually performed |
| `standard_errors` | `Float64Array \| null` | Always `null` (Batch only) |
| `confidence_lower` | `Float64Array \| null` | Always `null` (Batch only) |
| `confidence_upper` | `Float64Array \| null` | Always `null` (Batch only) |
| `prediction_lower` | `Float64Array \| null` | Always `null` (Batch only) |
| `prediction_upper` | `Float64Array \| null` | Always `null` (Batch only) |
| `residuals` | `Float64Array \| null` | Residuals (if `return_residuals`) |
| `robustness_weights` | `Float64Array \| null` | Robustness weights (if `return_robustness_weights`) |
| `cv_scores` | `Float64Array \| null` | Always `null` (Batch only) |
| `diagnostics` | `Diagnostics \| null` | Fit metrics (if `return_diagnostics`) |
| `dimensions` | `number` | Number of predictor dimensions |

### `Diagnostics`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `number` | Root Mean Squared Error |
| `mae` | `number` | Mean Absolute Error |
| `r_squared` | `number` | R-squared |
| `residual_sd` | `number` | Residual standard deviation |
| `effective_df` | `number \| null` | Always `null` (requires standard errors, Batch only) |
| `aic` | `number \| null` | Always `null` (requires `effective_df`, Batch only) |
| `aicc` | `number \| null` | Always `null` (requires `effective_df`, Batch only) |

---

:::caution[Always call finalize()]
The streaming adapter buffers overlap data. Call `finalize()` after the last chunk to retrieve the buffered tail.
:::
