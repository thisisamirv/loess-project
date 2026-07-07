# fastLoess R API Reference

The R bindings provide a high-performance interface to the core Rust library, mirroring the Rust API structure.

## Classes

### `Loess`

The `Loess` class allows configuring the LOESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

```r
model <- Loess(fraction = 0.5)
```

* `...`: Arguments corresponding to `LoessOptions` fields.

**Methods:**

```r
model <- Loess()
result <- model$fit(x, y)
# or with per-observation weights:
result <- model$fit(x, y, custom_weights = weights)
```

* Fits the model to the provided `x` and `y` numeric vectors.
* Returns a `LoessResult` S3 object containing the smoothed values and optional diagnostics.
* `print(model)`: Displays the model configuration.

### `StreamingLoess`

The `StreamingLoess` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```r
stream <- StreamingLoess(chunk_size = 10L)
```

* `...`: Arguments corresponding to `LoessOptions` and `StreamingOptions` fields.

**Methods:**

```r
stream <- StreamingLoess(chunk_size = 10L)
partial_result <- stream$process_chunk(x, y)
```

* Processes a chunk of data. Returns partial results.

```r
stream <- StreamingLoess(chunk_size = 10L)
stream$process_chunk(x, y)
final_result <- stream$finalize()
```

* Finalizes the smoothing process and returns any remaining buffered results.

### `OnlineLoess`

The `OnlineLoess` class updates the model incrementally with new data points.

**Constructor:**

```r
online <- OnlineLoess(window_capacity = 10L)
```

* `...`: Arguments corresponding to `LoessOptions` and `OnlineOptions` fields.

**Methods:**

```r
online <- OnlineLoess(window_capacity = 10L)
result <- online$add_points(x, y)
```

* Adds new points to the model and returns the smoothed values (retrospective or prospective depending on mode).

## Options Structures

### `LoessOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `numeric` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `integer` | `3` | Number of robustifying iterations |
| `weight_function` | `character` | `"tricube"` | Kernel weight function |
| `robustness_method` | `character` | `"bisquare"` | Robustness method |
| `scaling_method` | `character` | `"mad"` | Residual scaling method |
| `boundary_policy` | `character` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback` | `character` | `"use_local_mean"` | Zero-weight handling strategy |
| `auto_converge` | `numeric` | `NULL` | Auto-convergence tolerance |
| `custom_weights` | `numeric` | `NULL` | Per-observation case weights — passed to `$fit()`, not the constructor (Batch only) |
| `confidence_intervals` | `numeric` | `NULL` | Confidence level (e.g., 0.95) |
| `prediction_intervals` | `numeric` | `NULL` | Prediction level (e.g., 0.95) |
| `return_diagnostics` | `logical` | `FALSE` | Compute RMSE, MAE, R², AIC |
| `return_residuals` | `logical` | `FALSE` | Include residuals in result |
| `return_robustness_weights` | `logical` | `FALSE` | Include robustness weights in result |
| `return_se` | `logical` | `FALSE` | Compute hat-matrix statistics (enp, leverage …) |
| `parallel` | `logical` | `TRUE` | Enable parallel execution |
| `degree` | `character` | `"linear"` | Polynomial degree of local fit |
| `dimensions` | `integer` | `1L` | Number of predictor dimensions |
| `distance_metric` | `character` | `"normalized"` | Distance metric; use `"minkowski:p"` for custom p |
| `surface_mode` | `character` | `"interpolation"` | Surface computation mode |
| `weighted_metric_weights` | `numeric` | `NULL` | Per-dimension weights (used when `distance_metric = "weighted"`) |
| `cell` | `numeric` | `NULL` | Cell size for interpolation grid (smaller → more vertices, higher accuracy) |
| `interpolation_vertices` | `integer` | `NULL` | Number of interpolation vertices |
| `boundary_degree_fallback` | `logical` | `NULL` | Fall back to lower polynomial degree at boundaries when higher degrees fail |
| `cv_seed` | `integer` | `NULL` | Random seed for cross-validation shuffling (Batch only) |
| `cv_fractions` | `numeric` | `NULL` | Fractions to test for cross-validation (Batch only) |
| `cv_method` | `character` | `"kfold"` | CV method (`"kfold"` or `"loocv"`) (Batch only) |
| `cv_k` | `integer` | `5L` | Number of folds for k-fold CV (Batch only) |

### `StreamingOptions` (inherits `LoessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size` | `integer` | `5000L` | Data chunk size |
| `overlap` | `integer` | auto (10% of chunk) | Overlap between chunks (`NULL` for auto) |
| `merge_strategy` | `character` | `"weighted_average"` | Strategy for blending overlap: see Merge Strategies |

### `OnlineOptions` (inherits `LoessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `window_capacity` | `integer` | `100L` | Max points in sliding window |
| `min_points` | `integer` | `2L` | Min points before smoothing starts |
| `update_mode` | `character` | `"full"` | Update mode (`"full"` or `"incremental"`) |
| `parallel` | `logical` | `FALSE` | Enable parallel execution (off by default; online LOESS fits one point at a time) |

## Result Structure

### `LoessResult`

An S3 list with class `"LoessResult"` containing:

**Supported S3 Methods:** `print(result)`, `plot(result)`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `numeric` | Sorted x values |
| `y` | `numeric` | Smoothed y values |
| `fraction_used` | `numeric` | Fraction used (set or selected by CV) |
| `iterations_used` | `integer \| NULL` | Robustness iterations actually performed |
| `standard_errors` | `numeric \| NULL` | Per-point SE (if `return_se`) |
| `confidence_lower` | `numeric \| NULL` | Lower confidence bounds |
| `confidence_upper` | `numeric \| NULL` | Upper confidence bounds |
| `prediction_lower` | `numeric \| NULL` | Lower prediction bounds |
| `prediction_upper` | `numeric \| NULL` | Upper prediction bounds |
| `residuals` | `numeric \| NULL` | Residuals (if `return_residuals`) |
| `robustness_weights` | `numeric \| NULL` | Robustness weights (if `return_robustness_weights`) |
| `cv_scores` | `numeric \| NULL` | CV score per tested fraction |
| `diagnostics` | `list \| NULL` | Fit metrics (if `return_diagnostics`) |
| `enp` | `numeric \| NULL` | Equivalent number of parameters (if `return_se`) |
| `trace_hat` | `numeric \| NULL` | Trace of hat matrix (if `return_se`) |
| `delta1` | `numeric \| NULL` | First delta statistic (if `return_se`) |
| `delta2` | `numeric \| NULL` | Second delta statistic (if `return_se`) |
| `residual_scale` | `numeric \| NULL` | Residual scale estimate (if `return_se`) |
| `leverage` | `numeric \| NULL` | Per-point hat-matrix diagonal (if `return_se`) |
| `dimensions` | `integer` | Number of predictor dimensions |

### `Diagnostics`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `numeric` | Root Mean Squared Error |
| `mae` | `numeric` | Mean Absolute Error |
| `r_squared` | `numeric` | R-squared |
| `residual_sd` | `numeric` | Residual standard deviation |
| `effective_df` | `numeric` | Effective degrees of freedom (NaN if not computed) |
| `aic` | `numeric` | AIC (NaN if not computed) |
| `aicc` | `numeric` | AICc (NaN if not computed) |

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
* `"minkowski"` (Euclidean when no suffix; use `"minkowski:p"` for custom p, e.g. `"minkowski:3"`)
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

```r
library(rfastloess)

x <- seq(0, 10, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.2)

# Configure model
model <- Loess(fraction = 0.5)

# Fit data
result <- model$fit(x, y)

# Print summary
print(result)

# Plot result
plot(result)
```
