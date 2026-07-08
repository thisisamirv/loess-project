# fastLoess & loess-rs Rust API Reference

The Rust crates provide the core implementation and high-performance extensions.

## Structs & Usage

Both crates expose the same three entry types via their `prelude`: `Loess` for batch mode, `StreamingLoess` for chunked processing, and `OnlineLoess` for sliding-window updates.

```text
use fastLoess::prelude::*;  // or: use loess_rs::prelude::*;
```

### `Loess` (Batch)

Standard in-memory smoothing.

**Constructor:**

```rust
let model = Loess::new().build()?;
```

**Methods:**

```rust
let model = Loess::new().build()?;
let result = model.fit(&x, &y)?;
```

* Fits the model to the provided `x` and `y` arrays.
* Returns `Result<LoessResult<T>, LoessError>`.

### `StreamingLoess`

Streaming mode for large datasets.

**Constructor:**

```rust
let mut processor = StreamingLoess::new().build()?;
```

**Methods:**

```rust
let mut processor = StreamingLoess::new().build()?;
let result = processor.process_chunk(&x, &y)?;
```

* Processes a chunk of data. Returns `LoessResult<T>` with partial results.

```rust
let mut processor = StreamingLoess::new().build()?;
processor.process_chunk(&x_chunk, &y_chunk)?;
let final_result = processor.finalize()?;
```

* Finalizes processing and returns remaining buffered results.

### `OnlineLoess`

Online mode for real-time data.

**Constructor:**

```rust
let mut processor = OnlineLoess::new().build()?;
```

**Methods:**

```rust
let mut processor = OnlineLoess::new().build()?;
let output = processor.add_point(&[x[0]], y[0])?;
```

* Adds a single point `(x, y)` to the window.
* Returns `Result<Option<OnlineOutput<T>>, LoessError>`.

```rust
let mut processor = OnlineLoess::new().build()?;
processor.reset();
```

* Clears the internal window buffer.

## Builder Configuration

These chained methods configure the builder. They correspond to the "Options Structures" in other bindings.

### Loess Options

| Method | Default | Description |
| --- | --- | --- |
| `fraction(T)` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations(usize)` | `3` | Number of robustifying iterations |
| `weight_function(...)` | `"tricube"` | Kernel weight function |
| `robustness_method(...)` | `"bisquare"` | Robustness method |
| `scaling_method(...)` | `"mad"` | Residual scaling method |
| `boundary_policy(...)` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback(...)` | `"use_local_mean"` | Zero-weight handling |
| `auto_converge(T)` | disabled | Auto-convergence tolerance |
| `custom_weights(Vec<T>)` | disabled | Per-observation case weights (Batch only) |
| `confidence_intervals(T)` | disabled | Confidence level (e.g., 0.95) |
| `prediction_intervals(T)` | disabled | Prediction level (e.g., 0.95) |
| `return_diagnostics()` | `false` | Compute RMSE, MAE, R², AIC |
| `return_residuals()` | `false` | Include residuals in result |
| `return_robustness_weights()` | `false` | Include robustness weights in result |
| `return_se()` | `false` | Compute hat-matrix statistics (enp, leverage …) |
| `parallel(bool)` | `true` | Enable parallel execution |
| `degree(...)` | `"linear"` | Polynomial degree |
| `dimensions(usize)` | `1` | Number of predictor dimensions |
| `distance_metric(...)` | `"normalized"` | Distance metric |
| `weighted_metric_weights(Vec<T>)` | disabled | Per-dimension weights (used when `distance_metric = "weighted"`) |
| `surface_mode(...)` | `"interpolation"` | Surface computation mode |
| `cell(T)` | disabled | Cell size for interpolation grid (smaller → more vertices, higher accuracy) |
| `interpolation_vertices(usize)` | disabled | Number of interpolation vertices |
| `boundary_degree_fallback(bool)` | `true` | Fall back to lower polynomial degree at boundaries when higher degrees fail |
| `cv_method(...)` | disabled | Cross-validation method |
| `cv_k(...)` | disabled | Number of folds for K-fold cross-validation |
| `cv_fractions(...)` | disabled | Candidate fractions to evaluate during cross-validation |
| `cv_seed(...)` | disabled | Random seed for reproducible fold assignments |

### Streaming Options

| Method | Default | Description |
| --- | --- | --- |
| `chunk_size(usize)` | `5000` | Data chunk size |
| `overlap(usize)` | `500` | Overlap between chunks |
| `merge_strategy(...)` | `"weighted_average"` | Strategy for blending overlap regions |

### Online Options

| Method | Default | Description |
| --- | --- | --- |
| `window_capacity(usize)` | `1000` | Max points in sliding window |
| `min_points(usize)` | `3` | Min points before smoothing starts |
| `update_mode(...)` | `"full"` | Update strategy |
| `parallel(bool)` | `false` | Enable parallel execution (off by default; online LOESS fits one point at a time) |

## Result Structure

### `LoessResult<T>`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Vec<T>` | Sorted x values |
| `y` | `Vec<T>` | Smoothed y values |
| `fraction_used` | `T` | Fraction used (set or selected by CV) |
| `iterations_used` | `Option<usize>` | Robustness iterations actually performed |
| `standard_errors` | `Option<Vec<T>>` | Per-point SE (if `return_se()`) |
| `confidence_lower` | `Option<Vec<T>>` | Lower confidence bounds |
| `confidence_upper` | `Option<Vec<T>>` | Upper confidence bounds |
| `prediction_lower` | `Option<Vec<T>>` | Lower prediction bounds |
| `prediction_upper` | `Option<Vec<T>>` | Upper prediction bounds |
| `residuals` | `Option<Vec<T>>` | Residuals (if `return_residuals()`) |
| `robustness_weights` | `Option<Vec<T>>` | Robustness weights (if `return_robustness_weights()`) |
| `cv_scores` | `Option<Vec<T>>` | CV score per tested fraction |
| `diagnostics` | `Option<Diagnostics<T>>` | Fit metrics (if `return_diagnostics()`) |
| `enp` | `Option<T>` | Equivalent number of parameters (if `return_se()`) |
| `trace_hat` | `Option<T>` | Trace of hat matrix (if `return_se()`) |
| `delta1` | `Option<T>` | First delta statistic (if `return_se()`) |
| `delta2` | `Option<T>` | Second delta statistic (if `return_se()`) |
| `residual_scale` | `Option<T>` | Residual scale estimate (if `return_se()`) |
| `leverage` | `Option<Vec<T>>` | Per-point hat-matrix diagonal (if `return_se()`) |
| `dimensions` | `usize` | Number of predictor dimensions |
| `polynomial_degree` | `PolynomialDegree` (internal) | Polynomial degree used; implements `Display` (e.g. `"linear"`) |
| `distance_metric` | `DistanceMetric<T>` (internal) | Distance metric used; implements `Display` (e.g. `"normalized"`) |

### `Diagnostics<T>`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `T` | Root Mean Squared Error |
| `mae` | `T` | Mean Absolute Error |
| `r_squared` | `T` | R-squared |
| `residual_sd` | `T` | Residual standard deviation |
| `effective_df` | `Option<T>` | Effective degrees of freedom |
| `aic` | `Option<T>` | AIC |
| `aicc` | `Option<T>` | AICc |

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

### distance_metric

* `"normalized"` (default — scales each dimension by its range; alias: `"norm"`)
* `"euclidean"` (alias: `"euclid"`)
* `"manhattan"` (alias: `"l1"`)
* `"chebyshev"` (alias: `"linf"`)
* `"minkowski"` or `"minkowski:p"` for a custom exponent
* `"weighted"` plus `.weighted_metric_weights(vec![...])` (alias: `"weighted_euclidean"`)

### degree

* `"constant"` or `"0"` (degree 0)
* `"linear"` or `"1"` (default, degree 1)
* `"quadratic"` or `"2"` (degree 2)
* `"cubic"` or `"3"` (degree 3)
* `"quartic"` or `"4"` (degree 4)

### surface_mode

* `"interpolation"` (default)
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

```rust
let model = Loess::new()
    .fraction(0.5)
    .iterations(3)
    .build()?;

let result = model.fit(&x, &y)?;

println!("Smoothed Y: {:?}", result.y);
```
