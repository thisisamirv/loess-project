# fastLoess & loess-rs Rust API Reference

The Rust bindings provide the core implementation and high-performance extensions. The API uses a Builder pattern consistent across both the `loess-rs` (pure Rust) and `fastLoess` (accelerated) crates.

## Structs & Usage

Unlike other bindings which have distinct classes, Rust uses a single `Loess` builder that produces different model types based on the configured `.adapter()`.

### `Loess` (Batch)

Standard in-memory smoothing.

**Constructor:**

```rust
let model = Loess::<f64>::new().adapter(Batch).build()?;
```

**Methods:**

```rust
let model = Loess::new().adapter(Batch).build()?;
let result = model.fit(&x, &y)?;
```

* Fits the model to the provided `x` and `y` arrays.
* Returns `Result<LoessResult<T>, LoessError>`.

### `Loess` (Streaming)

Streaming mode for large datasets.

**Constructor:**

```rust
let mut processor = Loess::<f64>::new().adapter(Streaming).build()?;
```

**Methods:**

```rust
let mut processor = Loess::new().adapter(Streaming).build()?;
let result = processor.process_chunk(&x, &y)?;
```

* Processes a chunk of data. Returns `LoessResult<T>` with partial results.

```rust
let mut processor = Loess::new().adapter(Streaming).build()?;
processor.process_chunk(&x_chunk, &y_chunk)?;
let final_result = processor.finalize()?;
```

* Finalizes processing and returns remaining buffered results.

### `Loess` (Online)

Online mode for real-time data.

**Constructor:**

```rust
let mut processor = Loess::<f64>::new().adapter(Online).build()?;
```

**Methods:**

```rust
let mut processor = Loess::new().adapter(Online).build()?;
let output = processor.add_point(&[x[0]], y[0])?;
```

* Adds a single point `(x, y)` to the window.
* Returns `Result<Option<OnlineOutput<T>>, LoessError>`.

```rust
let mut processor = Loess::<f64>::new().adapter(Online).build()?;
processor.reset();
```

* Clears the internal window buffer.

## Builder Configuration

These chained methods configure the builder. They correspond to the "Options Structures" in other bindings.

### Loess Options

| Method | Argument Type | Default | Description |
| --- | --- | --- | --- |
| `fraction(T)` | `T: Float` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations(usize)` | `usize` | `3` | Number of robustifying iterations |
| `weight_function(...)` | `WeightFunction` | `Tricube` | Kernel weight function enum |
| `robustness_method(...)` | `RobustnessMethod` | `Bisquare` | Robustness method enum |
| `scaling_method(...)` | `ScalingMethod` | `MAD` | Residual scaling method enum |
| `boundary_policy(...)` | `BoundaryPolicy` | `Extend` | Boundary handling policy enum |
| `zero_weight_fallback(...)` | `ZeroWeightFallback` | `UseLocalMean` | Zero-weight handling enum |
| `auto_converge(T)` | `T: Float` | disabled | Auto-convergence tolerance |
| `custom_weights(Vec<T>)` | `Vec<T>` | disabled | Per-observation case weights (Batch only) |
| `confidence_intervals(T)` | `T: Float` | disabled | Confidence level (e.g., 0.95) |
| `prediction_intervals(T)` | `T: Float` | disabled | Prediction level (e.g., 0.95) |
| `return_diagnostics()` | — | `false` | Compute RMSE, MAE, R², AIC |
| `return_residuals()` | — | `false` | Include residuals in result |
| `return_robustness_weights()` | — | `false` | Include robustness weights in result |
| `return_se()` | — | `false` | Compute hat-matrix statistics (enp, leverage …) |
| `parallel(bool)` | `bool` | `true` | Enable parallel execution |
| `degree(...)` | `PolynomialDegree` | `Linear` | Polynomial degree enum |
| `dimensions(usize)` | `usize` | `1` | Number of predictor dimensions |
| `distance_metric(...)` | `DistanceMetric<T>` | `Normalized` | Distance metric enum |
| `surface_mode(...)` | `SurfaceMode` | `Interpolation` | Surface computation mode enum |
| `cell(T)` | `T: Float` | disabled | Cell size for interpolation grid (smaller → more vertices, higher accuracy) |
| `interpolation_vertices(usize)` | `usize` | disabled | Number of interpolation vertices |
| `boundary_degree_fallback(bool)` | `bool` | disabled | Fall back to lower polynomial degree at boundaries when higher degrees fail |
| `cross_validate(...)` | `impl CrossValidation` | disabled | CV strategy: `KFold { k, fractions, seed }` or `LOOCV { fractions }` |

### Streaming Options

| Method | Argument Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size(usize)` | `usize` | `5000` | Data chunk size |
| `overlap(usize)` | `usize` | auto (10%) | Overlap between chunks |
| `merge_strategy(...)` | `MergeStrategy` | `WeightedAverage` | Strategy for blending overlap regions |

### Online Options

| Method | Argument Type | Default | Description |
| --- | --- | --- | --- |
| `window_capacity(usize)` | `usize` | `100` | Max points in sliding window |
| `min_points(usize)` | `usize` | `2` | Min points before smoothing starts |
| `update_mode(...)` | `UpdateMode` | `Full` | Update strategy enum |

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
| `polynomial_degree` | `PolynomialDegree` | Polynomial degree used |
| `distance_metric` | `DistanceMetric<T>` | Distance metric used |

### `Diagnostics<T>`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `T` | Root Mean Squared Error |
| `mae` | `T` | Mean Absolute Error |
| `r_squared` | `T` | R-squared |
| `residual_sd` | `T` | Residual standard deviation |
| `effective_df` | `T` | Effective degrees of freedom |
| `aic` | `T` | AIC |
| `aicc` | `T` | AICc |

## Enum Options

### WeightFunction

* `Tricube` (default)
* `Epanechnikov`
* `Gaussian`
* `Uniform`
* `Biweight`
* `Triangle`
* `Cosine`

### RobustnessMethod

* `Bisquare` (default)
* `Huber`
* `Talwar`

### BoundaryPolicy

* `Extend` (default - linear extrapolation)
* `Reflect`
* `Zero`
* `NoBoundary`

### ScalingMethod

* `MAD` (default — Median Absolute Deviation)
* `MAR` (Median Absolute Residual)
* `Mean` (Mean Absolute Residual)

### ZeroWeightFallback

* `UseLocalMean` (default)
* `ReturnOriginal`
* `ReturnNone`

### DistanceMetric\<T\>

* `Normalized` (default — scales each dimension by its range)
* `Euclidean`
* `Manhattan`
* `Chebyshev`
* `Minkowski(T)` — weighted p-norm; use e.g. `Minkowski(3.0)` for a custom p value
* `Weighted(Vec<T>)` — weighted Euclidean distance; provide a `Vec` of per-dimension scaling weights

### PolynomialDegree

* `Constant` (degree 0)
* `Linear` (default, degree 1)
* `Quadratic` (degree 2)
* `Cubic` (degree 3)
* `Quartic` (degree 4)

### SurfaceMode

* `Interpolation` (default — faster, uses a spatial grid)
* `Direct` (fits every point exactly; slower but more accurate)

### DistanceMetric

* `Normalized` (default — scales each dimension by its range)
* `Euclidean`
* `Manhattan`
* `Chebyshev`
* `Minkowski(T)` (custom exponent)
* `Weighted(Vec<T>)` (per-dimension scale weights)

### MergeStrategy

* `WeightedAverage` (default — weighted blend of overlapping regions)
* `Average` (simple mean of overlapping regions)
* `TakeFirst` (keep values from the earlier chunk)
* `TakeLast` (keep values from the later chunk)

### UpdateMode

* `Full` (default — re-smooth entire window each update)
* `Incremental` (faster, O(1) incremental update)

## Example

```rust
let model = Loess::new()
    .fraction(0.5)
    .iterations(3)
    .adapter(Batch)
    .build()?;

let result = model.fit(&x, &y)?;

println!("Smoothed Y: {:?}", result.y);
```
