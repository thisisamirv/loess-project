# StreamingLoess API

See also: [fastLoess](crate::doc::api)

## When to Use

- Dataset >100,000 points
- Memory-constrained environments
- Batch processing pipelines

## Struct

### `StreamingLoess`

Streaming mode for large datasets.

**Constructor:**

```rust
use fastLoess::prelude::*;

fn main() -> Result<(), LoessError> {
    let mut processor = StreamingLoess::new();

    Ok(())
}
```

#### `process_chunk(x, y)`

Processes a chunk of data. Returns `LoessResult<T>` with partial results.

```rust
use fastLoess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let mut processor = StreamingLoess::new().fraction(0.5f64).chunk_size(50usize).overlap(10usize).build()?;
    let result = processor.process_chunk(&x[..50], &y[..50])?;
    println!("Fraction used: {}", result.fraction_used);

    Ok(())
}
```

```output
Fraction used: 0.5
```

#### `finalize()`

Finalizes processing and returns remaining buffered results.

```rust
use fastLoess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let mut processor = StreamingLoess::new().fraction(0.5f64).chunk_size(50usize).overlap(10usize).build()?;
    processor.process_chunk(&x[..50], &y[..50])?;
    processor.process_chunk(&x[50..], &y[50..])?;
    let final_result = processor.finalize()?;
    println!("Fraction used: {}", final_result.fraction_used);

    Ok(())
}
```

```output
Fraction used: 0.5
```

## Builder Options

### Streaming Options

| Method | Argument Type | Default | Description |
| --- | --- | --- | --- |
| `fraction(T)` | `T: Float` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations(usize)` | `usize` | `3` | Number of robustifying iterations |
| `weight_function(...)` | `weight_function` | `"tricube"` | Kernel weight function |
| `robustness_method(...)` | `robustness_method` | `"bisquare"` | Robustness method |
| `scaling_method(...)` | `scaling_method` | `"mad"` | Residual scaling method |
| `boundary_policy(...)` | `boundary_policy` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback(...)` | `zero_weight_fallback` | `"use_local_mean"` | Zero-weight handling |
| `auto_converge(T)` | `T: Float` | disabled | Auto-convergence tolerance |
| `return_diagnostics()` | `bool` | `false` | Compute RMSE, MAE, R2 |
| `return_residuals()` | `bool` | `false` | Include residuals in result |
| `return_robustness_weights()` | `bool` | `false` | Include weights in result |
| `parallel(bool)` | `bool` | `true` | Enable parallel execution across CPU cores |
| `degree(...)` | `degree` | `"linear"` | Polynomial degree |
| `dimensions(usize)` | `usize` | `1` | Number of predictor dimensions |
| `distance_metric(...)` | `distance_metric` | `"normalized"` | Distance metric |
| `weighted_metric_weights(Vec<T>)` | `Vec<T: Float>` | disabled | Per-dimension weights (used when `distance_metric = "weighted"`) |
| `surface_mode(...)` | `surface_mode` | `"interpolation"` | Surface computation mode |
| `cell(T)` | `T: Float` | disabled | Cell size for interpolation grid (smaller → more vertices, higher accuracy) |
| `interpolation_vertices(usize)` | `usize` | disabled | Number of interpolation vertices |
| `boundary_degree_fallback(bool)` | `bool` | `true` | Fall back to lower polynomial degree at boundaries when higher degrees fail |
| `chunk_size(usize)` | `usize` | `5000` | Data chunk size |
| `overlap(usize)` | `usize` | `chunk_size / 10` | Overlap between chunks |
| `merge_strategy(...)` | `merge_strategy` | `"weighted_average"` | Strategy for blending overlap regions |

Confidence/prediction intervals, standard errors, cross-validation, `return_sorted`, and `backend` are Batch-only and not available here; see [API](crate::doc::api) for those.

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

*See: [Weight Functions](crate::doc::weighting::kernels)*

- `"tricube"` (default)
- `"epanechnikov"`
- `"gaussian"`
- `"uniform"` (alias: `"boxcar"`)
- `"biweight"` (alias: `"bisquare"`)
- `"triangle"` (alias: `"triangular"`)
- `"cosine"`

### robustness_method

*See: [Robustness](crate::doc::weighting::robustness)*

- `"bisquare"` (default; alias: `"biweight"`)
- `"huber"`
- `"talwar"`

### scaling_method

*See: [Scaling Methods](crate::doc::weighting::scaling)*

- `"mad"` (default; alias: `"median_absolute_deviation"`)
- `"mar"` (alias: `"median_absolute_residual"`)
- `"mean"` (alias: `"mean_absolute_residual"`)

### boundary_policy

*See: [Boundary Handling](crate::doc::advanced::boundary)*

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

*See: [Robustness](crate::doc::weighting::robustness#auto-convergence)*

Convergence tolerance for early stopping of robustness iterations. Disabled by default.

### return_diagnostics

Populates `LoessResult::diagnostics` with RMSE, MAE, R², and residual_sd. `effective_df`/`aic`/`aicc` require standard errors, which are Batch-only, so they're always `None` here. `false` by default.

### return_residuals

Populates `LoessResult::residuals` (`y - fitted`). `false` by default.

### return_robustness_weights

Populates `LoessResult::robustness_weights` with the final per-point robustness weights (from the last robustness iteration). `false` by default.

### parallel

Enables multi-threaded execution via Rayon, parallelizing chunk processing across CPU cores. `true` by default; set to `false` to force single-threaded execution.

### degree

*See: [Polynomial Degree](crate::doc::advanced::degree)*

- `"constant"` or `"0"` (degree 0)
- `"linear"` or `"1"` (default, degree 1)
- `"quadratic"` or `"2"` (degree 2)
- `"cubic"` or `"3"` (degree 3)
- `"quartic"` or `"4"` (degree 4)

### dimensions

*See: [Multivariate LOESS](crate::doc::advanced::dimensions)*

Number of predictor dimensions. Set to match the number of columns in a multivariate `x` array. `1` (default) is univariate.

### distance_metric

*See: [Multivariate LOESS](crate::doc::advanced::dimensions)*

- `"normalized"` (default — scales each dimension by its range; alias: `"norm"`)
- `"euclidean"` (alias: `"euclid"`)
- `"manhattan"` (alias: `"l1"`)
- `"chebyshev"` (alias: `"linf"`)
- `"minkowski"` (use `"minkowski:p"` for custom exponent, e.g. `"minkowski:3"`)
- `"weighted"` plus `.weighted_metric_weights(vec![...])` (alias: `"weighted_euclidean"`)

### weighted_metric_weights

*See: [Multivariate LOESS](crate::doc::advanced::dimensions)*

Per-dimension weights, one per dimension declared in `dimensions`. Only used when `distance_metric` is `"weighted"`; calling `.distance_metric("weighted")` without also calling this returns a `LoessError`.

### surface_mode

*See: [Polynomial Degree](../advanced/degree.md#surface-mode)*

Controls whether the local polynomial is evaluated at every query point or at a sparser grid of anchor vertices with Hermite cubic interpolation in between.

| Mode | Behavior | Speed | Accuracy |
| --- | --- | --- | --- |
| `"interpolation"` (default) | Evaluate at vertices, interpolate between | Faster | Slight approximation |
| `"direct"` | Evaluate at every query point | Slower | Full precision |

### cell

Cell size for the interpolation grid, as a fraction of the data range in `(0, 1]`. Smaller values place more vertices (denser grid), improving accuracy at the cost of speed. Disabled by default (uses the library default `0.2`). Only applies when `surface_mode` is `"interpolation"`.

### interpolation_vertices

Caps the maximum number of interpolation vertices, overriding the count implied by `cell`. Disabled by default (no explicit cap). Only applies when `surface_mode` is `"interpolation"`.

### boundary_degree_fallback

Whether to reduce the polynomial degree at boundary vertices when the requested `degree` can't be fit there. `true` by default. Only applies when `surface_mode` is `"interpolation"`.

### chunk_size

Number of points processed per chunk. Larger chunks reduce per-chunk overhead and give each local fit more surrounding context, at the cost of higher peak memory; smaller chunks bound memory tightly but increase the fraction of points that fall in overlap regions. A good starting point is balancing available memory against how much processing overhead per chunk is acceptable — match it to your file-read buffer or message-batch size to avoid unnecessary copying.

### overlap

Number of points retained from the previous chunk as context, so the neighbourhood at chunk boundaries isn't artificially truncated. Points inside the overlap zone are fitted twice (once by each chunk) and reconciled via `merge_strategy`. A good starting point is 10–20% of `chunk_size`: too little overlap causes visible boundary artefacts, while too much wastes computation refitting the same points twice.

- Not called (default) — computes `chunk_size / 10`, clamped to at least 1 and at most `chunk_size - 10`
- Any `usize >= 1` and `< chunk_size`

### merge_strategy

*See: [Merge Strategies](crate::doc::advanced::merge)*

| Strategy | Alias | Behavior |
| --- | --- | --- |
| `"weighted_average"` (default) | `"weighted"` | Distance-weighted blend |
| `"average"` | `"mean"` | Average overlapping values |
| `"take_first"` | `"first"` | Keep left chunk values |
| `"take_last"` | `"last"` | Keep right chunk values |

![Merge Strategies](https://raw.githubusercontent.com/thisisamirv/loess-project/main/crates/fastLoess/assets/diagrams/merge_comparison.svg)

## Result Structure

### `LoessResult<T>`

Returned by `process_chunk()` and `finalize()`.

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Vec<T>` | x values (same order as input) |
| `y` | `Vec<T>` | Smoothed y values |
| `fraction_used` | `T` | Fraction used |
| `iterations_used` | `Option<usize>` | Robustness iterations actually performed |
| `standard_errors` | `Option<Vec<T>>` | Always `None` (Batch only) |
| `confidence_lower` | `Option<Vec<T>>` | Always `None` (Batch only) |
| `confidence_upper` | `Option<Vec<T>>` | Always `None` (Batch only) |
| `prediction_lower` | `Option<Vec<T>>` | Always `None` (Batch only) |
| `prediction_upper` | `Option<Vec<T>>` | Always `None` (Batch only) |
| `residuals` | `Option<Vec<T>>` | Residuals (if `return_residuals()`) |
| `robustness_weights` | `Option<Vec<T>>` | Robustness weights (if `return_robustness_weights()`) |
| `cv_scores` | `Option<Vec<T>>` | Always `None` (Batch only) |
| `diagnostics` | `Option<Diagnostics<T>>` | Fit metrics (if `return_diagnostics()`) |
| `dimensions` | `usize` | Number of predictor dimensions |

### `Diagnostics<T>`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `T` | Root Mean Squared Error |
| `mae` | `T` | Mean Absolute Error |
| `r_squared` | `T` | R-squared |
| `residual_sd` | `T` | Residual standard deviation |
| `effective_df` | `Option<T>` | Always `None` (requires standard errors, Batch only) |
| `aic` | `Option<T>` | Always `None` (requires `effective_df`, Batch only) |
| `aicc` | `Option<T>` | Always `None` (requires `effective_df`, Batch only) |

See [rust.md](crate::doc::api) for the full `LoessResult<T>` field reference.

---

!!! warning "Always call finalize()"
    The streaming adapter buffers overlap data. Call `finalize()` after the last chunk to retrieve the buffered tail.
