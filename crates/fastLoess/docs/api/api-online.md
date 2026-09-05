# OnlineLoess API

See also: [fastLoess](crate::doc::api)

## When to Use

- Data arrives incrementally (sensors, streams)
- Need real-time smoothed values
- Fixed memory budget

![Online Adapter](https://raw.githubusercontent.com/thisisamirv/loess-project/main/crates/fastLoess/assets/diagrams/online_comparison.svg)

## Struct

### `OnlineLoess`

Online mode for real-time data.

**Constructor:**

```rust
use fastLoess::prelude::*;

fn main() -> Result<(), LoessError> {
    let mut processor = OnlineLoess::new();

    Ok(())
}
```

#### `add_point(x, y)`

Adds a single point `(x, y)` to the window. `x` is a slice of predictor values (one per dimension). Returns `Result<Option<OnlineOutput<T>>, LoessError>`.

```rust
use fastLoess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let mut processor = OnlineLoess::new().fraction(0.5f64).window_capacity(50usize).min_points(3usize).build()?;;

    // Returns None until min_points (3) are reached
    let r1 = processor.add_point(&[x[0]], y[0])?;  // None
    let r2 = processor.add_point(&[x[1]], y[1])?;  // None

    // Returns Some(OnlineOutput) once enough points are available
    let r3 = processor.add_point(&[x[2]], y[2])?;
    if let Some(output) = r3 {
        println!("Smoothed value: {}", output.y);
    }

    Ok(())
}
```

```output
Smoothed value: 0.22659245357374927
```

#### `reset()`

Clears the internal window buffer. **Rust-only** — this method is not exposed in other language bindings, where creating a new instance is the idiomatic alternative.

```rust
use fastLoess::prelude::*;

fn main() -> Result<(), LoessError> {
    let mut processor = OnlineLoess::new().build()?;
    processor.reset();

    Ok(())
}
```

## Builder Options

### Online Options

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
| `return_robustness_weights()` | `bool` | `false` | Include `robustness_weight` in result |
| `degree(...)` | `degree` | `"linear"` | Polynomial degree |
| `dimensions(usize)` | `usize` | `1` | Number of predictor dimensions |
| `distance_metric(...)` | `distance_metric` | `"normalized"` | Distance metric |
| `weighted_metric_weights(Vec<T>)` | `Vec<T: Float>` | disabled | Per-dimension weights (used when `distance_metric = "weighted"`) |
| `surface_mode(...)` | `surface_mode` | `"interpolation"` | Surface computation mode |
| `cell(T)` | `T: Float` | disabled | Cell size for interpolation grid (smaller → more vertices, higher accuracy) |
| `interpolation_vertices(usize)` | `usize` | disabled | Number of interpolation vertices |
| `boundary_degree_fallback(bool)` | `bool` | `true` | Fall back to lower polynomial degree at boundaries when higher degrees fail |
| `window_capacity(usize)` | `usize` | `1000` | Max points in sliding window |
| `min_points(usize)` | `usize` | `2` | Min points before smoothing starts |
| `update_mode(...)` | `update_mode` | `"incremental"` | Update mode (`"full"` or `"incremental"`) |

Confidence/prediction intervals, standard errors, cross-validation, `return_diagnostics`, `return_residuals`, `parallel`, and `backend` are Batch-only (or Batch/Streaming-only) and not available here; see [API](crate::doc::api) for those. Online always runs sequentially.

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

### return_robustness_weights

Populates `OnlineOutput::robustness_weight` with the robustness weight for the latest point (from the last robustness iteration). `false` by default.

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

### window_capacity

Maximum number of most recent points kept in the sliding window; older points are discarded as new ones arrive. Each `add_point()` call costs O(`window_capacity`) rather than growing with total history.

### min_points

Minimum number of points required before smoothing starts. `add_point()` returns `None` until the window reaches this size.

### update_mode

*See: [Execution Modes](crate::doc::guide::adapter_choice)*

| Mode | Alias | Behavior | Speed |
| --- | --- | --- | --- |
| `"incremental"` (default) | `"single"` | Update only affected fits | Faster |
| `"full"` | `"resmooth"` | Recompute entire window | More accurate |

## Result Structure

### `OnlineOutput<T>`

Returned inside `Ok(Some(...))` by `add_point()`. `None` while the window is still filling.

| Field | Type | Description |
| --- | --- | --- |
| `y` | `T` | Smoothed value for the latest point |
| `standard_error` | `Option<T>` | Always `None` — standard errors require confidence intervals, which are Batch-only |
| `residual` | `Option<T>` | Residual y − smoothed; always present (there is no `return_residuals` option for Online) |
| `robustness_weight` | `Option<T>` | Robustness weight, if `return_robustness_weights` was set |
| `iterations_used` | `Option<usize>` | Robustness iterations performed |

There is no `Diagnostics` type or `return_diagnostics` option for `OnlineLoess`: `OnlineOutput` carries no diagnostics field, since diagnostics like RMSE/R² need more than one point's worth of history to be meaningful.
