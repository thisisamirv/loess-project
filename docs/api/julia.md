# FastLOESS Julia API Reference

The Julia bindings provide a modern interface to the core Rust library, mirroring the Rust API structure.

## Classes

### `Loess`

The `Loess` struct allows configuring the LOESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

```julia
model = Loess(; kwargs...)
```

* `kwargs`: Keyword arguments corresponding to `LoessOptions` fields.

**Methods:**

```julia
result = fit(model, x::Vector{Float64}, y::Vector{Float64}; custom_weights=nothing) :: LoessResult
```

* Fits the model to the provided `x` and `y` data vectors.
* Returns a `LoessResult` struct containing the smoothed values and optional diagnostics.

### `StreamingLoess`

The `StreamingLoess` struct processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```julia
stream = StreamingLoess(; kwargs...)
```

* `kwargs`: Keyword arguments corresponding to `StreamingOptions` fields.

**Methods:**

```julia
partial_result = process_chunk(stream, x::Vector{Float64}, y::Vector{Float64}) :: LoessResult
```

* Processes a chunk of data. Returns partial results.

```julia
final_result = finalize(stream) :: LoessResult
```

* Finalizes the smoothing process and returns any remaining buffered results.

### `OnlineLoess`

The `OnlineLoess` struct updates the model incrementally with new data points.

**Constructor:**

```julia
online = OnlineLoess(; kwargs...)
```

* `kwargs`: Keyword arguments corresponding to `OnlineOptions` fields.

**Methods:**

```julia
result = add_point(online, x::Float64, y::Float64) :: Union{OnlineOutput, Nothing}
```

* Adds a single point to the sliding window. Returns `nothing` while the window is still filling (fewer than `min_points` seen), and an `OnlineOutput` once smoothing begins.

## Options Structures

### `LoessOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `Float64` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `Int` | `3` | Number of robustifying iterations |
| `weight_function` | `String` | `"tricube"` | Kernel weight function |
| `robustness_method` | `String` | `"bisquare"` | Robustness method |
| `scaling_method` | `String` | `"mad"` | Residual scaling method |
| `boundary_policy` | `String` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback` | `String` | `"use_local_mean"` | Zero-weight handling |
| `auto_converge` | `Float64` | `NaN` | Auto-convergence tolerance (NaN to disable) |
| `custom_weights` | `Union{Vector{Float64}, Nothing}` | `nothing` | Per-observation case weights — passed to `fit()`, not the constructor (Batch only) |
| `confidence_intervals` | `Float64` | `NaN` | Confidence level (e.g., 0.95; NaN to disable) |
| `prediction_intervals` | `Float64` | `NaN` | Prediction level (e.g., 0.95; NaN to disable) |
| `return_diagnostics` | `Bool` | `false` | Compute RMSE, MAE, R², AIC |
| `return_residuals` | `Bool` | `false` | Include residuals in result |
| `return_robustness_weights` | `Bool` | `false` | Include robustness weights in result |
| `return_se` | `Bool` | `false` | Compute hat-matrix statistics (enp, leverage …) |
| `parallel` | `Bool` | `true` | Enable parallel execution |
| `degree` | `String` | `"linear"` | Polynomial degree of local fit |
| `dimensions` | `Int` | `1` | Number of predictor dimensions |
| `distance_metric` | `String` | `"normalized"` | Distance metric; use `"minkowski:p"` for custom p |
| `weighted_metric_weights` | `Union{Vector{Float64}, Nothing}` | `nothing` | Per-dimension weights (used when `distance_metric = "weighted"`) |
| `surface_mode` | `String` | `"interpolation"` | Surface computation mode |
| `cell` | `Union{Float64, Nothing}` | `nothing` | Cell size for interpolation grid (smaller → more vertices, higher accuracy) |
| `interpolation_vertices` | `Union{Int, Nothing}` | `nothing` | Number of interpolation vertices |
| `boundary_degree_fallback` | `Union{Bool, Nothing}` | `nothing` | Fall back to lower polynomial degree at boundaries when higher degrees fail |
| `cv_method` | `String` | `"kfold"` | CV method (`"kfold"` or `"loocv"`) (Batch only) |
| `cv_k` | `Int` | `5` | Number of folds for k-fold CV (Batch only) |
| `cv_fractions` | `Vector{Float64}` | `Float64[]` | Fractions to test for cross-validation (Batch only) |
| `cv_seed` | `Union{Int, Nothing}` | `nothing` | Random seed for cross-validation shuffling (Batch only) |

### `StreamingOptions` (inherits `LoessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size` | `Int` | `5000` | Data chunk size |
| `overlap` | `Int` | `500` | Overlap between chunks |
| `merge_strategy` | `String` | `"weighted_average"` | Strategy for blending overlap regions |

### `OnlineOptions` (inherits `LoessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `window_capacity` | `Int` | `1000` | Max points in sliding window |
| `min_points` | `Int` | `3` | Min points before smoothing starts |
| `update_mode` | `String` | `"full"` | Update mode (`"full"` or `"incremental"`) |
| `parallel` | `Bool` | `false` | Enable parallel execution (off by default; online LOESS fits one point at a time) |

## Result Structure

### `OnlineOutput`

Returned by `add_point()` once the window has enough points (`nothing` until then).

| Field | Type | Description |
| --- | --- | --- |
| `smoothed` | `Float64` | Smoothed value for the latest point |
| `std_error` | `Union{Float64, Nothing}` | Standard error (if requested) |
| `residual` | `Union{Float64, Nothing}` | Residual y − smoothed (if requested) |
| `robustness_weight` | `Union{Float64, Nothing}` | Robustness weight (if requested) |
| `iterations_used` | `Union{Int, Nothing}` | Robustness iterations performed |

### `LoessResult`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Vector{Float64}` | Sorted x values |
| `y` | `Vector{Float64}` | Smoothed y values |
| `fraction_used` | `Float64` | Fraction used (set or selected by CV) |
| `iterations_used` | `Int` | Robustness iterations actually performed (-1 = N/A) |
| `standard_errors` | `Union{Vector{Float64}, Nothing}` | Per-point SE (if `return_se`) |
| `confidence_lower` | `Union{Vector{Float64}, Nothing}` | Lower confidence bounds |
| `confidence_upper` | `Union{Vector{Float64}, Nothing}` | Upper confidence bounds |
| `prediction_lower` | `Union{Vector{Float64}, Nothing}` | Lower prediction bounds |
| `prediction_upper` | `Union{Vector{Float64}, Nothing}` | Upper prediction bounds |
| `residuals` | `Union{Vector{Float64}, Nothing}` | Residuals (if `return_residuals`) |
| `robustness_weights` | `Union{Vector{Float64}, Nothing}` | Robustness weights (if `return_robustness_weights`) |
| `cv_scores` | `Union{Vector{Float64}, Nothing}` | CV score per tested fraction |
| `diagnostics` | `Union{Diagnostics, Nothing}` | Fit metrics (if `return_diagnostics`) |
| `enp` | `Union{Float64, Nothing}` | Equivalent number of parameters (if `return_se`) |
| `trace_hat` | `Union{Float64, Nothing}` | Trace of hat matrix (if `return_se`) |
| `delta1` | `Union{Float64, Nothing}` | First delta statistic (if `return_se`) |
| `delta2` | `Union{Float64, Nothing}` | Second delta statistic (if `return_se`) |
| `residual_scale` | `Union{Float64, Nothing}` | Residual scale estimate (if `return_se`) |
| `leverage` | `Union{Vector{Float64}, Nothing}` | Per-point hat-matrix diagonal (if `return_se`) |
| `dimensions` | `Int` | Number of predictor dimensions |

### `Diagnostics`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `Float64` | Root Mean Squared Error |
| `mae` | `Float64` | Mean Absolute Error |
| `r_squared` | `Float64` | R-squared |
| `residual_sd` | `Float64` | Residual standard deviation |
| `effective_df` | `Float64` | Effective degrees of freedom (NaN if not computed) |
| `aic` | `Float64` | AIC (NaN if not computed) |
| `aicc` | `Float64` | AICc (NaN if not computed) |

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

```julia
using FastLOESS

x = collect(range(0, 10, length=100))
y = sin.(x) .+ randn(100) .* 0.2

# Configure model
model = Loess(fraction=0.5, iterations=3)

# Fit data (throws on error)
result = fit(model, x, y)

println("Smoothed Y: ", result.y)
```
