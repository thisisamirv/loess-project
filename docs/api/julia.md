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
result = fit(model, x::Vector{Float64}, y::Vector{Float64}) :: LoessResult
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
result = add_points(online, x::Vector{Float64}, y::Vector{Float64}) :: LoessResult
```

* Adds new points to the model and returns the smoothed values (retrospective or prospective depending on mode).

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
| `zero_weight_fallback` | `String` | `"use_local_mean"` | Zero-weight handling strategy |
| `auto_converge` | `Float64` | `NaN` | Auto-convergence tolerance (NaN to disable) |
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
| `surface_mode` | `String` | `"interpolation"` | Surface computation mode |
| `cv_fractions` | `Vector{Float64}` | `Float64[]` | Fractions to test for cross-validation |
| `cv_method` | `String` | `"kfold"` | CV method (`"kfold"` or `"loocv"`) |
| `cv_k` | `Int` | `5` | Number of folds for k-fold CV |

### `StreamingOptions` (inherits `LoessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size` | `Int` | `5000` | Data chunk size |
| `overlap` | `Int` | `auto (-1)` | Overlap between chunks (-1 for auto 10%) |
| `merge_strategy` | `String` | `"weighted_average"` | Strategy for blending overlap: see Merge Strategies |

### `OnlineOptions` (inherits `LoessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `window_capacity` | `Int` | `100` | Max points in sliding window |
| `min_points` | `Int` | `2` | Min points before smoothing starts |
| `update_mode` | `String` | `"full"` | Update mode (`"full"` or `"incremental"`) |

## Result Structure

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
| `effective_df` | `Float64` | Effective degrees of freedom |
| `aic` | `Float64` | AIC |
| `aicc` | `Float64` | AICc |

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
* `"incremental"` (faster, O(1) incremental update)`

## Example

```julia
using FastLOESS

x = collect(range(0, 10, length=100))
y = sin.(x) .+ randn(100) .* 0.2

# Configure model
model = Loess(fraction=0.3, iterations=3)

# Fit data
result = fit(model, x, y)

if !isempty(result.error)
    println("Error: ", result.error)
else
    println("Smoothed Y: ", result.y)
end
```
