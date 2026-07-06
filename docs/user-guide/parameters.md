<!-- markdownlint-disable MD024 -->
# Parameters

Complete reference for all LOESS configuration options.

## Quick Reference

=== "R"

    | Parameter | Default | Range/Options | Description | Adapter |
    | --- | --- | --- | --- | --- |
    | **fraction** | 0.67 | (0, 1] | Smoothing span | All |
    | **iterations** | 3 | [0, 1000] | Robustness iterations | All |
    | **degree** | 1 | 0–4 | Polynomial degree | All |
    | **surface_mode** | `"interpolation"` | 2 options | Fit vs interpolate | All |
    | **delta** | NULL (auto) | [0, ∞) | Interpolation threshold | All |
    | **weight_function** | `"tricube"` | 7 options | Distance kernel | All |
    | **robustness_method** | `"bisquare"` | 3 options | Outlier weighting | All |
    | **zero_weight_fallback** | `"use_local_mean"` | 3 options | Zero-weight behavior | All |
    | **boundary_policy** | `"extend"` | 4 options | Edge handling | All |
    | **scaling_method** | `"mad"` | 3 options | Scale estimation | All |
    | **auto_converge** | NULL | tolerance | Early stopping | All |
    | **custom_weights** | NULL | positive numeric | Per-observation weights | Batch |
    | **return_residuals** | FALSE | logical | Include residuals | All |
    | **return_robustness_weights** | FALSE | logical | Include weights | All |
    | **return_diagnostics** | FALSE | logical | Include metrics | All |
    | **confidence_intervals** | NULL | (0, 1) | CI level | All |
    | **prediction_intervals** | NULL | (0, 1) | PI level | All |
    | **weighted_metric_weights** | NULL | numeric | Per-dimension distance weights | All |
    | **cell** | NULL | (0, ∞) | Interpolation cell size | All |
    | **interpolation_vertices** | NULL | integer | Interpolation grid vertices | All |
    | **boundary_degree_fallback** | NULL | logical | Degree fallback at boundaries | All |
    | **cv_method** | NULL | method | Auto-select fraction | Batch |
    | **cv_k** | 5L | [2, ∞) | K-fold count | Batch |
    | **cv_fractions** | NULL | numeric | Fractions to evaluate | Batch |
    | **cv_seed** | NULL | integer | CV fold randomization seed | Batch |
    | **chunk_size** | 5000 | [10, ∞) | Points per chunk | Streaming |
    | **overlap** | 500 | [0, chunk) | Overlap between chunks | Streaming |
    | **merge_strategy** | `"weighted_average"` | 4 options | Merge overlaps | Streaming |
    | **window_capacity** | 1000 | [3, ∞) | Max window size | Online |
    | **min_points** | 2 | [2, window] | Min before output | Online |
    | **update_mode** | `"incremental"` | 2 options | Update strategy | Online |
=== "Python"

    | Parameter | Default | Range/Options | Description | Adapter |
    | --- | --- | --- | --- | --- |
    | **fraction** | 0.67 | (0, 1] | Smoothing span | All |
    | **iterations** | 3 | [0, 1000] | Robustness iterations | All |
    | **degree** | 1 | 0–4 | Polynomial degree | All |
    | **surface_mode** | `"interpolation"` | 2 options | Fit vs interpolate | All |
    | **delta** | None (auto) | [0, ∞) | Interpolation threshold | All |
    | **weight_function** | `"tricube"` | 7 options | Distance kernel | All |
    | **robustness_method** | `"bisquare"` | 3 options | Outlier weighting | All |
    | **zero_weight_fallback** | `"use_local_mean"` | 3 options | Zero-weight behavior | All |
    | **boundary_policy** | `"extend"` | 4 options | Edge handling | All |
    | **scaling_method** | `"mad"` | 3 options | Scale estimation | All |
    | **auto_converge** | None | tolerance | Early stopping | All |
    | **custom_weights** | None | positive float | Per-observation weights | Batch |
    | **return_residuals** | False | bool | Include residuals | All |
    | **return_robustness_weights** | False | bool | Include weights | All |
    | **return_diagnostics** | False | bool | Include metrics | All |
    | **confidence_intervals** | None | (0, 1) | CI level | All |
    | **prediction_intervals** | None | (0, 1) | PI level | All |
    | **weighted_metric_weights** | None | list[float] | Per-dimension distance weights | All |
    | **cell** | None | (0, ∞) | Interpolation cell size | All |
    | **interpolation_vertices** | None | int | Interpolation grid vertices | All |
    | **boundary_degree_fallback** | None | bool | Degree fallback at boundaries | All |
    | **cv_method** | None | method | Auto-select fraction | Batch |
    | **cv_k** | 5 | [2, ∞) | K-fold count | Batch |
    | **cv_fractions** | None | list[float] | Fractions to evaluate | Batch |
    | **cv_seed** | None | int | CV fold randomization seed | Batch |
    | **chunk_size** | 5000 | [10, ∞) | Points per chunk | Streaming |
    | **overlap** | 500 | [0, chunk) | Overlap between chunks | Streaming |
    | **merge_strategy** | `"weighted_average"` | 4 options | Merge overlaps | Streaming |
    | **window_capacity** | 1000 | [3, ∞) | Max window size | Online |
    | **min_points** | 2 | [2, window] | Min before output | Online |
    | **update_mode** | `"incremental"` | 2 options | Update strategy | Online |
=== "Rust"

    | Parameter | Default | Range/Options | Description | Adapter |
    | --- | --- | --- | --- | --- |
    | **fraction** | 0.67 | (0, 1] | Smoothing span | All |
    | **iterations** | 3 | [0, 1000] | Robustness iterations | All |
    | **degree** | 1 | 0–4 | Polynomial degree | All |
    | **surface_mode** | `Interpolation` | 2 options | Fit vs interpolate | All |
    | **delta** | auto | [0, ∞) | Interpolation threshold | All |
    | **weight_function** | `Tricube` | 7 options | Distance kernel | All |
    | **robustness_method** | `Bisquare` | 3 options | Outlier weighting | All |
    | **zero_weight_fallback** | `UseLocalMean` | 3 options | Zero-weight behavior | All |
    | **boundary_policy** | `Extend` | 4 options | Edge handling | All |
    | **scaling_method** | `MAD` | 3 options | Scale estimation | All |
    | **auto_converge** | None | tolerance | Early stopping | All |
    | **custom_weights** | — | `Vec<T>` | Per-observation weights | Batch |
    | **return_residuals** | false | bool | Include residuals | All |
    | **return_robustness_weights** | false | bool | Include weights | All |
    | **return_diagnostics** | false | bool | Include metrics | All |
    | **confidence_intervals** | None | (0, 1) | CI level | All |
    | **prediction_intervals** | None | (0, 1) | PI level | All |
    | **distance_metric** | `Normalized` | enum | Distance metric (use `Weighted(weights)` for per-dim) | All |
    | **cell** | — | `T: Float` | Interpolation cell size | All |
    | **interpolation_vertices** | — | `usize` | Interpolation grid vertices | All |
    | **boundary_degree_fallback** | — | `bool` | Degree fallback at boundaries | All |
    | **cross_validate** | None | method | Auto-select fraction (`KFold` / `LOOCV` both accept `.seed()`) | Batch |
    | **chunk_size** | 5000 | [10, ∞) | Points per chunk | Streaming |
    | **overlap** | 500 | [0, chunk) | Overlap between chunks | Streaming |
    | **merge_strategy** | `WeightedAverage` | 4 options | Merge overlaps | Streaming |
    | **window_capacity** | 1000 | [3, ∞) | Max window size | Online |
    | **min_points** | 2 | [2, window] | Min before output | Online |
    | **update_mode** | `Incremental` | 2 options | Update strategy | Online |
=== "Julia"

    | Parameter | Default | Range/Options | Description | Adapter |
    | --- | --- | --- | --- | --- |
    | **fraction** | 0.67 | (0, 1] | Smoothing span | All |
    | **iterations** | 3 | [0, 1000] | Robustness iterations | All |
    | **degree** | 1 | 0–4 | Polynomial degree | All |
    | **surface_mode** | `"interpolation"` | 2 options | Fit vs interpolate | All |
    | **delta** | `nothing` (auto) | [0, ∞) | Interpolation threshold | All |
    | **weight_function** | `"tricube"` | 7 options | Distance kernel | All |
    | **robustness_method** | `"bisquare"` | 3 options | Outlier weighting | All |
    | **zero_weight_fallback** | `"use_local_mean"` | 3 options | Zero-weight behavior | All |
    | **boundary_policy** | `"extend"` | 4 options | Edge handling | All |
    | **scaling_method** | `"mad"` | 3 options | Scale estimation | All |
    | **auto_converge** | `nothing` | tolerance | Early stopping | All |
    | **custom_weights** | `nothing` | positive float | Per-observation weights | Batch |
    | **return_residuals** | `false` | bool | Include residuals | All |
    | **return_robustness_weights** | `false` | bool | Include weights | All |
    | **return_diagnostics** | `false` | bool | Include metrics | All |
    | **confidence_intervals** | `nothing` | (0, 1) | CI level | All |
    | **prediction_intervals** | `nothing` | (0, 1) | PI level | All |
    | **weighted_metric_weights** | `nothing` | `Vector{Float64}` | Per-dimension distance weights | All |
    | **cell** | `nothing` | (0, ∞) | Interpolation cell size | All |
    | **interpolation_vertices** | `nothing` | `Int` | Interpolation grid vertices | All |
    | **boundary_degree_fallback** | `nothing` | `Bool` | Degree fallback at boundaries | All |
    | **cv_method** | `nothing` | method | Auto-select fraction | Batch |
    | **cv_k** | 5 | [2, ∞) | K-fold count | Batch |
    | **cv_fractions** | `nothing` | `Vector{Float64}` | Fractions to evaluate | Batch |
    | **cv_seed** | `nothing` | `Int` | CV fold randomization seed | Batch |
    | **chunk_size** | 5000 | [10, ∞) | Points per chunk | Streaming |
    | **overlap** | 500 | [0, chunk) | Overlap between chunks | Streaming |
    | **merge_strategy** | `"weighted_average"` | 4 options | Merge overlaps | Streaming |
    | **window_capacity** | 1000 | [3, ∞) | Max window size | Online |
    | **min_points** | 2 | [2, window] | Min before output | Online |
    | **update_mode** | `"incremental"` | 2 options | Update strategy | Online |
=== "Node.js"

    | Parameter | Default | Range/Options | Description | Adapter |
    | --- | --- | --- | --- | --- |
    | **fraction** | 0.67 | (0, 1] | Smoothing span | All |
    | **iterations** | 3 | [0, 1000] | Robustness iterations | All |
    | **degree** | 1 | 0–4 | Polynomial degree | All |
    | **surfaceMode** | `"interpolation"` | 2 options | Fit vs interpolate | All |
    | **delta** | auto | [0, ∞) | Interpolation threshold | All |
    | **weightFunction** | `"tricube"` | 7 options | Distance kernel | All |
    | **robustnessMethod** | `"bisquare"` | 3 options | Outlier weighting | All |
    | **zeroWeightFallback** | `"use_local_mean"` | 3 options | Zero-weight behavior | All |
    | **boundaryPolicy** | `"extend"` | 4 options | Edge handling | All |
    | **scalingMethod** | `"mad"` | 3 options | Scale estimation | All |
    | **autoConverge** | null | tolerance | Early stopping | All |
    | **customWeights** | null | positive number | Per-observation weights | Batch |
    | **returnResiduals** | false | bool | Include residuals | All |
    | **returnRobustnessWeights** | false | bool | Include weights | All |
    | **returnDiagnostics** | false | bool | Include metrics | All |
    | **confidenceIntervals** | null | (0, 1) | CI level | All |
    | **predictionIntervals** | null | (0, 1) | PI level | All |
    | **weightedMetricWeights** | null | number[] | Per-dimension distance weights | All |
    | **cell** | null | (0, ∞) | Interpolation cell size | All |
    | **interpolationVertices** | null | number | Interpolation grid vertices | All |
    | **boundaryDegreeFallback** | null | boolean | Degree fallback at boundaries | All |
    | **cvMethod** | null | method | Auto-select fraction | Batch |
    | **cvK** | 5 | [2, ∞) | K-fold count | Batch |
    | **cvFractions** | null | number[] | Fractions to evaluate | Batch |
    | **cvSeed** | null | number | CV fold randomization seed | Batch |
    | **chunkSize** | 5000 | [10, ∞) | Points per chunk | Streaming |
    | **overlap** | 500 | [0, chunk) | Overlap between chunks | Streaming |
    | **mergeStrategy** | `"weighted_average"` | 4 options | Merge overlaps | Streaming |
    | **windowCapacity** | 1000 | [3, ∞) | Max window size | Online |
    | **minPoints** | 2 | [2, window] | Min before output | Online |
    | **updateMode** | `"incremental"` | 2 options | Update strategy | Online |
=== "WebAssembly"

    | Parameter | Default | Range/Options | Description | Adapter |
    | --- | --- | --- | --- | --- |
    | **fraction** | 0.67 | (0, 1] | Smoothing span | All |
    | **iterations** | 3 | [0, 1000] | Robustness iterations | All |
    | **degree** | 1 | 0–4 | Polynomial degree | All |
    | **surfaceMode** | `"interpolation"` | 2 options | Fit vs interpolate | All |
    | **delta** | auto | [0, ∞) | Interpolation threshold | All |
    | **weightFunction** | `"tricube"` | 7 options | Distance kernel | All |
    | **robustnessMethod** | `"bisquare"` | 3 options | Outlier weighting | All |
    | **zeroWeightFallback** | `"use_local_mean"` | 3 options | Zero-weight behavior | All |
    | **boundaryPolicy** | `"extend"` | 4 options | Edge handling | All |
    | **scalingMethod** | `"mad"` | 3 options | Scale estimation | All |
    | **autoConverge** | null | tolerance | Early stopping | All |
    | **customWeights** | null | positive number | Per-observation weights | Batch |
    | **returnResiduals** | false | bool | Include residuals | All |
    | **returnRobustnessWeights** | false | bool | Include weights | All |
    | **returnDiagnostics** | false | bool | Include metrics | All |
    | **confidenceIntervals** | null | (0, 1) | CI level | All |
    | **predictionIntervals** | null | (0, 1) | PI level | All |
    | **weightedMetricWeights** | null | number[] | Per-dimension distance weights | All |
    | **cell** | null | (0, ∞) | Interpolation cell size | All |
    | **interpolationVertices** | null | number | Interpolation grid vertices | All |
    | **boundaryDegreeFallback** | null | boolean | Degree fallback at boundaries | All |
    | **cvMethod** | null | method | Auto-select fraction | Batch |
    | **cvK** | 5 | [2, ∞) | K-fold count | Batch |
    | **cvFractions** | null | number[] | Fractions to evaluate | Batch |
    | **cvSeed** | null | number | CV fold randomization seed | Batch |
    | **chunkSize** | 5000 | [10, ∞) | Points per chunk | Streaming |
    | **overlap** | 500 | [0, chunk) | Overlap between chunks | Streaming |
    | **mergeStrategy** | `"weighted_average"` | 4 options | Merge overlaps | Streaming |
    | **windowCapacity** | 1000 | [3, ∞) | Max window size | Online |
    | **minPoints** | 2 | [2, window] | Min before output | Online |
    | **updateMode** | `"incremental"` | 2 options | Update strategy | Online |
=== "C++"

    | Parameter | Default | Range/Options | Description | Adapter |
    | --- | --- | --- | --- | --- |
    | **fraction** | 0.67 | (0, 1] | Smoothing span | All |
    | **iterations** | 3 | [0, 1000] | Robustness iterations | All |
    | **degree** | 1 | 0–4 | Polynomial degree | All |
    | **surface_mode** | `"interpolation"` | 2 options | Fit vs interpolate | All |
    | **delta** | NAN (auto) | [0, ∞) | Interpolation threshold | All |
    | **weight_function** | `"tricube"` | 7 options | Distance kernel | All |
    | **robustness_method** | `"bisquare"` | 3 options | Outlier weighting | All |
    | **zero_weight_fallback** | `"use_local_mean"` | 3 options | Zero-weight behavior | All |
    | **boundary_policy** | `"extend"` | 4 options | Edge handling | All |
    | **scaling_method** | `"mad"` | 3 options | Scale estimation | All |
    | **auto_converge** | NAN | tolerance | Early stopping | All |
    | **custom_weights** | {} | positive double | Per-observation weights | Batch |
    | **return_residuals** | false | bool | Include residuals | All |
    | **return_robustness_weights** | false | bool | Include weights | All |
    | **return_diagnostics** | false | bool | Include metrics | All |
    | **confidence_intervals** | NAN | (0, 1) | CI level | All |
    | **prediction_intervals** | NAN | (0, 1) | PI level | All |
    | **weighted_metric_weights** | {} | vector<double> | Per-dimension distance weights | All |
    | **cell** | NAN | (0, ∞) | Interpolation cell size | All |
    | **interpolation_vertices** | 0 | int | Interpolation grid vertices | All |
    | **boundary_degree_fallback** | -1 | int | Degree fallback at boundaries | All |
    | **cv_method** | — | string | Auto-select fraction | Batch |
    | **cv_k** | 5 | [2, ∞) | K-fold count | Batch |
    | **cv_fractions** | {} | vector<double> | Fractions to evaluate | Batch |
    | **cv_seed** | 0 | uint64 | CV fold randomization seed | Batch |
    | **chunk_size** | 5000 | [10, ∞) | Points per chunk | Streaming |
    | **overlap** | -1 (auto) | [0, chunk) | Overlap between chunks | Streaming |
    | **merge_strategy** | `"weighted_average"` | 4 options | Merge overlaps | Streaming |
    | **window_capacity** | 1000 | [3, ∞) | Max window size | Online |
    | **min_points** | 2 | [2, window] | Min before output | Online |
    | **update_mode** | `"incremental"` | 2 options | Update strategy | Online |

---

## Parameter Options Summary

=== "R / Python / Julia / C++"

| Parameter | Available Options |
| --- | --- |
| **weight_function** | `"tricube"`, `"epanechnikov"`, `"gaussian"`, `"biweight"`, `"cosine"`, `"triangle"`, `"uniform"` |
| **robustness_method** | `"bisquare"`, `"huber"`, `"talwar"` |
| **zero_weight_fallback** | `"use_local_mean"`, `"return_original"`, `"return_none"` |
| **boundary_policy** | `"extend"`, `"reflect"`, `"zero"`, `"noboundary"` |
| **scaling_method** | `"mad"`, `"mar"`, `"mean"` |
| **surface_mode** | `"interpolation"`, `"direct"` |
| **distance_metric** | `"normalized"`, `"euclidean"`, `"manhattan"`, `"chebyshev"`, `"minkowski:p"`, `"weighted"` |
| **merge_strategy** | `"average"`, `"weighted_average"`, `"take_first"`, `"take_last"` |
| **update_mode** | `"incremental"`, `"full"` |

=== "Rust"

| Parameter | Available Options |
| --- | --- |
| **weight_function** | `Tricube`, `Epanechnikov`, `Gaussian`, `Biweight`, `Cosine`, `Triangle`, `Uniform` |
| **robustness_method** | `Bisquare`, `Huber`, `Talwar` |
| **zero_weight_fallback** | `UseLocalMean`, `ReturnOriginal`, `ReturnNone` |
| **boundary_policy** | `Extend`, `Reflect`, `Zero`, `NoBoundary` |
| **scaling_method** | `MAD`, `MAR`, `Mean` |
| **surface_mode** | `Interpolation`, `Direct` |
| **distance_metric** | `Normalized`, `Euclidean`, `Manhattan`, `Chebyshev`, `Minkowski(T)`, `Weighted(Vec<T>)` |
| **merge_strategy** | `Average`, `WeightedAverage`, `TakeFirst`, `TakeLast` |
| **update_mode** | `Incremental`, `Full` |

=== "Node.js / WebAssembly"

| Parameter | Available Options |
| --- | --- |
| **weightFunction** | `"tricube"`, `"epanechnikov"`, `"gaussian"`, `"biweight"`, `"cosine"`, `"triangle"`, `"uniform"` |
| **robustnessMethod** | `"bisquare"`, `"huber"`, `"talwar"` |
| **zeroWeightFallback** | `"use_local_mean"`, `"return_original"`, `"return_none"` |
| **boundaryPolicy** | `"extend"`, `"reflect"`, `"zero"`, `"noboundary"` |
| **scalingMethod** | `"mad"`, `"mar"`, `"mean"` |
| **surfaceMode** | `"interpolation"`, `"direct"` |
| **distanceMetric** | `"normalized"`, `"euclidean"`, `"manhattan"`, `"chebyshev"`, `"minkowski:p"`, `"weighted"` |
| **mergeStrategy** | `"average"`, `"weighted_average"`, `"take_first"`, `"take_last"` |
| **updateMode** | `"incremental"`, `"full"` |

---

## Core Parameters

### fraction

The proportion of data used for each local fit. **Most important parameter.**

| Value | Effect | Use Case |
| --- | --- | --- |
| 0.1–0.3 | Fine detail | Rapidly changing signals |
| 0.3–0.5 | Balanced | General purpose |
| 0.5–0.7 | Heavy smoothing | Noisy data |
| 0.7–1.0 | Very smooth | Trend extraction |

![Fraction Comparison](../assets/diagrams/fraction_comparison.svg)

=== "R"
    ```r
    result <- Loess(fraction = 0.3)$fit(x, y)
    ```

=== "Python"
    ```python
    result = fl.Loess(fraction=0.3).fit(x, y)
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .fraction(0.3)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = fit(Loess(fraction=0.3), x, y)
    ```

=== "Node.js"
    ```javascript
    const result = new Loess({ fraction: 0.3 }).fit(x, y);
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { fraction: 0.3 });
    ```

=== "C++"
    ```cpp
    fastloess::Loess model({ .fraction = 0.3 });
    auto result = model.fit(x, y).value();
    ```

---

### iterations

Number of robustness iterations for outlier resistance.

| Value | Effect | Performance |
| --- | --- | --- |
| 0 | No robustness | Fastest |
| 1–3 | Moderate | Recommended |
| 4–6 | Strong | Contaminated data |
| 7+ | Very strong | Heavy outliers |

=== "R"
    ```r
    result <- Loess(iterations = 5)$fit(x, y)
    ```

=== "Python"
    ```python
    result = fl.Loess(iterations=5).fit(x, y)
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .iterations(5)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = fit(Loess(iterations=5), x, y)
    ```

=== "Node.js"
    ```javascript
    const result = new Loess({ iterations: 5 }).fit(x, y);
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { iterations: 5 });
    ```

=== "C++"
    ```cpp
    fastloess::Loess model({ .iterations = 5 });
    auto result = model.fit(x, y).value();
    ```

---

### degree

Polynomial degree for the local regression fits.

![Degree Comparison](../assets/diagrams/degree_comparison.svg)

| Degree | Fit Type |
| --- | --- |
| `0` | Local constant |
| `1` | Local linear (Default) |
| `2` | Local quadratic |
| `3` | Local cubic |
| `4` | Local quartic |

Higher degrees capture curvature but can overfit with small fractions. Degree 1 is appropriate for most use cases.

See [Polynomial Degree](degree.md) for a detailed comparison.

---

### surface_mode

Controls whether the local polynomial is evaluated at every query point or at a sparser grid of anchor vertices with Hermite cubic interpolation in between.

| Mode | Behavior | Speed | Accuracy |
| --- | --- | --- | --- |
| `Interpolation` (default) | Evaluate at vertices, interpolate between | Faster | Slight approximation |
| `Direct` | Evaluate at every query point | Slower | Full precision |

=== "R / Python / Julia / Node.js / WebAssembly / C++"

| Mode | String value |
| --- | --- |
| Interpolation (default) | `"interpolation"` |
| Direct | `"direct"` |

=== "Rust"

| Mode | Enum value |
| --- | --- |
| Interpolation (default) | `SurfaceMode::Interpolation` |
| Direct | `SurfaceMode::Direct` |

See [Polynomial Degree](degree.md#surface-mode) for a visual comparison.

=== "R"
    ```r
    result <- Loess(surface_mode = "direct")$fit(x, y)
    ```

=== "Python"
    ```python
    result = fl.Loess(surface_mode="direct").fit(x, y)
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .surface_mode(SurfaceMode::Direct)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = fit(Loess(surface_mode="direct"), x, y)
    ```

=== "Node.js"
    ```javascript
    const result = new Loess({ surfaceMode: "direct" }).fit(x, y);
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { surfaceMode: "direct" });
    ```

=== "C++"
    ```cpp
    fastloess::Loess model({ .surface_mode = "direct" });
    auto result = model.fit(x, y).value();
    ```

---

### cell

Cell size for the interpolation grid. Controls the density of anchor vertices when `surface_mode = "interpolation"`. Smaller values produce a finer grid, increasing accuracy at the cost of memory and computation.

- **Default**: `0.2` (20% of x-range per dimension)
- **Range**: `(0, ∞)` — values close to 0 approach `"direct"` accuracy
- **Adapter**: All

| `cell` | Grid density | Accuracy | Speed |
| --- | --- | --- | --- |
| `0.05` | Very fine | Highest | Slowest |
| `0.2` | Moderate (default) | High | Fast |
| `0.5` | Coarse | Lower | Faster |

=== "R"
    ```r
    result <- Loess(cell = 0.05)$fit(x, y)
    ```

=== "Python"
    ```python
    result = fl.Loess(cell=0.05).fit(x, y)
    ```

=== "Rust"
    ```rust
    let model = Loess::new().cell(0.05).adapter(Batch).build()?;
    ```

=== "Julia"
    ```julia
    result = fit(Loess(cell=0.05), x, y)
    ```

=== "Node.js"
    ```javascript
    const result = new Loess({ cell: 0.05 }).fit(x, y);
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { cell: 0.05 });
    ```

=== "C++"
    ```cpp
    fastloess::Loess model({ .cell = 0.05 });
    auto result = model.fit(x, y).value();
    ```

---

### interpolation_vertices

Explicitly set the number of anchor vertices for the interpolation grid, overriding the `cell`-based automatic count. Use when you need a precise vertex budget.

- **Default**: auto (derived from `cell` and data range)
- **Adapter**: All

=== "R"
    ```r
    result <- Loess(interpolation_vertices = 50L)$fit(x, y)
    ```

=== "Python"
    ```python
    result = fl.Loess(interpolation_vertices=50).fit(x, y)
    ```

=== "Rust"
    ```rust
    let model = Loess::new().interpolation_vertices(50).adapter(Batch).build()?;
    ```

=== "Julia"
    ```julia
    result = fit(Loess(interpolation_vertices=50), x, y)
    ```

=== "Node.js"
    ```javascript
    const result = new Loess({ interpolationVertices: 50 }).fit(x, y);
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { interpolationVertices: 50 });
    ```

=== "C++"
    ```cpp
    fastloess::Loess model({ .interpolation_vertices = 50 });
    auto result = model.fit(x, y).value();
    ```

---

### dimensions

Number of predictor variables. Enables multivariate LOESS over an n-dimensional input space.

![Multivariate LOESS](../assets/diagrams/multivariate_loess.svg)

- **1** (default): Standard 1D smoothing over a single predictor
- **2**: Spatial or bi-predictor surface smoothing
- **3+**: High-dimensional local regression

See [Multivariate LOESS](dimensions.md) for detailed usage and distance metric options.

---

### distance_metric / weighted_metric_weights

Distance metric for neighbourhood calculation. Only meaningful when `dimensions > 1`. The `"weighted"` metric lets you assign per-dimension importance via `weighted_metric_weights`.

| Metric | Description |
| --- | --- |
| `"normalized"` | Each dimension scaled to unit range (default) |
| `"euclidean"` | Raw Euclidean distance |
| `"manhattan"` | City-block distance |
| `"chebyshev"` | Maximum coordinate difference |
| `"minkowski:p"` | Generalised $L_p$ norm — e.g. `"minkowski:3"` |
| `"weighted"` | Weighted Euclidean — set `weighted_metric_weights` to one weight per dimension |

=== "R"
    ```r
    result <- Loess(
        dimensions = 2L,
        distance_metric = "weighted",
        weighted_metric_weights = c(2.0, 0.5)  # x1 twice as important
    )$fit(x2d, y)
    ```

=== "Python"
    ```python
    result = fl.Loess(
        dimensions=2,
        distance_metric="weighted",
        weighted_metric_weights=[2.0, 0.5]
    ).fit(x2d, y)
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .dimensions(2)
        .distance_metric(DistanceMetric::Weighted(vec![2.0, 0.5]))
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = fit(Loess(
        dimensions=2,
        distance_metric="weighted",
        weighted_metric_weights=[2.0, 0.5]
    ), x2d, y)
    ```

=== "Node.js"
    ```javascript
    const result = new Loess({
        dimensions: 2,
        distanceMetric: "weighted",
        weightedMetricWeights: [2.0, 0.5]
    }).fit(x2d, y);
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x2d, y, {
        dimensions: 2,
        distanceMetric: "weighted",
        weightedMetricWeights: [2.0, 0.5]
    });
    ```

=== "C++"
    ```cpp
    fastloess::LoessOptions opts;
    opts.dimensions = 2;
    opts.distance_metric = "weighted";
    opts.weighted_metric_weights = {2.0, 0.5};
    fastloess::Loess model(opts);
    auto result = model.fit(x2d, y).value();
    ```

---

### weight_function

Distance weighting kernel for local fits.

=== "R / Python / Julia / Node.js / WebAssembly / C++"

| Kernel | Efficiency | Smoothness |
| --- | --- | --- |
| `"tricube"` | 0.998 | Very smooth |
| `"epanechnikov"` | 1.000 | Smooth |
| `"gaussian"` | 0.961 | Infinite |
| `"biweight"` | 0.995 | Very smooth |
| `"cosine"` | 0.999 | Smooth |
| `"triangle"` | 0.989 | Moderate |
| `"uniform"` | 0.943 | None |

=== "Rust"

| Kernel | Efficiency | Smoothness |
| --- | --- | --- |
| `Tricube` | 0.998 | Very smooth |
| `Epanechnikov` | 1.000 | Smooth |
| `Gaussian` | 0.961 | Infinite |
| `Biweight` | 0.995 | Very smooth |
| `Cosine` | 0.999 | Smooth |
| `Triangle` | 0.989 | Moderate |
| `Uniform` | 0.943 | None |

See [Weight Functions](kernels.md) for detailed comparison.

=== "R"
    ```r
    result <- Loess(weight_function = "epanechnikov")$fit(x, y)
    ```

=== "Python"
    ```python
    result = fl.Loess(weight_function="epanechnikov").fit(x, y)
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .weight_function(Epanechnikov)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = fit(Loess(weight_function="epanechnikov"), x, y)
    ```

=== "Node.js"
    ```javascript
    const result = new Loess({ weightFunction: "epanechnikov" }).fit(x, y);
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { weightFunction: "epanechnikov" });
    ```

=== "C++"
    ```cpp
    fastloess::Loess model({ .weight_function = "epanechnikov" });
    auto result = model.fit(x, y).value();
    ```

---

### robustness_method

Method for downweighting outliers during iterative refinement.

=== "R / Python / Julia / Node.js / WebAssembly / C++"

| Method | Behavior | Use Case |
| --- | --- | --- |
| `"bisquare"` | Smooth downweighting | General-purpose |
| `"huber"` | Linear beyond threshold | Moderate outliers |
| `"talwar"` | Hard threshold (0 or 1) | Extreme contamination |

=== "Rust"

| Method | Behavior | Use Case |
| --- | --- | --- |
| `Bisquare` | Smooth downweighting | General-purpose |
| `Huber` | Linear beyond threshold | Moderate outliers |
| `Talwar` | Hard threshold (0 or 1) | Extreme contamination |

See [Robustness](robustness.md) for detailed comparison.

=== "R"
    ```r
    result <- Loess(robustness_method = "talwar")$fit(x, y)
    ```

=== "Python"
    ```python
    result = fl.Loess(robustness_method="talwar").fit(x, y)
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .robustness_method(Talwar)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = fit(Loess(robustness_method="talwar"), x, y)
    ```

=== "Node.js"
    ```javascript
    const result = new Loess({ robustnessMethod: "talwar" }).fit(x, y);
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { robustnessMethod: "talwar" });
    ```

=== "C++"
    ```cpp
    fastloess::Loess model({ .robustness_method = "talwar" });
    auto result = model.fit(x, y).value();
    ```

---

### boundary_policy

Edge handling strategy to reduce boundary bias. See [Boundary Handling](boundary.md) for a detailed comparison.

![Boundary Policy](../assets/diagrams/boundary_comparison.svg)

=== "R / Python / Julia / Node.js / WebAssembly / C++"

| Policy | Behavior | Use Case |
| --- | --- | --- |
| `"extend"` | Pad with first/last values | Most cases (default) |
| `"reflect"` | Mirror data at boundaries | Periodic/symmetric data |
| `"zero"` | Pad with zeros | Data approaches zero |
| `"noboundary"` | No padding | Original Cleveland behavior |

=== "Rust"

| Policy | Behavior | Use Case |
| --- | --- | --- |
| `Extend` | Pad with first/last values | Most cases (default) |
| `Reflect` | Mirror data at boundaries | Periodic/symmetric data |
| `Zero` | Pad with zeros | Data approaches zero |
| `NoBoundary` | No padding | Original Cleveland behavior |

For example:

=== "R"
    ```r
    result <- Loess(boundary_policy = "reflect")$fit(x, y)
    ```

=== "Python"
    ```python
    result = fl.Loess(boundary_policy="reflect").fit(x, y)
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .boundary_policy(Reflect)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = fit(Loess(boundary_policy="reflect"), x, y)
    ```

=== "Node.js"
    ```javascript
    const result = new Loess({ boundaryPolicy: "reflect" }).fit(x, y);
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { boundaryPolicy: "reflect" });
    ```

=== "C++"
    ```cpp
    fastloess::Loess model({ .boundary_policy = "reflect" });
    auto result = model.fit(x, y).value();
    ```

---

### boundary_degree_fallback

When enabled, the polynomial degree is automatically reduced to the highest degree that can be stably estimated for points near the boundary (where the local neighbourhood is one-sided). Prevents numerical failures when `degree ≥ 2` at the edges.

- **Default**: `false`
- **Adapter**: All

!!! tip
    Enable this if you observe NaN values or instability at the edges of your data when using `degree = "quadratic"` or higher.

=== "R"
    ```r
    result <- Loess(degree = "quadratic", boundary_degree_fallback = TRUE)$fit(x, y)
    ```

=== "Python"
    ```python
    result = fl.Loess(degree="quadratic", boundary_degree_fallback=True).fit(x, y)
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .degree(Quadratic)
        .boundary_degree_fallback(true)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = fit(Loess(degree="quadratic", boundary_degree_fallback=true), x, y)
    ```

=== "Node.js"
    ```javascript
    const result = new Loess({ degree: "quadratic", boundaryDegreeFallback: true }).fit(x, y);
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { degree: "quadratic", boundaryDegreeFallback: true });
    ```

=== "C++"
    ```cpp
    fastloess::Loess model({ .degree = "quadratic", .boundary_degree_fallback = 1 });
    auto result = model.fit(x, y).value();
    ```

---

### scaling_method

Method for estimating residual scale during robustness iterations. See [Scaling Methods](scaling.md) for a detailed comparison.

![Scaling Methods](../assets/diagrams/scaling_comparison.svg)

=== "R / Python / Julia / Node.js / WebAssembly / C++"

| Method | Description | Robustness |
| --- | --- | --- |
| `"mad"` | Median Absolute Deviation | Very robust |
| `"mar"` | Median Absolute Residual | Robust |
| `"mean"` | Mean Absolute Residual | Less robust |

=== "Rust"

| Method | Description | Robustness |
| --- | --- | --- |
| `MAD` | Median Absolute Deviation | Very robust |
| `MAR` | Median Absolute Residual | Robust |
| `Mean` | Mean Absolute Residual | Less robust |

For example:

=== "R"
    ```r
    result <- Loess(scaling_method = "mad")$fit(x, y)
    ```

=== "Python"
    ```python
    result = fl.Loess(scaling_method="mad").fit(x, y)
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .scaling_method(MAD)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = fit(Loess(scaling_method="mad"), x, y)
    ```

=== "Node.js / WebAssembly"
    ```javascript
    const result = new Loess({ scalingMethod: "mad" }).fit(x, y);
    ```

=== "C++"
    ```cpp
    fastloess::Loess model({ .scaling_method = "mad" });
    auto result = model.fit(x, y).value();
    ```

---

### zero_weight_fallback

Behavior when all neighborhood weights are zero.

![Zero-Weight Fallback Policies](../assets/diagrams/zero_weight_comparison.svg)

=== "R / Python / Julia / Node.js / WebAssembly / C++"

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` | Use mean of neighborhood (default) |
| `"return_original"` | Return original y value |
| `"return_none"` | Return NaN |

=== "Rust"

| Option | Behavior |
| --- | --- |
| `UseLocalMean` | Use mean of neighborhood (default) |
| `ReturnOriginal` | Return original y value |
| `ReturnNone` | Return NaN |

For example:

=== "R"
    ```r
    result <- Loess(zero_weight_fallback = "use_local_mean")$fit(x, y)
    ```

=== "Python"
    ```python
    result = fl.Loess(zero_weight_fallback="use_local_mean").fit(x, y)
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .zero_weight_fallback(UseLocalMean)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = fit(Loess(zero_weight_fallback="use_local_mean"), x, y)
    ```

=== "Node.js"
    ```javascript
    const result = new Loess({ zeroWeightFallback: "use_local_mean" }).fit(x, y);
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { zeroWeightFallback: "use_local_mean" });
    ```

=== "C++"
    ```cpp
    fastloess::Loess model({ .zero_weight_fallback = "use_local_mean" });
    auto result = model.fit(x, y).value();
    ```

---

### auto_converge

Enable early stopping when robustness weights stabilize.

=== "R"
    ```r
    result <- Loess(iterations = 20, auto_converge = 1e-6)$fit(x, y)
    ```

=== "Python"
    ```python
    result = fl.Loess(iterations=20, auto_converge=1e-6).fit(x, y)
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .iterations(20)           // Maximum
        .auto_converge(1e-6)      // Stop when change < 1e-6
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = fit(Loess(iterations=20, auto_converge=1e-6), x, y)
    ```

=== "Node.js"
    ```javascript
    const result = new Loess({ iterations: 20, autoConverge: 1e-6 }).fit(x, y);
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { iterations: 20, autoConverge: 1e-6 });
    ```

=== "C++"
    ```cpp
    fastloess::Loess model({ .iterations = 20, .auto_converge = 1e-6 });
    auto result = model.fit(x, y).value();
    ```

---

### custom_weights

Per-observation case weights that scale each point's contribution to nearby local fits.
Equivalent to the `weights` argument in R's `stats::loess`.

**Formula:** `w_ij = custom_weights[j] × K(d_ij / h) × robustness_j`

where `K` is the distance kernel and `robustness_j` is the robustness weight (if `iterations > 0`).

| Value | Effect |
| --- | --- |
| `1.0` for all | Equivalent to no weights (uniform) |
| `0.0` | Excludes the observation from all local fits |
| `> 1.0` | Increases the observation's influence |
| `0 < v < 1.0` | Reduces the observation's influence |

!!! note "Batch only"
    `custom_weights` is applied in **Batch** mode only. It is ignored in Streaming and Online modes.

!!! warning "Length must match y"
    The weights vector must have the same length as `y`. A mismatch returns an error.

=== "R"
    ```r
    # Downweight an outlier at index 5
    weights <- rep(1, length(y))
    weights[5] <- 0
    result <- Loess(custom_weights = weights)$fit(x, y)
    ```

=== "Python"
    ```python
    import numpy as np
    weights = np.ones(len(y))
    weights[4] = 0  # Exclude 5th point
    result = fl.Loess().fit(x, y, custom_weights=weights)
    ```

=== "Rust"
    ```rust
    let mut weights = vec![1.0f64; y.len()];
    weights[4] = 0.0; // Exclude 5th point
    let model = Loess::new()
        .custom_weights(weights)
        .adapter(Batch)
        .build()?;
    let result = model.fit(&x, &y)?;
    ```

=== "Julia"
    ```julia
    weights = ones(length(y))
    weights[5] = 0.0  # Exclude 5th point
    result = fit(Loess(), x, y; custom_weights=weights)
    ```

=== "Node.js"
    ```javascript
    const weights = new Array(y.length).fill(1);
    weights[4] = 0; // Exclude 5th point
    const result = new Loess({ customWeights: weights }).fit(x, y);
    ```

=== "WebAssembly"
    ```javascript
    const weights = new Array(y.length).fill(1);
    weights[4] = 0; // Exclude 5th point
    const result = smooth(x, y, { customWeights: weights });
    ```

=== "C++"
    ```cpp
    std::vector<double> weights(y.size(), 1.0);
    weights[4] = 0.0; // Exclude 5th point
    fastloess::Loess model({ .custom_weights = weights });
    auto result = model.fit(x, y).value();
    ```

---

## Output Options

### return_residuals

Include residuals (`y - smoothed`) in the output.

=== "R"
    ```r
    result <- Loess(return_residuals = TRUE)$fit(x, y)
    print(result$residuals)
    ```

=== "Python"
    ```python
    result = fl.Loess(return_residuals=True).fit(x, y)
    print(result.residuals)
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .return_residuals()
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y)?;
    if let Some(residuals) = result.residuals {
        println!("Residuals: {:?}", residuals);
    }
    ```

=== "Julia"
    ```julia
    result = fit(Loess(return_residuals=true), x, y)
    println(result.residuals)
    ```

=== "Node.js"
    ```javascript
    const result = new Loess({ returnResiduals: true }).fit(x, y);
    console.log(result.residuals);
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { returnResiduals: true });
    console.log(result.residuals);
    ```

=== "C++"
    ```cpp
    fastloess::Loess model({ .return_residuals = true });
    auto result = model.fit(x, y).value();
    auto residuals = result.residuals();
    ```

---

### return_diagnostics

Include fit quality metrics (Batch and Streaming only).

| Metric | Description |
| --- | --- |
| `rmse` | Root Mean Square Error |
| `mae` | Mean Absolute Error |
| `r_squared` | R² coefficient |
| `residual_sd` | Residual standard deviation |
| `effective_df` | Effective degrees of freedom |
| `aic` | Akaike Information Criterion |
| `aicc` | Corrected AIC |

=== "R"
    ```r
    result <- Loess(return_diagnostics = TRUE)$fit(x, y)
    cat(sprintf("R²: %.4f\n", result$diagnostics$r_squared))
    ```

=== "Python"
    ```python
    result = fl.Loess(return_diagnostics=True).fit(x, y)
    print(f"R²: {result.diagnostics.r_squared:.4f}")
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .return_diagnostics()
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y)?;
    if let Some(diag) = result.diagnostics {
        println!("R²: {:.4}", diag.r_squared);
        println!("RMSE: {:.4}", diag.rmse);
    }
    ```

=== "Julia"
    ```julia
    result = fit(Loess(return_diagnostics=true), x, y)
    println("R²: ", result.diagnostics.r_squared)
    ```

=== "Node.js"
    ```javascript
    const result = new Loess({ returnDiagnostics: true }).fit(x, y);
    console.log("R²:", result.diagnostics.rSquared);
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { returnDiagnostics: true });
    console.log("R²:", result.diagnostics.rSquared);
    ```

=== "C++"
    ```cpp
    fastloess::Loess model({ .return_diagnostics = true });
    auto result = model.fit(x, y).value();
    auto diag = result.diagnostics();
    std::cout << "R²: " << diag.rSquared() << std::endl;
    ```

---

### return_robustness_weights

Include final robustness weights (useful for outlier detection).

=== "R"
    ```r
    result <- Loess(iterations = 3, return_robustness_weights = TRUE)$fit(x, y)
    outliers <- which(result$robustness_weights < 0.5)
    ```

=== "Python"
    ```python
    result = fl.Loess(iterations=3, return_robustness_weights=True).fit(x, y)
    outliers = [i for i, w in enumerate(result.robustness_weights) if w < 0.5]
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .iterations(3)
        .return_robustness_weights()
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y)?;
    // Points with weight < 0.5 are likely outliers
    ```

=== "Julia"
    ```julia
    result = fit(Loess(iterations=3, return_robustness_weights=true), x, y)
    # Points with result.robustness_weights < 0.5 are likely outliers
    ```

=== "Node.js"
    ```javascript
    const result = new Loess({ iterations: 3, returnRobustnessWeights: true }).fit(x, y);
    // result.robustnessWeights contains outlier weights
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { iterations: 3, returnRobustnessWeights: true });
    // result.robustnessWeights contains outlier weights
    ```

=== "C++"
    ```cpp
    fastloess::Loess model({ .iterations = 3, .return_robustness_weights = true });
    auto result = model.fit(x, y).value();
    auto weights = result.robustnessWeights();
    ```

---

### confidence_intervals / prediction_intervals

Request uncertainty estimates (Batch only).

See [Intervals](intervals.md) for detailed usage.

=== "R"
    ```r
    result <- Loess(confidence_intervals = 0.95, prediction_intervals = 0.95)$fit(x, y)
    ```

=== "Python"
    ```python
    result = fl.Loess(confidence_intervals=0.95, prediction_intervals=0.95).fit(x, y)
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = fit(Loess(confidence_intervals=0.95, prediction_intervals=0.95), x, y)
    ```

=== "Node.js"
    ```javascript
    const result = new Loess({ confidenceIntervals: 0.95, predictionIntervals: 0.95 }).fit(x, y);
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { confidenceIntervals: 0.95, predictionIntervals: 0.95 });
    ```

=== "C++"
    ```cpp
    fastloess::Loess model({ .confidence_intervals = 0.95, .prediction_intervals = 0.95 });
    auto result = model.fit(x, y).value();
    ```

---

## CV Methods

### cv_method

Selection strategy for automated parameter tuning.

| Method | Description | Speed |
| --- | --- | --- |
| `"kfold"` | K-Fold Cross-Validation | Fast |
| `"loocv"` | Leave-One-Out Cross-Validation | Slow |

=== "R"
    ```r
    result <- Loess(cv_method = "kfold", cv_k = 5)$fit(x, y)
    ```

=== "Python"
    ```python
    result = fl.Loess(cv_method="kfold", cv_k=5).fit(x, y)
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .cross_validate(KFold(5, &[0.1, 0.3, 0.5]))
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = fit(Loess(cv_method="kfold", cv_k=5), x, y)
    ```

=== "Node.js"
    ```javascript
    // Coming soon
    ```

=== "WebAssembly"
    ```javascript
    // Coming soon
    ```

=== "C++"
    ```cpp
    auto model = fastloess::Loess::new()
        .cross_validate(fastloess::KFold(5, {0.1, 0.3, 0.5}))
        .adapter(fastloess::Batch)
        .build();
    ```

---

## Adapter Parameters

### chunk_size

Points per chunk in Streaming mode.

=== "R"
    ```r
    result <- StreamingLoess(chunk_size = 10000)$process_chunk(x, y)
    ```

=== "Python"
    ```python
    model = fl.StreamingLoess(chunk_size=10000)
    model.process_chunk(x, y)
    result = model.finalize()
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .adapter(Streaming)
        .chunk_size(10000)
        .build()?;
    ```

=== "Julia"
    ```julia
    model = StreamingLoess(chunk_size=10000)
    process_chunk(model, x, y)
    result = finalize(model)
    ```

=== "Node.js"
    ```javascript
    const processor = new StreamingLoess({}, { chunkSize: 10000 });
    ```

=== "WebAssembly"
    ```javascript
    const processor = new StreamingLoessWasm({}, { chunkSize: 10000 });
    ```

=== "C++"
    ```cpp
    fastloess::StreamingOptions opts;
    opts.chunk_size = 10000;
    fastloess::StreamingLoess stream(opts);
    (void)stream.processChunk(x, y);
    auto result = stream.finalize().value();
    ```

---

### overlap

Overlap between chunks in Streaming mode.

=== "R"
    ```r
    result <- StreamingLoess(overlap = 1000)$process_chunk(x, y)
    ```

=== "Python"
    ```python
    model = fl.StreamingLoess(overlap=1000)
    model.process_chunk(x, y)
    result = model.finalize()
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .adapter(Streaming)
        .overlap(1000)
        .build()?;
    ```

=== "Julia"
    ```julia
    model = StreamingLoess(overlap=1000)
    process_chunk(model, x, y)
    result = finalize(model)
    ```

=== "Node.js"
    ```javascript
    const processor = new StreamingLoess({}, { overlap: 1000 });
    ```

=== "WebAssembly"
    ```javascript
    const processor = new StreamingLoessWasm({}, { overlap: 1000 });
    ```

=== "C++"
    ```cpp
    fastloess::StreamingOptions opts;
    opts.overlap = 1000;
    fastloess::StreamingLoess stream(opts);
    (void)stream.processChunk(x, y);
    auto result = stream.finalize().value();
    ```

---

### merge_strategy

Method for merging overlapping chunks. See [Merge Strategies](merge.md) for a detailed comparison.

=== "R / Python / Julia / Node.js / WebAssembly / C++"

| Strategy | Description | Robustness |
| --- | --- | --- |
| `"average"` | Average of overlapping chunks | Fastest, least robust |
| `"take_first"` | Left chunk only | Fastest, least robust |
| `"take_last"` | Right chunk only | Fastest, least robust |
| `"weighted_average"` | Weighted average of overlapping chunks | Most robust |

=== "Rust"

| Strategy | Description | Robustness |
| --- | --- | --- |
| `Average` | Average of overlapping chunks | Fastest, least robust |
| `TakeFirst` | Left chunk only | Fastest, least robust |
| `TakeLast` | Right chunk only | Fastest, least robust |
| `WeightedAverage` | Weighted average of overlapping chunks | Most robust |

For example:

=== "R"
    ```r
    result <- StreamingLoess(merge_strategy = "weighted_average")$process_chunk(x, y)
    ```

=== "Python"
    ```python
    model = fl.StreamingLoess(merge_strategy="weighted_average")
    model.process_chunk(x, y)
    result = model.finalize()
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .adapter(Streaming)
        .merge_strategy(WeightedAverage)
        .build()?;
    ```

=== "Julia"
    ```julia
    model = StreamingLoess(merge_strategy="weighted_average")
    process_chunk(model, x, y)
    result = finalize(model)
    ```

=== "Node.js"
    ```javascript
    const processor = new StreamingLoess({}, { mergeStrategy: "weighted_average" });
    ```

=== "WebAssembly"
    ```javascript
    const processor = new StreamingLoessWasm({}, { mergeStrategy: "weighted_average" });
    ```

=== "C++"
    ```cpp
    // merge_strategy is handled internally in C++
    fastloess::StreamingLoess stream({});
    (void)stream.processChunk(x, y);
    auto result = stream.finalize().value();
    ```

---

### window_capacity

Maximum points held in memory for Online mode.

=== "R"
    ```r
    result <- OnlineLoess(window_capacity = 500)$add_points(x, y)
    ```

=== "Python"
    ```python
    model = fl.OnlineLoess(window_capacity=500)
    result = model.add_points(x, y)
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .adapter(Online)
        .window_capacity(500)
        .build()?;
    ```

=== "Julia"
    ```julia
    model = OnlineLoess(window_capacity=500)
    result = add_points(model, x, y)
    ```

=== "Node.js"
    ```javascript
    const processor = new OnlineLoess({}, { windowCapacity: 500 });
    ```

=== "WebAssembly"
    ```javascript
    const processor = new OnlineLoessWasm({}, { windowCapacity: 500 });
    ```

=== "C++"
    ```cpp
    fastloess::OnlineOptions opts;
    opts.window_capacity = 500;
    fastloess::OnlineLoess model(opts);
    auto result = model.addPoints(x, y).value();
    ```

---

### min_points

Minimum points required before Online filter starts producing outputs.

=== "R"
    ```r
    result <- OnlineLoess(min_points = 10)$add_points(x, y)
    ```

=== "Python"
    ```python
    model = fl.OnlineLoess(min_points=10)
    result = model.add_points(x, y)
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .adapter(Online)
        .min_points(10)
        .build()?;
    ```

=== "Julia"
    ```julia
    model = OnlineLoess(min_points=10)
    result = add_points(model, x, y)
    ```

=== "Node.js"
    ```javascript
    const processor = new OnlineLoess({}, { minPoints: 10 });
    ```

=== "WebAssembly"
    ```javascript
    const processor = new OnlineLoessWasm({}, { minPoints: 10 });
    ```

=== "C++"
    ```cpp
    fastloess::OnlineOptions opts;
    opts.min_points = 10;
    fastloess::OnlineLoess model(opts);
    auto result = model.addPoints(x, y).value();
    ```

---

### update_mode

Optimization strategy for Online mode updates.

=== "R / Python / Julia / Node.js / WebAssembly / C++"

| Mode | Description | Speed |
| --- | --- | --- |
| `full` | Re-smooth entire window | Slow |
| `incremental` | Update only affected fits | Fast |

=== "Rust"

| Mode | Description | Speed |
| --- | --- | --- |
| `Full` | Re-smooth entire window | Slow |
| `Incremental` | Update only affected fits | Fast |

For example:

=== "R"
    ```r
    result <- OnlineLoess(update_mode = "full")$add_points(x, y)
    ```

=== "Python"
    ```python
    model = fl.OnlineLoess(update_mode="full")
    result = model.add_points(x, y)
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .adapter(Online)
        .update_mode(Full)
        .build()?;
    ```

=== "Julia"
    ```julia
    model = OnlineLoess(update_mode="full")
    result = add_points(model, x, y)
    ```

=== "Node.js"
    ```javascript
    const processor = new OnlineLoess({}, { updateMode: "full" });
    ```

=== "WebAssembly"
    ```javascript
    const processor = new OnlineLoessWasm({}, { updateMode: "full" });
    ```

=== "C++"
    ```cpp
    fastloess::OnlineOptions opts;
    opts.update_mode = "full";
    fastloess::OnlineLoess model(opts);
    auto result = model.addPoints(x, y).value();
    ```
