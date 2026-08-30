# API

The Rust crates provide the core implementation and high-performance extensions.

## When to Use Batch Adapter

- Dataset fits in memory
- Need intervals, cross-validation, or diagnostics
- Processing complete files

![Gap Handling](https://raw.githubusercontent.com/thisisamirv/loess-project/main/crates/fastLoess/assets/diagrams/gap_handling.svg)

## Structs & Usage

Both crates expose the same three entry types via their `prelude`: `Loess` for batch mode, `StreamingLoess` for chunked processing, and `OnlineLoess` for sliding-window updates.

> **StreamingLoess** and **OnlineLoess** are documented separately: [rust-streaming.md](api-streaming.md), [rust-online.md](api-online.md)

```text
use fastLoess::prelude::*;  // or: use loess_rs::prelude::*;
```

### `Loess` (Batch)

Standard in-memory smoothing.

**Constructor:**

```rust
use fastLoess::prelude::*;

fn main() -> Result<(), LoessError> {
    let builder = Loess::new(); // Batch is default

    Ok(())
}
```

**Methods:**

```rust
use fastLoess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Loess::new().fraction(0.5f64).build()?;
    let result = model.fit(&x, &y)?;
    println!("Fraction used: {}", result.fraction_used);
    println!("Iterations used: {:?}", result.iterations_used);

    Ok(())
}
```

```output
Fraction used: 0.5
Iterations used: Some(3)
```

- Fits the model to the provided `x` and `y` arrays.
- Returns `Result<LoessResult<T>, LoessError>`.

See [rust-streaming.md](api-streaming.md) for the `StreamingLoess` struct.

See [rust-online.md](api-online.md) for the `OnlineLoess` struct.

## Builder Configuration

These chained methods configure the builder. They correspond to the "Options Structures" in other bindings.

### Loess Options

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
| `custom_weights(Vec<T>)` | `Vec<T: Float>` | disabled | Per-observation case weights (Batch only) — see [Custom Weights](custom-weights.md) |
| `confidence_intervals(T)` | `T: Float` | disabled | Confidence level (e.g., 0.95) — see [Intervals](intervals.md) |
| `prediction_intervals(T)` | `T: Float` | disabled | Prediction level (e.g., 0.95) — see [Intervals](intervals.md) |
| `return_diagnostics()` | `bool` | `false` | Compute RMSE, MAE, R², AIC |
| `return_residuals()` | `bool` | `false` | Include residuals in result |
| `return_robustness_weights()` | `bool` | `false` | Include robustness weights in result |
| `return_se()` | `bool` | `false` | Compute hat-matrix statistics (enp, leverage …) |
| `parallel(bool)` | `bool` | `true` | Enable parallel execution |
| `degree(...)` | `degree` | `"linear"` | Polynomial degree — see [Polynomial Degree](degree.md) |
| `dimensions(usize)` | `usize` | `1` | Number of predictor dimensions — see [Multivariate LOESS](dimensions.md) |
| `distance_metric(...)` | `distance_metric` | `"normalized"` | Distance metric |
| `weighted_metric_weights(Vec<T>)` | `Vec<T: Float>` | disabled | Per-dimension weights (used when `distance_metric = "weighted"`) |
| `surface_mode(...)` | `surface_mode` | `"interpolation"` | Surface computation mode |
| `cell(T)` | `T: Float` | disabled | Cell size for interpolation grid (smaller → more vertices, higher accuracy) |
| `interpolation_vertices(usize)` | `usize` | disabled | Number of interpolation vertices |
| `boundary_degree_fallback(bool)` | `bool` | `true` | Fall back to lower polynomial degree at boundaries when higher degrees fail |
| `cv_method(...)` | `&str` | disabled | Cross-validation method — see [Cross-Validation](cross-validation.md) |
| `cv_k(...)` | `usize` | disabled | Number of folds for K-fold cross-validation |
| `cv_fractions(...)` | `Vec<f64>` | disabled | Candidate fractions to evaluate during cross-validation |
| `cv_seed(...)` | `u64` | disabled | Random seed for reproducible fold assignments |

`fraction` is the most important parameter: it controls the size of the local neighbourhood used at each point.

| Range | Effect | Use case |
| --- | --- | --- |
| 0.1-0.3 | Fine detail | Rapidly changing signals |
| 0.3-0.5 | Balanced | General purpose |
| 0.5-0.7 | Heavy smoothing | Noisy data |
| 0.7-1.0 | Very smooth | Trend extraction |

`iterations` controls robustness to outliers, at the cost of speed.

| Value | Effect | Performance |
| --- | --- | --- |
| 0 | No robustness | Fastest |
| 1-3 | Moderate | Recommended |
| 4-6 | Strong | Contaminated data |
| 7+ | Very strong | Heavy outliers |

See [rust-streaming.md](api-streaming.md) for Streaming Options.

See [rust-online.md](api-online.md) for Online Options.

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

*See: [Weight Functions](kernels.md)*

- `"tricube"` (default)
- `"epanechnikov"`
- `"gaussian"`
- `"uniform"` (alias: `"boxcar"`)
- `"biweight"` (alias: `"bisquare"`)
- `"triangle"` (alias: `"triangular"`)
- `"cosine"`

### robustness_method

*See: [Robustness](robustness.md)*

- `"bisquare"` (default; alias: `"biweight"`)
- `"huber"`
- `"talwar"`

### boundary_policy

*See: [Boundary Handling](boundary.md)*

- `"extend"` (default; alias: `"pad"`)
- `"reflect"` (alias: `"mirror"`)
- `"zero"`
- `"noboundary"` (alias: `"none"`)

### scaling_method

*See: [Scaling Methods](scaling.md)*

- `"mad"` (default; alias: `"median_absolute_deviation"`)
- `"mar"` (alias: `"median_absolute_residual"`)
- `"mean"` (alias: `"mean_absolute_residual"`)

### zero_weight_fallback

Behavior when all neighborhood weights are zero:

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`) | Use the mean of the neighborhood |
| `"return_original"` (alias: `"original"`) | Return the original y value |
| `"return_none"` (alias: `"none"`) | Return `NaN` |

### distance_metric

*See: [Multivariate LOESS](dimensions.md)*

- `"normalized"` (default — scales each dimension by its range; alias: `"norm"`)
- `"euclidean"` (alias: `"euclid"`)
- `"manhattan"` (alias: `"l1"`)
- `"chebyshev"` (alias: `"linf"`)
- `"minkowski"` or `"minkowski:p"` for a custom exponent
- `"weighted"` plus `.weighted_metric_weights(vec![...])` (alias: `"weighted_euclidean"`)

### degree

*See: [Polynomial Degree](degree.md)*

- `"constant"` or `"0"` (degree 0)
- `"linear"` or `"1"` (default, degree 1)
- `"quadratic"` or `"2"` (degree 2)
- `"cubic"` or `"3"` (degree 3)
- `"quartic"` or `"4"` (degree 4)

### surface_mode

*See: [Polynomial Degree](degree.md#surface-mode)*

Controls whether the local polynomial is evaluated at every query point or at a sparser grid of anchor vertices with Hermite cubic interpolation in between.

| Mode | Behavior | Speed | Accuracy |
| --- | --- | --- | --- |
| `"interpolation"` (default) | Evaluate at vertices, interpolate between | Faster | Slight approximation |
| `"direct"` | Evaluate at every query point | Slower | Full precision |

### merge_strategy

See [rust-streaming.md](api-streaming.md).

### update_mode

See [rust-online.md](api-online.md).

## Example

```rust
use fastLoess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Loess::new()
        .fraction(0.5)
        .iterations(3)
        .build()?;

    let result = model.fit(&x, &y)?;

    println!("Smoothed Y: {:?}", result.y);

    Ok(())
}
```

```output
Smoothed Y: [0.32737554007097225, 0.35073954500496546, 0.37739076052679205, 0.4079090586832767, 0.44013801836005667, 0.47175028090334875, 0.5030693120508469, 0.5344185775402449, 0.566121543109237, 0.5985016744955168, 0.6318824374367782, 0.6665872976707156, 0.7029397209350224, 0.7372002465968626, 0.7658825967811843, 0.789690011362359, 0.8093257302147573, 0.8254929932127506, 0.8388950402307097, 0.8502351111430055, 0.8602164458240097, 0.8695422841480926, 0.8789158659896257, 0.887153972292037, 0.892497810781526, 0.8947872615066219, 0.8938622045158529, 0.8895625198577481, 0.8817280875808359, 0.870198787733645, 0.8548145003647043, 0.8354151055225423, 0.8145176987587835, 0.7940917447778371, 0.7731556773574066, 0.7507279302751975, 0.7258269373089133, 0.697471132236259, 0.664678948834939, 0.6264688208826581, 0.58185918215712, 0.5338061063516419, 0.485746338309858, 0.4374193185349509, 0.38856448753010236, 0.3389212857984939, 0.28822915384330855, 0.23622753216772754, 0.1826558612749335, 0.12725358166810816, 0.07166276905292143, 0.01758784291743965, -0.03513824056156522, -0.08668252520731944, -0.13721205484305027, -0.18689387329198498, -0.2358950243773498, -0.2843825519223721, -0.3325234997502794, -0.38048491168429727, -0.42502115342682245, -0.4633916611067457, -0.496521084187035, -0.5253340721306595, -0.5507552744005872, -0.5737093404597874, -0.5951209197712283, -0.6159146617978789, -0.637015216002707, -0.6567417543108368, -0.672824952761191, -0.6853067169874377, -0.6942289526232445, -0.6996335653022793, -0.7015624606582102, -0.700057544324705, -0.6951607219354317, -0.686913899124058, -0.675358981524252, -0.6626748096467386, -0.6504220592509987, -0.6377153888438335, -0.6236694569320449, -0.6073989220224337, -0.588018442621802, -0.5646426772369509, -0.5363862843746818, -0.5023639225417965, -0.465957340092984, -0.43100065173974045, -0.3972870655240804, -0.3646097894880157, -0.3327620316735591, -0.30153700012272333, -0.2707279028775207, -0.2401279479799649, -0.20953034347206806, -0.1787282973958432, -0.15006975314180593, -0.12554500177583605]
```
