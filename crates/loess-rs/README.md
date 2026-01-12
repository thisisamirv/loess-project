# loess-rs

[![Crates.io](https://img.shields.io/crates/v/loess-rs.svg)](https://crates.io/crates/loess-rs)
[![Documentation](https://docs.rs/loess-rs/badge.svg)](https://docs.rs/loess-rs)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

A high-performance implementation of LOESS (Locally Estimated Scatterplot Smoothing) in Rust. This crate provides a robust, production-ready implementation with support for confidence intervals, multiple kernel functions, and optimized execution modes.

> [!IMPORTANT]
> For parallelization or `ndarray` support, use [`fastLoess`](https://github.com/av746/fastLoess).

## How LOESS works

LOESS creates smooth curves through scattered data using local weighted neighborhoods:

![LOESS Smoothing Concept](https://raw.githubusercontent.com/thisisamirv/loess-rs/main/docs/loess_concept.svg)

## LOESS vs. LOWESS

| Feature               | LOESS (This Crate)                | LOWESS                         |
|-----------------------|-----------------------------------|--------------------------------|
| **Polynomial Degree** | Linear, Quadratic, Cubic, Quartic | Linear (Degree 1)              |
| **Dimensions**        | Multivariate (n-D support)        | Univariate (1-D only)          |
| **Flexibility**       | High (Distance metrics)           | Standard                       |
| **Complexity**        | Higher (Matrix inversion)         | Lower (Weighted average/slope) |

LOESS can fit higher-degree polynomials for more complex data:

![Degree Comparison](https://raw.githubusercontent.com/thisisamirv/loess-rs/main/docs/degree_comparison.svg)

LOESS can also handle multivariate data (n-D), while LOWESS is limited to univariate data (1-D):

![Multivariate LOESS](https://raw.githubusercontent.com/thisisamirv/loess-rs/main/docs/multivariate_loess.svg)

> [!TIP]
> **Note:** For a simple, lightweight, and fast **LOWESS** implementation, use [`lowess`](https://github.com/thisisamirv/lowess) crate.

## Features

- **Robust Statistics**: IRLS with Bisquare, Huber, or Talwar weighting for outlier handling.
- **Multidimensional Smoothing**: Support for n-D data with customizable distance metrics (Euclidean, Manhattan, etc.).
- **Flexible Fitting**: Linear, Quadratic, Cubic, and Quartic local polynomials.
- **Uncertainty Quantification**: Point-wise standard errors, confidence intervals, and prediction intervals.
- **Optimized Performance**: Interpolation surface with Tensor Product Hermite interpolation and streaming/online modes for large or real-time datasets.
- **Parameter Selection**: Built-in cross-validation for automatic smoothing fraction selection.
- **Flexibility**: Multiple weight kernels (Tricube, Epanechnikov, etc.) and `no_std` support (requires `alloc`).
- **Validated**: Numerical twin of R's `stats::loess` with exact match (< 1e-12 diff).

## Performance

Benchmarked against R's `loess`. Achieves **4.4×–53× faster performance** across all tested scenarios. No regressions observed.

### Summary

| Category               | Matched | Median Speedup | Mean Speedup |
|------------------------|---------|----------------|--------------|
| **Polynomial Degrees** | 2       | **21.67×**     | 21.67×       |
| **Fraction**           | 6       | **13.36×**     | 19.70×       |
| **Iterations**         | 6       | **16.35×**     | 15.90×       |
| **Pathological**       | 4       | **16.14×**     | 16.27×       |
| **Dimensions**         | 3       | **7.98×**      | 8.01×        |
| **Scalability**        | 2       | **5.86×**      | 5.86×        |
| **Genomic**            | 2       | **5.27×**      | 5.27×        |
| **Financial**          | 3       | **4.88×**      | 5.97×        |
| **Scientific**         | 3       | **4.46×**      | 5.28×        |

### Top 10 Performance Wins

| Benchmark        | Rust   | R       | Speedup    |
|------------------|--------|---------|------------|
| fraction_0.67    | 0.83ms | 44.80ms | **53.71×** |
| fraction_0.5     | 1.22ms | 32.25ms | **26.50×** |
| degree_quadratic | 0.75ms | 19.21ms | **25.52×** |
| high_noise       | 1.60ms | 34.68ms | **21.68×** |
| iterations_1     | 0.80ms | 15.88ms | **19.94×** |
| iterations_0     | 0.75ms | 13.68ms | **18.25×** |
| degree_linear    | 0.78ms | 13.92ms | **17.81×** |
| clustered        | 1.15ms | 19.74ms | **17.17×** |
| iterations_2     | 0.97ms | 16.63ms | **17.13×** |
| iterations_3     | 1.13ms | 17.56ms | **15.57×** |

Check [Benchmarks](https://github.com/thisisamirv/loess-rs/tree/bench/benchmarks) for detailed results and reproducible benchmarking code.

## Robustness Advantages

This implementation includes several robustness features beyond R's `loess`:

### MAD-Based Scale Estimation

Uses **MAD-based scale estimation** for robustness weight calculations:

```text
s = median(|r_i - median(r)|)
```

MAD is a **breakdown-point-optimal** estimator—it remains valid even when up to 50% of data are outliers, compared to the median of absolute residuals used by some other implementations.

Median Absolute Residual (MAR), which is the default Cleveland's choice, is also available through the `scaling_method` parameter.

### Configurable Boundary Policies

R's `loess` uses asymmetric windows at data boundaries, which can introduce edge bias. This implementation offers configurable **boundary policies** to mitigate this:

- **Extend** (default): Pad with constant values for symmetric windows
- **Reflect**: Mirror data at boundaries (best for periodic data)
- **Zero**: Pad with zeros (signal processing applications)
- **NoBoundary**: Original R behavior (no padding)

### Boundary Degree Fallback

When using `Interpolation` mode with higher polynomial degrees (Quadratic, Cubic), vertices outside the tight data bounds can produce unstable extrapolation. This implementation offers a configurable **boundary degree fallback**:

- **`true`** (default): Reduce to Linear fits at boundary vertices (more stable)
- **`false`**: Use full requested degree everywhere (matches R exactly)

## Validation

The Rust `loess-rs` crate is a **numerical twin** of R's `loess` implementation:

| Aspect          | Status         | Details                                    |
|-----------------|----------------|--------------------------------------------|
| **Accuracy**    | ✅ EXACT MATCH | Max diff < 1e-12 across all scenarios      |
| **Consistency** | ✅ PERFECT     | 20/20 scenarios pass with strict tolerance |
| **Robustness**  | ✅ VERIFIED    | Robust smoothing matches R exactly         |

Check [Validation](https://github.com/thisisamirv/loess-rs/tree/bench/validation) for detailed scenario results.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
loess-rs = "0.2"
```

For `no_std` environments:

```toml
[dependencies]
loess-rs = { version = "0.2", default-features = false }
```

## Quick Start

```rust
use loess_rs::prelude::*;

fn main() -> Result<(), LoessError> {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];

    // Build and fit model
    let result = Loess::new()
        .fraction(0.5)      // Use 50% of data for each local fit
        .iterations(3)      // 3 robustness iterations
        .adapter(Batch)
        .build()?
        .fit(&x, &y)?;

    println!("{}", result);
    Ok(())
}
```

```text
Summary:
  Data points: 5
  Fraction: 0.5

Smoothed Data:
       X     Y_smooth
  --------------------
    1.00     2.00000
    2.00     4.10000
    3.00     5.90000
    4.00     8.20000
    5.00     9.80000
```

## Builder Methods

All builder parameters have sensible defaults. You only need to specify what you want to change.

```rust
use loess_rs::prelude::*;

Loess::new()
    // Smoothing span (0, 1] - default: 0.67
    .fraction(0.5)

    // Polynomial degree - default: Linear
    .degree(Quadratic)

    // Number of dimensions - default: 1
    .dimensions(2)

    // Distance metric - default: Euclidean
    .distance_metric(Manhattan)

    // Robustness iterations - default: 3
    .iterations(5)

    // Kernel selection - default: Tricube
    .weight_function(Epanechnikov)

    // Robustness method - default: Bisquare
    .robustness_method(Huber)

    // Boundary handling - default: Extend
    .boundary_policy(Reflect)

    // Boundary degree fallback - default: true
    .boundary_degree_fallback(true)

    // Confidence intervals (Batch only)
    .confidence_intervals(0.95)

    // Prediction intervals (Batch only)
    .prediction_intervals(0.95)

    // Include diagnostics
    .return_diagnostics()
    .return_residuals()
    .return_robustness_weights()

    // Cross-validation (Batch only)
    .cross_validate(KFold(5, &[0.3, 0.5, 0.7]).seed(123))

    // Auto-convergence
    .auto_converge(1e-4)

    // Interpolation settings
    .surface_mode(Interpolation)

    // Interpolation cell size - default: 0.2
    .cell(0.2)

    // Execution mode
    .adapter(Batch)

    // Build the model
    .build()?;
```

## Result Structure

```rust
pub struct LoessResult<T> {
    /// Sorted x values (independent variable)
    pub x: Vec<T>,

    /// Smoothed y values (dependent variable)
    pub y: Vec<T>,

    /// Point-wise standard errors of the fit
    pub standard_errors: Option<Vec<T>>,

    /// Confidence interval bounds (if computed)
    pub confidence_lower: Option<Vec<T>>,
    pub confidence_upper: Option<Vec<T>>,

    /// Prediction interval bounds (if computed)
    pub prediction_lower: Option<Vec<T>>,
    pub prediction_upper: Option<Vec<T>>,

    /// Residuals (y - fit)
    pub residuals: Option<Vec<T>>,

    /// Final robustness weights from outlier downweighting
    pub robustness_weights: Option<Vec<T>>,

    /// Detailed fit diagnostics (RMSE, R^2, Effective DF, etc.)
    pub diagnostics: Option<Diagnostics<T>>,

    /// Number of robustness iterations actually performed
    pub iterations_used: Option<usize>,

    /// Smoothing fraction used (optimal if selected via CV)
    pub fraction_used: T,

    /// RMSE scores for each fraction tested during CV
    pub cv_scores: Option<Vec<T>>,
}
```

> [!TIP]
> **Using with ndarray:** While the result struct uses `Vec<T>` for maximum compatibility, you can effortlessly convert any field to an `Array1` using `Array1::from_vec(result.y)`.

## Streaming Processing

For datasets that don't fit in memory:

```rust
let mut processor = Loess::new()
    .fraction(0.3)
    .iterations(2)
    .adapter(Streaming)
    .chunk_size(1000)
    .overlap(100)
    .build()?;

// Process data in chunks
let result1 = processor.process_chunk(&chunk1_x, &chunk1_y)?;
let result2 = processor.process_chunk(&chunk2_x, &chunk2_y)?;

// Finalize to get remaining buffered data
let final_result = processor.finalize()?;
```

## Online Processing

For real-time data streams:

```rust
let mut processor = Loess::new()
    .fraction(0.2)
    .iterations(1)
    .adapter(Online)
    .window_capacity(100)
    .build()?;

// Process points as they arrive
for i in 1..=10 {
    let x = i as f64;
    let y = 2.0 * x + 1.0;
    if let Some(output) = processor.add_point(&[x], y)? {
        println!("Smoothed: {:.2}", output.smoothed);
    }
}
```

## Parameter Selection Guide

### Fraction (Smoothing Span)

- **0.1-0.3**: Fine detail, may be noisy
- **0.3-0.5**: Moderate smoothing (good for most cases)
- **0.5-0.7**: Heavy smoothing, emphasizes trends
- **0.7-1.0**: Very smooth, may over-smooth
- **Default: 0.67** (Cleveland's choice)

### Robustness Iterations

- **0**: No robustness (fastest, sensitive to outliers)
- **1-3**: Light to moderate robustness (recommended)
- **4-6**: Strong robustness (for contaminated data)
- **7+**: Diminishing returns

### Polynomial Degree

- **Constant**: Local weighted mean (smoothing only)
- **Linear** (default): Standard LOESS, good bias-variance balance
- **Quadratic**: Better for peaks/valleys, higher variance
- **Cubic/Quartic**: Specialized high-order fitting

### Kernel Function

- **Tricube** (default): Best all-around, Cleveland's original choice
- **Epanechnikov**: Theoretically optimal MSE
- **Gaussian**: Maximum smoothness, no compact support
- **Uniform**: Fastest, least smooth (moving average)

### Boundary Policy

- **Extend** (default): Pad with constant values
- **Reflect**: Mirror data at boundaries (for periodic/symmetric data)
- **Zero**: Pad with zeros (signal processing)
- **NoBoundary**: Original Cleveland behavior

> **Note:** For nD data, `Extend` defaults to `NoBoundary` to preserve regression accuracy.

## Examples

```bash
cargo run --example batch_smoothing
cargo run --example online_smoothing
cargo run --example streaming_smoothing
```

## MSRV

Rust **1.85.0** or later (2024 Edition).

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Licensed under either of

- Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license
   ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## References

- Cleveland, W.S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots". *Journal of the American Statistical Association*.
- Cleveland, W.S. & Devlin, S.J. (1988). "Locally Weighted Regression: An Approach to Regression Analysis by Local Fitting". *Journal of the American Statistical Association*.
