# fastLoess

[![Crates.io](https://img.shields.io/crates/v/fastLoess.svg)](https://crates.io/crates/fastLoess)
[![Documentation](https://docs.rs/fastLoess/badge.svg)](https://docs.rs/fastLoess)
[![License](https://img.shields.io/badge/License-AGPL--3.0%20OR%20Commercial-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

A high-performance implementation of LOESS (Locally Estimated Scatterplot Smoothing) in Rust. This crate provides a robust, production-ready implementation with support for confidence intervals, multiple kernel functions, and optimized execution modes.

## How LOESS works

LOESS creates smooth curves through scattered data using local weighted neighborhoods:

![LOESS Smoothing Concept](https://raw.githubusercontent.com/thisisamirv/fastLoess/main/docs/loess_concept.svg)

## LOESS vs. LOWESS

| Feature               | LOESS (This Crate)                | LOWESS                         |
|-----------------------|-----------------------------------|--------------------------------|
| **Polynomial Degree** | Linear, Quadratic, Cubic, Quartic | Linear (Degree 1)              |
| **Dimensions**        | Multivariate (n-D support)        | Univariate (1-D only)          |
| **Flexibility**       | High (Distance metrics)           | Standard                       |
| **Complexity**        | Higher (Matrix inversion)         | Lower (Weighted average/slope) |

LOESS can fit higher-degree polynomials for more complex data:

![Degree Comparison](https://raw.githubusercontent.com/thisisamirv/fastLoess/main/docs/degree_comparison.svg)

LOESS can also handle multivariate data (n-D), while LOWESS is limited to univariate data (1-D):

![Multivariate LOESS](https://raw.githubusercontent.com/thisisamirv/fastLoess/main/docs/multivariate_loess.svg)

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

Benchmarked against R's `stats::loess`. The latest benchmarks comparing **Serial** vs **Parallel (Rayon)** execution modes show that the parallel implementation correctly leverages multiple cores to provide additional speedups, particularly for computationally heavier tasks (high dimensions, larger datasets).

Overall, Rust implementations achieve **3x to 54x** speedups over R.

### Comparison: R vs Rust (Serial) vs Rust (Parallel)

The table below shows the execution time and speedup relative to R.

| Name                           |      R       | Rust (Serial) | Rust (Parallel) |
|--------------------------------|--------------|---------------|-----------------|
| **Dimensions**                 |              |               |                 |
| 1d_linear                      |    4.18ms    |     7.2x      |      8.1x       |
| 2d_linear                      |   13.24ms    |     6.5x      |      10.1x      |
| 3d_linear                      |   28.37ms    |     7.9x      |      13.6x      |
| **Pathological**               |              |               |                 |
| clustered                      |   19.70ms    |     15.7x     |      21.5x      |
| constant_y                     |   13.61ms    |     13.6x     |      17.5x      |
| extreme_outliers               |   23.55ms    |     10.3x     |      11.7x      |
| high_noise                     |   34.96ms    |     19.9x     |      28.0x      |
| **Polynomial Degree**          |              |               |                 |
| degree_constant                |    8.50ms    |     10.0x     |      13.5x      |
| degree_linear                  |   13.47ms    |     16.2x     |      21.4x      |
| degree_quadratic               |   19.07ms    |     23.3x     |      29.7x      |
| **Scalability**                |              |               |                 |
| scale_1000                     |    1.09ms    |     4.3x      |      3.7x       |
| scale_5000                     |    8.63ms    |     7.2x      |      8.2x       |
| scale_10000                    |   28.68ms    |     10.4x     |      14.5x      |
| **Real-world Scenarios**       |              |               |                 |
| financial_1000                 |    1.11ms    |     4.8x      |      4.7x       |
| financial_5000                 |    8.28ms    |     7.6x      |      9.2x       |
| genomic_5000                   |    8.27ms    |     6.7x      |      7.5x       |
| scientific_5000                |   11.23ms    |     6.8x      |      10.1x      |
| **Parameter Sensitivity**      |              |               |                 |
| fraction_0.67                  |   44.96ms    |     54.0x     |      54.1x      |
| iterations_10                  |   23.31ms    |     10.9x     |      11.8x      |

*Note: "Rust (Parallel)" corresponds to the optimized CPU backend using Rayon.*

### Key Takeaways

1. **Parallel Wins on Load**: For computationally intensive tasks (e.g., `3d_linear`, `high_noise`, `scientific_5000`, `scale_10000`), the parallel backend provides significant additional speedup over the serial implementation (e.g., **13.6x vs 7.9x** for 3D data).
2. **Overhead on Small Data**: For very small or fast tasks (e.g., `scale_1000`, `financial_1000`), the serial implementation is comparable or slightly faster, indicating that thread management overhead is visible but minimal (often < 0.05ms difference).
3. **Consistent Superiority**: Both Rust implementations consistently outperform R, usually by an order of magnitude.

### Recommendation

- **Default to Parallel**: The overhead for small datasets is negligible (microseconds), while the gains for larger or more complex datasets are substantial (doubling the speedup factor in some cases).
- **Use Serial for Tiny Batches**: If processing millions of independent tiny datasets (< 1000 points) where calling `fit()` repeatedly, the serial backend might save thread pool overhead.

Check [Benchmarks](https://github.com/thisisamirv/fastLoess/tree/bench/benchmarks) for detailed results and reproducible benchmarking code.

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

The Rust `fastLoess` crate is a **numerical twin** of R's `loess` implementation:

| Aspect          | Status         | Details                                    |
|-----------------|----------------|--------------------------------------------|
| **Accuracy**    | ✅ EXACT MATCH | Max diff < 1e-12 across all scenarios      |
| **Consistency** | ✅ PERFECT     | 20/20 scenarios pass with strict tolerance |
| **Robustness**  | ✅ VERIFIED    | Robust smoothing matches R exactly         |

Check [Validation](https://github.com/thisisamirv/fastLoess/tree/bench/validation) for detailed scenario results.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
fastLoess = "0.1"
```

For `no_std` environments:

```toml
[dependencies]
fastLoess = { version = "0.1", default-features = false }
```

## Quick Start

```rust
use fastLoess::prelude::*;

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
use fastLoess::prelude::*;

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

    // Parallelism
    .parallel(true)

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
