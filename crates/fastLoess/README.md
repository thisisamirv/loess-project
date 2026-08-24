<!-- markdownlint-disable MD024 MD033 -->
# LOESS Project

<p align="center">
  <a href="https://crates.io/crates/fastLoess"><img src="https://img.shields.io/badge/fastLoess-000000?logo=rust&logoColor=white" alt="fastLoess"></a>
  <a href="https://github.com/thisisamirv/loess-project/actions/workflows/ci-rust.yml"><img src="https://github.com/thisisamirv/loess-project/actions/workflows/ci-rust.yml/badge.svg" alt="CI"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/thisisamirv/loess-project/main/dev/logo.png" alt="One LOESS to Rule Them All" width="400">
  <br>
  <em>One LOESS to Rule Them All</em>
</p>

The fastest, most robust, and most feature-complete language-agnostic LOESS (Locally Estimated Scatterplot Smoothing) implementation for **Rust**, **Python**, **R**, **Julia**, **JavaScript**, **C++**, and **WebAssembly**.

The `loess-project` also offers bindings for Rust, Python, R, Julia, Node.js, WebAssembly, and C++ — see the [full repository](https://github.com/thisisamirv/loess-project).

---

## Installation

> [!NOTE]
>
> Currently available for R, Python, Rust, Julia, Node.js, WebAssembly, and C++. See the [Installation Guide](https://docs.rs/fastLoess) for detailed installation instructions.

## Documentation

> [!NOTE]
>
> ### 📚 [View the full documentation](https://docs.rs/fastLoess)

---

## LOESS vs. LOWESS

| Feature | LOESS (This Crate) | LOWESS |
| --- | --- | --- |
| **Polynomial Degree** | Linear, Quadratic, Cubic, Quartic | Linear (Degree 1) |
| **Dimensions** | Multivariate (n-D support) | Univariate (1-D only) |
| **Flexibility** | High (Distance metrics) | Standard |
| **Complexity** | Higher (Matrix inversion) | Lower (Weighted average/slope) |

> [!TIP]
> **Note:** For a **LOWESS** implementation, use [`lowess-project`](https://github.com/thisisamirv/lowess-project).

---

## Why this package?

### Speed

The `loess` project beats the competition in terms of speed, whether in single-threaded or multi-threaded parallel execution. It is typically **5–20x faster** than R's `loess` in serial mode, and up to **200x faster** on large datasets with parallel execution.

For more details on the performance comparison, see the [Benchmarks](https://docs.rs/fastLoess) page.

### Robustness

This implementation is *more robust* than R's `loess` due to two key design choices:

**MAD-Based Scale Estimation:**

For robustness weight calculations, this crate uses *Median Absolute Deviation (MAD)* for scale estimation:

```text
s = median(|r_i - median(r)|)
```

In contrast, R's `loess` uses the median of absolute residuals (MAR):

```text
s = median(|r_i|)
```

- MAD is a *breakdown-point-optimal* estimator—it remains valid even when up to 50% of data are outliers.
- The median-centering step removes asymmetric bias from residual distributions.
- MAD provides consistent outlier detection regardless of whether residuals are centered around zero.

**Boundary Padding:**

This crate applies a range of different *boundary policies* at dataset edges:

- **Extend**: Repeats edge values to maintain local neighborhood size.
- **Reflect**: Mirrors data symmetrically around boundaries.
- **Zero**: Pads with zeros (useful for signal processing).
- **NoBoundary**: Original Cleveland behavior

R's `loess` does not apply boundary padding, which can lead to:

- Biased estimates near boundaries due to asymmetric local neighborhoods.
- Increased variance at the edges of the smoothed curve.

### Features

A variety of features, supporting a range of use cases:

| Feature                       | This package  | R (stats)        |
|-------------------------------|:-------------:|:----------------:|
| Polynomial Degree             | 5 (0–4)       | 2 (1 or 2)       |
| Kernel                        | 7 options     | only Tricube     |
| Robustness Weighting          | 3 options     | only Bisquare    |
| Scale Estimation              | 3 options     | only MAR         |
| Distance Metric               | 6 options     | normalized only  |
| Boundary Padding              | 4 options     | no padding       |
| Zero Weight Fallback          | 3 options     | no               |
| Auto Convergence              | yes           | no               |
| Online Mode                   | yes           | no               |
| Streaming Mode                | yes           | no               |
| Confidence Intervals          | yes           | no               |
| Prediction Intervals          | yes           | no               |
| Diagnostics (RMSE, R², AIC)   | yes           | no               |
| Cross-Validation              | 2 options     | no               |
| Parallel Execution            | yes           | no               |
| `no-std` Support              | yes           | no               |

## Validation

All implementations are **numerical twins** of R's `loess`:

| Aspect | Status | Details |
| --- | --- | --- |
| **Accuracy** | ✅ EXACT MATCH | Max diff < 1e-12 across all scenarios |
| **Consistency** | ✅ PERFECT | Multiple scenarios pass with strict tolerance |
| **Robustness** | ✅ VERIFIED | Robust smoothing matches R exactly |

## API Reference

```rust
use fastLoess::prelude::*;

let model = Loess::new()
    .fraction(0.67)
    .iterations(3)
    .weight_function("tricube")
    .robustness_method("bisquare")
    .zero_weight_fallback("use_local_mean")
    .boundary_policy("extend")
    .scaling_method("mad")
    .confidence_intervals(0.95)
    .prediction_intervals(0.95)
    .return_diagnostics()
    .return_residuals()
    .return_robustness_weights()
    .cv_fractions(vec![0.3, 0.5, 0.7])
    .cv_method("kfold")
    .cv_k(5)
    .cv_seed(123)
    .auto_converge(1e-4)
    .custom_weights(vec![1.0; x.len()])
    .parallel(true)
    .build()?;

let result = model.fit(&x, &y)?;

// Result structure:
pub struct LoessResult<T> {
    pub x: Vec<T>,                           // Sorted x values
    pub y: Vec<T>,                           // Smoothed y values
    pub standard_errors: Option<Vec<T>>,
    pub confidence_lower: Option<Vec<T>>,
    pub confidence_upper: Option<Vec<T>>,
    pub prediction_lower: Option<Vec<T>>,
    pub prediction_upper: Option<Vec<T>>,
    pub residuals: Option<Vec<T>>,
    pub robustness_weights: Option<Vec<T>>,
    pub diagnostics: Option<Diagnostics<T>>,
    pub iterations_used: Option<usize>,
    pub fraction_used: T,
    pub cv_scores: Option<Vec<T>>,
}
```

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](https://github.com/thisisamirv/loess-project/blob/main/CONTRIBUTING.md) for more information.

## Changelog

See [CHANGELOG.md](https://github.com/thisisamirv/loess-project/blob/main/CHANGELOG.md) for a history of changes.

## License

Licensed under [MIT](https://github.com/thisisamirv/loess-project/blob/main/LICENSE-MIT) or [Apache-2.0](https://github.com/thisisamirv/loess-project/blob/main/LICENSE-APACHE).

## Citation

If you use this software in your research, please cite it using the [CITATION.cff](https://github.com/thisisamirv/loess-project/blob/main/CITATION.cff) file or the BibTeX entry below:

```bibtex
@software{loess_project,
  author = {Valizadeh, Amir},
  title = {LOESS Project: High-Performance Locally Estimated Scatterplot Smoothing},
  year = {2026},
  url = {https://github.com/thisisamirv/loess-project},
  license = {MIT OR Apache-2.0}
}
```
