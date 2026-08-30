<!-- markdownlint-disable MD024 MD033 -->
# LOESS Project

<p align="center">
  <a href="https://crates.io/crates/loess-rs"><img src="https://img.shields.io/badge/loess--rs-000000?logo=rust&logoColor=white" alt="loess-rs"></a>
  <a href="https://crates.io/crates/fastLoess"><img src="https://img.shields.io/badge/fastLoess-000000?logo=rust&logoColor=white" alt="fastLoess"></a>
  <a href="https://pypi.org/project/fastloess/"><img src="https://img.shields.io/badge/PyPI-3775A9?logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://thisisamirv.r-universe.dev/rfastloess"><img src="https://img.shields.io/badge/R--universe-276DC3?logo=r&logoColor=white" alt="R-universe"></a>
  <a href="https://www.npmjs.com/package/fastloess"><img src="https://img.shields.io/badge/npm-CB3837?logo=npm&logoColor=white" alt="npm"></a>
  <a href="https://juliahub.com/ui/Packages/General/FastLOESS"><img src="https://img.shields.io/badge/Julia-9558B2?logo=julia&logoColor=white" alt="Julia"></a>
  <a href="https://www.npmjs.com/package/fastloess-wasm"><img src="https://img.shields.io/badge/WASM-654FF0?logo=webassembly&logoColor=white" alt="WASM"></a>
  <a href="https://github.com/thisisamirv/loess-project/releases/latest"><img src="https://img.shields.io/badge/C++-00599C?logo=cplusplus&logoColor=white" alt="C++"></a>
  <br>
  <a href="https://anaconda.org/conda-forge/fastloess"><img src="https://img.shields.io/badge/fastloess_(Python)-44A833?logo=anaconda&logoColor=white" alt="fastloess (Python)"></a>
  <a href="https://anaconda.org/conda-forge/libfastloess"><img src="https://img.shields.io/badge/libfastloess_(C++)-44A833?logo=anaconda&logoColor=white" alt="libfastloess (C++)"></a>
  <a href="https://anaconda.org/conda-forge/r-rfastloess"><img src="https://img.shields.io/badge/rfastloess_(R)-44A833?logo=anaconda&logoColor=white" alt="rfastloess (R)"></a>
  <br>
  <a href="https://github.com/thisisamirv/loess-project/actions/workflows/ci-rust.yml"><img src="https://github.com/thisisamirv/loess-project/actions/workflows/ci-rust.yml/badge.svg" alt="CI - Rust"></a>
  <a href="https://github.com/thisisamirv/loess-project/actions/workflows/ci-python.yml"><img src="https://github.com/thisisamirv/loess-project/actions/workflows/ci-python.yml/badge.svg" alt="CI - Python"></a>
  <a href="https://github.com/thisisamirv/loess-project/actions/workflows/ci-r.yml"><img src="https://github.com/thisisamirv/loess-project/actions/workflows/ci-r.yml/badge.svg" alt="CI - R"></a>
  <a href="https://github.com/thisisamirv/loess-project/actions/workflows/ci-julia.yml"><img src="https://github.com/thisisamirv/loess-project/actions/workflows/ci-julia.yml/badge.svg" alt="CI - Julia"></a>
  <a href="https://github.com/thisisamirv/loess-project/actions/workflows/ci-nodejs.yml"><img src="https://github.com/thisisamirv/loess-project/actions/workflows/ci-nodejs.yml/badge.svg" alt="CI - Node.js"></a>
  <a href="https://github.com/thisisamirv/loess-project/actions/workflows/ci-wasm.yml"><img src="https://github.com/thisisamirv/loess-project/actions/workflows/ci-wasm.yml/badge.svg" alt="CI - WASM"></a>
  <a href="https://github.com/thisisamirv/loess-project/actions/workflows/ci-cpp.yml"><img src="https://github.com/thisisamirv/loess-project/actions/workflows/ci-cpp.yml/badge.svg" alt="CI - C++"></a>
  <br>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/thisisamirv/loess-project/main/dev/logo.png" alt="One LOESS to Rule Them All" width="400">
  <br>
  <em>One LOESS to Rule Them All</em>
</p>

The fastest, most robust, and most feature-complete language-agnostic LOESS (Locally Estimated Scatterplot Smoothing) implementation for **Rust**, **Python**, **R**, **Julia**, **JavaScript**, **C++**, and **WebAssembly**.

> [!IMPORTANT]
>
> The `loess-project` contains a complete ecosystem for LOESS smoothing:
>
> - **[`loess-rs`](https://crates.io/crates/loess-rs)** - Core single-threaded Rust implementation with `no_std` support
> - **[`fastLoess`](https://crates.io/crates/fastLoess)** - Parallel Rust wrapper with ndarray integration  
> - **[`R bindings`](https://thisisamirv.r-universe.dev/rfastloess)** - extendr-based R binding
> - **[`Python bindings`](https://pypi.org/project/fastloess/)** - PyO3-based Python binding
> - **[`Julia bindings`](https://juliahub.com/ui/Packages/General/FastLOESS)** - Native Julia binding with C FFI
> - **[`JavaScript bindings`](https://www.npmjs.com/package/fastloess)** - Node.js binding
> - **[`WebAssembly bindings`](https://www.npmjs.com/package/fastloess-wasm)** - WASM binding
> - **[`C++ bindings`](https://github.com/thisisamirv/loess-project/releases/latest)** - Native C++ binding with CMake integration

---

## Installation & Documentation

> Currently available for R, Python, Rust, Julia, Node.js, WebAssembly, and C++. See the documentation for your binding/crate below for installation instructions.
>
> | Binding / Crate | Documentation |
> | --- | --- |
> | `loess-rs` (Rust) | [docs.rs/loess-rs](https://docs.rs/loess-rs) |
> | `fastLoess` (Rust) | [docs.rs/fastLoess](https://docs.rs/fastLoess) |
> | Python | [loess.readthedocs.io](https://loess.readthedocs.io/) |
> | R | [thisisamirv.github.io/loess-project/r](https://thisisamirv.github.io/loess-project/r/) |
> | Julia | [thisisamirv.github.io/loess-project/julia](https://thisisamirv.github.io/loess-project/julia/) |
> | Node.js | [thisisamirv.github.io/loess-project/nodejs](https://thisisamirv.github.io/loess-project/nodejs/) |
> | WebAssembly | [thisisamirv.github.io/loess-project/wasm](https://thisisamirv.github.io/loess-project/wasm/) |
> | C++ | [thisisamirv.github.io/loess-project/cpp](https://thisisamirv.github.io/loess-project/cpp/) |

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

For more details on the performance comparison, see the Benchmarks page in the documentation for your binding/crate.

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
