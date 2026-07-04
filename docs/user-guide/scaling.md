<!-- markdownlint-disable MD024 -->
# Scaling Methods

Residual scale estimation during robustness iterations.

## Overview

When `iterations > 0`, LOESS computes robustness weights by comparing each residual to the current residual scale estimate. The `scaling_method` parameter controls how that scale is measured.

The robustness weight for point $i$ is:

$$w_i = B\!\left(\frac{|r_i|}{6 \cdot \hat{\sigma}}\right)$$

where $B$ is the bisquare function and $\hat{\sigma}$ is the scale estimate. A larger $\hat{\sigma}$ makes the algorithm more tolerant of large residuals; a smaller one makes it more aggressive.

| Method | Formula | Robustness | Speed |
| --- | --- | --- | --- |
| `"mad"` | Median of \|residuals − median(residuals)\| | Very robust | Moderate |
| `"mar"` | Median of \|residuals\| | Robust | Fast |
| `"mean"` | Mean of \|residuals\| | Less robust | Fastest |

![Scaling Methods Comparison](../assets/diagrams/scaling_comparison.svg)

---

## MAD — Median Absolute Deviation (Default)

$$\hat{\sigma} = \text{median}(|r_i - \text{median}(r_i)|)$$

First centers residuals at their median, then takes the median of the absolute deviations. Double use of the median makes it highly resistant to extreme outliers. This is the standard choice for robust regression.

**Use when**: Data may contain outliers (default for most applications).

=== "R"
    ```r
    result <- Loess(iterations = 3, scaling_method = "mad")$fit(x, y)
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, iterations=3, scaling_method="mad")
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .iterations(3)
        .scaling_method(ScalingMethod::MAD)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = fit(Loess(; iterations=3, scaling_method="mad"), x, y)
    ```

=== "Node.js"
    ```javascript
    const result = smooth(x, y, { iterations: 3, scalingMethod: "mad" });
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { iterations: 3, scalingMethod: "mad" });
    ```

=== "C++"
    ```cpp
    auto result = fastloess::smooth(x, y, {
        .iterations = 3,
        .scaling_method = "mad"
    });
    ```

---

## MAR — Median Absolute Residual

$$\hat{\sigma} = \text{median}(|r_i|)$$

Uses the uncentered median — unlike MAD it does not subtract the residual median first. Still robust (median-based) but slightly less resistant than MAD when residuals are systematically shifted. Faster than MAD in practice because it requires only one partial sort.

**Use when**: Speed matters and data have minimal systematic bias in residuals.

=== "R"
    ```r
    result <- Loess(iterations = 3, scaling_method = "mar")$fit(x, y)
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, iterations=3, scaling_method="mar")
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .iterations(3)
        .scaling_method(ScalingMethod::MAR)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = fit(Loess(; iterations=3, scaling_method="mar"), x, y)
    ```

=== "Node.js"
    ```javascript
    const result = smooth(x, y, { iterations: 3, scalingMethod: "mar" });
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { iterations: 3, scalingMethod: "mar" });
    ```

=== "C++"
    ```cpp
    auto result = fastloess::smooth(x, y, {
        .iterations = 3,
        .scaling_method = "mar"
    });
    ```

---

## Mean — Mean Absolute Residual

$$\hat{\sigma} = \frac{1}{n}\sum_i |r_i|$$

Arithmetic mean of absolute residuals. Non-robust: a single extreme outlier inflates $\hat{\sigma}$, causing the algorithm to under-downweight it. Fastest to compute (no sort required). Useful when data are believed to be clean and speed is a priority.

**Use when**: Clean data with no outliers; maximum computation speed required.

=== "R"
    ```r
    result <- Loess(iterations = 3, scaling_method = "mean")$fit(x, y)
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, iterations=3, scaling_method="mean")
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .iterations(3)
        .scaling_method(ScalingMethod::Mean)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = fit(Loess(; iterations=3, scaling_method="mean"), x, y)
    ```

=== "Node.js"
    ```javascript
    const result = smooth(x, y, { iterations: 3, scalingMethod: "mean" });
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { iterations: 3, scalingMethod: "mean" });
    ```

=== "C++"
    ```cpp
    auto result = fastloess::smooth(x, y, {
        .iterations = 3,
        .scaling_method = "mean"
    });
    ```

---

## Choosing a Scaling Method

| Situation | Recommended Method |
| --- | --- |
| General purpose, possible outliers | `"mad"` (default) |
| Robust but faster, low systematic bias | `"mar"` |
| Clean data, no outliers | `"mean"` |

See [Robustness](robustness.md) for a broader discussion of outlier handling.
