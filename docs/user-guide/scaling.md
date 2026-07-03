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
| `"mad"` | Median of \|residuals\| | Very robust | Moderate |
| `"mar"` | Mean of \|residuals\| | Moderate | Fast |
| `"mean"` | Mean of residuals² / n | Least robust | Fastest |

---

## MAD — Median Absolute Deviation (Default)

$$\hat{\sigma} = \text{median}(|r_i|)$$

The median is resistant to extreme outliers, so a few very large residuals do not inflate $\hat{\sigma}$ and cause the algorithm to under-downweight them. This is the standard choice for robust regression.

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

## MAR — Mean Absolute Residual

$$\hat{\sigma} = \frac{1}{n}\sum_i |r_i|$$

Uses the mean rather than the median, so large outliers have some influence on the scale estimate. Faster to compute than MAD (no sort required) and can converge faster when the data are only mildly contaminated.

**Use when**: Data are only mildly non-Gaussian; speed matters more than maximum robustness.

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

## Mean — Root Mean Square Residual

$$\hat{\sigma} = \sqrt{\frac{1}{n}\sum_i r_i^2}$$

Equivalent to the RMSE of the current fit. Sensitive to large residuals (they are squared), which inflates the scale estimate and makes the algorithm more tolerant. Useful when data are believed to be Gaussian and speed is a priority.

**Use when**: Clean, near-Gaussian data; iterations used mainly for convergence rather than outlier rejection.

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
| Mild contamination, speed preferred | `"mar"` |
| Clean / Gaussian data | `"mean"` |

See [Robustness](robustness.md) for a broader discussion of outlier handling.
