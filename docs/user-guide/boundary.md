<!-- markdownlint-disable MD024 -->
# Boundary Handling

Edge strategies that reduce bias near the ends of the data range.

## Overview

Standard LOESS neighbourhoods become asymmetric at the boundaries: fewer points exist on one side, pulling the local fit toward the data interior. The `boundary_policy` parameter controls how the data is padded to mitigate this effect.

![Boundary Handling](../assets/diagrams/boundary_comparison.svg)

| Policy | Padding Strategy | Best For |
| --- | --- | --- |
| `"extend"` | Repeat first / last value | Most datasets (default) |
| `"reflect"` | Mirror data at boundaries | Periodic or symmetric data |
| `"zero"` | Pad with zeros | Data known to approach zero |
| `"no_boundary"` | No padding (Cleveland original) | Reproducing reference behaviour |

---

## Extend (Default)

Pads beyond both endpoints by replicating the first and last observed values. Prevents the fit from curling toward zero and is a safe default for nearly all use cases.

**Use when**: No strong prior on boundary behaviour; general-purpose smoothing.

=== "R"
    ```r
    result <- Loess(boundary_policy = "extend")$fit(x, y)
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, boundary_policy="extend")
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .boundary_policy(BoundaryPolicy::Extend)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = fit(Loess(; boundary_policy="extend"), x, y)
    ```

=== "Node.js"
    ```javascript
    const result = smooth(x, y, { boundaryPolicy: "extend" });
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { boundaryPolicy: "extend" });
    ```

=== "C++"
    ```cpp
    auto result = fastloess::smooth(x, y, { .boundary_policy = "extend" });
    ```

---

## Reflect

Mirrors the data about both endpoints before fitting, then discards the reflected region from the output. Preserves continuity of derivatives, making it ideal for periodic or spatially symmetric signals.

**Use when**: Circular data (e.g., angle, day-of-year), symmetric physical quantities, or when the derivative at the boundary should be near zero.

=== "R"
    ```r
    result <- Loess(boundary_policy = "reflect")$fit(x, y)
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, boundary_policy="reflect")
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .boundary_policy(BoundaryPolicy::Reflect)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = fit(Loess(; boundary_policy="reflect"), x, y)
    ```

=== "Node.js"
    ```javascript
    const result = smooth(x, y, { boundaryPolicy: "reflect" });
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { boundaryPolicy: "reflect" });
    ```

=== "C++"
    ```cpp
    auto result = fastloess::smooth(x, y, { .boundary_policy = "reflect" });
    ```

---

## Zero

Pads with zeros beyond both endpoints. Appropriate when the underlying process is known to be zero outside the observation window (e.g., a pulse signal or a bounded physical quantity).

**Use when**: Signal decays to zero at both ends; zero is a meaningful boundary value.

=== "R"
    ```r
    result <- Loess(boundary_policy = "zero")$fit(x, y)
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, boundary_policy="zero")
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .boundary_policy(BoundaryPolicy::Zero)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = fit(Loess(; boundary_policy="zero"), x, y)
    ```

=== "Node.js"
    ```javascript
    const result = smooth(x, y, { boundaryPolicy: "zero" });
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { boundaryPolicy: "zero" });
    ```

=== "C++"
    ```cpp
    auto result = fastloess::smooth(x, y, { .boundary_policy = "zero" });
    ```

---

## No Boundary

Applies no padding. Each local fit uses only the points that are actually available, which may be fewer than the requested neighbourhood at the endpoints. This reproduces the original Cleveland (1979) algorithm exactly.

**Use when**: Reproducing reference results; you prefer the raw LOESS boundary behaviour.

!!! note
    Without padding, boundary fits can have higher variance and visible edge artefacts, particularly with small `fraction` values.

=== "R"
    ```r
    result <- Loess(boundary_policy = "no_boundary")$fit(x, y)
    ```

=== "Python"
    ```python
    result = fl.smooth(x, y, boundary_policy="no_boundary")
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .boundary_policy(BoundaryPolicy::NoBoundary)
        .adapter(Batch)
        .build()?;
    ```

=== "Julia"
    ```julia
    result = fit(Loess(; boundary_policy="no_boundary"), x, y)
    ```

=== "Node.js"
    ```javascript
    const result = smooth(x, y, { boundaryPolicy: "no_boundary" });
    ```

=== "WebAssembly"
    ```javascript
    const result = smooth(x, y, { boundaryPolicy: "no_boundary" });
    ```

=== "C++"
    ```cpp
    auto result = fastloess::smooth(x, y, { .boundary_policy = "no_boundary" });
    ```

---

## Choosing a Policy

| Situation | Recommended Policy |
| --- | --- |
| General purpose | `"extend"` (default) |
| Periodic signal (angle, day-of-year) | `"reflect"` |
| Signal known to be zero at boundaries | `"zero"` |
| Replicating original Cleveland behaviour | `"no_boundary"` |
