# OnlineLoess — Rust API Reference

See also: [fastLoess & loess-rs Rust API Reference](rust.md)

## Struct

### `OnlineLoess`

Online mode for real-time data.

**Constructor:**

```rust
use fastLoess::prelude::*;

fn main() -> Result<(), LoessError> {
    let mut processor = OnlineLoess::new();

    Ok(())
}
```

**Methods:**

```rust
use fastLoess::prelude::*;

fn main() -> Result<(), LoessError> {
    let mut processor = OnlineLoess::new().build()?;
    let output = processor.add_point(&[1.0f64], 2.0f64)?;

    Ok(())
}
```

* Adds a single point `(x, y)` to the window. `x` is a slice of predictor values (one per dimension).
* Returns `Result<Option<OnlineOutput<T>>, LoessError>`.

```rust
use fastLoess::prelude::*;

fn main() -> Result<(), LoessError> {
    let mut processor = OnlineLoess::new().build()?;
    processor.reset();

    Ok(())
}
```

* Clears the internal window buffer.
* **Rust-only** — this method is not exposed in other language bindings, where creating a new instance is the idiomatic alternative.

## Result Structure

### `OnlineOutput<T>`

Returned inside `Ok(Some(...))` by `add_point()`. `None` while the window is still filling.

| Field | Type | Description |
| --- | --- | --- |
| `smoothed` | `T` | Smoothed value for the latest point |
| `std_error` | `Option<T>` | Standard error (if requested) |
| `residual` | `Option<T>` | Residual y − smoothed (if requested) |
| `robustness_weight` | `Option<T>` | Robustness weight (if requested) |
| `iterations_used` | `Option<usize>` | Robustness iterations performed |

## Builder Options

### Online Options

| Method | Argument Type | Default | Description |
| --- | --- | --- | --- |
| `window_capacity(usize)` | `usize` | `1000` | Max points in sliding window |
| `min_points(usize)` | `usize` | `3` | Min points before smoothing starts |
| `update_mode(...)` | `update_mode` | `"full"` | Update strategy |
| `parallel(bool)` | `bool` | `false` | Enable parallel execution (off by default; online LOESS fits one point at a time) |

## Options

### update_mode

*See: [Execution Modes](../user-guide/adapters.md)*

* `"full"` (default; alias: `"resmooth"`)
* `"incremental"` (alias: `"single"`)
