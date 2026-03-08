# Rust Examples

Complete Rust examples demonstrating fastLoess with the builder pattern and type safety.

## Batch Smoothing (fastLoess)

Parallel batch processing with confidence intervals, diagnostics, and cross-validation.

```rust
--8<-- "examples/fastLoess/fast_batch_smoothing.rs"
```

[:material-download: Download fast_batch_smoothing.rs](https://github.com/thisisamirv/loess-project/blob/main/examples/fastLoess/fast_batch_smoothing.rs)

---

## Streaming Smoothing (fastLoess)

Process large datasets in memory-efficient chunks with parallel processing.

```rust
--8<-- "examples/fastLoess/fast_streaming_smoothing.rs"
```

[:material-download: Download fast_streaming_smoothing.rs](https://github.com/thisisamirv/loess-project/blob/main/examples/fastLoess/fast_streaming_smoothing.rs)

---

## Online Smoothing (fastLoess)

Real-time smoothing with sliding window for streaming data applications.

```rust
--8<-- "examples/fastLoess/fast_online_smoothing.rs"
```

[:material-download: Download fast_online_smoothing.rs](https://github.com/thisisamirv/loess-project/blob/main/examples/fastLoess/fast_online_smoothing.rs)

---

## Core loess Examples

The core `loess` crate provides single-threaded, `no_std`-compatible implementations.

### Batch Smoothing (loess)

```rust
--8<-- "examples/loess/batch_smoothing.rs"
```

### Streaming Smoothing (loess)

```rust
--8<-- "examples/loess/streaming_smoothing.rs"
```

### Online Smoothing (loess)

```rust
--8<-- "examples/loess/online_smoothing.rs"
```

---

## Running the Examples

```bash
# Run fastLoess examples (parallel)
cargo run --example fast_batch_smoothing -p examples
cargo run --example fast_streaming_smoothing -p examples
cargo run --example fast_online_smoothing -p examples

# Run loess examples (single-threaded)
cargo run --example batch_smoothing -p examples
cargo run --example streaming_smoothing -p examples
cargo run --example online_smoothing -p examples
```

## Quick Start

```rust
use fastLoess::prelude::*;

fn main() -> Result<(), LoessError> {
    let x: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    // Build and fit the model
    let model = Loess::new()
        .fraction(0.3)
        .iterations(3)
        .confidence_intervals(0.95)
        .return_diagnostics()
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y)?;
    
    println!("R²: {:.4}", result.diagnostics.unwrap().r_squared);
    Ok(())
}
```
