# StreamingLoess — Rust API Reference

See also: [fastLoess & loess-rs Rust API Reference](rust.md)

## Struct

### `StreamingLoess`

Streaming mode for large datasets.

**Constructor:**

```rust
use fastLoess::prelude::*;

fn main() -> Result<(), LoessError> {
    let mut processor = StreamingLoess::new();

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

    let mut processor = StreamingLoess::new().build()?;
    let result = processor.process_chunk(&x, &y)?;

    Ok(())
}
```

* Processes a chunk of data. Returns `LoessResult<T>` with partial results.

```rust
use fastLoess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let mut processor = StreamingLoess::new().build()?;
    processor.process_chunk(&x, &y)?;
    let final_result = processor.finalize()?;

    Ok(())
}
```

* Finalizes processing and returns remaining buffered results.

## Builder Options

### Streaming Options

| Method | Argument Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size(usize)` | `usize` | `5000` | Data chunk size |
| `overlap(usize)` | `usize` | `500` | Overlap between chunks |
| `merge_strategy(...)` | `merge_strategy` | `"weighted_average"` | Strategy for blending overlap regions |

## Options

### merge_strategy

*See: [Merge Strategies](../user-guide/merge.md)*

* `"weighted_average"` (default; alias: `"weighted"`)
* `"average"` (alias: `"mean"`)
* `"take_first"` (alias: `"first"`)
* `"take_last"` (alias: `"last"`)
