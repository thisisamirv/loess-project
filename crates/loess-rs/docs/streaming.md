# Streaming Adapter

Process large datasets in chunks with configurable overlap.

## When to Use

- Dataset >100,000 points
- Memory-constrained environments
- Batch processing pipelines

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `chunk_size` | 5000 | Points per chunk |
| `overlap` | 500 | Overlap between chunks |
| `merge_strategy` | `"weighted_average"` | How to merge overlaps |

### Merge Strategies

| Strategy | Behavior |
| --- | --- |
| `"average"` | Average overlapping values |
| `"weighted_average"` | Distance-weighted blend (default) |
| `"take_first"` | Keep left chunk values |
| `"take_last"` | Keep right chunk values |

![Merge Strategies](../assets/diagrams/merge_comparison.svg)

## Example

```rust
use loess_rs::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x_chunk: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y_chunk: Vec<f64> = x_chunk.iter().map(|&xi| xi.sin() + 0.1).collect();

    let mut processor = StreamingLoess::new()
        .fraction(0.3)
        .iterations(2)
        .chunk_size(50)
        .overlap(10)
        .merge_strategy("average")
        .build()?;

    let result = processor.process_chunk(&x_chunk, &y_chunk)?;
    println!("Chunk processed: {} points", result.y.len());

    let final_result = processor.finalize()?;
    println!("Final: {} points", final_result.y.len());

    Ok(())
}
```

```output
Chunk processed: 90 points
Final: 10 points
```

---

!!! warning "Always call finalize()"
    In Rust, always call `processor.finalize()` after processing all chunks to retrieve buffered overlap data.
