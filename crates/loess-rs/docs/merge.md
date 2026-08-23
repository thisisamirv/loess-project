```rust
use loess_rs::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x_chunk: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y_chunk: Vec<f64> = x_chunk.iter().map(|&xi| xi.sin() + 0.1).collect();

    let mut model = StreamingLoess::new()
        .merge_strategy("average")
        .chunk_size(5000)
        .overlap(500)
        .build()?;
    let result = model.process_chunk(&x_chunk, &y_chunk)?;;

    Ok(())
}
```