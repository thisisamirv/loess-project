```rust
use fastLoess::prelude::*;
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