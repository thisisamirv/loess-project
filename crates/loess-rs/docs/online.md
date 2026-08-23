```rust
use loess_rs::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let mut processor = OnlineLoess::new()
        .fraction(0.2)
        .iterations(1)
        .window_capacity(100)
        .min_points(5)
        .update_mode("incremental")
        .build()?;

    for i in 0..x.len() {
        if let Some(output) = processor.add_point(&[x[i]], y[i])? {
            println!("Smoothed: {:.2}", output.y);
        }
    }

    Ok(())
}
```