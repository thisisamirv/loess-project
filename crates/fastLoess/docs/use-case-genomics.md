```rust
use fastLoess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let positions = x.clone();
    let observed = y.clone();

    let model = Loess::new()
        .fraction(0.1)
        .iterations(3)
        .confidence_intervals(0.95)
        .build()?;

    let result = model.fit(&positions, &observed)?;
    // result.y contains smoothed methylation profile
    // result.confidence_lower/upper contain 95% CI bounds

    Ok(())
}
```