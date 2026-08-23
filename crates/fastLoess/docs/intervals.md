```rust
use fastLoess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Loess::new()
        .fraction(0.5)
        .confidence_intervals(0.95)  // 95% CI
        .build()?;

    let result = model.fit(&x, &y)?;

    // Access intervals
    if let (Some(lower), Some(upper)) = (&result.confidence_lower, &result.confidence_upper) {
        for i in 0..result.y.len() {
            println!("x={:.2}: y={:.2} [{:.2}, {:.2}]",
                result.x[i], result.y[i], lower[i], upper[i]);
        }
    }

    Ok(())
}
```