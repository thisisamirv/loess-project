```rust
use loess_rs::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    // 100-point noisy sine wave (deterministic)
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().enumerate()
        .map(|(i, &xi)| xi.sin() + ((i * 7 + 3) as f64 % 1.7 - 0.85) * 0.3)
        .collect();

    let model = Loess::new()
        .fraction(0.3)
        .iterations(3)
        .build()?;

    let result = model.fit(&x, &y)?;
    println!("First smoothed: {:.4}  (true: {:.4})", result.y[0], x[0].sin());
    Ok(())
}
```