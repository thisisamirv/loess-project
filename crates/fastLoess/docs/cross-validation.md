```rust
use fastLoess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Loess::new()
        .cv_method("kfold")
        .cv_k(5)
        .cv_fractions(vec![0.2, 0.3, 0.5, 0.7])
        .build()?;

    let result = model.fit(&x, &y)?;

    // The best fraction was automatically selected
    println!("Selected fraction: {}", result.fraction_used);

    if let Some(cv_scores) = &result.cv_scores {
        println!("CV scores: {:?}", cv_scores);
    }

    Ok(())
}
```