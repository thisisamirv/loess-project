```rust
use fastLoess::prelude::*;

fn main() -> Result<(), LoessError> {
    let mut processor = OnlineLoess::new()
        .fraction(0.3)
        .iterations(1)
        .window_capacity(25)
        .min_points(5)
        .update_mode("incremental")
        .build()?;

    for i in 0..100 {
        let xi = i as f64;
        let yi = 20.0 + 5.0 * (xi / 10.0).sin() + (xi * 1.7).sin() * 0.5;

        if let Some(output) = processor.add_point(&[xi], yi)? {
            println!("Time {}: smoothed = {:.2}", xi, output.y);
        }
    }

    Ok(())
}
```