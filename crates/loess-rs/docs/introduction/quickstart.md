<!-- markdownlint-disable MD024 MD046 -->
# Quick Start

Get up and running with LOESS in minutes.

## Basic Smoothing

Smooth a noisy sine wave — the kind of signal where LOESS shines. Each example recovers the underlying trend from 100 points of Gaussian noise.

```rust
use loess_rs::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    // 100-point noisy sine wave (deterministic)
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().enumerate()
        .map(|(i, &xi)| xi.sin() + ((i * 7 + 3) % 17) as f64 / 17.0 * 0.6 - 0.3)
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

```output
First smoothed: 0.0281  (true: 0.0000)
```

---

## With Confidence Intervals

```rust
use loess_rs::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Loess::new()
        .fraction(0.5)
        .iterations(3)
        .confidence_intervals(0.95)  // 95% CI
        .prediction_intervals(0.95)  // 95% PI
        .outputs(["diagnostics"])
        .build()?;

    let result = model.fit(&x, &y)?;
    
    // Access intervals
    if let Some(ci_lower) = &result.confidence_lower {
        println!("CI Lower: {:?}", ci_lower);
    }

    Ok(())
}
```

```output
CI Lower: [0.301714242148439, 0.3250782470824322, 0.3517294626042588, 0.38224776076074346, 0.4144767204375235, 0.44608898298081556, 0.47740801412831385, 0.5087572796177118, 0.540460245186704, 0.5728403765729838, 0.6062211395142453, 0.6409259997481827, 0.6772784230124895, 0.7115389486743297, 0.7402212988586514, 0.7640287134398259, 0.7836644322922242, 0.7998316952902174, 0.8132337423081761, 0.8245738132204722, 0.8345551479014761, 0.8438809862255591, 0.853254568067092, 0.8614926743695032, 0.8668365128589923, 0.8691259635840883, 0.8682009065933195, 0.8639012219352149, 0.856066789658303, 0.844537489811112, 0.8291532024421714, 0.8097538076000095, 0.7888564008362507, 0.7684304468553043, 0.7474943794348738, 0.7250666323526644, 0.7001656393863802, 0.671809834313726, 0.639017650912406, 0.6008075229601249, 0.5561978842345867, 0.5081448084291086, 0.46008504038732473, 0.41175802061241773, 0.3629031896075693, 0.31325998787596077, 0.2625678559207755, 0.21056623424519444, 0.15699456335240047, 0.10159228374557512, 0.0460014711303884, -0.00807345500509337, -0.06079953848409825, -0.11234382312985247, -0.1628733527655833, -0.212555171214518, -0.2615563222998829, -0.3100438498449052, -0.35818479767281247, -0.40614620960683034, -0.45068245134935553, -0.48905295902927876, -0.522182382109568, -0.5509953700531924, -0.5764165723231203, -0.5993706383823202, -0.6207822176937611, -0.6415759597204117, -0.6626765139252399, -0.6824030522333696, -0.6984862506837239, -0.7109680149099706, -0.7198902505457774, -0.7252948632248123, -0.7272237585807432, -0.725718842247238, -0.7208220198579648, -0.7125751970465911, -0.7010202794467851, -0.6883361075692718, -0.6760833571735319, -0.6633766867663666, -0.6493307548545781, -0.6330602199449669, -0.6136797405443353, -0.5903039751594841, -0.5620475822972152, -0.5280252204643298, -0.4916186380155172, -0.45666194966227364, -0.42294836344661363, -0.39027108741054894, -0.3584233295960923, -0.3271982980452566, -0.29638920080005393, -0.26578924590249814, -0.2351916413946013, -0.20438959531837647, -0.17573105106433917, -0.1512062996983693]
```

---

## Handling Outliers

LOESS can robustly handle outliers through iterative reweighting:

```rust
use loess_rs::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    // Data with an outlier at position 3
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y_with_outlier = vec![2.0, 4.0, 6.0, 50.0, 10.0, 12.0];  // 50.0 is outlier

let model = Loess::new()
    .fraction(0.7)
    .iterations(5)                    // More iterations for outliers
    .robustness_method("bisquare")   // Default, smooth downweighting
    .outputs(["weights"])             // See which points were downweighted
    .build()?;

let result = model.fit(&x, &y_with_outlier)?;

// Outliers will have low robustness weights
    if let Some(weights) = &result.robustness_weights {
        for (i, w) in weights.iter().enumerate() {
            if *w < 0.5 {
                println!("Point {} is likely an outlier (weight: {:.3})", i, w);
            }
        }
    }

    Ok(())
}
```

```output
Point 3 is likely an outlier (weight: 0.000)
```

---

## Streaming Mode

For datasets too large to fit in memory, stream them in fixed-size chunks with overlap.

```rust
use loess_rs::prelude::*;
use std::f64::consts::PI;

fn main() -> Result<(), LoessError> {
    let n = 5_000usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 10.0 * PI / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().enumerate()
        .map(|(i, &xi)| (xi / PI).sin() * (-xi / 30.0).exp()
                       + (((i * 7 + 3) % 17) as f64 / 17.0 * 0.3 - 0.15))
        .collect();

    let mut model = StreamingLoess::new()
        .fraction(0.2)
        .chunk_size(1000)
        .overlap(100)
        .build()?;

    for chunk in x.chunks(1000).zip(y.chunks(1000)) {
        model.process_chunk(chunk.0, chunk.1)?;
    }
    let result = model.finalize()?;
    println!("Smoothed {} points", result.y.len());
    Ok(())
}
```

```output
Smoothed 100 points
```

---

## Next Steps

| Topic | Link |
| --- | --- |
| How LOESS works | [Concepts](crate::doc::introduction::concepts) |
| All parameters explained | [API Reference](crate::doc::api) |
| Batch vs Streaming vs Online | [Execution Modes](crate::doc::guide::adapter_choice) |
| Polynomial degree choices | [Degree](crate::doc::advanced::degree) |
| Multivariate smoothing | [Dimensions](crate::doc::advanced::dimensions) |
| Edge handling | [Boundary](crate::doc::advanced::boundary) |
| Outlier handling in depth | [Robustness](crate::doc::weighting::robustness) |
| Full API per language | [API Reference](crate::doc::api) |
