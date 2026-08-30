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

```output
First smoothed: 0.2376  (true: 0.0000)
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
        .return_diagnostics()
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
CI Lower: [0.3017142421484391, 0.3250782470824323, 0.3517294626042589, 0.38224776076074357, 0.41447672043752354, 0.4460889829808156, 0.4774080141283138, 0.5087572796177118, 0.5404602451867039, 0.5728403765729837, 0.606221139514245, 0.6409259997481824, 0.6772784230124893, 0.7115389486743294, 0.7402212988586512, 0.7640287134398258, 0.7836644322922242, 0.7998316952902175, 0.8132337423081766, 0.8245738132204724, 0.8345551479014766, 0.8438809862255595, 0.8532545680670925, 0.8614926743695038, 0.8668365128589929, 0.8691259635840888, 0.8682009065933197, 0.863901221935215, 0.8560667896583027, 0.8445374898111119, 0.8291532024421712, 0.8097538076000091, 0.7888564008362504, 0.7684304468553039, 0.7474943794348735, 0.7250666323526643, 0.7001656393863802, 0.6718098343137259, 0.6390176509124059, 0.600807522960125, 0.5561978842345868, 0.5081448084291088, 0.46008504038732484, 0.4117580206124178, 0.36290318960756923, 0.31325998787596077, 0.2625678559207754, 0.21056623424519444, 0.1569945633524004, 0.10159228374557505, 0.04600147113038832, -0.00807345500509346, -0.06079953848409833, -0.11234382312985255, -0.16287335276558337, -0.21255517121451808, -0.2615563222998829, -0.31004384984490524, -0.3581847976728125, -0.4061462096068304, -0.4506824513493556, -0.4890529590292788, -0.5221823821095681, -0.5509953700531927, -0.5764165723231204, -0.5993706383823205, -0.6207822176937614, -0.641575959720412, -0.6626765139252402, -0.6824030522333699, -0.6984862506837242, -0.7109680149099709, -0.7198902505457776, -0.7252948632248124, -0.7272237585807433, -0.7257188422472381, -0.7208220198579648, -0.7125751970465911, -0.7010202794467851, -0.6883361075692718, -0.6760833571735319, -0.6633766867663666, -0.649330754854578, -0.6330602199449669, -0.6136797405443352, -0.5903039751594841, -0.562047582297215, -0.5280252204643296, -0.49161863801551714, -0.4566619496622736, -0.4229483634466135, -0.39027108741054883, -0.35842332959609224, -0.32719829804525646, -0.2963892008000538, -0.26578924590249803, -0.23519164139460116, -0.2043895953183763, -0.17573105106433903, -0.15120629969836916]
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
    .fraction(0.5)
    .iterations(5)                    // More iterations for outliers
    .robustness_method("bisquare")   // Default, smooth downweighting
    .return_robustness_weights()      // See which points were downweighted
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
                       + ((i * 7 + 3) as f64 % 1.7 - 0.85) * 0.15)
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
| How LOESS works | [Concepts](concepts.md) |
| All parameters explained | [API Reference](api.md) |
| Batch vs Streaming vs Online | [Execution Modes](adapter-choice.md) |
| Polynomial degree choices | [Degree](degree.md) |
| Multivariate smoothing | [Dimensions](dimensions.md) |
| Edge handling | [Boundary](boundary.md) |
| Outlier handling in depth | [Robustness](robustness.md) |
| Full API per language | [API Reference](api.md) |
