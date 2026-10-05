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
        .outputs(["diagnostics"])
        .intervals(IntervalsBuilder::new().confidence(0.95).prediction(0.95))  // 95% PI
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
CI Lower: [0.3017142421484391, 0.32507824708243227, 0.35172946260425886, 0.3822477607607435, 0.41447672043752354, 0.4460889829808156, 0.4774080141283139, 0.5087572796177121, 0.5404602451867042, 0.5728403765729841, 0.6062211395142457, 0.640925999748183, 0.67727842301249, 0.7115389486743301, 0.7402212988586517, 0.7640287134398264, 0.7836644322922247, 0.7998316952902178, 0.8132337423081767, 0.8245738132204726, 0.8345551479014768, 0.8438809862255595, 0.8532545680670925, 0.8614926743695037, 0.8668365128589928, 0.8691259635840884, 0.8682009065933194, 0.8639012219352146, 0.8560667896583023, 0.8445374898111112, 0.8291532024421705, 0.8097538076000085, 0.7888564008362497, 0.7684304468553035, 0.7474943794348732, 0.7250666323526641, 0.70016563938638, 0.6718098343137259, 0.6390176509124061, 0.6008075229601251, 0.5561978842345869, 0.5081448084291088, 0.4600850403873249, 0.41175802061241784, 0.3629031896075693, 0.3132599878759608, 0.2625678559207756, 0.21056623424519455, 0.15699456335240053, 0.10159228374557516, 0.04600147113038844, -0.008073455005093343, -0.06079953848409822, -0.11234382312985243, -0.16287335276558323, -0.21255517121451797, -0.2615563222998828, -0.31004384984490513, -0.3581847976728124, -0.4061462096068303, -0.4506824513493554, -0.48905295902927864, -0.5221823821095679, -0.5509953700531922, -0.57641657232312, -0.5993706383823199, -0.6207822176937607, -0.6415759597204113, -0.6626765139252394, -0.6824030522333693, -0.6984862506837235, -0.7109680149099703, -0.719890250545777, -0.7252948632248121, -0.7272237585807431, -0.7257188422472379, -0.7208220198579648, -0.712575197046591, -0.7010202794467851, -0.6883361075692718, -0.6760833571735319, -0.6633766867663667, -0.649330754854578, -0.633060219944967, -0.6136797405443353, -0.5903039751594841, -0.5620475822972151, -0.5280252204643298, -0.49161863801551725, -0.4566619496622737, -0.4229483634466137, -0.3902710874105489, -0.3584233295960923, -0.32719829804525646, -0.2963892008000539, -0.265789245902498, -0.23519164139460116, -0.20438959531837628, -0.175731051064339, -0.15120629969836913]
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
