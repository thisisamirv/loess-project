<!-- markdownlint-disable MD024 MD046 -->
# Quick Start

Get up and running with LOESS in minutes.

## Basic Smoothing

Smooth a noisy sine wave — the kind of signal where LOESS shines. Each example recovers the underlying trend from 100 points of Gaussian noise.

```rust
use fastLoess::prelude::*;
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
use fastLoess::prelude::*;
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
CI Lower: [0.2577313994160416, 0.28065564316417135, 0.3067998772748776, 0.3368545611486861, 0.36857523666947994, 0.3996318458148274, 0.43034653861933597, 0.46101784764264053, 0.4920046355995411, 0.5236113831576814, 0.5562427641174795, 0.5902361304990443, 0.625946609687663, 0.6596619734295803, 0.6879852804169622, 0.7115325738662553, 0.7311406555834993, 0.7475122055013467, 0.7612440853784198, 0.7732858005986858, 0.7840874634022451, 0.794570343166188, 0.8053910109160316, 0.8151763023892671, 0.8224570763481013, 0.8267136883458364, 0.8275497889302532, 0.8209537786761842, 0.8111841214544561, 0.7979167131271738, 0.7809615208225402, 0.7603768292962768, 0.7384615745641331, 0.7172969325534325, 0.6960024262947815, 0.6735078809134224, 0.6487372102497242, 0.6207999911028678, 0.5883402565964531, 0.5505419427924512, 0.5064482763005476, 0.4589319604440169, 0.4114796908829429, 0.3637787224982863, 0.31550035685601696, 0.26642460205245344, 0.2161839376269688, 0.16457677399867415, 0.11125970579850057, 0.05600483902044749, 0.00041543067589756844, -0.05379886080186893, -0.10676813531930139, -0.15868041301689448, -0.20963183556764198, -0.2598299191890814, -0.3093392085512675, -0.3583959840505485, -0.40706404550476993, -0.4554901212122665, -0.5004738586642552, -0.5392046956092924, -0.572655685595772, -0.601857973187632, -0.6273883014537248, -0.6502696424299358, -0.6713303392600497, -0.6914036524554246, -0.7115141251185615, -0.7300820081329651, -0.7446220627178773, -0.7553989905227363, -0.7624083155622648, -0.7655496772964432, -0.769032590206554, -0.7695636035247404, -0.7665804516228186, -0.7598561821730663, -0.7497856544873446, -0.7382325095688936, -0.7268033456664437, -0.7147788381705173, -0.7010582267792572, -0.6849909117637988, -0.6655991368446305, -0.6419921549940061, -0.6134168929454223, -0.5789030123947878, -0.5419164177527114, -0.5063161080333289, -0.47192195554132643, -0.4385396817515972, -0.4060371668046828, -0.3741883867428736, -0.34282077322986704, -0.31170235851562, -0.2806303406056458, -0.2493864611334142, -0.22024306336128008, -0.195293973266987]
```

---

## Handling Outliers

LOESS can robustly handle outliers through iterative reweighting:

```rust
use fastLoess::prelude::*;
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
use fastLoess::prelude::*;
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
