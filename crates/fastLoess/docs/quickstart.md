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
use fastLoess::prelude::*;
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
CI Lower: [0.25695793316718074, 0.28013360799811393, 0.306526959386537, 0.3367099823610436, 0.3685197219985789, 0.3996243989519645, 0.43034601134864664, 0.4610102683715577, 0.4919471395711389, 0.5234913643527839, 0.5559830751397056, 0.5897682331748719, 0.625198327066367, 0.6585658506534955, 0.6864172774180312, 0.7094851020368323, 0.7284938998724966, 0.7441571374690453, 0.7571764253892405, 0.7682433303978466, 0.7780426945988385, 0.7872562536698389, 0.7965657456991511, 0.8047687396967826, 0.8100910272032527, 0.8123617815142499, 0.8114153292110882, 0.8070911640785605, 0.7992339906355472, 0.7876937464088198, 0.7723254373759071, 0.7529885923659494, 0.7322232897194271, 0.7120232186652751, 0.6914291364802129, 0.669475925096847, 0.6451885536910311, 0.6175791623256435, 0.5856464615257944, 0.5483780032774144, 0.5047546551183593, 0.45769335148574175, 0.41059436716384506, 0.36316585229406906, 0.31512188265271496, 0.26618201379885637, 0.21607034855545676, 0.16451464987548653, 0.11124581993591023, 0.05599782292835259, 0.0004104796937605959, -0.053810675616622526, -0.10682816369826723, -0.1588011969654406, -0.20988591195799827, -0.26023564915155084, -0.3100009999903099, -0.35932941874818536, -0.40836458338827447, -0.45724595018111447, -0.502696452697978, -0.5419420095572668, -0.5758748932503562, -0.6053926161506987, -0.6314017279052533, -0.6548199668594891, -0.6765760862524244, -0.6976076891825762, -0.7188579715819946, -0.7386668595654166, -0.7547851680271045, -0.7672723231023992, -0.7761838913321041, -0.7815709253665157, -0.7834797037423049, -0.7819517252537315, -0.7770237672693092, -0.7687278788218782, -0.7570918543973238, -0.7442771507318324, -0.7318253230454418, -0.7188323931626669, -0.7043977692933998, -0.687627474163232, -0.6676369196670344, -0.6435521676873293, -0.6145088460062821, -0.5796489757765437, -0.542383982327747, -0.5065762115038127, -0.4720430760826475, -0.43859681973450154, -0.4060445221661609, -0.3741887043708325, -0.3428280099155515, -0.3117576760241456, -0.28076994181851267, -0.24965459128353595, -0.22075430430539328, -0.19605285129411093]
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
use fastLoess::prelude::*;
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
| How LOESS works | [Concepts](crate::doc::concepts) |
| All parameters explained | [API Reference](crate::doc::api) |
| Batch vs Streaming vs Online | [Execution Modes](crate::doc::adapter_choice) |
| Polynomial degree choices | [Degree](crate::doc::degree) |
| Multivariate smoothing | [Dimensions](crate::doc::dimensions) |
| Edge handling | [Boundary](crate::doc::boundary) |
| Outlier handling in depth | [Robustness](crate::doc::robustness) |
| Full API per language | [API Reference](crate::doc::api) |
