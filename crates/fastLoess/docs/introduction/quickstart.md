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
        .intervals(IntervalsBuilder::new().confidence(0.95).prediction(0.95))  // 95% PI
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
CI Lower: [0.2569579331671805, 0.2801336079981137, 0.3065269593865368, 0.3367099823610434, 0.36851972199857874, 0.3996243989519644, 0.43034601134864664, 0.4610102683715577, 0.491947139571139, 0.523491364352784, 0.5559830751397058, 0.5897682331748721, 0.6251983270663672, 0.6585658506534957, 0.6864172774180314, 0.7094851020368325, 0.7284938998724965, 0.7441571374690452, 0.75717642538924, 0.7682433303978463, 0.7780426945988379, 0.7872562536698384, 0.7965657456991505, 0.804768739696782, 0.8100910272032521, 0.8123617815142494, 0.8114153292110879, 0.8070911640785604, 0.7992339906355475, 0.7876937464088201, 0.7723254373759072, 0.7529885923659496, 0.7322232897194274, 0.7120232186652754, 0.6914291364802132, 0.669475925096847, 0.645188553691031, 0.6175791623256436, 0.5856464615257945, 0.5483780032774143, 0.5047546551183592, 0.45769335148574153, 0.4105943671638449, 0.363165852294069, 0.31512188265271496, 0.2661820137988563, 0.21607034855545676, 0.16451464987548653, 0.11124581993591033, 0.05599782292835266, 0.0004104796937606514, -0.053810675616622415, -0.10682816369826711, -0.15880119696544054, -0.20988591195799816, -0.2602356491515508, -0.31000099999030983, -0.35932941874818536, -0.4083645833882744, -0.45724595018111447, -0.5026964526979779, -0.5419420095572668, -0.5758748932503562, -0.6053926161506985, -0.6314017279052532, -0.6548199668594887, -0.676576086252424, -0.6976076891825759, -0.7188579715819943, -0.7386668595654161, -0.7547851680271042, -0.767272323102399, -0.7761838913321039, -0.7815709253665155, -0.7834797037423047, -0.7819517252537312, -0.7770237672693091, -0.7687278788218782, -0.7570918543973237, -0.7442771507318323, -0.7318253230454417, -0.7188323931626669, -0.7043977692933999, -0.687627474163232, -0.6676369196670345, -0.6435521676873293, -0.6145088460062823, -0.5796489757765437, -0.542383982327747, -0.5065762115038128, -0.47204307608264756, -0.43859681973450165, -0.40604452216616094, -0.3741887043708326, -0.3428280099155516, -0.3117576760241457, -0.28076994181851284, -0.24965459128353615, -0.22075430430539345, -0.1960528512941111]
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
