#![cfg(feature = "dev")]
use approx::assert_abs_diff_eq;
use fastLoess::prelude::*;

#[test]
fn test_smooth_pass_consistency_robust() {
    let n = 50;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();

    // Sequential fit with 3 iterations
    let seq_res = Loess::new()
        .fraction(0.3)
        .iterations(3)
        .parallel(false)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Parallel fit with 3 iterations
    let par_res = Loess::new()
        .fraction(0.3)
        .iterations(3)
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    for i in 0..n {
        assert_abs_diff_eq!(seq_res.y[i], par_res.y[i], epsilon = 1e-12);
    }
    println!("Robust smooth pass consistency (3 iters): OK");
}

/// Parallel fit with Normalized distance exercises the Normalized arm of
/// `LoessDistanceCalculator::distance_squared` and `split_distance_squared`.
#[test]
fn test_parallel_normalized_distance() {
    let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi * 1.5).collect();

    let res = Loess::new()
        .fraction(0.5)
        .distance_metric("normalized")
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert!(!res.y.is_empty());
}

/// Parallel fit with Manhattan distance.
#[test]
fn test_parallel_manhattan_distance() {
    let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi * 1.5).collect();

    let res = Loess::new()
        .fraction(0.5)
        .distance_metric("manhattan")
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert!(!res.y.is_empty());
}

/// Parallel fit with Chebyshev distance.
#[test]
fn test_parallel_chebyshev_distance() {
    let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi * 1.5).collect();

    let res = Loess::new()
        .fraction(0.5)
        .distance_metric("chebyshev")
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert!(!res.y.is_empty());
}

/// Parallel fit with Minkowski(3) distance.
#[test]
fn test_parallel_minkowski_distance() {
    let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi * 1.5).collect();

    let res = Loess::new()
        .fraction(0.5)
        .distance_metric("minkowski:3.0")
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert!(!res.y.is_empty());
}
