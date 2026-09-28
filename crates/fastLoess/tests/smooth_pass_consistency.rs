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

/// Rayon-parallel `gradient_pass_parallel` should agree with the serial
/// gradient pass (Direct mode) to within numerical precision.
#[test]
fn test_gradient_pass_consistency() {
    let n = 60;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1 * xi).collect();

    let seq_res = Loess::new()
        .fraction(0.3)
        .surface_mode("direct")
        .return_gradient()
        .parallel(false)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let par_res = Loess::new()
        .fraction(0.3)
        .surface_mode("direct")
        .return_gradient()
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let seq_gradient = seq_res.gradient.expect("serial gradient should be Some");
    let par_gradient = par_res.gradient.expect("parallel gradient should be Some");
    assert_eq!(seq_gradient.len(), par_gradient.len());
    for (s, p) in seq_gradient.iter().zip(par_gradient.iter()) {
        assert_abs_diff_eq!(s, p, epsilon = 1e-9);
    }
}

#[test]
fn test_parallel_gradient_retains_small_local_spread() {
    let x: Vec<f64> = (0..40)
        .map(|index| {
            if index < 20 {
                index as f64 * 1e-5
            } else {
                (index - 19) as f64
            }
        })
        .collect();
    let y: Vec<f64> = x.iter().map(|&value| 3.0 * value + 1.0).collect();

    let result = Loess::new()
        .fraction(0.1)
        .surface_mode("direct")
        .boundary_policy("noboundary")
        .return_gradient()
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let gradient = result.gradient.expect("gradient should be Some");
    assert_abs_diff_eq!(gradient[10], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(gradient[30], 3.0, epsilon = 1e-6);
}
