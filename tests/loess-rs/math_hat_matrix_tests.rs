#![cfg(feature = "dev")]

use approx::assert_relative_eq;
use loess_rs::internals::math::hat_matrix::HatMatrixStats;

// ============================================================================
// HatMatrixStats::from_leverage Tests
// ============================================================================

#[test]
fn test_from_leverage_basic() {
    // Single point: leverage = 1.0, trace = 1.0
    let leverage = vec![1.0f64];
    let stats = HatMatrixStats::from_leverage(leverage);
    assert_relative_eq!(stats.trace, 1.0);
    // delta1 = 1 - 2*1 + 1^2 = 0
    assert_relative_eq!(stats.delta1, 0.0);
}

#[test]
fn test_from_leverage_uniform() {
    // n points each with leverage 1/n (uniform smoother)
    let n = 5usize;
    let lev = 1.0f64 / n as f64;
    let leverage = vec![lev; n];
    let stats = HatMatrixStats::from_leverage(leverage.clone());

    let expected_trace = leverage.iter().sum::<f64>();
    assert_relative_eq!(stats.trace, expected_trace, epsilon = 1e-12);

    // delta1 = n - 2*trace + sum(l^2)
    let trace_sq: f64 = leverage.iter().map(|&l| l * l).sum();
    let expected_delta1 = n as f64 - 2.0 * expected_trace + trace_sq;
    assert_relative_eq!(stats.delta1, expected_delta1, epsilon = 1e-12);

    // delta2 = delta1^2 / n
    let expected_delta2 = expected_delta1 * expected_delta1 / n as f64;
    assert_relative_eq!(stats.delta2, expected_delta2, epsilon = 1e-12);
}

#[test]
fn test_from_leverage_varied() {
    let leverage = vec![0.1f64, 0.2, 0.3, 0.4];
    let stats = HatMatrixStats::from_leverage(leverage.clone());

    let expected_trace: f64 = leverage.iter().sum();
    assert_relative_eq!(stats.trace, expected_trace, epsilon = 1e-12);
    assert_eq!(stats.leverage, leverage);
}

#[test]
fn test_from_leverage_stores_values() {
    let leverage = vec![0.25f64, 0.5, 0.75];
    let stats = HatMatrixStats::from_leverage(leverage.clone());
    assert_eq!(stats.leverage, leverage);
}

#[test]
fn test_from_leverage_large_n() {
    let n = 100usize;
    let leverage: Vec<f64> = (0..n)
        .map(|i| (i + 1) as f64 / (n * (n + 1) / 2) as f64)
        .collect();
    let stats = HatMatrixStats::from_leverage(leverage.clone());
    let expected_trace: f64 = leverage.iter().sum();
    assert_relative_eq!(stats.trace, expected_trace, epsilon = 1e-10);
    assert!(stats.delta1.is_finite());
    assert!(stats.delta2.is_finite());
}

// ============================================================================
// HatMatrixStats::compute_residual_scale Tests
// ============================================================================

#[test]
fn test_compute_residual_scale_basic() {
    // delta1 = 4, rss = 16 => sigma = sqrt(16/4) = 2
    let leverage = vec![0.0f64; 4]; // all zero leverage
    let stats = HatMatrixStats::from_leverage(leverage);
    // With all-zero leverage: delta1 = 4 - 0 + 0 = 4
    let sigma = stats.compute_residual_scale(16.0);
    assert_relative_eq!(sigma, 2.0, epsilon = 1e-12);
}

#[test]
fn test_compute_residual_scale_zero_delta1() {
    // When delta1 = 0, should return 0 rather than NaN
    let leverage = vec![1.0f64]; // n=1, trace=1, delta1 = 1 - 2 + 1 = 0
    let stats = HatMatrixStats::from_leverage(leverage);
    let sigma = stats.compute_residual_scale(100.0);
    assert_relative_eq!(sigma, 0.0);
}

#[test]
fn test_compute_residual_scale_zero_rss() {
    let leverage = vec![0.1f64, 0.2, 0.3];
    let stats = HatMatrixStats::from_leverage(leverage);
    let sigma = stats.compute_residual_scale(0.0);
    assert_relative_eq!(sigma, 0.0);
}

#[test]
fn test_from_leverage_clone_and_debug() {
    let leverage = vec![0.1f64, 0.2];
    let stats = HatMatrixStats::from_leverage(leverage);
    let cloned = stats.clone();
    assert_eq!(stats, cloned);
    // Ensure Debug is implemented (won't panic)
    let _ = format!("{:?}", cloned);
}
