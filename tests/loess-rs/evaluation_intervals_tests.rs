#![cfg(feature = "dev")]
//! Tests for confidence and prediction interval computation.
//!
//! These tests verify the interval estimation functionality used in LOESS for:
//! - Standard error computation
//! - Confidence intervals
//! - Prediction intervals
//! - Z-score approximation
//! - Interval validation
//!
//! ## Test Organization
//!
//! 1. **Standard Error** - Point and window SE computation
//! 2. **Z-Score** - Normal distribution quantiles
//! 3. **Confidence Intervals** - CI computation and validation
//! 4. **Prediction Intervals** - PI computation and validation
//! 5. **Edge Cases** - Zero bandwidth, zero weights, invalid levels

use approx::assert_relative_eq;
use loess_rs::prelude::*;

use loess_rs::internals::engine::validator::Validator;
use loess_rs::internals::evaluation::intervals::IntervalMethod;
use loess_rs::internals::primitives::errors::LoessError;

// ============================================================================
// Helper Functions
// ============================================================================

fn uniform_weight_fn<T: num_traits::Float>(_u: T) -> T {
    T::one()
}

// ============================================================================
// Standard Error Tests
// ============================================================================

/// Test point SE computation at center point.
///
/// Verifies correct SE calculation for middle point.
#[test]
fn test_compute_point_se_center() {
    let x = vec![0.0f64, 1.0, 2.0];
    let y = vec![0.0f64, 1.0, 0.0];
    let y_smooth = vec![0.0f64, 0.0, 0.0];
    let robustness = vec![1.0f64; 3];

    let est = IntervalMethod::se();
    let mut std_errors = vec![0.0; x.len()];

    est.compute_window_se(
        &x,
        &y,
        &y_smooth,
        3,
        &robustness,
        &mut std_errors,
        &uniform_weight_fn,
    );

    let se = std_errors[1];
    let expected = (1.0f64 / 3.0f64).sqrt();

    assert_relative_eq!(se, expected, epsilon = 1e-12);
}

/// Test SE with zero bandwidth (identical x values).
///
/// Verifies that zero bandwidth produces zero SE.
#[test]
fn test_compute_se_zero_bandwidth() {
    let x = vec![1.0f64, 1.0];
    let y = vec![1.0f64, 2.0];
    let y_smooth = vec![1.0f64, 2.0];
    let robustness = vec![1.0f64; 2];

    let est = IntervalMethod::se();
    let mut std_errors = vec![0.0; x.len()];

    est.compute_window_se(
        &x,
        &y,
        &y_smooth,
        2,
        &robustness,
        &mut std_errors,
        &|_: f64| 1.0,
    );

    assert_eq!(std_errors[0], 0.0, "SE should be zero for zero bandwidth");
}

/// Test SE with zero weights.
///
/// Verifies that zero robustness weights produce zero SE.
#[test]
fn test_compute_se_zero_weights() {
    let x = vec![0.0f64, 1.0];
    let y = vec![0.0f64, 1.0];
    let ys = vec![0.0f64, 1.0];
    let robustness_zero = vec![0.0f64; 2];

    let est = IntervalMethod::se();
    let mut std_errors = vec![0.0; x.len()];

    est.compute_window_se(
        &x,
        &y,
        &ys,
        2,
        &robustness_zero,
        &mut std_errors,
        &|_: f64| 1.0,
    );

    assert_eq!(std_errors[0], 0.0, "SE should be zero for zero weights");
}

/// Test SE with insufficient degrees of freedom.
///
/// Verifies that df <= 0 produces zero SE.
#[test]
fn test_compute_se_insufficient_df() {
    let x = vec![0.0f64, 1.0];
    let y = vec![0.0f64, 1.0];
    let ys = vec![0.0f64, 1.0];
    let robustness_ones = vec![1.0f64; 2];

    let est = IntervalMethod::se();
    let mut std_errors = vec![0.0; x.len()];

    // With n=2 and p=2 (local linear), df = 0 => SE = 0
    est.compute_window_se(
        &x,
        &y,
        &ys,
        2,
        &robustness_ones,
        &mut std_errors,
        &|_: f64| 1.0,
    );

    assert_eq!(std_errors[0], 0.0, "SE should be zero for df <= 0");
}

/// Test window SE computation.
///
/// Verifies that SE vector is correctly populated.
#[test]
fn test_compute_window_se_vector() {
    let x = vec![0.0f64, 1.0, 2.0];
    let y = vec![0.0f64, 1.0, 0.0];
    let y_smooth = vec![0.0f64, 0.0, 0.0];
    let robustness = vec![1.0f64; 3];
    let mut std_err = vec![0.0f64; 3];

    let estimator = IntervalMethod::se();
    estimator.compute_window_se(
        &x,
        &y,
        &y_smooth,
        3,
        &robustness,
        &mut std_err,
        &uniform_weight_fn,
    );

    // Middle element should match expected value
    let expected_mid = (1.0f64 / 3.0f64).sqrt();
    assert_relative_eq!(std_err[1], expected_mid, epsilon = 1e-12);
    assert_eq!(std_err.len(), 3, "SE vector should have correct length");
}

// ============================================================================
// Z-Score Tests
// ============================================================================

/// Test z-score for common confidence levels.
///
/// Verifies correct z-scores for 90%, 95%, 99%.
#[test]
fn test_z_score_common_levels() {
    let z95 = IntervalMethod::approximate_z_score(0.95f64).expect("z95");
    assert_relative_eq!(z95, 1.96f64, epsilon = 1e-6);

    let z99 = IntervalMethod::approximate_z_score(0.99f64).expect("z99");
    assert_relative_eq!(z99, 2.576f64, epsilon = 1e-6);

    let z90 = IntervalMethod::approximate_z_score(0.90f64).expect("z90");
    assert_relative_eq!(z90, 1.645f64, epsilon = 1e-6);
}

/// Test z-score for arbitrary level.
///
/// Verifies that arbitrary confidence levels produce finite z-scores.
#[test]
fn test_z_score_arbitrary() {
    let z = IntervalMethod::approximate_z_score(0.87f64).expect("z");

    assert!(z.is_finite(), "Z-score should be finite");
    assert!(z > 0.0, "Z-score should be positive");
}

/// Test very high confidence level to hit Acklam's tail regions.
#[test]
fn test_acklam_tails() {
    let z_999 = IntervalMethod::<f64>::approximate_z_score(0.999).unwrap();
    assert!(z_999 > 3.0, "z_999 was {}", z_999);

    let z_001 = IntervalMethod::<f64>::approximate_z_score(0.001).unwrap();
    assert!(z_001 > 0.0);
}

// ============================================================================
// Confidence Interval Tests
// ============================================================================

/// Test confidence interval computation.
///
/// Verifies that CI is computed correctly.
#[test]
fn test_confidence_intervals() {
    let y_smooth = vec![10.0f64, 20.0];
    let std_err = vec![1.0f64, 2.0];
    let level = 0.95f64;
    let residuals = vec![0.0f64; 2];

    let estimator = IntervalMethod::confidence(level);
    let (cl, cu, _, _) = estimator
        .compute_intervals(&y_smooth, &std_err, &residuals, None, None)
        .expect("intervals");

    assert!(cl.is_some(), "CI lower should be computed");
    assert!(cu.is_some(), "CI upper should be computed");

    let lower = cl.unwrap();
    let upper = cu.unwrap();

    // Verify intervals contain smoothed values
    for i in 0..y_smooth.len() {
        assert!(
            lower[i] <= y_smooth[i] && y_smooth[i] <= upper[i],
            "CI should contain smoothed value"
        );
    }
}

/// Test prediction interval computation.
///
/// Verifies that PI is computed correctly.
#[test]
fn test_prediction_intervals() {
    let y_smooth = vec![10.0f64, 20.0];
    let std_err = vec![1.0f64, 2.0];
    let level = 0.95f64;
    let residuals = vec![0.0f64; 2];

    let estimator = IntervalMethod::prediction(level);
    let (_, _, pl, pu) = estimator
        .compute_intervals(&y_smooth, &std_err, &residuals, None, None)
        .expect("intervals");

    assert!(pl.is_some(), "PI lower should be computed");
    assert!(pu.is_some(), "PI upper should be computed");
}

/// Test that PI is wider than CI.
///
/// Verifies that prediction intervals include residual variance.
#[test]
fn test_pi_wider_than_ci() {
    let y_smooth = vec![10.0f64, 20.0];
    let std_err = vec![1.0f64, 2.0];
    let level = 0.95f64;
    let residuals_noisy = vec![3.0, -3.0]; // Non-zero residuals

    let estimator_ci = IntervalMethod::confidence(level);
    let (cl, cu, _, _) = estimator_ci
        .compute_intervals(&y_smooth, &std_err, &residuals_noisy, None, None)
        .expect("CI");

    let estimator_pi = IntervalMethod::prediction(level);
    let (_, _, pl, pu) = estimator_pi
        .compute_intervals(&y_smooth, &std_err, &residuals_noisy, None, None)
        .expect("PI");

    let w_ci = cu.unwrap()[0] - cl.unwrap()[0];
    let w_pi = pu.unwrap()[0] - pl.unwrap()[0];

    assert!(w_pi > w_ci, "PI should be wider than CI");
}

// ============================================================================
// Integration Tests
// ============================================================================

/// Test complete interval method workflow.
///
/// Verifies SE computation and interval calculation together.
#[test]
fn test_interval_method_workflow() {
    let x = vec![0.0f64, 1.0, 2.0];
    let y = vec![0.0f64, 1.0, 0.0];
    let y_smooth = vec![0.0f64, 0.0, 0.0];
    let robustness = vec![1.0f64; 3];
    let level = 0.95f64;

    let estimator = IntervalMethod::confidence(level);

    // Compute SE
    let mut std_errors = vec![0.0f64; 3];
    estimator.compute_window_se(
        &x,
        &y,
        &y_smooth,
        3,
        &robustness,
        &mut std_errors,
        &uniform_weight_fn,
    );

    let expected_se_mid = (1.0f64 / 3.0f64).sqrt();
    assert_relative_eq!(std_errors[1], expected_se_mid, epsilon = 1e-12);

    // Compute intervals
    let residuals = vec![0.0f64; 3];
    let (ci_lower, ci_upper, _, _) = estimator
        .compute_intervals(&y_smooth, &std_errors, &residuals, None, None)
        .expect("Intervals");

    // Verify lengths
    assert_eq!(ci_lower.unwrap().len(), 3, "CI lower should have 3 values");
    assert_eq!(ci_upper.unwrap().len(), 3, "CI upper should have 3 values");

    // Verify level
    assert_relative_eq!(estimator.level, 0.95, epsilon = 1e-12);
}

/// Test residual SD with minimum required points.
#[test]
fn test_interval_edge_cases() {
    let loess_rs = Loess::<f64>::new()
        .fraction(1.0)
        .confidence_intervals(0.95)
        .adapter(Batch)
        .build()
        .unwrap();

    let x = [1.0, 2.0];
    let y = [2.0, 4.0];
    let result = loess_rs.fit(&x, &y).unwrap();
    assert!(result.standard_errors.is_some());
}

// ============================================================================
// Validation Tests
// ============================================================================

/// Test interval level validation.
///
/// Verifies that validator correctly checks interval levels.
#[test]
fn test_validate_level() {
    assert!(
        Validator::validate_interval_level(0.95).is_ok(),
        "Valid level should pass"
    );

    match Validator::validate_interval_level(1.5) {
        Err(LoessError::InvalidIntervals(v)) => {
            assert!(
                (v - 1.5).abs() < 1e-10,
                "Error should contain invalid value"
            )
        }
        _ => panic!("Expected InvalidIntervals error"),
    }

    match Validator::validate_interval_level(-0.1) {
        Err(LoessError::InvalidIntervals(v)) => {
            assert!(
                (v + 0.1).abs() < 1e-10,
                "Error should contain invalid value"
            )
        }
        _ => panic!("Expected InvalidIntervals error"),
    }

    match Validator::validate_interval_level(f64::NAN) {
        Err(LoessError::InvalidIntervals(v)) => {
            assert!(v.is_nan(), "Error should contain NaN")
        }
        _ => panic!("Expected InvalidIntervals error"),
    }
}

/// Test interval method constructors.
///
/// Verifies that different constructors produce correct configurations.
#[test]
fn test_interval_method_constructors() {
    let ci = IntervalMethod::confidence(0.95);
    assert_relative_eq!(ci.level, 0.95, epsilon = 1e-12);

    let pi = IntervalMethod::prediction(0.99);
    assert_relative_eq!(pi.level, 0.99, epsilon = 1e-12);

    let se: IntervalMethod<f64> = IntervalMethod::se();
    // SE method should have default level
    assert!(
        se.level > 0.0 && se.level < 1.0,
        "SE should have valid level"
    );

    // Test Default trait
    let d = IntervalMethod::<f64>::default();
    assert!(!d.confidence);
}

// ============================================================================
// Internal Interval Edge Cases
// ============================================================================

/// Test residual SD with edge point counts (n=0, n=1).
#[test]
fn test_residual_sd_edge_points() {
    // Internal method access via IntervalMethod
    // We can't call private methods directly from integration tests unless we use internals re-export
    // Let's check if compute_intervals handles them correctly or if we can test calculate_residual_sd

    // Actually compute_intervals uses calculate_residual_sd internally.
    let ys = vec![10.0f64];
    let ses = vec![0.1f64];
    let residuals = vec![0.0f64];

    let estimator = IntervalMethod::prediction(0.95);
    let result = estimator.compute_intervals(&ys, &ses, &residuals, None, None);
    assert!(result.is_ok());
}

/// Test Z-score with extremely high precision.
#[test]
fn test_z_score_high_precision() {
    // Test very close to 1.0
    let z_extreme = IntervalMethod::<f64>::approximate_z_score(0.9999).unwrap();
    assert!(z_extreme > 3.8); // z for 0.9999 is ~3.89

    // Fast path for 0.95
    let z_95 = IntervalMethod::<f64>::approximate_z_score(0.95).unwrap();
    assert_relative_eq!(z_95, 1.96, epsilon = 1e-6);
}

/// Test intervals when standard error is zero.
#[test]
fn test_intervals_degenerate_se() {
    let ys = vec![10.0];
    let ses = vec![0.0]; // Zero SE
    let residuals = vec![1.0]; // Non-zero residual

    let estimator = IntervalMethod::confidence(0.95);
    let (cl, cu, _, _) = estimator
        .compute_intervals(&ys, &ses, &residuals, None, None)
        .unwrap();

    let clv = cl.unwrap();
    let cuv = cu.unwrap();

    // Width should be clamped to EPS
    assert!(cuv[0] > clv[0]);
    assert_relative_eq!(cuv[0] - clv[0], 1e-12, epsilon = 1e-15);
}

// ============================================================================
// T-Score and Delta-DF Tests
// ============================================================================

/// Test approximate_t_score for various degrees of freedom.
///
/// Verifies that the t-score is always >= z-score and converges for large df.
#[test]
fn test_t_score_various_df() {
    // For large df, t ≈ z
    let z_95 = IntervalMethod::<f64>::approximate_z_score(0.95).unwrap();
    let t_large = IntervalMethod::<f64>::approximate_t_score(0.95, 1000.0).unwrap();
    assert_relative_eq!(t_large, z_95, epsilon = 0.01);

    // For small df > 2, t > z
    let t_small = IntervalMethod::<f64>::approximate_t_score(0.95, 5.0).unwrap();
    assert!(t_small > z_95, "t-score should exceed z-score for df=5");

    // For df <= 2, fallback branch: t = z * 1.5
    let t_degenerate = IntervalMethod::<f64>::approximate_t_score(0.95, 1.0).unwrap();
    assert_relative_eq!(t_degenerate, z_95 * 1.5, epsilon = 1e-10);
}

/// Test t_score for f32 type.
#[test]
fn test_t_score_f32() {
    let t = IntervalMethod::<f32>::approximate_t_score(0.95, 10.0).unwrap();
    assert!(t > 1.5 && t < 5.0, "t-score should be in valid range");
}

/// Test compute_intervals activates t-score path when delta1 and delta2 are given.
///
/// Verifies that the `compute_confidence_intervals_impl` and
/// `compute_prediction_intervals_impl` functions are exercised with df != None.
#[test]
fn test_compute_intervals_with_df() {
    let ys = vec![5.0f64, 6.0, 7.0];
    let ses = vec![0.5f64, 0.5, 0.5];
    let residuals = vec![0.1f64, -0.1, 0.2];

    let estimator = IntervalMethod::confidence(0.95);

    // Provide delta1 and delta2 so df = delta1^2 / delta2 is computed (> 2)
    let delta1 = Some(10.0f64);
    let delta2 = Some(3.0f64); // df = 100 / 3 ≈ 33.3 → t-score path

    let result = estimator.compute_intervals(&ys, &ses, &residuals, delta1, delta2);
    assert!(
        result.is_ok(),
        "compute_intervals with delta should succeed"
    );
    let (cl, cu, _, _) = result.unwrap();
    let cl = cl.unwrap();
    let cu = cu.unwrap();
    assert_eq!(cl.len(), 3);
    assert_eq!(cu.len(), 3);
    for (l, h) in cl.iter().zip(cu.iter()) {
        assert!(h > l, "upper CI should exceed lower CI");
        assert!(l.is_finite() && h.is_finite());
    }
}

/// Test prediction intervals with df (t-score path in compute_prediction_intervals_impl).
#[test]
fn test_prediction_intervals_with_df() {
    let ys = vec![10.0f64, 12.0, 14.0];
    let ses = vec![1.0f64, 1.0, 1.0];
    let residuals = vec![0.5f64, -0.5, 1.0];

    let estimator = IntervalMethod::prediction(0.90);

    let delta1 = Some(6.0f64);
    let delta2 = Some(2.0f64); // df = 36 / 2 = 18 → t-score path

    let result = estimator.compute_intervals(&ys, &ses, &residuals, delta1, delta2);
    assert!(result.is_ok());
    let (_, _, pl, pu) = result.unwrap();
    let pl = pl.unwrap();
    let pu = pu.unwrap();
    assert_eq!(pl.len(), 3);
    assert_eq!(pu.len(), 3);
    for (l, h) in pl.iter().zip(pu.iter()) {
        assert!(h > l, "upper PI should exceed lower PI");
        assert!(l.is_finite() && h.is_finite());
    }
}

/// Test that Default trait creates an interval method with no CI or PI.
///
/// The Default impl delegates to the private none() constructor, so this
/// exercises that code path.
#[test]
fn test_interval_method_default_is_none() {
    let m = IntervalMethod::<f64>::default();
    assert!(!m.confidence, "default() should have confidence=false");
    assert!(!m.prediction, "default() should have prediction=false");

    // compute_intervals should return all-None
    let ys = vec![1.0f64, 2.0, 3.0];
    let ses = vec![0.1f64, 0.1, 0.1];
    let residuals = vec![0.01f64, 0.01, 0.01];

    let (cl, cu, pl, pu) = m
        .compute_intervals(&ys, &ses, &residuals, None, None)
        .unwrap();
    assert!(cl.is_none(), "CI lower should be None for default()");
    assert!(cu.is_none(), "CI upper should be None for default()");
    assert!(pl.is_none(), "PI lower should be None for default()");
    assert!(pu.is_none(), "PI upper should be None for default()");
}

/// Test calculate_residual_sd function via various compute_intervals paths.
///
/// Verifies the residual SD is correctly computed from residuals and delta1.
#[test]
fn test_residual_sd_via_prediction_intervals() {
    // With known residuals we can check the PI width is > CI width
    let ys = vec![0.0f64, 0.0, 0.0];
    let ses = vec![0.2f64, 0.2, 0.2];
    let residuals = vec![1.0f64, 2.0, 3.0]; // spread → larger PI

    let ci_estimator = IntervalMethod::confidence(0.95);
    let pi_estimator = IntervalMethod::prediction(0.95);

    let (ccl, ccu, _, _) = ci_estimator
        .compute_intervals(&ys, &ses, &residuals, None, None)
        .unwrap();
    let (_, _, pcl, pcu) = pi_estimator
        .compute_intervals(&ys, &ses, &residuals, None, None)
        .unwrap();

    let ci_width = ccu.unwrap()[0] - ccl.unwrap()[0];
    let pi_width = pcu.unwrap()[0] - pcl.unwrap()[0];

    assert!(
        pi_width > ci_width,
        "Prediction interval should be wider than confidence interval"
    );
}
