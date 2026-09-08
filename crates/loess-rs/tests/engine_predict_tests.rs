#![cfg(feature = "dev")]
//! Tests for out-of-sample prediction on the Batch adapter (`LoessResult::predict()`).
//!
//! ## Test Organization
//!
//! 1. **Retain-Model Gating** - `PredictionUnavailable` when not retained
//! 2. **1D Linear Reproduction** - exact/near-exact recovery on linear data
//! 3. **Input Validation** - `new_x` length must be a multiple of `dimensions`
//! 4. **Multivariate (nD)** - basic sanity check with `dimensions = 2`
//! 5. **Polynomial Degree** - predict works with a non-default degree
//! 6. **Standard Errors & Intervals** - SE, confidence intervals, prediction intervals
//! 7. **Derivative** - local fit gradient at a query point
//! 8. **Extrapolation Policy** - Clamp, Linear, Error

use approx::assert_relative_eq;

use loess_rs::internals::engine::predict::{ExtrapolationPolicy, PredictOptions};
use loess_rs::internals::primitives::errors::LoessError;
use loess_rs::prelude::*;

fn linear_series(n: usize, slope: f64, intercept: f64) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| slope * xi + intercept).collect();
    (x, y)
}

// ============================================================================
// Retain-Model Gating
// ============================================================================

#[test]
fn test_predict_without_retain_model_errors() {
    let (x, y) = linear_series(20, 2.0, 1.0);
    let result = Loess::new()
        .fraction(0.5)
        .iterations(0)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let err = result
        .predict(&[5.5], &PredictOptions::default())
        .unwrap_err();
    assert!(matches!(err, LoessError::PredictionUnavailable));
}

#[test]
fn test_predict_after_cross_validation() {
    // `.retain_model(true)` combined with CV bandwidth selection: only the final fit (at
    // the CV-selected best fraction) should retain model state; candidate fold/fraction
    // fits must not pay for building it (see engine::executor::run_with_config).
    let (x, y) = linear_series(40, 2.0, 1.0);
    let result = Loess::new()
        .iterations(0)
        .cv_method("kfold")
        .cv_k(3)
        .cv_fractions(vec![0.3, 0.5, 0.7])
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert!(result.has_cv_scores());
    let output = result
        .predict(&[10.5], &PredictOptions::default())
        .expect("predict should succeed after CV selected the best fraction");
    assert_relative_eq!(output.y[0], 2.0 * 10.5 + 1.0, epsilon = 1e-1);
}

// ============================================================================
// 1D Linear Reproduction
// ============================================================================

#[test]
fn test_predict_reproduces_linear_data() {
    let (x, y) = linear_series(50, 2.0, 1.0);
    let result = Loess::new()
        .fraction(0.5)
        .iterations(0)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Out-of-sample query points strictly inside the training range.
    let new_x = vec![10.5, 20.25, 30.75];
    let output = result
        .predict(&new_x, &PredictOptions::default())
        .expect("predict should succeed");

    assert_eq!(output.y.len(), new_x.len());
    for (&q, &p) in new_x.iter().zip(output.y.iter()) {
        let expected = 2.0 * q + 1.0;
        // Local WLS solves on raw (uncentered) x-values are mildly ill-conditioned,
        // so allow a small numerical tolerance rather than requiring exact recovery.
        assert_relative_eq!(p, expected, epsilon = 1e-2);
    }
    assert!(output.standard_errors.is_none());
    assert!(output.derivative.is_none());
}

#[test]
fn test_predict_at_training_points_matches_fit() {
    let (x, y) = linear_series(40, 3.0, -2.0);
    let result = Loess::new()
        .fraction(0.6)
        .iterations(0)
        .surface_mode("direct")
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Predicting exactly at the training x-values should closely match `fit()`'s
    // own smoothed values (same local WLS fit, evaluated at the same points).
    // Uses Direct surface mode so `fit()` itself computes a per-point local fit
    // rather than an interpolated surface (the default), matching `predict()`'s
    // own always-direct evaluation.
    let output = result
        .predict(&x, &PredictOptions::default())
        .expect("predict should succeed");
    for (&fitted, &pred) in result.y.iter().zip(output.y.iter()) {
        assert_relative_eq!(fitted, pred, epsilon = 1e-4);
    }
}

// ============================================================================
// Input Validation
// ============================================================================

#[test]
fn test_predict_rejects_mismatched_new_x_length() {
    let x: Vec<f64> = (0..40).flat_map(|i| [i as f64, (i * 2) as f64]).collect();
    let y: Vec<f64> = (0..40).map(|i| i as f64).collect();

    let result = Loess::new()
        .fraction(0.5)
        .iterations(0)
        .dimensions(2)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // 3 values isn't a multiple of dimensions=2.
    let err = result
        .predict(&[1.0, 2.0, 3.0], &PredictOptions::default())
        .unwrap_err();
    assert!(matches!(err, LoessError::InvalidInput(_)));
}

// ============================================================================
// Multivariate (nD)
// ============================================================================

#[test]
fn test_predict_multivariate() {
    // z = x + y (dimensions = 2), fit and predict at a held-out point.
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..10 {
        for j in 0..10 {
            x.push(i as f64);
            x.push(j as f64);
            y.push((i + j) as f64);
        }
    }

    let result = Loess::new()
        .fraction(0.5)
        .iterations(0)
        .dimensions(2)
        .degree("linear")
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let output = result
        .predict(&[4.5, 4.5], &PredictOptions::default())
        .expect("predict should succeed");
    assert_eq!(output.y.len(), 1);
    assert_relative_eq!(output.y[0], 9.0, epsilon = 1.0);
}

// ============================================================================
// Polynomial Degree
// ============================================================================

#[test]
fn test_predict_with_quadratic_degree() {
    let x: Vec<f64> = (0..40).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();

    let result = Loess::new()
        .fraction(0.7)
        .iterations(0)
        .degree("quadratic")
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let output = result
        .predict(&[15.5], &PredictOptions::default())
        .expect("predict should succeed");
    assert_relative_eq!(output.y[0], 15.5 * 15.5, epsilon = 2.0);
}

// ============================================================================
// Standard Errors & Intervals
// ============================================================================

#[test]
fn test_predict_standard_errors() {
    let (x, y) = linear_series(50, 2.0, 1.0);
    let result = Loess::new()
        .fraction(0.5)
        .iterations(0)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let options = PredictOptions {
        return_se: true,
        ..PredictOptions::default()
    };
    let output = result
        .predict(&[10.0, 20.0], &options)
        .expect("predict should succeed");

    let se = output.standard_errors.expect("standard errors requested");
    assert_eq!(se.len(), 2);
    for &s in &se {
        assert!(s.is_finite() && s >= 0.0);
    }
}

#[test]
fn test_predict_confidence_and_prediction_intervals() {
    let (x, y) = linear_series(50, 2.0, 1.0);
    let result = Loess::new()
        .fraction(0.5)
        .iterations(0)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let options = PredictOptions {
        confidence_level: Some(0.95),
        prediction_level: Some(0.95),
        ..PredictOptions::default()
    };
    let output = result
        .predict(&[10.0, 20.0], &options)
        .expect("predict should succeed");

    let cl = output.confidence_lower.expect("confidence lower");
    let cu = output.confidence_upper.expect("confidence upper");
    let pl = output.prediction_lower.expect("prediction lower");
    let pu = output.prediction_upper.expect("prediction upper");

    for i in 0..2 {
        assert!(cl[i] <= output.y[i] && output.y[i] <= cu[i]);
        assert!(pl[i] <= output.y[i] && output.y[i] <= pu[i]);
        // Prediction intervals must be at least as wide as confidence intervals
        // (they add residual variance on top of the mean's standard error).
        assert!(pu[i] - pl[i] >= cu[i] - cl[i]);
    }
}

// ============================================================================
// Derivative
// ============================================================================

#[test]
fn test_predict_derivative_matches_slope() {
    let (x, y) = linear_series(50, 2.0, 1.0);
    let result = Loess::new()
        .fraction(0.5)
        .iterations(0)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let options = PredictOptions {
        return_derivative: true,
        ..PredictOptions::default()
    };
    let output = result
        .predict(&[25.0], &options)
        .expect("predict should succeed");

    // dimensions = 1, so the derivative is a single-element gradient per query point.
    let derivative = output.derivative.expect("derivative requested");
    assert_eq!(derivative.len(), 1);
    assert_relative_eq!(derivative[0], 2.0, epsilon = 1e-2);
}

// ============================================================================
// Extrapolation Policy
// ============================================================================

#[test]
fn test_predict_extrapolation_clamp_default() {
    let (x, y) = linear_series(30, 2.0, 1.0);
    let result = Loess::new()
        .fraction(0.5)
        .iterations(0)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Default (Clamp) should not error on an out-of-range query.
    let output = result
        .predict(&[1000.0], &PredictOptions::default())
        .expect("clamp should not error");
    assert!(output.y[0].is_finite());
}

#[test]
fn test_predict_extrapolation_error_policy() {
    let (x, y) = linear_series(30, 2.0, 1.0);
    let result = Loess::new()
        .fraction(0.5)
        .iterations(0)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let options = PredictOptions {
        extrapolation: ExtrapolationPolicy::Error,
        ..PredictOptions::default()
    };
    let err = result.predict(&[1000.0], &options).unwrap_err();
    assert!(matches!(err, LoessError::PredictOutOfRange { .. }));
}

#[test]
fn test_predict_extrapolation_linear_policy() {
    // Use noboundary to avoid BoundaryPolicy::Extend's flat-padding biasing the
    // boundary-region slope estimate toward zero, which would confound this test.
    let (x, y) = linear_series(30, 2.0, 1.0);
    let result = Loess::new()
        .fraction(0.5)
        .iterations(0)
        .boundary_policy("noboundary")
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let options = PredictOptions {
        extrapolation: ExtrapolationPolicy::Linear,
        ..PredictOptions::default()
    };
    let output = result
        .predict(&[35.0], &options)
        .expect("linear extrapolation should not error");

    let expected = 2.0 * 35.0 + 1.0;
    assert_relative_eq!(output.y[0], expected, epsilon = 1.0);
}

// ============================================================================
// Repeated Calls (cached KD-tree)
// ============================================================================

#[test]
fn test_predict_repeated_calls_are_consistent() {
    // `PredictState`'s KD-tree is built once at `retain_model` time and reused by every
    // `predict()` call; repeated calls (including single-point ones) must keep returning
    // identical results rather than drifting due to any rebuild-related state issues.
    let (x, y) = linear_series(50, 2.0, 1.0);
    let result = Loess::new()
        .fraction(0.5)
        .iterations(0)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let first = result.predict(&[10.5], &PredictOptions::default()).unwrap();
    for _ in 0..5 {
        let again = result.predict(&[10.5], &PredictOptions::default()).unwrap();
        assert_relative_eq!(again.y[0], first.y[0], epsilon = 1e-12);
    }
}
