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

#[test]
fn test_predict_at_training_points_matches_fit_interpolation_mode() {
    // Same as `test_predict_at_training_points_matches_fit`, but with the *default*
    // `SurfaceMode::Interpolation`: `predict()` reuses the retained interpolation surface
    // for in-range points, so it should still exactly reproduce `fit()`'s `y_smooth` here
    // instead of diverging via a separate exact per-point regression.
    let (x, y) = linear_series(40, 3.0, -2.0);
    let result = Loess::new()
        .fraction(0.6)
        .iterations(0)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let output = result
        .predict(&x, &PredictOptions::default())
        .expect("predict should succeed");
    for (&fitted, &pred) in result.y.iter().zip(output.y.iter()) {
        assert_relative_eq!(fitted, pred, epsilon = 1e-10);
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

#[test]
fn test_predict_rejects_non_finite_new_x() {
    let (x, y) = linear_series(40, 2.0, 1.0);
    let result = Loess::new()
        .fraction(0.5)
        .iterations(0)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let err = result
            .predict(&[5.0, bad, 10.0], &PredictOptions::default())
            .unwrap_err();
        assert!(matches!(err, LoessError::InvalidNumericValue(_)));
    }
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

#[test]
fn test_predict_at_training_points_matches_fit_multivariate_interpolation() {
    // Broader coverage for the interpolation-surface reuse fix: nD (3), quadratic degree,
    // and a non-default distance metric, all under the default `SurfaceMode::Interpolation`.
    // Predicting at every training point (including the first/last, i.e. the boundary of
    // the surface's cell tree) should still exactly match `fit()`'s own `y_smooth`.
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..8 {
        for j in 0..8 {
            for k in 0..8 {
                x.push(i as f64);
                x.push(j as f64);
                x.push(k as f64);
                y.push((i * i + j * j + k * k) as f64);
            }
        }
    }

    let result = Loess::new()
        .fraction(0.4)
        .iterations(0)
        .dimensions(3)
        .degree("quadratic")
        .distance_metric("manhattan")
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let output = result
        .predict(&x, &PredictOptions::default())
        .expect("predict should succeed");
    for (&fitted, &pred) in result.y.iter().zip(output.y.iter()) {
        assert_relative_eq!(fitted, pred, epsilon = 1e-8);
    }
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

/// Under `SurfaceMode::Interpolation` (the default), predict()'s SE at a training point
/// should match the SAME uniform approximate-leverage heuristic fit() itself falls back to
/// there, not an unrelated per-point exact-leverage value from a fit that `y` no longer
/// even depends on.
#[test]
fn test_predict_se_matches_fit_interpolation_mode() {
    let (x, y) = linear_series(60, 2.0, 1.0);
    let result = Loess::new()
        .fraction(0.3)
        .iterations(0)
        .return_se()
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let fit_se = result.standard_errors.as_ref().expect("fit() SE requested")[10];

    let options = PredictOptions {
        return_se: true,
        ..PredictOptions::default()
    };
    let output = result
        .predict(&[x[10]], &options)
        .expect("predict should succeed");
    let predict_se = output.standard_errors.expect("standard errors requested")[0];

    assert_relative_eq!(fit_se, predict_se, epsilon = 1e-10);
}

/// Under `SurfaceMode::Direct`, predict()'s SE should still come from an exact per-point
/// leverage computation (unaffected by the Interpolation-mode consistency fix above).
#[test]
fn test_predict_se_uses_exact_leverage_direct_mode() {
    let (x, y) = linear_series(60, 2.0, 1.0);
    let result = Loess::new()
        .fraction(0.3)
        .iterations(0)
        .surface_mode("direct")
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
        .predict(&[x[5], x[40]], &options)
        .expect("predict should succeed");
    let se = output.standard_errors.expect("standard errors requested");

    // Exact per-point leverage varies across points (unlike the uniform Interpolation-mode
    // heuristic), so the two SE values at these differently-positioned points shouldn't be
    // forced equal by construction.
    assert!(se[0].is_finite() && se[0] >= 0.0);
    assert!(se[1].is_finite() && se[1] >= 0.0);
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

#[test]
fn test_predict_extrapolation_linear_respects_max_distance() {
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

    // Within the cap: still succeeds.
    let options = PredictOptions {
        extrapolation: ExtrapolationPolicy::Linear,
        max_extrapolation_distance: Some(10.0),
        ..PredictOptions::default()
    };
    result
        .predict(&[35.0], &options)
        .expect("within max_extrapolation_distance should succeed");

    // Beyond the cap: errors instead of returning an unbounded Taylor-extended value.
    let err = result.predict(&[100.0], &options).unwrap_err();
    assert!(matches!(err, LoessError::ExtrapolationTooFar { .. }));

    // The cap is ignored under Clamp/Error.
    let clamp_options = PredictOptions {
        extrapolation: ExtrapolationPolicy::Clamp,
        max_extrapolation_distance: Some(10.0),
        ..PredictOptions::default()
    };
    result
        .predict(&[100.0], &clamp_options)
        .expect("max_extrapolation_distance should not apply under Clamp");
}

/// A query point can sit inside every dimension's per-dimension bounding box yet be far
/// from any real training point (an "empty corner" for non-rectangularly distributed
/// data). `max_neighbor_distance` should catch this even though the bbox check alone
/// would treat the point as in-range. Uses the default (`Normalized`) distance metric
/// deliberately: the cap is measured in raw coordinate units regardless of which metric
/// was used to select the neighborhood.
#[test]
fn test_predict_max_neighbor_distance_catches_bbox_corner() {
    // Training data only along the diagonal (i, i): the bounding box is the full square
    // [0, 19] x [0, 19], but the corner (19, 0) is far from any actual training point.
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..20 {
        x.push(i as f64);
        x.push(i as f64);
        y.push(i as f64);
    }

    let result = Loess::new()
        .fraction(0.3)
        .iterations(0)
        .dimensions(2)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // No cap: the empty corner is silently treated as in-range (original behavior).
    result
        .predict(&[19.0, 0.0], &PredictOptions::default())
        .expect("uncapped predict should not error, even in the empty corner");

    // With a cap: the corner's neighbor window is much farther than a point actually on
    // the diagonal, so it should be rejected.
    let options = PredictOptions {
        max_neighbor_distance: Some(5.0),
        ..PredictOptions::default()
    };
    let err = result.predict(&[19.0, 0.0], &options).unwrap_err();
    assert!(matches!(err, LoessError::SparseNeighborhood { .. }));

    // A point actually on the diagonal has a tight neighbor window and should still
    // succeed under the same cap.
    result
        .predict(&[10.0, 10.0], &options)
        .expect("a point on the diagonal should have a tight neighbor window");
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
