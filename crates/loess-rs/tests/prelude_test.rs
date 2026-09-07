#![cfg(feature = "dev")]
//! Tests for the prelude module.
//!
//! These tests verify that the prelude exports all necessary types and traits
//! for convenient usage of the LOESS API. The prelude should provide a
//! one-stop import for common LOESS functionality.
//!
//! ## Test Organization
//!
//! 1. **Import Verification** - All prelude exports are accessible
//! 2. **Type Usage** - Types can be used without qualification
//! 3. **Builder Pattern** - Complete workflows work with prelude imports

use loess_rs::prelude::*;

// ============================================================================
// Import Verification Tests
// ============================================================================

/// Test that all prelude imports work correctly.
///
/// Verifies that the prelude exports all necessary types for LOESS usage.
#[test]
fn test_prelude_imports() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

    // Verify primary entrypoint and result types are usable
    let result = Loess::new().build().unwrap().fit(&x, &y);

    assert!(result.is_ok(), "Basic fit should work with prelude imports");
}

/// Test RobustnessMethod is available.
///
/// Verifies that RobustnessMethod enum is exported.
#[test]
fn test_prelude_robustness_method() {
    let _ = Loess::<f64>::new().robustness_method("bisquare");
    let _ = Loess::<f64>::new().robustness_method("huber");
    let _ = Loess::<f64>::new().robustness_method("talwar");
}

/// Test WeightFunction is available.
///
/// Verifies that WeightFunction enum is exported.
#[test]
fn test_prelude_weight_function() {
    let _ = Loess::<f64>::new().weight_function("tricube");
    let _ = Loess::<f64>::new().weight_function("epanechnikov");
    let _ = Loess::<f64>::new().weight_function("gaussian");
    let _ = Loess::<f64>::new().weight_function("biweight");
}

/// Test CrossValidationStrategy is available.
///
/// Verifies string-based cross-validation builder API is available.
#[test]
fn test_prelude_cross_validation() {
    let _ = Loess::<f64>::new()
        .cv_method("kfold")
        .cv_k(5)
        .cv_fractions(vec![0.5]);
    let _ = Loess::<f64>::new()
        .cv_method("loocv")
        .cv_fractions(vec![0.5]);
}

/// Test ZeroWeightFallback is available.
///
/// Verifies that ZeroWeightFallback enum is exported.
#[test]
fn test_prelude_zero_weight_fallback() {
    let _ = Loess::<f64>::new().zero_weight_fallback("use_local_mean");
    let _ = Loess::<f64>::new().zero_weight_fallback("return_original");
    let _ = Loess::<f64>::new().zero_weight_fallback("return_none");
}

/// Test mode entry points are available.
///
/// Verifies that Loess/StreamingLoess/OnlineLoess constructors are exported.
#[test]
fn test_prelude_adapters() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![2.0, 4.0, 6.0];

    // Batch mode
    let _ = Loess::<f64>::new().build().unwrap().fit(&x, &y);

    // Streaming mode
    let _ = StreamingLoess::<f64>::new().build();

    // Online mode
    let _ = OnlineLoess::<f64>::new().build();
}

/// Test complete workflow with prelude.
///
/// Verifies that a complete LOESS workflow works with only prelude imports.
#[test]
fn test_prelude_complete_workflow() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0];

    let result = Loess::<f64>::new()
        .fraction(0.5)
        .iterations(3)
        .robustness_method("bisquare")
        .weight_function("tricube")
        .confidence_intervals(0.95)
        .return_diagnostics()
        .return_residuals()
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("Complete workflow should succeed");

    // Verify all requested outputs are present
    assert_eq!(result.y.len(), x.len());
    assert!(result.has_confidence_intervals());
    assert!(result.diagnostics.is_some());
    assert!(result.residuals.is_some());
}

/// Test error types are available.
///
/// Verifies that error handling works with prelude imports.
#[test]
fn test_prelude_error_handling() {
    let x: Vec<f64> = vec![];
    let y: Vec<f64> = vec![];

    let result = Loess::<f64>::new().build().unwrap().fit(&x, &y);

    // Should be able to match on error types from prelude
    assert!(result.is_err());
}
