#![cfg(feature = "dev")]
//! Tests for Median Absolute Deviation (MAD) computation.
//!
//! These tests verify the MAD calculation used in LOESS for:
//! - Robust scale estimation
//! - Outlier detection in robustness iterations
//! - Residual normalization
//!
//! ## Test Organization
//!
//! 1. **Basic Computation** - MAD calculation for various data sizes
//! 2. **Edge Cases** - Empty, single, and special inputs
//! 3. **Statistical Properties** - Correctness and robustness

use approx::assert_relative_eq;

use loess_rs::internals::algorithms::robustness::RobustnessMethod;
use loess_rs::internals::math::scaling::ScalingMethod;

// ============================================================================
// Basic MAD Computation Tests
// ============================================================================

/// Test MAD computation with even-length input.
///
/// Verifies correct median and MAD calculation.
#[test]
fn test_mad_even_length() {
    // Even-length: [1, 2, 3, 4]
    // Median = (2 + 3) / 2 = 2.5
    // Deviations: [1.5, 0.5, 0.5, 1.5]
    // Sorted Deviations: [0.5, 0.5, 1.5, 1.5]
    // MAD = (0.5 + 1.5) / 2 = 1.0
    let mut data = [1.0, 2.0, 3.0, 4.0];
    let mad = ScalingMethod::MAD.compute(&mut data);
    assert_relative_eq!(mad, 1.0);
}

/// Test MAD computation with odd-length input.
#[test]
fn test_mad_odd_length() {
    // Odd-length: [1, 1, 2, 2, 4]
    // Median = 2
    // Deviations: [1, 1, 0, 0, 2]
    // Sorted Deviations: [0, 0, 1, 1, 2]
    // MAD = 1
    let mut data = [1.0, 1.0, 2.0, 2.0, 4.0];
    let mad = ScalingMethod::MAD.compute(&mut data);
    assert_relative_eq!(mad, 1.0);
}

/// Test MAD computation with identical values.
#[test]
fn test_mad_identical_values() {
    let mut data = [5.0, 5.0, 5.0, 5.0, 5.0];
    let mad = ScalingMethod::MAD.compute(&mut data);
    assert_relative_eq!(mad, 0.0);
}

/// Test MAD computation with zeros.
#[test]
fn test_mad_with_zeros() {
    let mut data = [0.0, 0.0, 0.0, 0.0];
    let mad = ScalingMethod::MAD.compute(&mut data);
    assert_relative_eq!(mad, 0.0);
}

/// Test MAD with mixed positive and negative values.
#[test]
fn test_mad_mixed_signs() {
    // Data: [-10, 0, 10]
    // Median: 0
    // Deviations: [10, 0, 10]
    // MAD: 10
    let mut data = [-10.0, 0.0, 10.0];
    let mad = ScalingMethod::MAD.compute(&mut data);
    assert_relative_eq!(mad, 10.0);
}

/// Test MAD with negative values.
#[test]
fn test_mad_negative_values() {
    // Data: [-4, -3, -2, -1]
    // Median: -2.5
    // Deviations: [1.5, 0.5, 0.5, 1.5]
    // MAD: 1.0
    let mut data = [-4.0, -3.0, -2.0, -1.0];
    let mad = ScalingMethod::MAD.compute(&mut data);
    assert_relative_eq!(mad, 1.0);
}

/// Test MAD with symmetrical distribution.
#[test]
fn test_mad_symmetric_distribution() {
    let mut data = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mad = ScalingMethod::MAD.compute(&mut data);
    assert_relative_eq!(mad, 1.0);
}

/// Test MAD with outliers.
#[test]
fn test_mad_with_outliers() {
    // Data with high outlier: [1, 2, 3, 100]
    // Median: 2.5
    // Deviations: [1.5, 0.5, 0.5, 97.5]
    // MAD: 1.0
    let mut data = [1.0, 2.0, 3.0, 100.0];
    let mad = ScalingMethod::MAD.compute(&mut data);
    assert_relative_eq!(mad, 1.0);
}

/// Test MAD scale invariance: MAD(k*X) = |k| * MAD(X).
#[test]
fn test_mad_scale_invariance() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0];
    let data_scaled = [10.0, 20.0, 30.0, 40.0, 50.0];

    let mut data_vec = data.to_vec();
    let mut data_scaled_vec = data_scaled.to_vec();

    let mad = ScalingMethod::MAD.compute(&mut data_vec);
    let mad_scaled = ScalingMethod::MAD.compute(&mut data_scaled_vec);

    assert_relative_eq!(mad_scaled, 10.0 * mad);
}

/// Test MAD with small values.
#[test]
fn test_mad_small_values() {
    let mut data = [1e-10, 2e-10, 3e-10];
    let mad = ScalingMethod::MAD.compute(&mut data);
    assert_relative_eq!(mad, 1e-10);
}

/// Test MAD with large values.
#[test]
fn test_mad_large_values() {
    let mut data = [1e10, 2e10, 3e10];
    let mad = ScalingMethod::MAD.compute(&mut data);
    assert_relative_eq!(mad, 1e10);
}

/// Test MAD with unsorted input.
#[test]
fn test_mad_unsorted_input() {
    let mut data = [5.0, 1.0, 4.0, 2.0, 3.0];
    let mad = ScalingMethod::MAD.compute(&mut data);
    assert_relative_eq!(mad, 1.0);
}

/// Test MAD with two values.
#[test]
fn test_mad_two_values() {
    let mut data = [1.0, 10.0];
    // Median: 5.5
    // Deviations: [4.5, 4.5]
    // MAD: 4.5
    let mad = ScalingMethod::MAD.compute(&mut data);
    assert_relative_eq!(mad, 4.5);
}

// ============================================================================
// Scaling Method Integration Tests
// ============================================================================

/// Test that MAR and MAD produce distinct results for robustness weighting.
///
/// This test verifies that the chosen scaling method correctly affects
/// the downweighting of outliers.
#[test]
fn test_scaling_method_differences_direct() {
    // Residuals where median is far from 0
    // [10.0, 11.0, 12.0, 13.0, 14.0]
    // Median = 12.0
    // MAR = median(|10, 11, 12, 13, 14|) = 12.0
    // MAD = median(|-2, -1, 0, 1, 2|) = 1.0
    let residuals = vec![10.0, 11.0, 12.0, 13.0, 14.0];

    let mut weights_mar = vec![0.0; 5];
    let mut scratch = vec![0.0; 5];

    RobustnessMethod::Bisquare.apply_robustness_weights(
        &residuals,
        &mut weights_mar,
        ScalingMethod::MAR,
        &mut scratch,
    );

    let mut weights_mad = vec![0.0; 5];
    RobustnessMethod::Bisquare.apply_robustness_weights(
        &residuals,
        &mut weights_mad,
        ScalingMethod::MAD,
        &mut scratch,
    );

    // With MAR (scale=12), tuned_scale = 72.
    // u = [10/72, 11/72, 12/72, 13/72, 14/72] all < 1.0, so weights > 0.

    // With MAD (scale=1), tuned_scale = 6.
    // u = [10/6, 11/6, 12/6, 13/6, 14/6] all > 1.0, so weights = 0.

    assert!(weights_mar[0] > 0.0, "MAR weights should be non-zero");
    assert_eq!(
        weights_mad[0], 0.0,
        "MAD weights should be zero for these residuals"
    );
}
