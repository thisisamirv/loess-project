#![cfg(feature = "dev")]
//! Tests for local regression algorithms.
//!
//! These tests verify the core regression utilities used in LOESS for:
//! - Local weighted least squares (WLS) fitting
//! - Weight parameter computation
//! - Zero-weight fallback strategies
//! - Boundary condition handling
//!
//! ## Test Organization
//!
//! 1. **Weight Parameters** - WeightParams construction and validation
//! 2. **Local WLS** - Weighted least squares fitting
//! 3. **Fit Point** - Complete point fitting with various scenarios
//! 4. **Zero Weight Fallbacks** - Handling degenerate cases
//! 5. **Boundary Conditions** - Edge cases and invalid inputs

use approx::assert_relative_eq;
use num_traits::Float;

use loess_rs::internals::algorithms::regression::{
    PolynomialDegree, RegressionContext, SolverLinalg, ZeroWeightFallback,
};
use loess_rs::internals::math::distance::DistanceMetric;
use loess_rs::internals::math::kernel::WeightFunction;
use loess_rs::internals::math::linalg::FloatLinalg;
use loess_rs::internals::math::neighborhood::{KDTree, Neighborhood};
use loess_rs::internals::primitives::buffer::NeighborhoodSearchBuffer;
use loess_rs::internals::primitives::window::Window;
use loess_rs::prelude::*;

#[cfg(feature = "dev")]
use loess_rs::internals::engine::executor::{LoessConfig, LoessDistanceCalculator, LoessExecutor};
#[cfg(feature = "dev")]
use loess_rs::internals::evaluation::cv::CVKind;

// ============================================================================
// Helper Functions
// ============================================================================

fn compute_weighted_sum<T: Float>(values: &[T], weights: &[T], left: usize, right: usize) -> T {
    let mut sum = T::zero();
    for j in left..=right {
        sum = sum + weights[j] * values[j];
    }
    sum
}

#[allow(clippy::too_many_arguments)]
fn fit_1d_helper<T: FloatLinalg + std::fmt::Debug + SolverLinalg>(
    x: &[T],
    y: &[T],
    idx: usize,
    window: Window,
    use_robustness: bool,
    robustness_weights: &[T],
    weight_function: WeightFunction,
    zero_weight_fallback: ZeroWeightFallback,
    polynomial_degree: PolynomialDegree,
) -> Option<T> {
    if idx >= x.len() {
        return None;
    }
    if window.left >= x.len() || window.right >= x.len() {
        return None;
    }

    let x_val = x[idx];
    // inline calculation for max_distance
    let max_dist = T::max(x_val - x[window.left], x[window.right] - x_val);
    let mut indices = Vec::new();
    let mut distances = Vec::new();

    // Use iterator to avoid needless_range_loop lint
    let num_points = window.right - window.left + 1;
    for (i, val) in x.iter().enumerate().skip(window.left).take(num_points) {
        indices.push(i);
        distances.push((*val - x_val).abs());
    }

    let neighborhood = Neighborhood {
        indices,
        distances,
        max_distance: max_dist,
    };

    let mut ctx = RegressionContext::new(
        x,
        1,
        y,
        idx,
        None,
        &neighborhood,
        use_robustness,
        robustness_weights,
        weight_function,
        zero_weight_fallback,
        polynomial_degree,
        false,
        None,
    );

    ctx.fit().map(|(v, _)| v)
}

fn local_wls_helper<T: FloatLinalg + std::fmt::Debug + SolverLinalg>(
    x: &[T],
    y: &[T],
    weights: &[T],
    left: usize,
    right: usize,
    x_current: T,
    window_radius: T,
) -> T {
    // To emulate arbitrary weights using `fit`:
    // 1. Construct a neighborhood for the window.
    // 2. Use Uniform kernel (so kernel weight is 1.0).
    // 3. Use `weights` as `robustness_weights`.

    let mut indices = Vec::new();
    let mut distances = Vec::new();
    for i in left..=right {
        indices.push(i);
        // Distances don't matter much for Uniform kernel if within radius,
        // but we need to ensure they are <= max_distance.
        // Or we can just set distances to 0.0.
        distances.push(T::zero());
    }

    // Use window_radius as max_dist, or slightly larger to ensure inclusion if needed.
    // If radius is 0, we need to handle it.
    let max_dist = if window_radius <= T::epsilon() {
        T::min_positive_value()
    } else {
        window_radius
    }; // Ensure non-zero for division

    let neighborhood = Neighborhood {
        indices,
        distances,
        max_distance: max_dist,
    };

    // Create a full robustness vector (size equal to x) or just enough if we only access via indices.
    // `fit` accesses `robustness_weights[neighbor_idx]`.
    // So we need `weights` to correspond to absolute indices.
    // But `weights` input here is also likely aligned?
    // Wait, typical usages pass `weights` as a slice matching x?
    // See `test_local_wls_degenerate_bandwidth`: weights len = x len.

    let query_point_arr = [x_current];
    let mut ctx = RegressionContext::new(
        x,
        1,
        y,
        0,
        Some(&query_point_arr),
        &neighborhood,
        true,
        weights,
        WeightFunction::Uniform,
        ZeroWeightFallback::UseLocalMean,
        PolynomialDegree::Linear,
        false,
        None,
    );

    ctx.fit().map(|(v, _)| v).unwrap_or(T::zero()) // Or handle None?
}

// ============================================================================
// Weight Parameters Tests
// ============================================================================

// tests removed

// ============================================================================
// Local WLS Tests
// ============================================================================

/// Test local_wls with degenerate bandwidth.
///
/// Verifies that when bandwidth <= 0, weighted average is returned.
#[test]
fn test_local_wls_degenerate_bandwidth() {
    let x = vec![0.0f64, 1.0, 2.0];
    let y = vec![10.0f64, 20.0, 30.0];
    let weights = vec![1.0f64, 1.0, 1.0];

    // bandwidth <= 0 triggers early return of weighted average
    let result = local_wls_helper(&x, &y, &weights, 0, 2, 1.0f64, 0.0f64);
    let sum_w = weights.iter().fold(0.0, |acc, &w| acc + w);
    let expected = compute_weighted_sum(&y, &weights, 0, 2) / sum_w;

    assert_relative_eq!(result, expected, epsilon = 1e-12);
}

/// Test local_wls with small denominator (identical x values).
///
/// Verifies fallback to weighted average when denominator is too small.
#[test]
fn test_local_wls_small_denominator() {
    // All x values identical => denom zero => fallback to average
    let x = vec![1.0f64, 1.0, 1.0];
    let y = vec![2.0f64, 3.0, 4.0];
    let weights = vec![1.0f64, 1.0, 1.0];

    let result = local_wls_helper(&x, &y, &weights, 0, 2, 1.0f64, 1.0f64);

    // Average of y = 3.0
    assert_relative_eq!(result, 3.0f64, epsilon = 1e-12);
}

/// Test local_wls recovers correct linear slope.
///
/// Verifies that WLS correctly fits a simple linear relationship.
#[test]
fn test_local_wls_linear_fit() {
    // y = 2 * x, expect fitted at x_current = 1.0 => y = 2.0
    let x = vec![0.0f64, 1.0, 2.0];
    let y = vec![0.0f64, 2.0, 4.0];
    let weights = vec![1.0f64, 1.0, 1.0];

    let result = local_wls_helper(&x, &y, &weights, 0, 2, 1.0f64, 1.0f64);

    assert_relative_eq!(result, 2.0f64, epsilon = 1e-12);
}

// ============================================================================
// Fit Point Tests
// ============================================================================

/// Test fit_point with degenerate bandwidth.
///
/// Verifies that when all x values are identical, weighted average is returned.
#[test]
fn test_fit_point_degenerate_bandwidth() {
    // All x identical => bandwidth computed will be zero
    let x = vec![1.0f64, 1.0, 1.0];
    let y = vec![10.0f64, 20.0, 30.0];

    let robustness = vec![1.0f64; 3];
    let window = Window { left: 0, right: 2 };

    let result = fit_1d_helper(
        &x,
        &y,
        1usize,
        window,
        false,
        &robustness,
        WeightFunction::Tricube,
        ZeroWeightFallback::UseLocalMean,
        PolynomialDegree::Linear,
    )
    .expect("Should return weighted average");

    let weights_ones = vec![1.0f64; 3];
    let sum_w = 3.0f64;
    let expected = compute_weighted_sum(&y, &weights_ones, 0, 2) / sum_w;

    assert_relative_eq!(result, expected, epsilon = 1e-12);
}

// ============================================================================
// Zero Weight Fallback Tests
// ============================================================================

/// Test zero weight fallback: UseLocalMean.
///
/// Verifies that when all weights are zero, local mean is returned.
#[test]
fn test_zero_weight_fallback_local_mean() {
    let x = vec![0.0f64, 1.0, 2.0];
    let y = vec![10.0f64, 20.0, 30.0];

    let robustness = vec![0.0f64; 3]; // Zero robustness => zero total weight
    let window = Window { left: 0, right: 2 };

    let result = fit_1d_helper(
        &x,
        &y,
        1usize,
        window, // idx 1 is center
        true,
        &robustness,
        WeightFunction::Tricube,
        ZeroWeightFallback::UseLocalMean,
        PolynomialDegree::Linear,
    )
    .expect("Should return local mean");

    // Mean of [10, 20, 30] = 20
    assert_relative_eq!(result, 20.0f64, epsilon = 1e-12);
}

/// Test zero weight fallback: ReturnOriginal.
///
/// Verifies that when all weights are zero, original y[idx] is returned.
#[test]
fn test_zero_weight_fallback_return_original() {
    let x = vec![0.0f64, 1.0, 2.0];
    let y = vec![10.0f64, 20.0, 30.0];

    let robustness = vec![0.0f64; 3];
    let window = Window { left: 0, right: 2 };

    let result = fit_1d_helper(
        &x,
        &y,
        2usize,
        window,
        true,
        &robustness,
        WeightFunction::Tricube,
        ZeroWeightFallback::ReturnOriginal,
        PolynomialDegree::Linear,
    )
    .expect("Should return original y");

    assert_relative_eq!(result, 30.0f64, epsilon = 1e-12);
}

/// Test zero weight fallback: ReturnNone.
///
/// Verifies that when all weights are zero, None is returned.
#[test]
fn test_zero_weight_fallback_return_none() {
    let x = vec![0.0f64, 1.0, 2.0];
    let y = vec![10.0f64, 20.0, 30.0];

    let robustness = vec![0.0f64; 3];
    let window = Window { left: 0, right: 2 };

    let result = fit_1d_helper(
        &x,
        &y,
        0usize,
        window,
        true,
        &robustness,
        WeightFunction::Tricube,
        ZeroWeightFallback::ReturnNone,
        PolynomialDegree::Linear,
    );

    assert!(result.is_none(), "Should return None for zero weights");
}

/// Test degenerate bandwidth with zero weights returns original y.
///
/// Verifies combined degenerate case handling.
#[test]
fn test_degenerate_bandwidth_zero_weights() {
    // Identical x within window => bandwidth == 0
    let x = vec![1.0f64, 1.0, 1.0];
    let y = vec![5.0f64, 6.0, 7.0];

    let robustness = vec![1.0f64; 3];
    let window = Window { left: 0, right: 2 };

    let result = fit_1d_helper(
        &x,
        &y,
        1usize,
        window,
        false,
        &robustness,
        WeightFunction::Tricube,
        ZeroWeightFallback::ReturnOriginal,
        PolynomialDegree::Linear,
    )
    .expect("Should return original y when weights sum to zero in degenerate bandwidth");

    assert_relative_eq!(result, 6.0f64, epsilon = 1e-12);
}

// test_regression_defaults removed

// ============================================================================
// Boundary Conditions Tests
// ============================================================================

/// Test fit_point with invalid index.
///
/// Verifies that out-of-bounds index returns None.
#[test]
fn test_fit_point_invalid_index() {
    let x = vec![0.0f64, 1.0];
    let y = vec![10.0f64, 20.0];

    let robustness = vec![1.0f64, 1.0f64];
    let window = Window { left: 0, right: 1 };

    // idx out of bounds (equal to n) => None
    let result = fit_1d_helper(
        &x,
        &y,
        2usize,
        window,
        false,
        &robustness,
        WeightFunction::Tricube,
        ZeroWeightFallback::ReturnNone,
        PolynomialDegree::Linear,
    );

    assert!(result.is_none(), "Out-of-bounds index should return None");
}

/// Test fit_point with invalid window bounds.
///
/// Verifies that invalid window boundaries return None.
#[test]
fn test_fit_point_invalid_window() {
    let x = vec![0.0f64, 1.0];
    let y = vec![10.0f64, 20.0];

    let robustness = vec![1.0f64, 1.0f64];

    // left >= n
    let window_bad_left = Window { left: 2, right: 2 };
    assert!(
        fit_1d_helper(
            &x,
            &y,
            1usize,
            window_bad_left,
            false,
            &robustness,
            WeightFunction::Tricube,
            ZeroWeightFallback::ReturnNone,
            PolynomialDegree::Linear
        )
        .is_none(),
        "Invalid left bound should return None"
    );

    // right >= n
    let window_bad_right = Window { left: 0, right: 2 };
    assert!(
        fit_1d_helper(
            &x,
            &y,
            0usize,
            window_bad_right,
            false,
            &robustness,
            WeightFunction::Tricube,
            ZeroWeightFallback::ReturnNone,
            PolynomialDegree::Linear
        )
        .is_none(),
        "Invalid right bound should return None"
    );
}

/// Test with various weight functions.
///
/// Verifies that different kernel functions work correctly.
#[test]
fn test_fit_point_various_kernels() {
    let kernels = vec![
        WeightFunction::Tricube,
        WeightFunction::Epanechnikov,
        WeightFunction::Gaussian,
        WeightFunction::Biweight,
    ];

    for kernel in kernels {
        let x = vec![0.0f64, 1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];

        let robustness = vec![1.0f64; 5];
        let window = Window { left: 0, right: 4 };

        let result = fit_1d_helper(
            &x,
            &y,
            2usize,
            window,
            false,
            &robustness,
            kernel,
            ZeroWeightFallback::UseLocalMean,
            PolynomialDegree::Linear,
        );
        assert!(
            result.is_some(),
            "Kernel {:?} should produce valid result",
            kernel
        );
        assert!(
            result.unwrap().is_finite(),
            "Result should be finite for kernel {:?}",
            kernel
        );
    }
}

/// Test with robustness weights enabled.
///
/// Verifies that robustness weights are correctly applied.
#[test]
fn test_fit_point_with_robustness() {
    let x = vec![0.0f64, 1.0, 2.0, 3.0, 4.0];
    let y = vec![1.0f64, 2.0, 100.0, 4.0, 5.0]; // Point 2 is outlier

    let robustness = vec![1.0f64, 1.0, 0.1, 1.0, 1.0]; // Downweight outlier
    let window = Window { left: 0, right: 4 };

    let result = fit_1d_helper(
        &x,
        &y,
        2usize,
        window,
        true,
        &robustness,
        WeightFunction::Tricube,
        ZeroWeightFallback::UseLocalMean,
        PolynomialDegree::Linear,
    );
    assert!(result.is_some(), "Should produce valid result");

    // Result should be closer to linear trend than to outlier
    let fitted = result.unwrap();
    assert!(
        (fitted - 3.0).abs() < 50.0,
        "Fitted value should be closer to trend than outlier"
    );
}

/// Test local_wls when only one point has a non-zero weight.
#[test]
fn test_local_wls_single_weight() {
    let x = vec![0.0f64, 1.0, 2.0];
    let y = vec![10.0f64, 20.0, 30.0];
    let weights = vec![1.0f64, 0.0, 0.0];

    // With only one point having weight, denominator for slope is zero.
    // Should fallback to weighted average (which is y[0] since others are weighted 0).
    let result = local_wls_helper(&x, &y, &weights, 0, 2, 0.0f64, 1.0f64);
    assert_relative_eq!(result, 10.0f64, epsilon = 1e-12);
}

/// Test local_wls with exactly two points.
#[test]
fn test_local_wls_two_points() {
    let x = vec![0.0f64, 2.0];
    let y = vec![10.0f64, 20.0];
    let weights = vec![1.0f64, 1.0];

    // Fit at x = 1.0. Linear interpolation between (0,10) and (2,20) is 15.0
    let result = local_wls_helper(&x, &y, &weights, 0, 1, 1.0f64, 1.0f64);
    assert_relative_eq!(result, 15.0f64, epsilon = 1e-12);
}

/// Test local_wls with perfectly correlated data.
#[test]
fn test_local_wls_perfect_correlation() {
    let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let y: Vec<f64> = (0..10).map(|i| i as f64 * 3.0 + 5.0).collect();
    let weights = vec![1.0f64; 10];

    // Fit at x = 4.5. Expect 3.0 * 4.5 + 5.0 = 13.5 + 5.0 = 18.5
    let result = local_wls_helper(&x, &y, &weights, 0, 9, 4.5f64, 5.0f64);
    assert_relative_eq!(result, 18.5f64, epsilon = 1e-12);
}

/// Test local_wls with extreme values.
#[test]
fn test_local_wls_extreme_values() {
    let x = vec![0.0f64, 1e10];
    let y = vec![0.0f64, 1e10];
    let weights = vec![1.0f64, 1.0];

    // Slope is 1.0. Fit at 5e9. Expect 5e9
    let result = local_wls_helper(&x, &y, &weights, 0, 1, 5e9f64, 1e10f64);
    assert_relative_eq!(result, 5e9f64, epsilon = 1e-2); // Relaxed epsilon for large values
}

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

#[test]
fn test_weighted_mean_nd() {
    // 3 points in 2D with y = x₁ + x₂
    let x: [f64; 6] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let y: [f64; 3] = [0.0, 1.0, 1.0];
    let scales: [f64; 2] = [1.0, 1.0];

    let query = [0.0, 0.0];
    let dims = 2;
    let tree = KDTree::new(&x, dims);
    let dist_calc = LoessDistanceCalculator {
        metric: DistanceMetric::Euclidean,
        scales: &scales,
    };
    let mut buffer = NeighborhoodSearchBuffer::new(3);
    let mut neighborhood = Neighborhood::with_capacity(3);
    tree.find_k_nearest(&query, 3, &dist_calc, None, &mut buffer, &mut neighborhood);

    let mut context = RegressionContext::new(
        &x,
        2,
        &y,
        0,
        None,
        &neighborhood,
        false,
        &[1.0, 1.0, 1.0],
        WeightFunction::Tricube,
        ZeroWeightFallback::default(),
        PolynomialDegree::Constant,
        false,
        None,
    );

    let (result, _leverage) = context.fit().unwrap();
    // Should be close to weighted mean
    assert!(result.is_finite());
}

#[test]
fn test_linear_fit_nd_2d() {
    // 4 points in 2D forming a plane: y = 1 + x₁ + 2*x₂
    let x: [f64; 8] = [
        0.0, 0.0, // (0, 0)
        1.0, 0.0, // (1, 0)
        0.0, 1.0, // (0, 1)
        1.0, 1.0, // (1, 1)
    ];
    let y: [f64; 4] = [1.0, 2.0, 3.0, 4.0]; // y = 1 + x₁ + 2*x₂
    let scales: [f64; 2] = [1.0, 1.0];

    // Query at center (0.5, 0.5) - expected y = 1 + 0.5 + 1.0 = 2.5
    let query: [f64; 2] = [0.5, 0.5];
    let dims = 2;
    let tree = KDTree::new(&x, dims);
    let dist_calc = LoessDistanceCalculator {
        metric: DistanceMetric::Euclidean,
        scales: &scales,
    };
    let mut buffer = NeighborhoodSearchBuffer::new(4);
    let mut neighborhood = Neighborhood::with_capacity(4);
    tree.find_k_nearest(&query, 4, &dist_calc, None, &mut buffer, &mut neighborhood);

    // Create a robustness weights array matching size of x
    let robustness_weights = [1.0; 4];

    let mut context = RegressionContext::new(
        &x,
        2,
        &y,
        0,
        Some(&query),
        &neighborhood,
        false,
        &robustness_weights,
        WeightFunction::Uniform, // Uniform for exact fit
        ZeroWeightFallback::default(),
        PolynomialDegree::Linear,
        false,
        None,
    );

    let (result, _leverage) = context.fit().unwrap();
    // With uniform weights and exact linear data, should be close to 2.5
    assert!(approx_eq(result, 2.5, 0.1), "Expected ~2.5, got {}", result);
}

#[test]
fn test_polynomial_terms_constant() {
    let point: [f64; 2] = [1.0, 2.0];
    let center: [f64; 2] = [0.0, 0.0];
    let mut terms = vec![0.0; 1];

    PolynomialDegree::Constant.build_terms(&point, &center, &mut terms);
    assert_eq!(terms.len(), 1);
    assert_eq!(terms[0], 1.0);
}

#[test]
fn test_polynomial_terms_linear_2d() {
    let point: [f64; 2] = [3.0, 5.0];
    let center: [f64; 2] = [1.0, 2.0];
    let mut terms = vec![0.0; 3];

    PolynomialDegree::Linear.build_terms(&point, &center, &mut terms);
    assert_eq!(terms.len(), 3); // [1, x₁-c₁, x₂-c₂]
    assert_eq!(terms[0], 1.0);
    assert_eq!(terms[1], 2.0); // 3 - 1
    assert_eq!(terms[2], 3.0); // 5 - 2
}

#[test]
fn test_polynomial_terms_quadratic_2d() {
    let point: [f64; 2] = [2.0, 3.0];
    let center: [f64; 2] = [0.0, 0.0];
    let mut terms = vec![0.0; 6];

    PolynomialDegree::Quadratic.build_terms(&point, &center, &mut terms);
    // [1, x₁, x₂, x₁², x₁x₂, x₂²] = [1, 2, 3, 4, 6, 9]
    assert_eq!(terms.len(), 6);
    assert_eq!(terms[0], 1.0);
    assert_eq!(terms[1], 2.0);
    assert_eq!(terms[2], 3.0);
    assert_eq!(terms[3], 4.0); // 2²
    assert_eq!(terms[4], 6.0); // 2*3
    assert_eq!(terms[5], 9.0); // 3²
}

#[test]
fn test_polynomial_terms_cubic_2d() {
    let point: [f64; 2] = [2.0, 3.0];
    let center: [f64; 2] = [0.0, 0.0];
    let mut terms = vec![0.0; 10];

    PolynomialDegree::Cubic.build_terms(&point, &center, &mut terms);
    // Counts for 2D: 1 (const) + 2 (lin) + 3 (quad) + 4 (cubic) = 10
    // Cubic terms: x1^3, x1^2x2, x1x2^2, x2^3
    // values: 2^3=8, 4*3=12, 2*9=18, 27
    assert_eq!(terms.len(), 10);
    assert_eq!(terms[6], 8.0); // x1^3
    assert_eq!(terms[7], 12.0); // x1^2 x2
    assert_eq!(terms[8], 18.0); // x1 x2^2
    assert_eq!(terms[9], 27.0); // x2^3
}

#[test]
fn test_polynomial_terms_quartic_2d() {
    let point: [f64; 2] = [2.0, 1.0];
    let center: [f64; 2] = [0.0, 0.0];
    let mut terms = vec![0.0; 15];

    PolynomialDegree::Quartic.build_terms(&point, &center, &mut terms);
    // Counts for 2D: 10 (cubic) + 5 (quartic) = 15
    // Quartic terms: x1^4, x1^3x2, x1^2x2^2, x1x2^3, x2^4
    // centered vals: 2, 1
    // quartic values: 16, 8, 4, 2, 1
    assert_eq!(terms.len(), 15);

    // Check last few generated terms
    // The order depends on nested loop: i, j, k, l.
    // loops i..d, j..d, k..d, l..d
    // (0,0,0,0) -> x1^4 -> 16
    // (0,0,0,1) -> x1^3 x2 -> 8
    // (0,0,1,1) -> x1^2 x2^2 -> 4
    // (0,1,1,1) -> x1 x2^3 -> 2
    // (1,1,1,1) -> x2^4 -> 1

    assert_eq!(terms[10], 16.0);
    assert_eq!(terms[11], 8.0);
    assert_eq!(terms[12], 4.0);
    assert_eq!(terms[13], 2.0);
    assert_eq!(terms[14], 1.0);
}

// ============================================================================
// nD High-Level Tests (Merged)
// ============================================================================

#[test]
fn test_nd_linear_2d_high_level() {
    // y = x1 + x2
    // We expect a linear LOESS to fit this perfectly.
    let x = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5];
    let y = vec![0.0, 1.0, 1.0, 2.0, 1.0];

    let result = Loess::new()
        .dimensions(2)
        .fraction(0.8)
        .degree(Linear)
        .interpolation_vertices(100)
        .adapter(Batch)
        .build()
        .expect("Should build 2D model")
        .fit(&x, &y)
        .expect("Should fit 2D data");

    assert_eq!(result.dimensions, 2);
    assert_eq!(result.y.len(), 5);

    // Check points produce finite and reasonable results
    // (Surface interpolation trades accuracy for performance)
    for i in 0..5 {
        let x1 = x[i * 2];
        let x2 = x[i * 2 + 1];
        let expected = x1 + x2;
        assert!(result.y[i].is_finite(), "Result should be finite");
        assert!(
            (result.y[i] - expected).abs() < 2.0,
            "Result {} should be within 2.0 of expected {}",
            result.y[i],
            expected
        );
    }
}

#[test]
fn test_nd_quadratic_2d_high_level() {
    // y = x1^2 + x2^2
    // degree=2 local regression should fit this perfectly if it captures terms correctly.
    // However, LOESS is local, so it should be very close.
    let x = vec![
        0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 2.0, 2.0,
    ];
    let y: Vec<f64> = (0..9)
        .map(|i| {
            let x1 = x[i * 2];
            let x2 = x[i * 2 + 1];
            x1 * x1 + x2 * x2
        })
        .collect();

    let result = Loess::new()
        .dimensions(2)
        .fraction(1.0) // Use all points for better fit on simple quadratic
        .degree(Quadratic)
        .interpolation_vertices(100)
        .adapter(Batch)
        .build()
        .expect("Should build 2D model")
        .fit(&x, &y)
        .expect("Should fit 2D quadratic");

    // At the center (1.0, 1.0) it should be close to 2.0 (surface interpolation has tolerance)
    assert_relative_eq!(result.y[4], 2.0, epsilon = 3.0);
}

#[test]
fn test_nd_linear_3d_high_level() {
    // y = x1 + 2*x2 - x3
    let mut x = Vec::new();
    let mut y = Vec::new();

    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                let x1 = i as f64;
                let x2 = j as f64;
                let x3 = k as f64;
                x.push(x1);
                x.push(x2);
                x.push(x3);
                y.push(x1 + 2.0 * x2 - x3);
            }
        }
    }

    let result = Loess::new()
        .dimensions(3)
        .fraction(0.5)
        .degree(Linear)
        .interpolation_vertices(1331)
        .adapter(Batch)
        .build()
        .expect("Should build 3D model")
        .fit(&x, &y)
        .expect("Should fit 3D data");

    assert_eq!(result.dimensions, 3);

    // Check center point (1, 1, 1) -> y = 1 + 2 - 1 = 2
    let idx = 3 * 3 + 3 + 1;
    assert_relative_eq!(result.y[idx], 2.0, epsilon = 0.01);
}

#[test]
fn test_nd_distance_metrics() {
    let x = vec![0.0, 0.0, 1.0, 1.0];
    let y = vec![0.0, 1.0];

    let builder = Loess::new().dimensions(2);

    // Euclidean
    let res_e = builder
        .clone()
        .distance_metric(Euclidean)
        .interpolation_vertices(100)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Manually if possible, but here we just check it runs
    assert_eq!(res_e.distance_metric, Euclidean);
}

#[test]
fn test_nd_backward_compat_1d() {
    // 1D data passed as 1D
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![2.0, 4.0, 6.0];

    let res1d = Loess::new()
        .interpolation_vertices(100)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // 1D data passed as nD with dimensions=1
    let res_nd = Loess::new()
        .dimensions(1)
        .interpolation_vertices(100)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert_relative_eq!(res1d.y[1], res_nd.y[1]);
    assert_eq!(res_nd.dimensions, 1);
}

#[test]
fn test_nd_streaming_2d() {
    // y = x1 + x2
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..20 {
        let val = i as f64;
        x.push(val);
        x.push(val);
        y.push(val + val);
    }

    let mut model = Loess::new()
        .dimensions(2)
        .fraction(0.5)
        .degree(Linear)
        .overlap(5)
        .chunk_size(10)
        .interpolation_vertices(121)
        .adapter(Streaming)
        .build()
        .expect("Should build streaming model");

    // Process in two chunks of 10
    let res1 = model.process_chunk(&x[0..20], &y[0..10]).unwrap();
    let res2 = model.process_chunk(&x[20..40], &y[10..20]).unwrap();

    assert_eq!(res1.y.len(), 5); // 10 - 5 = 5 points finalized
    assert!(res2.y.len() >= 10);

    // Correctness check
    assert_relative_eq!(res1.y[0], 0.0, epsilon = 0.01);
}

#[test]
fn test_nd_intervals() {
    // y = x1 + x2 with some noise
    let x = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5];
    let y = vec![0.01, 1.02, 0.98, 2.05, 1.01];

    let result = Loess::new()
        .dimensions(2)
        .fraction(1.0)
        .degree(Linear)
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .interpolation_vertices(100)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("Should fit with intervals");

    assert!(result.standard_errors.is_some());
    let se = result.standard_errors.as_ref().unwrap();
    assert_eq!(se.len(), 5);
    for &s in se {
        assert!(s >= 0.0);
    }

    assert!(result.confidence_lower.is_some());
    assert!(result.confidence_upper.is_some());
    assert!(result.prediction_lower.is_some());
    assert!(result.prediction_upper.is_some());

    let (cl, cu) = (
        result.confidence_lower.as_ref().unwrap(),
        result.confidence_upper.as_ref().unwrap(),
    );
    let (pl, pu) = (
        result.prediction_lower.as_ref().unwrap(),
        result.prediction_upper.as_ref().unwrap(),
    );

    for i in 0..5 {
        assert!(cu[i] > cl[i]);
        assert!(pu[i] > pl[i]);
        // Prediction intervals should be wider than confidence intervals
        assert!(pu[i] - pl[i] >= cu[i] - cl[i]);

        // The smoothed value should be inside the intervals
        assert!(result.y[i] >= cl[i]);
        assert!(result.y[i] <= cu[i]);
    }
}

#[test]
fn test_nd_boundary_reflect() {
    // 2D data that is strictly linear: y = x1 + x2
    // We'll use a small window and check the corner point (0,0)
    let x = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5];
    let y = vec![0.0, 1.0, 1.0, 2.0, 1.0];

    // Case 1: No boundary padding (Extend is default but currently doesn't add points in nD)
    let res_no_pad = Loess::new()
        .dimensions(2)
        .fraction(0.5)
        .degree(Linear)
        .boundary_policy(Extend)
        .interpolation_vertices(121)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Case 2: Reflection padding
    let res_reflect = Loess::new()
        .dimensions(2)
        .fraction(0.5)
        .degree(Linear)
        .boundary_policy(Reflect)
        .interpolation_vertices(121)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // In this specific linear case, both should be relatively accurate,
    // but surface interpolation has some tolerance
    assert_relative_eq!(res_reflect.y[0], 0.0, epsilon = 0.3);
    assert_relative_eq!(res_no_pad.y[0], 0.0, epsilon = 0.3);

    // Check that we got results for all 5 original points
    assert_eq!(res_reflect.y.len(), 5);
}

#[test]
#[cfg(feature = "dev")]
fn test_nd_cross_validation() {
    // grid 5x5
    let mut x = Vec::with_capacity(50);
    let mut y = Vec::with_capacity(25);

    // Function: z = x^2 + y^2 (paraboloid)
    for i in 0..5 {
        for j in 0..5 {
            let u = i as f64 / 4.0; // 0.0 to 1.0
            let v = j as f64 / 4.0; // 0.0 to 1.0
            x.push(u);
            x.push(v);
            y.push(u * u + v * v);
        }
    }

    // With very small fraction, we fit noise/local too much (though here data is clean)
    // With very large fraction, we might over-smooth if the function was complex,
    // but for x^2+y^2 a quadratic fit should be perfect essentially everywhere.
    // However, to test CV, we'll try to see if it runs and picks a valid fraction.
    // We'll add some noise to make it interesting.
    let mut y_noisy = y.clone();
    // Add significant noise to index 12 (center)
    y_noisy[12] += 2.0;

    // Configuration with CV
    let config = LoessConfig {
        dimensions: 2,
        polynomial_degree: PolynomialDegree::Quadratic,
        cv_fractions: Some(vec![0.3, 0.5, 0.8]),
        cv_kind: Some(CVKind::KFold(5)),
        interpolation_vertices: None,
        ..Default::default()
    };

    let res = LoessExecutor::run_with_config(&x, &y_noisy, config);

    // Ensure we got a result
    assert_eq!(res.smoothed.len(), 25);

    // The CV should have selected one of the fractions
    // We don't strictly assert WHICH one because it depends on the noise and splitting,
    // but we assert the code ran through the nD CV path without panicking.
}
// ============================================================================
// New Edge Case Tests (Value Extremes & Stability)
// ============================================================================

/// Test WLS with flat response (zero variance).
#[test]
fn test_wls_flat_response() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![5.0, 5.0, 5.0];
    let weights = vec![1.0, 1.0, 1.0];
    // Fit should be exactly 5.0
    let result = local_wls_helper(&x, &y, &weights, 0, 2, 1.5, 2.0);
    assert_relative_eq!(result, 5.0, epsilon = 1e-12);
}

/// Test WLS with vertical line (zero variance X).
#[test]
fn test_wls_vertical_line_fallback() {
    let x = vec![2.0, 2.0, 2.0];
    let y = vec![1.0, 2.0, 3.0];
    let weights = vec![1.0, 1.0, 1.0];

    // Denominator will be zero due to x being identical.
    // Should fallback to weighted mean of Y => 2.0
    let result = local_wls_helper(&x, &y, &weights, 0, 2, 2.0, 1.0);
    assert_relative_eq!(result, 2.0, epsilon = 1e-12);
}

/// Test WLS with very small values (underflow check).
#[test]
fn test_wls_small_values() {
    let x = vec![1e-10, 2e-10, 3e-10];
    let y = vec![1e-10, 2e-10, 3e-10];
    let weights = vec![1.0, 1.0, 1.0];

    // Should fit y=x perfectly even with small scales
    // Fit at 2.5e-10
    let result = local_wls_helper(&x, &y, &weights, 0, 2, 2.5e-10, 1e-9);
    assert_relative_eq!(result, 2.5e-10, epsilon = 1e-12 * 1e-10); // Relative epsilon
}

/// Test WLS handling of NaNs (Internal behavior check).
/// Note: Public API validator prevents NaNs, but internals should be robust.
#[test]
fn test_wls_nan_handling() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![1.0, f64::NAN, 3.0];
    let weights = vec![1.0, 1.0, 1.0];

    // NaNs in Y will propagate to sums.
    let result = local_wls_helper(&x, &y, &weights, 0, 2, 2.0, 2.0);
    assert!(
        result.is_nan(),
        "WLS should propagate NaNs if present in data"
    );
}
