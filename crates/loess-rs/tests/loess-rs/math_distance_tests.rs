#![cfg(feature = "dev")]

use approx::assert_relative_eq;
use loess_rs::internals::math::distance::DistanceMetric;

// ============================================================================
// Euclidean Distance Tests
// ============================================================================

#[test]
fn test_euclidean_distance_1d() {
    let a = [1.0];
    let b = [4.0];
    let dist = DistanceMetric::euclidean(&a, &b);
    assert_relative_eq!(dist, 3.0);
}

#[test]
fn test_euclidean_distance_2d() {
    let a = [0.0, 0.0];
    let b = [3.0, 4.0];
    let dist = DistanceMetric::euclidean(&a, &b);
    assert_relative_eq!(dist, 5.0);
}

#[test]
fn test_euclidean_distance_3d() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 6.0, 8.0];
    // diffs: 3, 4, 5. sum_sq: 9+16+25=50. sqrt(50) approx 7.071
    let dist = DistanceMetric::euclidean(&a, &b);
    assert_relative_eq!(dist, 50.0f64.sqrt());
}

#[test]
fn test_euclidean_distance_f32() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 6.0, 8.0];
    let dist = DistanceMetric::euclidean(&a, &b);
    assert_relative_eq!(dist, 50.0f32.sqrt(), epsilon = 1e-6);
}

#[test]
fn test_euclidean_distance_high_dim() {
    let a: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let b: Vec<f64> = (0..10).map(|i| (i + 1) as f64).collect();
    let dist = DistanceMetric::euclidean(&a, &b);
    // All diffs are 1, so sqrt(10) = 3.162...
    assert_relative_eq!(dist, 10.0f64.sqrt());
}

#[test]
fn test_euclidean_distance_zero() {
    let a = [1.0, 2.0, 3.0];
    let b = [1.0, 2.0, 3.0];
    let dist = DistanceMetric::euclidean(&a, &b);
    assert_relative_eq!(dist, 0.0);
}

#[test]
fn test_euclidean_distance_large_values() {
    let a = [1e6, 2e6];
    let b = [1.1e6, 2.1e6];
    let dist: f64 = DistanceMetric::euclidean(&a, &b);
    assert!(dist > 0.0);
    assert!(dist.is_finite());
}

// ============================================================================
// Normalized Distance Tests
// ============================================================================

#[test]
fn test_normalized_distance() {
    let a = [0.0, 10.0];
    let b = [10.0, 20.0];
    let scales = [0.1, 0.05]; // range x: 10, range y: 20

    // diffs: 10, 10
    // scaled: 1.0, 0.5
    // sum_sq: 1.0 + 0.25 = 1.25
    // sqrt(1.25) approx 1.118

    let dist = DistanceMetric::normalized(&a, &b, &scales);
    assert_relative_eq!(dist, 1.25f64.sqrt());
}

#[test]
fn test_normalized_distance_f32() {
    let a = [0.0f32, 10.0];
    let b = [10.0f32, 20.0];
    let scales = [0.1f32, 0.05];

    let dist = DistanceMetric::normalized(&a, &b, &scales);
    assert_relative_eq!(dist, 1.25f32.sqrt(), epsilon = 1e-6);
}

#[test]
fn test_normalized_distance_uniform_scales() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    let scales = [1.0, 1.0, 1.0];

    let dist = DistanceMetric::normalized(&a, &b, &scales);
    let euclidean = DistanceMetric::euclidean(&a, &b);
    assert_relative_eq!(dist, euclidean);
}

#[test]
fn test_normalized_distance_high_dim() {
    let a: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let b: Vec<f64> = (0..20).map(|i| (i + 1) as f64).collect();
    let scales: Vec<f64> = vec![0.5; 20];

    let dist = DistanceMetric::normalized(&a, &b, &scales);
    assert!(dist > 0.0);
    assert!(dist.is_finite());
}

// ============================================================================
// Manhattan Distance Tests
// ============================================================================

#[test]
fn test_manhattan_distance() {
    let a = [1.0, 2.0];
    let b = [4.0, 6.0];
    // |1-4| + |2-6| = 3 + 4 = 7
    let dist = DistanceMetric::manhattan(&a, &b);
    assert_relative_eq!(dist, 7.0);
}

#[test]
fn test_manhattan_distance_f32() {
    let a = [1.0f32, 2.0];
    let b = [4.0f32, 6.0];
    let dist = DistanceMetric::manhattan(&a, &b);
    assert_relative_eq!(dist, 7.0f32);
}

#[test]
fn test_manhattan_distance_1d() {
    let a = [5.0];
    let b = [2.0];
    let dist = DistanceMetric::manhattan(&a, &b);
    assert_relative_eq!(dist, 3.0);
}

#[test]
fn test_manhattan_distance_3d() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 6.0, 9.0];
    // |1-4| + |2-6| + |3-9| = 3 + 4 + 6 = 13
    let dist = DistanceMetric::manhattan(&a, &b);
    assert_relative_eq!(dist, 13.0);
}

#[test]
fn test_manhattan_distance_negative() {
    let a = [-1.0, -2.0];
    let b = [1.0, 2.0];
    // |-1-1| + |-2-2| = 2 + 4 = 6
    let dist = DistanceMetric::manhattan(&a, &b);
    assert_relative_eq!(dist, 6.0);
}

#[test]
fn test_manhattan_distance_high_dim() {
    let a: Vec<f64> = vec![0.0; 50];
    let b: Vec<f64> = vec![1.0; 50];
    let dist = DistanceMetric::manhattan(&a, &b);
    assert_relative_eq!(dist, 50.0);
}

// ============================================================================
// Chebyshev Distance Tests
// ============================================================================

#[test]
fn test_chebyshev_distance() {
    let a = [1.0, 2.0];
    let b = [4.0, 7.0];
    // |1-4|=3, |2-7|=5. max(3, 5) = 5
    let dist = DistanceMetric::chebyshev(&a, &b);
    assert_relative_eq!(dist, 5.0);
}

#[test]
fn test_chebyshev_distance_f32() {
    let a = [1.0f32, 2.0];
    let b = [4.0f32, 7.0];
    let dist = DistanceMetric::chebyshev(&a, &b);
    assert_relative_eq!(dist, 5.0f32);
}

#[test]
fn test_chebyshev_distance_1d() {
    let a = [10.0];
    let b = [3.0];
    let dist = DistanceMetric::chebyshev(&a, &b);
    assert_relative_eq!(dist, 7.0);
}

#[test]
fn test_chebyshev_distance_3d() {
    let a = [1.0, 2.0, 3.0];
    let b = [2.0, 5.0, 4.0];
    // |1-2|=1, |2-5|=3, |3-4|=1. max = 3
    let dist = DistanceMetric::chebyshev(&a, &b);
    assert_relative_eq!(dist, 3.0);
}

#[test]
fn test_chebyshev_distance_all_equal_diffs() {
    let a = [0.0, 0.0, 0.0];
    let b = [5.0, 5.0, 5.0];
    let dist = DistanceMetric::chebyshev(&a, &b);
    assert_relative_eq!(dist, 5.0);
}

#[test]
fn test_chebyshev_distance_high_dim() {
    let mut a = vec![0.0; 100];
    let mut b = vec![1.0; 100];
    a[50] = 0.0;
    b[50] = 10.0; // This will be the max difference

    let dist = DistanceMetric::chebyshev(&a, &b);
    assert_relative_eq!(dist, 10.0);
}

// ============================================================================
// Minkowski Distance Tests
// ============================================================================

#[test]
fn test_minkowski_distance() {
    let a = [1.0, 2.0];
    let b = [4.0, 6.0];
    let p = 3.0;
    // |3|^3 + |4|^3 = 27 + 64 = 91. 91^(1/3) approx 4.4979
    let dist: f64 = DistanceMetric::minkowski(&a, &b, p);
    assert_relative_eq!(dist, 91.0f64.powf(1.0 / 3.0));
}

#[test]
fn test_minkowski_distance_f32() {
    let a = [1.0f32, 2.0];
    let b = [4.0f32, 6.0];
    let p = 3.0f32;
    let dist: f32 = DistanceMetric::minkowski(&a, &b, p);
    assert_relative_eq!(dist, 91.0f32.powf(1.0 / 3.0), epsilon = 1e-5);
}

#[test]
fn test_minkowski_p1_equals_manhattan() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 6.0, 9.0];

    let minkowski = DistanceMetric::minkowski(&a, &b, 1.0);
    let manhattan = DistanceMetric::manhattan(&a, &b);

    assert_relative_eq!(minkowski, manhattan);
}

#[test]
fn test_minkowski_p2_equals_euclidean() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 6.0, 8.0];

    let minkowski = DistanceMetric::minkowski(&a, &b, 2.0);
    let euclidean = DistanceMetric::euclidean(&a, &b);

    assert_relative_eq!(minkowski, euclidean);
}

#[test]
fn test_minkowski_large_p() {
    let a = [1.0, 2.0];
    let b = [4.0, 7.0];
    let p = 100.0;

    // As p -> infinity, Minkowski approaches Chebyshev
    let minkowski: f64 = DistanceMetric::minkowski(&a, &b, p);
    let chebyshev: f64 = DistanceMetric::chebyshev(&a, &b);

    // Should be close to Chebyshev for large p
    assert!((minkowski - chebyshev).abs() < 0.1f64);
}

#[test]
fn test_minkowski_fractional_p() {
    let a = [1.0, 2.0];
    let b = [3.0, 5.0];
    let p = 0.5;

    let dist: f64 = DistanceMetric::minkowski(&a, &b, p);
    assert!(dist > 0.0);
    assert!(dist.is_finite());
}

#[test]
fn test_minkowski_high_dim() {
    let a: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let b: Vec<f64> = (0..20).map(|i| (i + 1) as f64).collect();
    let p = 3.0;

    let dist: f64 = DistanceMetric::minkowski(&a, &b, p);
    assert!(dist > 0.0);
    assert!(dist.is_finite());
}

// ============================================================================
// Weighted Distance Tests
// ============================================================================

#[test]
fn test_weighted_distance() {
    let a = [1.0, 2.0];
    let b = [2.0, 3.0];
    let weights = [4.0, 1.0]; // Weight X more heavily

    // diffs: 1, 1
    // weighted sq: 4*(1)^2 + 1*(1)^2 = 5
    // dist = sqrt(5) approx 2.236

    let dist = DistanceMetric::weighted(&a, &b, &weights);
    assert_relative_eq!(dist, 5.0f64.sqrt());
}

#[test]
fn test_weighted_distance_f32() {
    let a = [1.0f32, 2.0];
    let b = [2.0f32, 3.0];
    let weights = [4.0f32, 1.0];

    let dist = DistanceMetric::weighted(&a, &b, &weights);
    assert_relative_eq!(dist, 5.0f32.sqrt(), epsilon = 1e-6);
}

#[test]
fn test_weighted_distance_uniform_weights() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    let weights = [1.0, 1.0, 1.0];

    let weighted = DistanceMetric::weighted(&a, &b, &weights);
    let euclidean = DistanceMetric::euclidean(&a, &b);

    assert_relative_eq!(weighted, euclidean);
}

#[test]
fn test_weighted_distance_zero_weight() {
    let a = [1.0, 2.0, 3.0];
    let b = [10.0, 20.0, 30.0];
    let weights = [1.0, 0.0, 1.0]; // Middle dimension ignored

    // Only dimensions 0 and 2 contribute
    // (10-1)^2 * 1 + (30-3)^2 * 1 = 81 + 729 = 810
    let dist = DistanceMetric::weighted(&a, &b, &weights);
    assert_relative_eq!(dist, 810.0f64.sqrt());
}

#[test]
fn test_weighted_distance_high_dim() {
    let a: Vec<f64> = vec![0.0; 50];
    let b: Vec<f64> = vec![1.0; 50];
    let weights: Vec<f64> = (0..50).map(|i| (i + 1) as f64).collect();

    let dist = DistanceMetric::weighted(&a, &b, &weights);
    assert!(dist > 0.0);
    assert!(dist.is_finite());
}

#[test]
fn test_weighted_distance_varying_weights() {
    let a = [1.0, 1.0, 1.0];
    let b = [2.0, 2.0, 2.0];
    let weights = [1.0, 4.0, 9.0];

    // All diffs are 1
    // weighted sq: 1*1 + 4*1 + 9*1 = 14
    let dist = DistanceMetric::weighted(&a, &b, &weights);
    assert_relative_eq!(dist, 14.0f64.sqrt());
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_distance_empty_arrays() {
    let a: Vec<f64> = vec![];
    let b: Vec<f64> = vec![];

    let euclidean = DistanceMetric::euclidean(&a, &b);
    assert_eq!(euclidean, 0.0);

    let manhattan = DistanceMetric::manhattan(&a, &b);
    assert_eq!(manhattan, 0.0);

    let chebyshev = DistanceMetric::chebyshev(&a, &b);
    assert_eq!(chebyshev, 0.0);
}

#[test]
fn test_distance_single_element() {
    let a = [5.0];
    let b = [2.0];

    let euclidean = DistanceMetric::euclidean(&a, &b);
    let manhattan = DistanceMetric::manhattan(&a, &b);
    let chebyshev = DistanceMetric::chebyshev(&a, &b);

    assert_relative_eq!(euclidean, 3.0);
    assert_relative_eq!(manhattan, 3.0);
    assert_relative_eq!(chebyshev, 3.0);
}

#[test]
fn test_distance_very_small_values() {
    let a = [1e-10, 2e-10];
    let b = [3e-10, 4e-10];

    let dist: f64 = DistanceMetric::euclidean(&a, &b);
    assert!(dist > 0.0);
    assert!(dist.is_finite());
}

#[test]
fn test_distance_mixed_signs() {
    let a = [-5.0, 3.0, -2.0];
    let b = [5.0, -3.0, 2.0];

    let euclidean = DistanceMetric::euclidean(&a, &b);
    // diffs: 10, 6, 4. sum_sq: 100+36+16=152
    assert_relative_eq!(euclidean, 152.0f64.sqrt());

    let manhattan = DistanceMetric::manhattan(&a, &b);
    assert_relative_eq!(manhattan, 20.0);

    let chebyshev = DistanceMetric::chebyshev(&a, &b);
    assert_relative_eq!(chebyshev, 10.0);
}

#[test]
fn test_distance_symmetry() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];

    let dist_ab = DistanceMetric::euclidean(&a, &b);
    let dist_ba = DistanceMetric::euclidean(&b, &a);

    assert_relative_eq!(dist_ab, dist_ba);
}

#[test]
fn test_distance_triangle_inequality() {
    let a = [0.0, 0.0];
    let b = [3.0, 4.0];
    let c = [6.0, 8.0];

    let ab = DistanceMetric::euclidean(&a, &b);
    let bc = DistanceMetric::euclidean(&b, &c);
    let ac = DistanceMetric::euclidean(&a, &c);

    // Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
    assert!(ac <= ab + bc + 1e-10);
}

// ============================================================================
// Squared Distance Methods Tests
// ============================================================================

#[test]
fn test_euclidean_squared_basic() {
    let a = [0.0f64, 0.0];
    let b = [3.0f64, 4.0];
    // sqrt(sq) should equal euclidean
    let sq = DistanceMetric::euclidean_squared(&a, &b);
    let eu = DistanceMetric::euclidean(&a, &b);
    assert_relative_eq!(sq, eu * eu, epsilon = 1e-12);
    assert_relative_eq!(sq, 25.0, epsilon = 1e-12);
}

#[test]
fn test_euclidean_squared_f32() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 6.0, 8.0];
    let sq = DistanceMetric::euclidean_squared(&a, &b);
    assert_relative_eq!(sq, 50.0f32, epsilon = 1e-5);
}

#[test]
fn test_euclidean_squared_high_dim() {
    // Exercises SIMD path (>= 4 for f64)
    let a: Vec<f64> = vec![0.0; 8];
    let b: Vec<f64> = vec![1.0; 8];
    let sq = DistanceMetric::euclidean_squared(&a, &b);
    assert_relative_eq!(sq, 8.0, epsilon = 1e-12);
}

#[test]
fn test_normalized_squared_basic() {
    let a = [0.0f64, 10.0];
    let b = [10.0f64, 20.0];
    let scales = [0.1f64, 0.05];
    let sq = DistanceMetric::normalized_squared(&a, &b, &scales);
    let dist = DistanceMetric::normalized(&a, &b, &scales);
    assert_relative_eq!(sq, dist * dist, epsilon = 1e-12);
}

#[test]
fn test_normalized_squared_f32() {
    let a = [0.0f32, 10.0];
    let b = [10.0f32, 20.0];
    let scales = [0.1f32, 0.05];
    let sq = DistanceMetric::normalized_squared(&a, &b, &scales);
    assert_relative_eq!(sq, 1.25f32, epsilon = 1e-5);
}

#[test]
fn test_normalized_squared_high_dim() {
    // Exercises f64 SIMD path (8 elements)
    let a: Vec<f64> = vec![0.0; 8];
    let b: Vec<f64> = vec![1.0; 8];
    let scales: Vec<f64> = vec![0.5; 8];
    let sq = DistanceMetric::normalized_squared(&a, &b, &scales);
    // each diff=1, scaled=0.5, sq=0.25, sum=2.0
    assert_relative_eq!(sq, 2.0, epsilon = 1e-12);
}

#[test]
fn test_weighted_squared_basic() {
    let a = [1.0f64, 1.0];
    let b = [2.0f64, 2.0];
    let weights = [2.0f64, 3.0];
    // diffs: 1, 1; weighted sq: 2+3 = 5
    let sq = DistanceMetric::weighted_squared(&a, &b, &weights);
    let dist = DistanceMetric::weighted(&a, &b, &weights);
    assert_relative_eq!(sq, dist * dist, epsilon = 1e-12);
    assert_relative_eq!(sq, 5.0, epsilon = 1e-12);
}

#[test]
fn test_weighted_squared_f32_simd() {
    // f32 SIMD path requires >= 8 elements
    let a: Vec<f32> = vec![0.0; 8];
    let b: Vec<f32> = vec![1.0; 8];
    let weights: Vec<f32> = vec![2.0; 8];
    // each diff=1, weight=2, sq contribution=2, total=16
    let sq = DistanceMetric::weighted_squared(&a, &b, &weights);
    assert_relative_eq!(sq, 16.0f32, epsilon = 1e-5);
}

#[test]
fn test_weighted_squared_f64_simd() {
    // f64 SIMD path requires >= 4 elements
    let a: Vec<f64> = vec![0.0; 4];
    let b: Vec<f64> = vec![1.0; 4];
    let weights: Vec<f64> = vec![3.0; 4];
    let sq = DistanceMetric::weighted_squared(&a, &b, &weights);
    assert_relative_eq!(sq, 12.0, epsilon = 1e-12);
}

#[test]
fn test_manhattan_squared_basic() {
    // manhattan = 7, so squared = 49
    let a = [1.0f64, 2.0];
    let b = [4.0f64, 6.0];
    let sq = DistanceMetric::manhattan_squared(&a, &b);
    assert_relative_eq!(sq, 49.0, epsilon = 1e-12);
}

#[test]
fn test_manhattan_squared_f32() {
    let a = [1.0f32, 2.0];
    let b = [4.0f32, 6.0];
    let sq = DistanceMetric::manhattan_squared(&a, &b);
    assert_relative_eq!(sq, 49.0f32, epsilon = 1e-5);
}

#[test]
fn test_chebyshev_squared_basic() {
    // chebyshev = 5, so squared = 25
    let a = [1.0f64, 2.0];
    let b = [4.0f64, 7.0];
    let sq = DistanceMetric::chebyshev_squared(&a, &b);
    assert_relative_eq!(sq, 25.0, epsilon = 1e-12);
}

#[test]
fn test_chebyshev_squared_f32() {
    let a = [1.0f32, 2.0];
    let b = [4.0f32, 7.0];
    let sq = DistanceMetric::chebyshev_squared(&a, &b);
    assert_relative_eq!(sq, 25.0f32, epsilon = 1e-5);
}

#[test]
fn test_minkowski_squared_basic() {
    let a = [0.0f64, 0.0];
    let b = [3.0f64, 4.0];
    let p = 2.0;
    let sq = DistanceMetric::minkowski_squared(&a, &b, p);
    // minkowski_p2 = euclidean = 5, squared = 25
    assert_relative_eq!(sq, 25.0, epsilon = 1e-10);
}

#[test]
fn test_minkowski_squared_f32() {
    let a = [0.0f32, 0.0];
    let b = [3.0f32, 4.0];
    let sq = DistanceMetric::minkowski_squared(&a, &b, 2.0f32);
    assert_relative_eq!(sq, 25.0f32, epsilon = 1e-4);
}

// ============================================================================
// simd_distance Module Direct Tests
// ============================================================================

use loess_rs::internals::math::distance::simd_distance;

#[test]
fn test_simd_euclidean_sq_f64_simd_path() {
    // >= 4 elements triggers SIMD
    let a = [1.0f64, 2.0, 3.0, 4.0];
    let b = [2.0f64, 3.0, 4.0, 5.0];
    let sq = simd_distance::euclidean_sq_f64(&a, &b);
    assert_relative_eq!(sq, 4.0, epsilon = 1e-12); // 4 * 1^2
}

#[test]
fn test_simd_euclidean_sq_f32_simd_path() {
    // >= 8 elements triggers SIMD
    let a: [f32; 8] = [1.0; 8];
    let b: [f32; 8] = [2.0; 8];
    let sq = simd_distance::euclidean_sq_f32(&a, &b);
    assert_relative_eq!(sq, 8.0f32, epsilon = 1e-5);
}

#[test]
fn test_simd_normalized_sq_f64_simd_path() {
    let a = [0.0f64, 0.0, 0.0, 0.0];
    let b = [2.0f64, 2.0, 2.0, 2.0];
    let scales = [0.5f64, 0.5, 0.5, 0.5];
    let sq = simd_distance::normalized_sq_f64(&a, &b, &scales);
    // diff=2, scale=0.5 => 1 each, 4 total
    assert_relative_eq!(sq, 4.0, epsilon = 1e-12);
}

#[test]
fn test_simd_normalized_sq_f32_simd_path() {
    let a: [f32; 8] = [0.0; 8];
    let b: [f32; 8] = [2.0; 8];
    let scales: [f32; 8] = [0.5; 8];
    let sq = simd_distance::normalized_sq_f32(&a, &b, &scales);
    assert_relative_eq!(sq, 8.0f32, epsilon = 1e-5);
}

#[test]
fn test_simd_weighted_sq_f64_simd_path() {
    let a = [0.0f64, 0.0, 0.0, 0.0];
    let b = [1.0f64, 1.0, 1.0, 1.0];
    let w = [2.0f64, 2.0, 2.0, 2.0];
    let sq = simd_distance::weighted_sq_f64(&a, &b, &w);
    assert_relative_eq!(sq, 8.0, epsilon = 1e-12);
}

#[test]
fn test_simd_weighted_sq_f32_simd_path() {
    let a: [f32; 8] = [0.0; 8];
    let b: [f32; 8] = [1.0; 8];
    let w: [f32; 8] = [3.0; 8];
    let sq = simd_distance::weighted_sq_f32(&a, &b, &w);
    assert_relative_eq!(sq, 24.0f32, epsilon = 1e-5);
}

#[test]
fn test_simd_manhattan_f64_simd_path() {
    let a = [1.0f64, 2.0, 3.0, 4.0];
    let b = [4.0f64, 1.0, 5.0, 2.0];
    let result = simd_distance::manhattan_f64(&a, &b);
    // |3| + |1| + |2| + |2| = 8
    assert_relative_eq!(result, 8.0, epsilon = 1e-12);
}

#[test]
fn test_simd_manhattan_f32_simd_path() {
    let a: [f32; 8] = [1.0; 8];
    let b: [f32; 8] = [3.0; 8];
    let result = simd_distance::manhattan_f32(&a, &b);
    assert_relative_eq!(result, 16.0f32, epsilon = 1e-5);
}

#[test]
fn test_simd_chebyshev_f64_simd_path() {
    let a = [1.0f64, 2.0, 10.0, 4.0];
    let b = [4.0f64, 5.0, 3.0, 4.0];
    let result = simd_distance::chebyshev_f64(&a, &b);
    // diffs: 3, 3, 7, 0 => max = 7
    assert_relative_eq!(result, 7.0, epsilon = 1e-12);
}

#[test]
fn test_simd_chebyshev_f32_simd_path() {
    let a: [f32; 8] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0];
    let b: [f32; 8] = [1.0; 8];
    let result = simd_distance::chebyshev_f32(&a, &b);
    assert_relative_eq!(result, 99.0f32, epsilon = 1e-5);
}

#[test]
fn test_simd_minkowski_f64_general() {
    // p=3, not 1 or 2
    let a = [0.0f64, 0.0];
    let b = [3.0f64, 4.0];
    let result = simd_distance::minkowski_f64(&a, &b, 3.0);
    assert_relative_eq!(
        result,
        (3.0f64.powi(3) + 4.0f64.powi(3)).powf(1.0 / 3.0),
        epsilon = 1e-10
    );
}

#[test]
fn test_simd_minkowski_f32_general() {
    let a = [0.0f32, 0.0];
    let b = [3.0f32, 4.0];
    let result = simd_distance::minkowski_f32(&a, &b, 3.0);
    assert_relative_eq!(
        result,
        (3.0f32.powi(3) + 4.0f32.powi(3)).powf(1.0 / 3.0),
        epsilon = 1e-5
    );
}

#[test]
fn test_simd_minkowski_f64_p1_shortcut() {
    let a = [1.0f64, 2.0];
    let b = [4.0f64, 6.0];
    let result = simd_distance::minkowski_f64(&a, &b, 1.0);
    assert_relative_eq!(result, 7.0, epsilon = 1e-12);
}

#[test]
fn test_simd_minkowski_f32_p2_shortcut() {
    let a = [0.0f32, 0.0];
    let b = [3.0f32, 4.0];
    let result = simd_distance::minkowski_f32(&a, &b, 2.0);
    assert_relative_eq!(result, 5.0f32, epsilon = 1e-5);
}
