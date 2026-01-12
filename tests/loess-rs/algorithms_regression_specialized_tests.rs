#![cfg(feature = "dev")]

//! Comprehensive tests for specialized regression accumulators (1D, 2D, 3D).
//!
//! Tests SIMD and scalar paths for all polynomial degrees via SolverLinalg trait.

use loess_rs::internals::algorithms::regression::SolverLinalg;

// ============================================================================
// 1D Linear Tests
// ============================================================================

#[test]
fn test_1d_linear_f64() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let indices = vec![0, 1, 2, 3, 4];
    let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let query = 3.0;

    let mut xtwx = [0.0; 4];
    let mut xtwy = [0.0; 2];

    f64::accumulate_1d_linear(&x, &y, &indices, &weights, query, &mut xtwx, &mut xtwy);

    // Verify accumulation happened
    assert!(xtwx[0] > 0.0);
    assert!(xtwy[0] > 0.0);
}

#[test]
fn test_1d_linear_f32() {
    let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0f32, 4.0, 6.0, 8.0, 10.0];
    let indices = vec![0, 1, 2, 3, 4];
    let weights = vec![1.0f32; 5];
    let query = 3.0f32;

    let mut xtwx = [0.0f32; 4];
    let mut xtwy = [0.0f32; 2];

    f32::accumulate_1d_linear(&x, &y, &indices, &weights, query, &mut xtwx, &mut xtwy);

    assert!(xtwx[0] > 0.0);
    assert!(xtwy[0] > 0.0);
}

#[test]
fn test_1d_linear_zero_weights() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![2.0, 4.0, 6.0];
    let indices = vec![0, 1, 2];
    let weights = vec![0.0, 0.0, 0.0];
    let query = 2.0;

    let mut xtwx = [0.0; 4];
    let mut xtwy = [0.0; 2];

    f64::accumulate_1d_linear(&x, &y, &indices, &weights, query, &mut xtwx, &mut xtwy);

    // All should be zero with zero weights
    assert_eq!(xtwx[0], 0.0);
    assert_eq!(xtwy[0], 0.0);
}

#[test]
fn test_1d_linear_large_dataset() {
    let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
    let indices: Vec<usize> = (0..100).collect();
    let weights = vec![1.0; 100];
    let query = 50.0;

    let mut xtwx = [0.0; 4];
    let mut xtwy = [0.0; 2];

    f64::accumulate_1d_linear(&x, &y, &indices, &weights, query, &mut xtwx, &mut xtwy);

    assert!(xtwx[0] > 0.0);
    assert!(xtwy[0] > 0.0);
}

// ============================================================================
// 1D Quadratic Tests
// ============================================================================

#[test]
fn test_1d_quadratic_f64() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![1.0, 4.0, 9.0, 16.0, 25.0]; // y = x^2
    let indices = vec![0, 1, 2, 3, 4];
    let weights = vec![1.0; 5];
    let query = 3.0;

    let mut xtwx = [0.0; 9];
    let mut xtwy = [0.0; 3];

    f64::accumulate_1d_quadratic(&x, &y, &indices, &weights, query, &mut xtwx, &mut xtwy);

    assert!(xtwx[0] > 0.0);
    assert!(xtwy[0] > 0.0);
}

#[test]
fn test_1d_quadratic_f32() {
    let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let y = vec![1.0f32, 4.0, 9.0, 16.0, 25.0];
    let indices = vec![0, 1, 2, 3, 4];
    let weights = vec![1.0f32; 5];
    let query = 3.0f32;

    let mut xtwx = [0.0f32; 9];
    let mut xtwy = [0.0f32; 3];

    f32::accumulate_1d_quadratic(&x, &y, &indices, &weights, query, &mut xtwx, &mut xtwy);

    assert!(xtwx[0] > 0.0);
    assert!(xtwy[0] > 0.0);
}

#[test]
fn test_1d_quadratic_varying_weights() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![1.0, 4.0, 9.0, 16.0, 25.0];
    let indices = vec![0, 1, 2, 3, 4];
    let weights = vec![1.0, 0.5, 1.0, 0.5, 1.0];
    let query = 3.0;

    let mut xtwx = [0.0; 9];
    let mut xtwy = [0.0; 3];

    f64::accumulate_1d_quadratic(&x, &y, &indices, &weights, query, &mut xtwx, &mut xtwy);

    assert!(xtwx[0] > 0.0);
    assert!(xtwy[0] > 0.0);
}

// ============================================================================
// 1D Cubic Tests
// ============================================================================

#[test]
fn test_1d_cubic_f64() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y = vec![1.0, 8.0, 27.0, 64.0, 125.0, 216.0]; // y = x^3
    let indices = vec![0, 1, 2, 3, 4, 5];
    let weights = vec![1.0; 6];
    let query = 3.5;

    let mut xtwx = [0.0; 16];
    let mut xtwy = [0.0; 4];

    f64::accumulate_1d_cubic(&x, &y, &indices, &weights, query, &mut xtwx, &mut xtwy);

    assert!(xtwx[0] > 0.0);
    assert!(xtwy[0] > 0.0);
}

#[test]
fn test_1d_cubic_f32() {
    let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y = vec![1.0f32, 8.0, 27.0, 64.0, 125.0, 216.0];
    let indices = vec![0, 1, 2, 3, 4, 5];
    let weights = vec![1.0f32; 6];
    let query = 3.5f32;

    let mut xtwx = [0.0f32; 16];
    let mut xtwy = [0.0f32; 4];

    f32::accumulate_1d_cubic(&x, &y, &indices, &weights, query, &mut xtwx, &mut xtwy);

    assert!(xtwx[0] > 0.0);
    assert!(xtwy[0] > 0.0);
}

// ============================================================================
// 2D Linear Tests
// ============================================================================

#[test]
fn test_2d_linear_f64() {
    // Flattened 2D data: [x1, y1, x2, y2, ...]
    let x = vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0];
    let y = vec![2.0, 3.0, 4.0, 5.0]; // 4 points
    let indices = vec![0, 1, 2, 3];
    let weights = vec![1.0; 4];
    let query_x = 1.5;
    let query_y = 1.5;

    let mut xtwx = [0.0; 9];
    let mut xtwy = [0.0; 3];

    f64::accumulate_2d_linear(
        &x, &y, &indices, &weights, query_x, query_y, &mut xtwx, &mut xtwy,
    );

    assert!(xtwx[0] > 0.0);
    assert!(xtwy[0] > 0.0);
}

#[test]
fn test_2d_linear_f32() {
    let x = vec![1.0f32, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0];
    let y = vec![2.0f32, 3.0, 4.0, 5.0];
    let indices = vec![0, 1, 2, 3];
    let weights = vec![1.0f32; 4];
    let query_x = 1.5f32;
    let query_y = 1.5f32;

    let mut xtwx = [0.0f32; 9];
    let mut xtwy = [0.0f32; 3];

    f32::accumulate_2d_linear(
        &x, &y, &indices, &weights, query_x, query_y, &mut xtwx, &mut xtwy,
    );

    assert!(xtwx[0] > 0.0);
    assert!(xtwy[0] > 0.0);
}

#[test]
fn test_2d_linear_large_dataset() {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..50 {
        for j in 0..50 {
            x.push(i as f64);
            x.push(j as f64);
            y.push((i + j) as f64);
        }
    }
    let indices: Vec<usize> = (0..2500).collect();
    let weights = vec![1.0; 2500];
    let query_x = 25.0;
    let query_y = 25.0;

    let mut xtwx = [0.0; 9];
    let mut xtwy = [0.0; 3];

    f64::accumulate_2d_linear(
        &x, &y, &indices, &weights, query_x, query_y, &mut xtwx, &mut xtwy,
    );

    assert!(xtwx[0] > 0.0);
    assert!(xtwy[0] > 0.0);
}

// ============================================================================
// 2D Quadratic Tests
// ============================================================================

#[test]
fn test_2d_quadratic_f64() {
    let x = vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0];
    let y = vec![2.0, 3.0, 4.0, 5.0, 6.0];
    let indices = vec![0, 1, 2, 3, 4];
    let weights = vec![1.0; 5];
    let query_x = 2.0;
    let query_y = 2.0;

    let mut xtwx = [0.0; 36];
    let mut xtwy = [0.0; 6];

    f64::accumulate_2d_quadratic(
        &x, &y, &indices, &weights, query_x, query_y, &mut xtwx, &mut xtwy,
    );

    assert!(xtwx[0] > 0.0);
    assert!(xtwy[0] > 0.0);
}

#[test]
fn test_2d_quadratic_f32() {
    let x = vec![1.0f32, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0];
    let y = vec![2.0f32, 3.0, 4.0, 5.0, 6.0];
    let indices = vec![0, 1, 2, 3, 4];
    let weights = vec![1.0f32; 5];
    let query_x = 2.0f32;
    let query_y = 2.0f32;

    let mut xtwx = [0.0f32; 36];
    let mut xtwy = [0.0f32; 6];

    f32::accumulate_2d_quadratic(
        &x, &y, &indices, &weights, query_x, query_y, &mut xtwx, &mut xtwy,
    );

    assert!(xtwx[0] > 0.0);
    assert!(xtwy[0] > 0.0);
}

// ============================================================================
// 2D Cubic Tests
// ============================================================================

#[test]
fn test_2d_cubic_f64() {
    let x = vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 3.0, 2.0];
    let y = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let indices = vec![0, 1, 2, 3, 4, 5];
    let weights = vec![1.0; 6];
    let query_x = 2.0;
    let query_y = 1.5;

    let mut xtwx = [0.0; 100];
    let mut xtwy = [0.0; 10];

    f64::accumulate_2d_cubic(
        &x, &y, &indices, &weights, query_x, query_y, &mut xtwx, &mut xtwy,
    );

    assert!(xtwx[0] > 0.0);
    assert!(xtwy[0] > 0.0);
}

#[test]
fn test_2d_cubic_f32() {
    let x = vec![
        1.0f32, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 3.0, 2.0,
    ];
    let y = vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0];
    let indices = vec![0, 1, 2, 3, 4, 5];
    let weights = vec![1.0f32; 6];
    let query_x = 2.0f32;
    let query_y = 1.5f32;

    let mut xtwx = [0.0f32; 100];
    let mut xtwy = [0.0f32; 10];

    f32::accumulate_2d_cubic(
        &x, &y, &indices, &weights, query_x, query_y, &mut xtwx, &mut xtwy,
    );

    assert!(xtwx[0] > 0.0);
    assert!(xtwy[0] > 0.0);
}

// ============================================================================
// 3D Linear Tests
// ============================================================================

#[test]
fn test_3d_linear_f64() {
    // Flattened 3D data: [x1, y1, z1, x2, y2, z2, ...]
    let x = vec![1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 2.0];
    let y = vec![3.0, 4.0, 5.0, 6.0]; // 4 points
    let indices = vec![0, 1, 2, 3];
    let weights = vec![1.0; 4];
    let query_x = 1.5;
    let query_y = 1.5;
    let query_z = 1.5;

    let mut xtwx = [0.0; 16];
    let mut xtwy = [0.0; 4];

    f64::accumulate_3d_linear(
        &x, &y, &indices, &weights, query_x, query_y, query_z, &mut xtwx, &mut xtwy,
    );

    assert!(xtwx[0] > 0.0);
    assert!(xtwy[0] > 0.0);
}

#[test]
fn test_3d_linear_f32() {
    let x = vec![
        1.0f32, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 2.0,
    ];
    let y = vec![3.0f32, 4.0, 5.0, 6.0];
    let indices = vec![0, 1, 2, 3];
    let weights = vec![1.0f32; 4];
    let query_x = 1.5f32;
    let query_y = 1.5f32;
    let query_z = 1.5f32;

    let mut xtwx = [0.0f32; 16];
    let mut xtwy = [0.0f32; 4];

    f32::accumulate_3d_linear(
        &x, &y, &indices, &weights, query_x, query_y, query_z, &mut xtwx, &mut xtwy,
    );

    assert!(xtwx[0] > 0.0);
    assert!(xtwy[0] > 0.0);
}

// ============================================================================
// 3D Quadratic Tests
// ============================================================================

#[test]
fn test_3d_quadratic_f64() {
    let x = vec![
        1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0,
    ];
    let y = vec![3.0, 4.0, 5.0, 6.0, 7.0];
    let indices = vec![0, 1, 2, 3, 4];
    let weights = vec![1.0; 5];
    let query_x = 2.0;
    let query_y = 2.0;
    let query_z = 2.0;

    let mut xtwx = [0.0; 100];
    let mut xtwy = [0.0; 10];

    f64::accumulate_3d_quadratic(
        &x, &y, &indices, &weights, query_x, query_y, query_z, &mut xtwx, &mut xtwy,
    );

    assert!(xtwx[0] > 0.0);
    assert!(xtwy[0] > 0.0);
}

#[test]
fn test_3d_quadratic_f32() {
    let x = vec![
        1.0f32, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0,
    ];
    let y = vec![3.0f32, 4.0, 5.0, 6.0, 7.0];
    let indices = vec![0, 1, 2, 3, 4];
    let weights = vec![1.0f32; 5];
    let query_x = 2.0f32;
    let query_y = 2.0f32;
    let query_z = 2.0f32;

    let mut xtwx = [0.0f32; 100];
    let mut xtwy = [0.0f32; 10];

    f32::accumulate_3d_quadratic(
        &x, &y, &indices, &weights, query_x, query_y, query_z, &mut xtwx, &mut xtwy,
    );

    assert!(xtwx[0] > 0.0);
    assert!(xtwy[0] > 0.0);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_1d_linear_single_point() {
    let x = vec![5.0];
    let y = vec![10.0];
    let indices = vec![0];
    let weights = vec![1.0];
    let query = 5.0;

    let mut xtwx = [0.0; 4];
    let mut xtwy = [0.0; 2];

    f64::accumulate_1d_linear(&x, &y, &indices, &weights, query, &mut xtwx, &mut xtwy);

    assert!(xtwx[0] > 0.0);
    assert!(xtwy[0] > 0.0);
}

#[test]
fn test_2d_linear_partial_weights() {
    let x = vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0];
    let y = vec![2.0, 3.0, 4.0, 5.0];
    let indices = vec![0, 1, 2, 3];
    let weights = vec![1.0, 0.0, 1.0, 0.0]; // Half zero
    let query_x = 1.5;
    let query_y = 1.5;

    let mut xtwx = [0.0; 9];
    let mut xtwy = [0.0; 3];

    f64::accumulate_2d_linear(
        &x, &y, &indices, &weights, query_x, query_y, &mut xtwx, &mut xtwy,
    );

    assert!(xtwx[0] > 0.0);
    assert!(xtwy[0] > 0.0);
}

#[test]
fn test_1d_cubic_large_values() {
    let x = vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0];
    let y = vec![1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0];
    let indices = vec![0, 1, 2, 3, 4, 5];
    let weights = vec![1.0; 6];
    let query = 350.0;

    let mut xtwx = [0.0; 16];
    let mut xtwy = [0.0; 4];

    f64::accumulate_1d_cubic(&x, &y, &indices, &weights, query, &mut xtwx, &mut xtwy);

    assert!(xtwx[0] > 0.0);
    assert!(xtwy[0] > 0.0);
    assert!(xtwx[0].is_finite());
}

// ============================================================================
// Solver Tests
// ============================================================================

#[test]
fn test_solve_2x2_f64() {
    let a = [2.0, 1.0, 1.0, 2.0];
    let b = [3.0, 3.0];

    let result = f64::solve_2x2(a, b);
    assert!(result.is_some());
    let sol = result.unwrap();
    assert!((sol[0] - 1.0).abs() < 1e-10);
    assert!((sol[1] - 1.0).abs() < 1e-10);
}

#[test]
fn test_solve_3x3_f64() {
    let a = [2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0];
    let b = [3.0, 4.0, 3.0];

    let result = f64::solve_3x3(a, b);
    assert!(result.is_some());
    let sol = result.unwrap();
    assert!(sol[0].is_finite());
    assert!(sol[1].is_finite());
    assert!(sol[2].is_finite());
}

#[test]
fn test_solve_6x6_f64() {
    // Identity matrix
    let mut a = [0.0; 36];
    for i in 0..6 {
        a[i * 6 + i] = 1.0;
    }
    let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let result = f64::solve_6x6(a, b);
    assert!(result.is_some());
    let sol = result.unwrap();
    for i in 0..6 {
        assert!((sol[i] - b[i]).abs() < 1e-10);
    }
}

#[test]
fn test_solve_2x2_singular() {
    let a = [1.0, 1.0, 1.0, 1.0]; // Singular matrix
    let b = [1.0, 2.0];

    let result = f64::solve_2x2(a, b);
    assert!(result.is_none());
}
