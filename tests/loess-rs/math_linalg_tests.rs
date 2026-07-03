#![cfg(feature = "dev")]

use approx::assert_relative_eq;
use loess_rs::internals::math::linalg::{FloatLinalg, nalgebra_backend, simd_batch};

// ============================================================================
// nalgebra_backend: solve_normal_equations (f64)
// ============================================================================

#[test]
fn test_solve_normal_f64_identity() {
    // Identity matrix * [1, 2] = [1, 2]
    let a = vec![1.0f64, 0.0, 0.0, 1.0]; // column-major: [[1,0],[0,1]]
    let b = vec![3.0f64, 7.0];
    let result = nalgebra_backend::solve_normal_equations_f64(&a, &b, 2);
    assert!(result.is_some());
    let sol = result.unwrap();
    assert_relative_eq!(sol[0], 3.0, epsilon = 1e-10);
    assert_relative_eq!(sol[1], 7.0, epsilon = 1e-10);
}

#[test]
fn test_solve_normal_f64_2x2() {
    // System: [2 1; 1 3] * [x; y] = [5; 10]
    // Solution: x = 1, y = 3
    // column-major: [2, 1, 1, 3]
    let a = vec![2.0f64, 1.0, 1.0, 3.0];
    let b = vec![5.0f64, 10.0];
    let result = nalgebra_backend::solve_normal_equations_f64(&a, &b, 2);
    assert!(result.is_some());
    let sol = result.unwrap();
    assert_relative_eq!(sol[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(sol[1], 3.0, epsilon = 1e-10);
}

#[test]
fn test_solve_normal_f64_1x1() {
    let a = vec![4.0f64];
    let b = vec![8.0f64];
    let result = nalgebra_backend::solve_normal_equations_f64(&a, &b, 1);
    assert!(result.is_some());
    assert_relative_eq!(result.unwrap()[0], 2.0, epsilon = 1e-10);
}

#[test]
fn test_solve_normal_f64_singular_svd_fallback() {
    // Singular matrix (all zeros) — SVD fallback should handle gracefully
    let a = vec![0.0f64; 4];
    let b = vec![0.0f64; 2];
    // Should return Some([0, 0]) or None without panicking
    let result = nalgebra_backend::solve_normal_equations_f64(&a, &b, 2);
    if let Some(sol) = result {
        // If it returns something, it should be near-zero
        assert!(sol.iter().all(|&v| v.abs() < 1e-6));
    }
}

// ============================================================================
// nalgebra_backend: solve_normal_equations (f32)
// ============================================================================

#[test]
fn test_solve_normal_f32_identity() {
    let a = vec![1.0f32, 0.0, 0.0, 1.0];
    let b = vec![3.0f32, 7.0];
    let result = nalgebra_backend::solve_normal_equations_f32(&a, &b, 2);
    assert!(result.is_some());
    let sol = result.unwrap();
    assert_relative_eq!(sol[0], 3.0f32, epsilon = 1e-5);
    assert_relative_eq!(sol[1], 7.0f32, epsilon = 1e-5);
}

#[test]
fn test_solve_normal_f32_2x2() {
    let a = vec![2.0f32, 1.0, 1.0, 3.0];
    let b = vec![5.0f32, 10.0];
    let result = nalgebra_backend::solve_normal_equations_f32(&a, &b, 2);
    assert!(result.is_some());
    let sol = result.unwrap();
    assert_relative_eq!(sol[0], 1.0f32, epsilon = 1e-5);
    assert_relative_eq!(sol[1], 3.0f32, epsilon = 1e-5);
}

#[test]
fn test_solve_normal_f32_singular_svd_fallback() {
    let a = vec![0.0f32; 4];
    let b = vec![0.0f32; 2];
    let result = nalgebra_backend::solve_normal_equations_f32(&a, &b, 2);
    if let Some(sol) = result {
        assert!(sol.iter().all(|&v| v.abs() < 1e-5));
    }
}

// ============================================================================
// nalgebra_backend: compute_leverage (f64)
// ============================================================================

#[test]
fn test_compute_leverage_f64_identity() {
    // x = [1, 0], A^-1 = identity => leverage = x' * I * x = 1
    let design_vec = vec![1.0f64, 0.0];
    let xtw_x_inv = vec![1.0f64, 0.0, 0.0, 1.0]; // column-major identity
    let leverage = nalgebra_backend::compute_leverage_f64(&design_vec, &xtw_x_inv, 2);
    assert_relative_eq!(leverage, 1.0, epsilon = 1e-10);
}

#[test]
fn test_compute_leverage_f64_scaled() {
    // x = [2, 0], A^-1 = identity => leverage = 4
    let design_vec = vec![2.0f64, 0.0];
    let xtw_x_inv = vec![1.0f64, 0.0, 0.0, 1.0];
    let leverage = nalgebra_backend::compute_leverage_f64(&design_vec, &xtw_x_inv, 2);
    assert_relative_eq!(leverage, 4.0, epsilon = 1e-10);
}

#[test]
fn test_compute_leverage_f64_1d() {
    // x = [3], A^-1 = [0.5] => leverage = 3 * 0.5 * 3 = 4.5
    let design_vec = vec![3.0f64];
    let xtw_x_inv = vec![0.5f64];
    let leverage = nalgebra_backend::compute_leverage_f64(&design_vec, &xtw_x_inv, 1);
    assert_relative_eq!(leverage, 4.5, epsilon = 1e-10);
}

// ============================================================================
// nalgebra_backend: compute_leverage (f32)
// ============================================================================

#[test]
fn test_compute_leverage_f32_identity() {
    let design_vec = vec![1.0f32, 0.0];
    let xtw_x_inv = vec![1.0f32, 0.0, 0.0, 1.0];
    let leverage = nalgebra_backend::compute_leverage_f32(&design_vec, &xtw_x_inv, 2);
    assert_relative_eq!(leverage, 1.0f32, epsilon = 1e-5);
}

#[test]
fn test_compute_leverage_f32_1d() {
    let design_vec = vec![3.0f32];
    let xtw_x_inv = vec![0.5f32];
    let leverage = nalgebra_backend::compute_leverage_f32(&design_vec, &xtw_x_inv, 1);
    assert_relative_eq!(leverage, 4.5f32, epsilon = 1e-5);
}

// ============================================================================
// nalgebra_backend: invert_normal_matrix (f64)
// ============================================================================

#[test]
fn test_invert_normal_f64_identity() {
    // Inverse of identity is identity
    let a = vec![1.0f64, 0.0, 0.0, 1.0];
    let result = nalgebra_backend::invert_normal_matrix_f64(&a, 2);
    assert!(result.is_some());
    let inv = result.unwrap();
    assert_relative_eq!(inv[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(inv[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(inv[2], 0.0, epsilon = 1e-10);
    assert_relative_eq!(inv[3], 1.0, epsilon = 1e-10);
}

#[test]
fn test_invert_normal_f64_diagonal() {
    // Diagonal [4, 0; 0, 2] => inverse [0.25, 0; 0, 0.5]
    let a = vec![4.0f64, 0.0, 0.0, 2.0];
    let result = nalgebra_backend::invert_normal_matrix_f64(&a, 2);
    assert!(result.is_some());
    let inv = result.unwrap();
    assert_relative_eq!(inv[0], 0.25, epsilon = 1e-10);
    assert_relative_eq!(inv[3], 0.5, epsilon = 1e-10);
}

#[test]
fn test_invert_normal_f64_1x1() {
    let a = vec![5.0f64];
    let result = nalgebra_backend::invert_normal_matrix_f64(&a, 1);
    assert!(result.is_some());
    assert_relative_eq!(result.unwrap()[0], 0.2, epsilon = 1e-10);
}

#[test]
fn test_invert_normal_f64_singular_pseudo_inverse() {
    // Singular — pseudo-inverse should be returned
    let a = vec![1.0f64, 1.0, 1.0, 1.0];
    let result = nalgebra_backend::invert_normal_matrix_f64(&a, 2);
    // Should not panic; may or may not return Some
    let _ = result;
}

// ============================================================================
// nalgebra_backend: invert_normal_matrix (f32)
// ============================================================================

#[test]
fn test_invert_normal_f32_identity() {
    let a = vec![1.0f32, 0.0, 0.0, 1.0];
    let result = nalgebra_backend::invert_normal_matrix_f32(&a, 2);
    assert!(result.is_some());
    let inv = result.unwrap();
    assert_relative_eq!(inv[0], 1.0f32, epsilon = 1e-5);
    assert_relative_eq!(inv[3], 1.0f32, epsilon = 1e-5);
}

#[test]
fn test_invert_normal_f32_1x1() {
    let a = vec![5.0f32];
    let result = nalgebra_backend::invert_normal_matrix_f32(&a, 1);
    assert!(result.is_some());
    assert_relative_eq!(result.unwrap()[0], 0.2f32, epsilon = 1e-5);
}

#[test]
fn test_invert_normal_f32_singular_pseudo_inverse() {
    let a = vec![1.0f32, 1.0, 1.0, 1.0];
    let result = nalgebra_backend::invert_normal_matrix_f32(&a, 2);
    let _ = result;
}

// ============================================================================
// FloatLinalg trait dispatch (f64)
// ============================================================================

#[test]
fn test_floatlinalg_f64_solve_normal() {
    let a = vec![2.0f64, 0.0, 0.0, 3.0];
    let b = vec![4.0f64, 9.0];
    let result = f64::solve_normal(&a, &b, 2);
    assert!(result.is_some());
    let sol = result.unwrap();
    assert_relative_eq!(sol[0], 2.0, epsilon = 1e-10);
    assert_relative_eq!(sol[1], 3.0, epsilon = 1e-10);
}

#[test]
fn test_floatlinalg_f64_invert_normal() {
    let a = vec![2.0f64, 0.0, 0.0, 4.0];
    let result = f64::invert_normal(&a, 2);
    assert!(result.is_some());
    let inv = result.unwrap();
    assert_relative_eq!(inv[0], 0.5, epsilon = 1e-10);
    assert_relative_eq!(inv[3], 0.25, epsilon = 1e-10);
}

#[test]
fn test_floatlinalg_f64_compute_leverage() {
    let design_vec = vec![1.0f64];
    let xtw_x_inv = vec![2.0f64];
    let lev = f64::compute_leverage(&design_vec, &xtw_x_inv, 1);
    assert_relative_eq!(lev, 2.0, epsilon = 1e-10);
}

// ============================================================================
// FloatLinalg trait dispatch (f32)
// ============================================================================

#[test]
fn test_floatlinalg_f32_solve_normal() {
    let a = vec![2.0f32, 0.0, 0.0, 3.0];
    let b = vec![4.0f32, 9.0];
    let result = f32::solve_normal(&a, &b, 2);
    assert!(result.is_some());
    let sol = result.unwrap();
    assert_relative_eq!(sol[0], 2.0f32, epsilon = 1e-5);
    assert_relative_eq!(sol[1], 3.0f32, epsilon = 1e-5);
}

#[test]
fn test_floatlinalg_f32_invert_normal() {
    let a = vec![2.0f32, 0.0, 0.0, 4.0];
    let result = f32::invert_normal(&a, 2);
    assert!(result.is_some());
    let inv = result.unwrap();
    assert_relative_eq!(inv[0], 0.5f32, epsilon = 1e-5);
}

#[test]
fn test_floatlinalg_f32_compute_leverage() {
    let design_vec = vec![1.0f32];
    let xtw_x_inv = vec![2.0f32];
    let lev = f32::compute_leverage(&design_vec, &xtw_x_inv, 1);
    assert_relative_eq!(lev, 2.0f32, epsilon = 1e-5);
}

// ============================================================================
// simd_batch: batch_abs_residuals (f64)
// ============================================================================

#[test]
fn test_batch_abs_residuals_f64_small() {
    // Length < 4: scalar path
    let a = [1.0f64, 5.0, 3.0];
    let b = [4.0f64, 2.0, 7.0];
    let mut out = [0.0f64; 3];
    simd_batch::batch_abs_residuals_f64(&a, &b, &mut out);
    assert_relative_eq!(out[0], 3.0);
    assert_relative_eq!(out[1], 3.0);
    assert_relative_eq!(out[2], 4.0);
}

#[test]
fn test_batch_abs_residuals_f64_simd() {
    // Length = 8: exercises SIMD chunks (2 full chunks of 4)
    let a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = [8.0f64, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let mut out = [0.0f64; 8];
    simd_batch::batch_abs_residuals_f64(&a, &b, &mut out);
    for (i, o) in out.iter().enumerate() {
        assert_relative_eq!(*o, (a[i] - b[i]).abs(), epsilon = 1e-12);
    }
}

#[test]
fn test_batch_abs_residuals_f64_with_remainder() {
    // Length = 5: 1 SIMD chunk + 1 remainder
    let a = [10.0f64, 20.0, 30.0, 40.0, 50.0];
    let b = [5.0f64, 25.0, 28.0, 45.0, 60.0];
    let mut out = [0.0f64; 5];
    simd_batch::batch_abs_residuals_f64(&a, &b, &mut out);
    for (i, o) in out.iter().enumerate() {
        assert_relative_eq!(*o, (a[i] - b[i]).abs(), epsilon = 1e-12);
    }
}

#[test]
fn test_floatlinalg_f64_batch_abs_residuals() {
    let a = [1.0f64, 4.0, 9.0, 16.0, 25.0];
    let b = [0.0f64, 5.0, 7.0, 20.0, 15.0];
    let mut out = [0.0f64; 5];
    f64::batch_abs_residuals(&a, &b, &mut out);
    assert_relative_eq!(out[0], 1.0);
    assert_relative_eq!(out[1], 1.0);
    assert_relative_eq!(out[2], 2.0);
    assert_relative_eq!(out[3], 4.0);
    assert_relative_eq!(out[4], 10.0);
}

// ============================================================================
// simd_batch: batch_abs_residuals (f32)
// ============================================================================

#[test]
fn test_batch_abs_residuals_f32_small() {
    // Length < 8: scalar path
    let a = [1.0f32, 5.0, 3.0];
    let b = [4.0f32, 2.0, 7.0];
    let mut out = [0.0f32; 3];
    simd_batch::batch_abs_residuals_f32(&a, &b, &mut out);
    assert_relative_eq!(out[0], 3.0f32);
    assert_relative_eq!(out[1], 3.0f32);
    assert_relative_eq!(out[2], 4.0f32);
}

#[test]
fn test_batch_abs_residuals_f32_simd() {
    // Length = 8: one full SIMD chunk
    let a: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b: [f32; 8] = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let mut out = [0.0f32; 8];
    simd_batch::batch_abs_residuals_f32(&a, &b, &mut out);
    for (i, o) in out.iter().enumerate() {
        assert_relative_eq!(*o, (a[i] - b[i]).abs(), epsilon = 1e-6);
    }
}

#[test]
fn test_batch_abs_residuals_f32_with_remainder() {
    // Length = 9: 1 SIMD chunk + 1 remainder
    let a: Vec<f32> = (0..9).map(|i| i as f32 * 2.0).collect();
    let b: Vec<f32> = (0..9).map(|i| i as f32).collect();
    let mut out = vec![0.0f32; 9];
    simd_batch::batch_abs_residuals_f32(&a, &b, &mut out);
    for (i, o) in out.iter().enumerate() {
        assert_relative_eq!(*o, (a[i] - b[i]).abs(), epsilon = 1e-6);
    }
}

#[test]
fn test_floatlinalg_f32_batch_abs_residuals() {
    let a: [f32; 8] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let b: [f32; 8] = [5.0, 25.0, 28.0, 45.0, 60.0, 55.0, 80.0, 75.0];
    let mut out = [0.0f32; 8];
    f32::batch_abs_residuals(&a, &b, &mut out);
    for (i, o) in out.iter().enumerate() {
        assert_relative_eq!(*o, (a[i] - b[i]).abs(), epsilon = 1e-6);
    }
}

// ============================================================================
// simd_batch: batch_sqrt_scale (f64)
// ============================================================================

#[test]
fn test_batch_sqrt_scale_f64_small() {
    // Length < 4: scalar path
    let input = [4.0f64, 9.0, 16.0];
    let mut out = [0.0f64; 3];
    simd_batch::batch_sqrt_scale_f64(&input, 2.0, &mut out);
    assert_relative_eq!(out[0], 4.0); // 2 * sqrt(4)
    assert_relative_eq!(out[1], 6.0); // 2 * sqrt(9)
    assert_relative_eq!(out[2], 8.0); // 2 * sqrt(16)
}

#[test]
fn test_batch_sqrt_scale_f64_simd() {
    // Length = 8: 2 full SIMD chunks of 4
    let input: [f64; 8] = [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0];
    let mut out = [0.0f64; 8];
    simd_batch::batch_sqrt_scale_f64(&input, 3.0, &mut out);
    let expected: [f64; 8] = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0];
    for (o, e) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(o, e, epsilon = 1e-10);
    }
}

#[test]
fn test_batch_sqrt_scale_f64_with_remainder() {
    // Length = 5: 1 SIMD chunk + 1 remainder
    let input = [4.0f64, 9.0, 16.0, 25.0, 36.0];
    let mut out = [0.0f64; 5];
    simd_batch::batch_sqrt_scale_f64(&input, 1.0, &mut out);
    let expected: Vec<f64> = input.iter().map(|&v| v.sqrt()).collect();
    for (o, e) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(o, e, epsilon = 1e-10);
    }
}

#[test]
fn test_floatlinalg_f64_batch_sqrt_scale() {
    let input = [1.0f64, 4.0, 9.0];
    let mut out = [0.0f64; 3];
    f64::batch_sqrt_scale(&input, 2.0, &mut out);
    assert_relative_eq!(out[0], 2.0);
    assert_relative_eq!(out[1], 4.0);
    assert_relative_eq!(out[2], 6.0);
}

// ============================================================================
// simd_batch: batch_sqrt_scale (f32)
// ============================================================================

#[test]
fn test_batch_sqrt_scale_f32_small() {
    let input = [4.0f32, 9.0, 16.0];
    let mut out = [0.0f32; 3];
    simd_batch::batch_sqrt_scale_f32(&input, 2.0, &mut out);
    assert_relative_eq!(out[0], 4.0f32, epsilon = 1e-6);
    assert_relative_eq!(out[1], 6.0f32, epsilon = 1e-6);
    assert_relative_eq!(out[2], 8.0f32, epsilon = 1e-6);
}

#[test]
fn test_batch_sqrt_scale_f32_simd() {
    // Length = 8: 1 full SIMD chunk
    let input: [f32; 8] = [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0];
    let mut out = [0.0f32; 8];
    simd_batch::batch_sqrt_scale_f32(&input, 0.5, &mut out);
    let expected: Vec<f32> = input.iter().map(|&v| 0.5 * v.sqrt()).collect();
    for (o, e) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(o, e, epsilon = 1e-5);
    }
}

#[test]
fn test_batch_sqrt_scale_f32_with_remainder() {
    // Length = 9: 1 SIMD chunk + 1 remainder
    let input: [f32; 9] = [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0];
    let mut out = [0.0f32; 9];
    simd_batch::batch_sqrt_scale_f32(&input, 1.0, &mut out);
    for (i, o) in out.iter().enumerate() {
        assert_relative_eq!(*o, input[i].sqrt(), epsilon = 1e-5);
    }
}

#[test]
fn test_floatlinalg_f32_batch_sqrt_scale() {
    let input: [f32; 8] = [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0];
    let mut out = [0.0f32; 8];
    f32::batch_sqrt_scale(&input, 2.0, &mut out);
    let expected: Vec<f32> = input.iter().map(|&v| 2.0 * v.sqrt()).collect();
    for (o, e) in out.iter().zip(expected.iter()) {
        assert_relative_eq!(o, e, epsilon = 1e-5);
    }
}
