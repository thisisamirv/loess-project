//! Linear algebra backend abstraction for LOESS.
//!
//! ## Purpose
//!
//! This module provides a trait-based abstraction over linear algebra operations,
//! standardizing on the optimized nalgebra backend.
//!
//! ## Design notes
//!
//! * Uses QR decomposition (Householder reflections) instead of Cholesky for better
//!   numerical stability with ill-conditioned systems.
//! * Fallback to SVD for rank-deficient matrices.
//! * Generic over `FloatLinalg` types (f32 and f64) which delegate to nalgebra.

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use nalgebra::{DMatrix, DVector};
use num_traits::Float;
use wide::{f32x8, f64x4};

// ============================================================================
// FloatLinalg Trait
// ============================================================================

/// Helper trait to bridge generic Float types to the optimized Nalgebra backend.
///
/// This trait provides type-specific optimized implementations for linear algebra
/// operations. Implemented for f32 and f64.
pub trait FloatLinalg: Float + 'static {
    /// Solve normal equations X'WX * beta = X'Wy.
    fn solve_normal(a: &[Self], b: &[Self], n: usize) -> Option<Vec<Self>>;
    /// Compute leverage (hat matrix diagonal element) for a query point.
    fn compute_leverage(design_vec: &[Self], xtw_x_inv: &[Self], n: usize) -> Self;
    /// Invert the normal matrix X'WX.
    fn invert_normal(a: &[Self], n: usize) -> Option<Vec<Self>>;

    // ========================================================================
    // Batch SIMD Operations
    // ========================================================================

    /// Batch compute absolute residuals: `out[i] = |a[i] - b[i]|`
    fn batch_abs_residuals(a: &[Self], b: &[Self], out: &mut [Self]);

    /// Batch compute sqrt-scaled values: `out[i] = scale * sqrt(input[i])`
    fn batch_sqrt_scale(input: &[Self], scale: Self, out: &mut [Self]);
}

impl FloatLinalg for f64 {
    #[inline]
    fn solve_normal(a: &[Self], b: &[Self], n: usize) -> Option<Vec<Self>> {
        nalgebra_backend::solve_normal_equations_f64(a, b, n)
    }
    #[inline]
    fn compute_leverage(design_vec: &[Self], xtw_x_inv: &[Self], n: usize) -> Self {
        nalgebra_backend::compute_leverage_f64(design_vec, xtw_x_inv, n)
    }
    #[inline]
    fn invert_normal(a: &[Self], n: usize) -> Option<Vec<Self>> {
        nalgebra_backend::invert_normal_matrix_f64(a, n)
    }
    #[inline]
    fn batch_abs_residuals(a: &[Self], b: &[Self], out: &mut [Self]) {
        simd_batch::batch_abs_residuals_f64(a, b, out)
    }
    #[inline]
    fn batch_sqrt_scale(input: &[Self], scale: Self, out: &mut [Self]) {
        simd_batch::batch_sqrt_scale_f64(input, scale, out)
    }
}

impl FloatLinalg for f32 {
    #[inline]
    fn solve_normal(a: &[Self], b: &[Self], n: usize) -> Option<Vec<Self>> {
        nalgebra_backend::solve_normal_equations_f32(a, b, n)
    }
    #[inline]
    fn compute_leverage(design_vec: &[Self], xtw_x_inv: &[Self], n: usize) -> Self {
        nalgebra_backend::compute_leverage_f32(design_vec, xtw_x_inv, n)
    }
    #[inline]
    fn invert_normal(a: &[Self], n: usize) -> Option<Vec<Self>> {
        nalgebra_backend::invert_normal_matrix_f32(a, n)
    }
    #[inline]
    fn batch_abs_residuals(a: &[Self], b: &[Self], out: &mut [Self]) {
        simd_batch::batch_abs_residuals_f32(a, b, out)
    }
    #[inline]
    fn batch_sqrt_scale(input: &[Self], scale: Self, out: &mut [Self]) {
        simd_batch::batch_sqrt_scale_f32(input, scale, out)
    }
}

// ============================================================================
// Nalgebra Backend Implementation
// ============================================================================

/// Nalgebra-based linear algebra operations.
pub mod nalgebra_backend {
    use super::*;

    /// Solve normal equations X'WX * beta = X'Wy using f64 precision.
    pub fn solve_normal_equations_f64(
        xtw_x: &[f64],
        xtw_y: &[f64],
        n_coeffs: usize,
    ) -> Option<Vec<f64>> {
        let matrix = DMatrix::from_column_slice(n_coeffs, n_coeffs, xtw_x);
        let rhs = DVector::from_column_slice(xtw_y);

        let qr = matrix.clone().qr();
        if let Some(solution) = qr.solve(&rhs) {
            return Some(solution.as_slice().to_vec());
        }

        matrix
            .svd(true, true)
            .solve(&rhs, f64::EPSILON * 100.0)
            .ok()
            .map(|s: DVector<f64>| s.as_slice().to_vec())
    }

    /// Compute leverage for a query point using f64 precision.
    pub fn compute_leverage_f64(design_vec: &[f64], xtw_x_inv: &[f64], n_coeffs: usize) -> f64 {
        let x = DVector::from_column_slice(design_vec);
        let inv = DMatrix::from_column_slice(n_coeffs, n_coeffs, xtw_x_inv);
        (x.transpose() * &inv * &x)[(0, 0)]
    }

    /// Invert normal matrix X'WX using f64 precision.
    pub fn invert_normal_matrix_f64(xtw_x: &[f64], n_coeffs: usize) -> Option<Vec<f64>> {
        let matrix = DMatrix::from_column_slice(n_coeffs, n_coeffs, xtw_x);
        let qr = matrix.clone().qr();
        let identity = DMatrix::identity(n_coeffs, n_coeffs);

        if let Some(inv) = qr.solve(&identity) {
            return Some(inv.as_slice().to_vec());
        }

        matrix
            .pseudo_inverse(f64::EPSILON * 100.0)
            .ok()
            .map(|inv: DMatrix<f64>| inv.as_slice().to_vec())
    }

    /// Solve normal equations X'WX * beta = X'Wy using f32 precision.
    pub fn solve_normal_equations_f32(
        xtw_x: &[f32],
        xtw_y: &[f32],
        n_coeffs: usize,
    ) -> Option<Vec<f32>> {
        let matrix = DMatrix::from_column_slice(n_coeffs, n_coeffs, xtw_x);
        let rhs = DVector::from_column_slice(xtw_y);

        let qr = matrix.clone().qr();
        if let Some(solution) = qr.solve(&rhs) {
            return Some(solution.as_slice().to_vec());
        }

        matrix
            .svd(true, true)
            .solve(&rhs, f32::EPSILON * 100.0)
            .ok()
            .map(|s: DVector<f32>| s.as_slice().to_vec())
    }

    /// Compute leverage for a query point using f32 precision.
    pub fn compute_leverage_f32(design_vec: &[f32], xtw_x_inv: &[f32], n_coeffs: usize) -> f32 {
        let x = DVector::from_column_slice(design_vec);
        let inv = DMatrix::from_column_slice(n_coeffs, n_coeffs, xtw_x_inv);
        (x.transpose() * &inv * &x)[(0, 0)]
    }

    /// Invert normal matrix X'WX using f32 precision.
    pub fn invert_normal_matrix_f32(xtw_x: &[f32], n_coeffs: usize) -> Option<Vec<f32>> {
        let matrix = DMatrix::from_column_slice(n_coeffs, n_coeffs, xtw_x);
        let qr = matrix.clone().qr();
        let identity = DMatrix::identity(n_coeffs, n_coeffs);

        if let Some(inv) = qr.solve(&identity) {
            return Some(inv.as_slice().to_vec());
        }

        matrix
            .pseudo_inverse(f32::EPSILON * 100.0)
            .ok()
            .map(|inv: DMatrix<f32>| inv.as_slice().to_vec())
    }
}

// ============================================================================
// SIMD Batch Operations
// ============================================================================

/// SIMD-optimized batch operations for array processing.
pub mod simd_batch {
    use super::*;

    /// Batch compute `|a[i] - b[i]|` for f64, SIMD-optimized.
    #[inline]
    pub fn batch_abs_residuals_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert!(out.len() >= a.len());

        let n = a.len();
        let chunks = n / 4;
        let remainder = n % 4;

        for i in 0..chunks {
            let base = i * 4;
            let va = f64x4::new([a[base], a[base + 1], a[base + 2], a[base + 3]]);
            let vb = f64x4::new([b[base], b[base + 1], b[base + 2], b[base + 3]]);
            let diff = (va - vb).abs();
            let arr = diff.to_array();
            out[base] = arr[0];
            out[base + 1] = arr[1];
            out[base + 2] = arr[2];
            out[base + 3] = arr[3];
        }

        let base = chunks * 4;
        for i in 0..remainder {
            out[base + i] = (a[base + i] - b[base + i]).abs();
        }
    }

    /// Batch compute `|a[i] - b[i]|` for f32, SIMD-optimized.
    #[inline]
    pub fn batch_abs_residuals_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert!(out.len() >= a.len());

        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;

        for i in 0..chunks {
            let base = i * 8;
            let va = f32x8::new([
                a[base],
                a[base + 1],
                a[base + 2],
                a[base + 3],
                a[base + 4],
                a[base + 5],
                a[base + 6],
                a[base + 7],
            ]);
            let vb = f32x8::new([
                b[base],
                b[base + 1],
                b[base + 2],
                b[base + 3],
                b[base + 4],
                b[base + 5],
                b[base + 6],
                b[base + 7],
            ]);
            let diff = (va - vb).abs();
            let arr = diff.to_array();
            out[base] = arr[0];
            out[base + 1] = arr[1];
            out[base + 2] = arr[2];
            out[base + 3] = arr[3];
            out[base + 4] = arr[4];
            out[base + 5] = arr[5];
            out[base + 6] = arr[6];
            out[base + 7] = arr[7];
        }

        let base = chunks * 8;
        for i in 0..remainder {
            out[base + i] = (a[base + i] - b[base + i]).abs();
        }
    }

    /// Batch compute `scale * sqrt(input[i])` for f64, SIMD-optimized.
    #[inline]
    pub fn batch_sqrt_scale_f64(input: &[f64], scale: f64, out: &mut [f64]) {
        debug_assert!(out.len() >= input.len());

        let n = input.len();
        let chunks = n / 4;
        let remainder = n % 4;
        let scale_vec = f64x4::splat(scale);

        for i in 0..chunks {
            let base = i * 4;
            let v = f64x4::new([
                input[base],
                input[base + 1],
                input[base + 2],
                input[base + 3],
            ]);
            let result = scale_vec * v.sqrt();
            let arr = result.to_array();
            out[base] = arr[0];
            out[base + 1] = arr[1];
            out[base + 2] = arr[2];
            out[base + 3] = arr[3];
        }

        let base = chunks * 4;
        for i in 0..remainder {
            out[base + i] = scale * input[base + i].sqrt();
        }
    }

    /// Batch compute `scale * sqrt(input[i])` for f32, SIMD-optimized.
    #[inline]
    pub fn batch_sqrt_scale_f32(input: &[f32], scale: f32, out: &mut [f32]) {
        debug_assert!(out.len() >= input.len());

        let n = input.len();
        let chunks = n / 8;
        let remainder = n % 8;
        let scale_vec = f32x8::splat(scale);

        for i in 0..chunks {
            let base = i * 8;
            let v = f32x8::new([
                input[base],
                input[base + 1],
                input[base + 2],
                input[base + 3],
                input[base + 4],
                input[base + 5],
                input[base + 6],
                input[base + 7],
            ]);
            let result = scale_vec * v.sqrt();
            let arr = result.to_array();
            out[base] = arr[0];
            out[base + 1] = arr[1];
            out[base + 2] = arr[2];
            out[base + 3] = arr[3];
            out[base + 4] = arr[4];
            out[base + 5] = arr[5];
            out[base + 6] = arr[6];
            out[base + 7] = arr[7];
        }

        let base = chunks * 8;
        for i in 0..remainder {
            out[base + i] = scale * input[base + i].sqrt();
        }
    }
}
