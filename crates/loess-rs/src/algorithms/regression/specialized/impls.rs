//! Specialized Solver Implementations
//!
//! ## Purpose
//!
//! This module provides the concrete implementations of the `SolverLinalg` trait for `f32` (standard precision) and `f64` (double precision) types.

// Modular dependencies
use super::SolverLinalg;
use super::accumulators_1d::{
    accumulate_1d_cubic_scalar, accumulate_1d_cubic_simd, accumulate_1d_linear_scalar,
    accumulate_1d_linear_simd, accumulate_1d_quadratic_scalar, accumulate_1d_quadratic_simd,
};
use super::accumulators_2d::{
    accumulate_2d_cubic_scalar, accumulate_2d_cubic_simd, accumulate_2d_linear_scalar,
    accumulate_2d_linear_simd, accumulate_2d_quadratic_scalar, accumulate_2d_quadratic_simd,
};
use super::accumulators_3d::{
    accumulate_3d_linear_scalar, accumulate_3d_linear_simd, accumulate_3d_quadratic_scalar,
    accumulate_3d_quadratic_simd,
};

impl SolverLinalg for f64 {
    #[inline]
    fn solve_2x2(a: [f64; 4], b: [f64; 2]) -> Option<[f64; 2]> {
        let det = a[0] * a[3] - a[1] * a[2];
        if det.abs() < f64::EPSILON * 1e-10 {
            return None;
        }
        let inv_det = 1.0 / det;
        Some([
            (b[0] * a[3] - a[1] * b[1]) * inv_det,
            (a[0] * b[1] - b[0] * a[2]) * inv_det,
        ])
    }

    #[inline]
    fn solve_3x3(a: [f64; 9], b: [f64; 3]) -> Option<[f64; 3]> {
        let det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6])
            + a[2] * (a[3] * a[7] - a[4] * a[6]);

        if det.abs() < f64::EPSILON * 1e-10 {
            return None;
        }
        let inv_det = 1.0 / det;

        let res0 = (b[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (b[1] * a[8] - a[5] * b[2])
            + a[2] * (b[1] * a[7] - a[4] * b[2]))
            * inv_det;
        let res1 = (a[0] * (b[1] * a[8] - a[5] * b[2]) - b[0] * (a[3] * a[8] - a[5] * a[6])
            + a[2] * (a[3] * b[2] - b[1] * a[6]))
            * inv_det;
        let res2 = (a[0] * (a[4] * b[2] - b[1] * a[7]) - a[1] * (a[3] * b[2] - b[1] * a[6])
            + b[0] * (a[3] * a[7] - a[4] * a[6]))
            * inv_det;

        Some([res0, res1, res2])
    }

    #[inline]
    fn solve_6x6(a: [f64; 36], b: [f64; 6]) -> Option<[f64; 6]> {
        let mut mat = [[0.0; 7]; 6];
        for i in 0..6 {
            for j in 0..6 {
                mat[i][j] = a[i * 6 + j];
            }
            mat[i][6] = b[i];
        }

        for i in 0..6 {
            let mut max_row = i;
            let mut max_val = mat[i][i].abs();
            for (k, row) in mat.iter().enumerate().skip(i + 1) {
                if row[i].abs() > max_val {
                    max_val = row[i].abs();
                    max_row = k;
                }
            }

            if max_val < f64::EPSILON * 1e-10 {
                return None;
            }

            if max_row != i {
                mat.swap(i, max_row);
            }

            let (upper, lower) = mat.split_at_mut(i + 1);
            let pivot_row = &upper[i];
            for row in lower.iter_mut() {
                let factor = row[i] / pivot_row[i];
                for (dest, src) in row[i..7].iter_mut().zip(&pivot_row[i..7]) {
                    *dest -= factor * *src;
                }
            }
        }

        let mut x = [0.0; 6];
        for i in (0..6).rev() {
            let mut sum = 0.0;
            for j in i + 1..6 {
                sum += mat[i][j] * x[j];
            }
            x[i] = (mat[i][6] - sum) / mat[i][i];
        }

        Some(x)
    }

    #[inline]
    fn accumulate_1d_linear(
        x: &[f64],
        y: &[f64],
        indices: &[usize],
        weights: &[f64],
        query: f64,
        xtwx: &mut [f64; 4],
        xtwy: &mut [f64; 2],
    ) {
        accumulate_1d_linear_simd(x, y, indices, weights, query, xtwx, xtwy)
    }

    #[inline]
    fn accumulate_1d_quadratic(
        x: &[f64],
        y: &[f64],
        indices: &[usize],
        weights: &[f64],
        query: f64,
        xtwx: &mut [f64; 9],
        xtwy: &mut [f64; 3],
    ) {
        accumulate_1d_quadratic_simd(x, y, indices, weights, query, xtwx, xtwy)
    }

    fn accumulate_1d_cubic(
        x: &[f64],
        y: &[f64],
        indices: &[usize],
        weights: &[f64],
        query: f64,
        xtwx: &mut [f64; 16],
        xtwy: &mut [f64; 4],
    ) {
        // cast to f64 slice since accumulators expect f64
        accumulate_1d_cubic_simd(x, y, indices, weights, query, xtwx, xtwy)
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn accumulate_2d_linear(
        x: &[f64],
        y: &[f64],
        indices: &[usize],
        weights: &[f64],
        query_x: f64,
        query_y: f64,
        xtwx: &mut [f64; 9],
        xtwy: &mut [f64; 3],
    ) {
        accumulate_2d_linear_simd(x, y, indices, weights, query_x, query_y, xtwx, xtwy)
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn accumulate_2d_quadratic(
        x: &[f64],
        y: &[f64],
        indices: &[usize],
        weights: &[f64],
        query_x: f64,
        query_y: f64,
        xtwx: &mut [f64; 36],
        xtwy: &mut [f64; 6],
    ) {
        accumulate_2d_quadratic_simd(x, y, indices, weights, query_x, query_y, xtwx, xtwy)
    }

    fn accumulate_2d_cubic(
        x: &[f64],
        y: &[f64],
        indices: &[usize],
        weights: &[f64],
        query_x: f64,
        query_y: f64,
        xtwx: &mut [f64; 100],
        xtwy: &mut [f64; 10],
    ) {
        accumulate_2d_cubic_simd(x, y, indices, weights, query_x, query_y, xtwx, xtwy)
    }

    fn accumulate_3d_linear(
        x: &[f64],
        y: &[f64],
        indices: &[usize],
        weights: &[f64],
        query_x: f64,
        query_y: f64,
        query_z: f64,
        xtwx: &mut [f64; 16],
        xtwy: &mut [f64; 4],
    ) {
        accumulate_3d_linear_simd(
            x, y, indices, weights, query_x, query_y, query_z, xtwx, xtwy,
        )
    }

    fn accumulate_3d_quadratic(
        x: &[f64],
        y: &[f64],
        indices: &[usize],
        weights: &[f64],
        query_x: f64,
        query_y: f64,
        query_z: f64,
        xtwx: &mut [f64; 100],
        xtwy: &mut [f64; 10],
    ) {
        accumulate_3d_quadratic_simd(
            x, y, indices, weights, query_x, query_y, query_z, xtwx, xtwy,
        )
    }
}

impl SolverLinalg for f32 {
    #[inline]
    fn solve_2x2(a: [f32; 4], b: [f32; 2]) -> Option<[f32; 2]> {
        let det = a[0] * a[3] - a[1] * a[2];
        if det.abs() < f32::EPSILON * 1e-10 {
            return None;
        }
        let inv_det = 1.0 / det;
        Some([
            (b[0] * a[3] - a[1] * b[1]) * inv_det,
            (a[0] * b[1] - b[0] * a[2]) * inv_det,
        ])
    }

    #[inline]
    fn solve_3x3(a: [f32; 9], b: [f32; 3]) -> Option<[f32; 3]> {
        let det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6])
            + a[2] * (a[3] * a[7] - a[4] * a[6]);

        if det.abs() < f32::EPSILON * 1e-10 {
            return None;
        }
        let inv_det = 1.0 / det;

        let res0 = (b[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (b[1] * a[8] - a[5] * b[2])
            + a[2] * (b[1] * a[7] - a[4] * b[2]))
            * inv_det;
        let res1 = (a[0] * (b[1] * a[8] - a[5] * b[2]) - b[0] * (a[3] * a[8] - a[5] * a[6])
            + a[2] * (a[3] * b[2] - b[1] * a[6]))
            * inv_det;
        let res2 = (a[0] * (a[4] * b[2] - b[1] * a[7]) - a[1] * (a[3] * b[2] - b[1] * a[6])
            + b[0] * (a[3] * a[7] - a[4] * a[6]))
            * inv_det;

        Some([res0, res1, res2])
    }

    #[inline]
    fn solve_6x6(a: [f32; 36], b: [f32; 6]) -> Option<[f32; 6]> {
        let mut mat = [[0.0; 7]; 6];
        for i in 0..6 {
            for j in 0..6 {
                mat[i][j] = a[i * 6 + j];
            }
            mat[i][6] = b[i];
        }

        for i in 0..6 {
            let mut max_row = i;
            let mut max_val = mat[i][i].abs();
            for (k, row) in mat.iter().enumerate().skip(i + 1) {
                if row[i].abs() > max_val {
                    max_val = row[i].abs();
                    max_row = k;
                }
            }

            if max_val < f32::EPSILON * 1e-10 {
                return None;
            }

            if max_row != i {
                mat.swap(i, max_row);
            }

            let (upper, lower) = mat.split_at_mut(i + 1);
            let pivot_row = &upper[i];
            for row in lower.iter_mut() {
                let factor = row[i] / pivot_row[i];
                for (dest, src) in row[i..7].iter_mut().zip(&pivot_row[i..7]) {
                    *dest -= factor * *src;
                }
            }
        }

        let mut x = [0.0; 6];
        for i in (0..6).rev() {
            let mut sum = 0.0;
            for j in i + 1..6 {
                sum += mat[i][j] * x[j];
            }
            x[i] = (mat[i][6] - sum) / mat[i][i];
        }

        Some(x)
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn accumulate_1d_linear(
        x: &[Self],
        y: &[Self],
        indices: &[usize],
        weights: &[Self],
        query: Self,
        xtwx: &mut [Self; 4],
        xtwy: &mut [Self; 2],
    ) {
        accumulate_1d_linear_scalar(x, y, indices, weights, query, xtwx, xtwy)
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn accumulate_1d_quadratic(
        x: &[Self],
        y: &[Self],
        indices: &[usize],
        weights: &[Self],
        query: Self,
        xtwx: &mut [Self; 9],
        xtwy: &mut [Self; 3],
    ) {
        accumulate_1d_quadratic_scalar(x, y, indices, weights, query, xtwx, xtwy)
    }

    fn accumulate_1d_cubic(
        x: &[f32],
        y: &[f32],
        indices: &[usize],
        weights: &[f32],
        query: f32,
        xtwx: &mut [f32; 16],
        xtwy: &mut [f32; 4],
    ) {
        accumulate_1d_cubic_scalar(x, y, indices, weights, query, xtwx, xtwy)
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn accumulate_2d_linear(
        x: &[Self],
        y: &[Self],
        indices: &[usize],
        weights: &[Self],
        query_x: Self,
        query_y: Self,
        xtwx: &mut [Self; 9],
        xtwy: &mut [Self; 3],
    ) {
        accumulate_2d_linear_scalar(x, y, indices, weights, query_x, query_y, xtwx, xtwy)
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn accumulate_2d_quadratic(
        x: &[Self],
        y: &[Self],
        indices: &[usize],
        weights: &[Self],
        query_x: Self,
        query_y: Self,
        xtwx: &mut [Self; 36],
        xtwy: &mut [Self; 6],
    ) {
        accumulate_2d_quadratic_scalar(x, y, indices, weights, query_x, query_y, xtwx, xtwy)
    }

    fn accumulate_2d_cubic(
        x: &[f32],
        y: &[f32],
        indices: &[usize],
        weights: &[f32],
        query_x: f32,
        query_y: f32,
        xtwx: &mut [f32; 100],
        xtwy: &mut [f32; 10],
    ) {
        accumulate_2d_cubic_scalar(x, y, indices, weights, query_x, query_y, xtwx, xtwy)
    }

    fn accumulate_3d_linear(
        x: &[f32],
        y: &[f32],
        indices: &[usize],
        weights: &[f32],
        query_x: f32,
        query_y: f32,
        query_z: f32,
        xtwx: &mut [f32; 16],
        xtwy: &mut [f32; 4],
    ) {
        accumulate_3d_linear_scalar(
            x, y, indices, weights, query_x, query_y, query_z, xtwx, xtwy,
        )
    }

    fn accumulate_3d_quadratic(
        x: &[f32],
        y: &[f32],
        indices: &[usize],
        weights: &[f32],
        query_x: f32,
        query_y: f32,
        query_z: f32,
        xtwx: &mut [f32; 100],
        xtwy: &mut [f32; 10],
    ) {
        accumulate_3d_quadratic_scalar(
            x, y, indices, weights, query_x, query_y, query_z, xtwx, xtwy,
        )
    }
}
