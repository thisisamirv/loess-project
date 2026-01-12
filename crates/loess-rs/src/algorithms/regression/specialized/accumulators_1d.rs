//! Specialized 1D Accumulators
//!
//! ## Purpose
//!
//! This module provides optimized scalar and SIMD accumulation functions for building Normal Equations matrices in 1D linear, quadratic, and cubic regression.

// External dependencies
use num_traits::Float;
use wide::f64x2;

// ============================================================================
// Specialized Accumulation Functions
// ============================================================================

/// Optimized accumulation for 1D Linear Case (Scalar).
#[allow(clippy::too_many_arguments)]
pub fn accumulate_1d_linear_scalar<T: Float>(
    x: &[T],
    y: &[T],
    indices: &[usize],
    weights: &[T],
    query: T,
    xtwx: &mut [T; 4],
    xtwy: &mut [T; 2],
) {
    let n = indices.len();
    let mut s_w = T::zero();
    let mut s_dx = T::zero();
    let mut s_dx2 = T::zero();
    let mut s_wy = T::zero();
    let mut s_wdxy = T::zero();

    // SAFETY: Invariants guaranteed by KD-tree.
    unsafe {
        for i in 0..n {
            let w = *weights.get_unchecked(i);
            if w <= T::epsilon() {
                continue;
            }
            let idx = *indices.get_unchecked(i);
            let dx = *x.get_unchecked(idx) - query;
            let y_val = *y.get_unchecked(idx);
            let wdx = w * dx;
            s_w = s_w + w;
            s_dx = s_dx + wdx;
            s_dx2 = s_dx2 + wdx * dx;
            s_wy = s_wy + w * y_val;
            s_wdxy = s_wdxy + wdx * y_val;
        }
    }
    xtwx[0] = s_w;
    xtwx[1] = s_dx;
    xtwx[2] = s_dx;
    xtwx[3] = s_dx2;
    xtwy[0] = s_wy;
    xtwy[1] = s_wdxy;
}

/// Optimized accumulation for 1D Linear Case using SIMD.
#[allow(clippy::too_many_arguments)]
pub fn accumulate_1d_linear_simd(
    x: &[f64],
    y: &[f64],
    indices: &[usize],
    weights: &[f64],
    query: f64,
    xtwx: &mut [f64; 4],
    xtwy: &mut [f64; 2],
) {
    let n = indices.len();
    let mut i = 0;

    let mut s_w = f64x2::splat(0.0);
    let mut s_dx = f64x2::splat(0.0);
    let mut s_dx2 = f64x2::splat(0.0);
    let mut s_wy = f64x2::splat(0.0);
    let mut s_wdxy = f64x2::splat(0.0);

    let q = f64x2::splat(query);

    // SAFETY:
    // 1. `indices` and `weights` have the same length `n`.
    // 2. `indices` contains valid indices into `x` and `y` (guaranteed by KD-tree construction).
    // 3. Loops are bounded by `n`.
    unsafe {
        while i + 2 <= n {
            let idx0 = *indices.get_unchecked(i);
            let idx1 = *indices.get_unchecked(i + 1);

            let w = f64x2::new([*weights.get_unchecked(i), *weights.get_unchecked(i + 1)]);
            let x_val = f64x2::new([*x.get_unchecked(idx0), *x.get_unchecked(idx1)]);
            let y_val = f64x2::new([*y.get_unchecked(idx0), *y.get_unchecked(idx1)]);

            let dx = x_val - q;
            let wdx = w * dx;

            s_w += w;
            s_dx += wdx;
            s_dx2 += wdx * dx;
            s_wy += w * y_val;
            s_wdxy += wdx * y_val;

            i += 2;
        }
    }

    let mut a_w = s_w.reduce_add();
    let mut a_dx = s_dx.reduce_add();
    let mut a_dx2 = s_dx2.reduce_add();
    let mut a_wy = s_wy.reduce_add();
    let mut a_wdxy = s_wdxy.reduce_add();

    // Tail
    unsafe {
        for k in i..n {
            let w = *weights.get_unchecked(k);
            if w <= f64::EPSILON {
                continue;
            }

            let idx = *indices.get_unchecked(k);
            let dx = *x.get_unchecked(idx) - query;
            let y_val = *y.get_unchecked(idx);
            let wdx = w * dx;

            a_w += w;
            a_dx += wdx;
            a_dx2 += wdx * dx;
            a_wy += w * y_val;
            a_wdxy += wdx * y_val;
        }
    }

    xtwx[0] = a_w;
    xtwx[1] = a_dx;
    xtwx[2] = a_dx;
    xtwx[3] = a_dx2;
    xtwy[0] = a_wy;
    xtwy[1] = a_wdxy;
}

/// Optimized accumulation for 1D Quadratic Case (Scalar).
#[allow(clippy::too_many_arguments)]
pub fn accumulate_1d_quadratic_scalar<T: Float>(
    x: &[T],
    y: &[T],
    indices: &[usize],
    weights: &[T],
    query: T,
    xtwx: &mut [T; 9],
    xtwy: &mut [T; 3],
) {
    let n = indices.len();
    let mut s_w = T::zero();
    let mut s_dx = T::zero();
    let mut s_dx2 = T::zero();
    let mut s_dx3 = T::zero();
    let mut s_dx4 = T::zero();
    let mut s_wy = T::zero();
    let mut s_wdx_y = T::zero();
    let mut s_wdx2_y = T::zero();

    // SAFETY: Invariants guaranteed by KD-tree.
    unsafe {
        for i in 0..n {
            let w = *weights.get_unchecked(i);
            if w <= T::epsilon() {
                continue;
            }
            let idx = *indices.get_unchecked(i);
            let dx = *x.get_unchecked(idx) - query;
            let y_val = *y.get_unchecked(idx);
            let dx2 = dx * dx;

            let wdx = w * dx;
            let wdx2 = w * dx2;

            s_w = s_w + w;
            s_dx = s_dx + wdx;
            s_dx2 = s_dx2 + wdx2;
            s_dx3 = s_dx3 + wdx2 * dx;
            s_dx4 = s_dx4 + wdx2 * dx2;

            s_wy = s_wy + w * y_val;
            s_wdx_y = s_wdx_y + wdx * y_val;
            s_wdx2_y = s_wdx2_y + wdx2 * y_val;
        }
    }

    // Matrix X'WX (symmetric)
    // Row 0: 1, x, x^2
    xtwx[0] = s_w;
    xtwx[1] = s_dx;
    xtwx[2] = s_dx2;
    // Row 1: x, x^2, x^3
    xtwx[3] = s_dx;
    xtwx[4] = s_dx2;
    xtwx[5] = s_dx3;
    // Row 2: x^2, x^3, x^4
    xtwx[6] = s_dx2;
    xtwx[7] = s_dx3;
    xtwx[8] = s_dx4;

    xtwy[0] = s_wy;
    xtwy[1] = s_wdx_y;
    xtwy[2] = s_wdx2_y;
}

/// Optimized accumulation for 1D Quadratic Case using SIMD.
#[allow(clippy::too_many_arguments)]
pub fn accumulate_1d_quadratic_simd(
    x: &[f64],
    y: &[f64],
    indices: &[usize],
    weights: &[f64],
    query: f64,
    xtwx: &mut [f64; 9],
    xtwy: &mut [f64; 3],
) {
    let n = indices.len();
    let mut i = 0;

    let mut s_w = f64x2::splat(0.0);
    let mut s_dx = f64x2::splat(0.0);
    let mut s_dx2 = f64x2::splat(0.0);
    let mut s_dx3 = f64x2::splat(0.0);
    let mut s_dx4 = f64x2::splat(0.0);

    let mut s_wy = f64x2::splat(0.0);
    let mut s_wdx_y = f64x2::splat(0.0);
    let mut s_wdx2_y = f64x2::splat(0.0);

    let q = f64x2::splat(query);

    // SAFETY: Use unchecked access for performance. Invariants guaranteed by KD-tree.
    unsafe {
        while i + 2 <= n {
            let idx0 = *indices.get_unchecked(i);
            let idx1 = *indices.get_unchecked(i + 1);

            let w = f64x2::new([*weights.get_unchecked(i), *weights.get_unchecked(i + 1)]);
            let x_val = f64x2::new([*x.get_unchecked(idx0), *x.get_unchecked(idx1)]);
            let y_val = f64x2::new([*y.get_unchecked(idx0), *y.get_unchecked(idx1)]);

            let dx = x_val - q;
            let dx2 = dx * dx;

            let wdx = w * dx;
            let wdx2 = w * dx2;

            s_w += w;
            s_dx += wdx;
            s_dx2 += wdx2;
            s_dx3 += wdx2 * dx;
            s_dx4 += wdx2 * dx2;

            s_wy += w * y_val;
            s_wdx_y += wdx * y_val;
            s_wdx2_y += wdx2 * y_val;

            i += 2;
        }
    }

    let mut a_w = s_w.reduce_add();
    let mut a_dx = s_dx.reduce_add();
    let mut a_dx2 = s_dx2.reduce_add();
    let mut a_dx3 = s_dx3.reduce_add();
    let mut a_dx4 = s_dx4.reduce_add();

    let mut a_wy = s_wy.reduce_add();
    let mut a_wdx_y = s_wdx_y.reduce_add();
    let mut a_wdx2_y = s_wdx2_y.reduce_add();

    // Tail
    unsafe {
        for k in i..n {
            let w = *weights.get_unchecked(k);
            if w <= f64::EPSILON {
                continue;
            }

            let idx = *indices.get_unchecked(k);
            let dx = *x.get_unchecked(idx) - query;
            let y_val = *y.get_unchecked(idx);
            let dx2 = dx * dx;

            let wdx = w * dx;
            let wdx2 = w * dx2;

            a_w += w;
            a_dx += wdx;
            a_dx2 += wdx2;
            a_dx3 += wdx2 * dx;
            a_dx4 += wdx2 * dx2;

            a_wy += w * y_val;
            a_wdx_y += wdx * y_val;
            a_wdx2_y += wdx2 * y_val;
        }
    }

    xtwx[0] = a_w;
    xtwx[1] = a_dx;
    xtwx[2] = a_dx2;
    xtwx[3] = a_dx;
    xtwx[4] = a_dx2;
    xtwx[5] = a_dx3;
    xtwx[6] = a_dx2;
    xtwx[7] = a_dx3;
    xtwx[8] = a_dx4;

    xtwy[0] = a_wy;
    xtwy[1] = a_wdx_y;
    xtwy[2] = a_wdx2_y;
}

/// Optimized accumulation for 1D Cubic Case (Scalar).
#[allow(clippy::too_many_arguments)]
pub fn accumulate_1d_cubic_scalar<T: Float>(
    x: &[T],
    y: &[T],
    indices: &[usize],
    weights: &[T],
    query: T,
    xtwx: &mut [T; 16],
    xtwy: &mut [T; 4],
) {
    let n = indices.len();
    let mut s_w = T::zero();
    let mut s_dx = T::zero();
    let mut s_dx2 = T::zero();
    let mut s_dx3 = T::zero();
    let mut s_dx4 = T::zero();
    let mut s_dx5 = T::zero();
    let mut s_dx6 = T::zero();
    let mut s_wy = T::zero();
    let mut s_wdx_y = T::zero();
    let mut s_wdx2_y = T::zero();
    let mut s_wdx3_y = T::zero();

    // SAFETY: Invariants guaranteed by KD-tree.
    unsafe {
        for i in 0..n {
            let w = *weights.get_unchecked(i);
            if w <= T::epsilon() {
                continue;
            }
            let idx = *indices.get_unchecked(i);
            let dx = *x.get_unchecked(idx) - query;
            let y_val = *y.get_unchecked(idx);
            let dx2 = dx * dx;
            let dx3 = dx2 * dx;

            let wdx = w * dx;
            let wdx2 = w * dx2;
            let wdx3 = w * dx3;

            s_w = s_w + w;
            s_dx = s_dx + wdx;
            s_dx2 = s_dx2 + wdx2;
            s_dx3 = s_dx3 + wdx3;
            s_dx4 = s_dx4 + wdx3 * dx;
            s_dx5 = s_dx5 + wdx3 * dx2;
            s_dx6 = s_dx6 + wdx3 * dx3;

            s_wy = s_wy + w * y_val;
            s_wdx_y = s_wdx_y + wdx * y_val;
            s_wdx2_y = s_wdx2_y + wdx2 * y_val;
            s_wdx3_y = s_wdx3_y + wdx3 * y_val;
        }
    }

    // Matrix X'WX (symmetric)
    // Row 0: 1, x, x^2, x^3
    xtwx[0] = s_w;
    xtwx[1] = s_dx;
    xtwx[2] = s_dx2;
    xtwx[3] = s_dx3;
    // Row 1: x, x^2, x^3, x^4
    xtwx[4] = s_dx;
    xtwx[5] = s_dx2;
    xtwx[6] = s_dx3;
    xtwx[7] = s_dx4;
    // Row 2: x^2, x^3, x^4, x^5
    xtwx[8] = s_dx2;
    xtwx[9] = s_dx3;
    xtwx[10] = s_dx4;
    xtwx[11] = s_dx5;
    // Row 3: x^3, x^4, x^5, x^6
    xtwx[12] = s_dx3;
    xtwx[13] = s_dx4;
    xtwx[14] = s_dx5;
    xtwx[15] = s_dx6;

    xtwy[0] = s_wy;
    xtwy[1] = s_wdx_y;
    xtwy[2] = s_wdx2_y;
    xtwy[3] = s_wdx3_y;
}

/// Optimized accumulation for 1D Cubic Case using SIMD.
#[allow(clippy::too_many_arguments)]
pub fn accumulate_1d_cubic_simd(
    x: &[f64],
    y: &[f64],
    indices: &[usize],
    weights: &[f64],
    query: f64,
    xtwx: &mut [f64; 16],
    xtwy: &mut [f64; 4],
) {
    let n = indices.len();
    let mut i = 0;

    let mut s_w = f64x2::splat(0.0);
    let mut s_dx = f64x2::splat(0.0);
    let mut s_dx2 = f64x2::splat(0.0);
    let mut s_dx3 = f64x2::splat(0.0);
    let mut s_dx4 = f64x2::splat(0.0);
    let mut s_dx5 = f64x2::splat(0.0);
    let mut s_dx6 = f64x2::splat(0.0);

    let mut s_wy = f64x2::splat(0.0);
    let mut s_wdx_y = f64x2::splat(0.0);
    let mut s_wdx2_y = f64x2::splat(0.0);
    let mut s_wdx3_y = f64x2::splat(0.0);

    let q = f64x2::splat(query);

    // SAFETY: Use unchecked access for performance. Invariants guaranteed by KD-tree.
    unsafe {
        while i + 2 <= n {
            let idx0 = *indices.get_unchecked(i);
            let idx1 = *indices.get_unchecked(i + 1);

            let w = f64x2::new([*weights.get_unchecked(i), *weights.get_unchecked(i + 1)]);
            let x_val = f64x2::new([*x.get_unchecked(idx0), *x.get_unchecked(idx1)]);
            let y_val = f64x2::new([*y.get_unchecked(idx0), *y.get_unchecked(idx1)]);

            let dx = x_val - q;
            let dx2 = dx * dx;
            let dx3 = dx2 * dx;

            let wdx = w * dx;
            let wdx2 = w * dx2;
            let wdx3 = w * dx3;

            s_w += w;
            s_dx += wdx;
            s_dx2 += wdx2;
            s_dx3 += wdx3;
            s_dx4 += wdx3 * dx;
            s_dx5 += wdx3 * dx2;
            s_dx6 += wdx3 * dx3;

            s_wy += w * y_val;
            s_wdx_y += wdx * y_val;
            s_wdx2_y += wdx2 * y_val;
            s_wdx3_y += wdx3 * y_val;

            i += 2;
        }
    }

    let mut a_w = s_w.reduce_add();
    let mut a_dx = s_dx.reduce_add();
    let mut a_dx2 = s_dx2.reduce_add();
    let mut a_dx3 = s_dx3.reduce_add();
    let mut a_dx4 = s_dx4.reduce_add();
    let mut a_dx5 = s_dx5.reduce_add();
    let mut a_dx6 = s_dx6.reduce_add();

    let mut a_wy = s_wy.reduce_add();
    let mut a_wdx_y = s_wdx_y.reduce_add();
    let mut a_wdx2_y = s_wdx2_y.reduce_add();
    let mut a_wdx3_y = s_wdx3_y.reduce_add();

    // Tail
    unsafe {
        for k in i..n {
            let w = *weights.get_unchecked(k);
            if w <= f64::EPSILON {
                continue;
            }

            let idx = *indices.get_unchecked(k);
            let dx = *x.get_unchecked(idx) - query;
            let y_val = *y.get_unchecked(idx);
            let dx2 = dx * dx;
            let dx3 = dx2 * dx;

            let wdx = w * dx;
            let wdx2 = w * dx2;
            let wdx3 = w * dx3;

            a_w += w;
            a_dx += wdx;
            a_dx2 += wdx2;
            a_dx3 += wdx3;
            a_dx4 += wdx3 * dx;
            a_dx5 += wdx3 * dx2;
            a_dx6 += wdx3 * dx3;

            a_wy += w * y_val;
            a_wdx_y += wdx * y_val;
            a_wdx2_y += wdx2 * y_val;
            a_wdx3_y += wdx3 * y_val;
        }
    }

    xtwx[0] = a_w;
    xtwx[1] = a_dx;
    xtwx[2] = a_dx2;
    xtwx[3] = a_dx3;

    xtwx[4] = a_dx;
    xtwx[5] = a_dx2;
    xtwx[6] = a_dx3;
    xtwx[7] = a_dx4;

    xtwx[8] = a_dx2;
    xtwx[9] = a_dx3;
    xtwx[10] = a_dx4;
    xtwx[11] = a_dx5;

    xtwx[12] = a_dx3;
    xtwx[13] = a_dx4;
    xtwx[14] = a_dx5;
    xtwx[15] = a_dx6;

    xtwy[0] = a_wy;
    xtwy[1] = a_wdx_y;
    xtwy[2] = a_wdx2_y;
    xtwy[3] = a_wdx3_y;
}
