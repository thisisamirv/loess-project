//! Specialized 2D Accumulators
//!
//! ## Purpose
//!
//! This module provides optimized scalar and SIMD accumulation functions
//! for building Normal Equations matrices in 2D linear, quadratic,
//! and cubic regression.

// External dependencies
use num_traits::Float;
use wide::f64x2;

// ============================================================================
// Specialized Accumulation Functions
// ============================================================================

/// Optimized accumulation for 2D Linear Case (Scalar).
#[allow(clippy::too_many_arguments)]
pub fn accumulate_2d_linear_scalar<T: Float>(
    x: &[T],
    y: &[T],
    indices: &[usize],
    weights: &[T],
    query_x: T,
    query_y: T,
    xtwx: &mut [T; 9],
    xtwy: &mut [T; 3],
) {
    let n = indices.len();
    let mut s_w = T::zero();
    let mut s_dx = T::zero();
    let mut s_dy = T::zero();
    let mut s_dx2 = T::zero();
    let mut s_dy2 = T::zero();
    let mut s_dxdy = T::zero();
    let mut s_wy = T::zero();
    let mut s_wdxy = T::zero();
    let mut s_wdyy = T::zero();

    // SAFETY: Invariants guaranteed by KD-tree.
    unsafe {
        for i in 0..n {
            let w = *weights.get_unchecked(i);
            if w <= T::epsilon() {
                continue;
            }
            let idx = *indices.get_unchecked(i);
            let dx = *x.get_unchecked(idx * 2) - query_x;
            let dy = *x.get_unchecked(idx * 2 + 1) - query_y;
            let y_val = *y.get_unchecked(idx);
            let wdx = w * dx;
            let wdy = w * dy;
            s_w = s_w + w;
            s_dx = s_dx + wdx;
            s_dy = s_dy + wdy;
            s_dx2 = s_dx2 + wdx * dx;
            s_dy2 = s_dy2 + wdy * dy;
            s_dxdy = s_dxdy + wdx * dy;
            s_wy = s_wy + w * y_val;
            s_wdxy = s_wdxy + wdx * y_val;
            s_wdyy = s_wdyy + wdy * y_val;
        }
    }
    xtwx[0] = s_w;
    xtwx[1] = s_dx;
    xtwx[2] = s_dy;
    xtwx[3] = s_dx;
    xtwx[4] = s_dx2;
    xtwx[5] = s_dxdy;
    xtwx[6] = s_dy;
    xtwx[7] = s_dxdy;
    xtwx[8] = s_dy2;
    xtwy[0] = s_wy;
    xtwy[1] = s_wdxy;
    xtwy[2] = s_wdyy;
}

/// Optimized accumulation for 2D Linear Case using SIMD.
#[allow(clippy::too_many_arguments)]
pub fn accumulate_2d_linear_simd(
    x: &[f64],
    y: &[f64],
    indices: &[usize],
    weights: &[f64],
    query_x: f64,
    query_y: f64,
    xtwx: &mut [f64; 9],
    xtwy: &mut [f64; 3],
) {
    let n = indices.len();
    let mut i = 0;

    let mut s_w = f64x2::splat(0.0);
    let mut s_dx = f64x2::splat(0.0);
    let mut s_dy = f64x2::splat(0.0);
    let mut s_dx2 = f64x2::splat(0.0);
    let mut s_dy2 = f64x2::splat(0.0);
    let mut s_dxdy = f64x2::splat(0.0);
    let mut s_wy = f64x2::splat(0.0);
    let mut s_wdxy = f64x2::splat(0.0);
    let mut s_wdyy = f64x2::splat(0.0);

    let qx = f64x2::splat(query_x);
    let qy = f64x2::splat(query_y);

    // SAFETY: Use unchecked access for performance. Invariants guaranteed by KD-tree.
    unsafe {
        while i + 2 <= n {
            let idx0 = *indices.get_unchecked(i);
            let idx1 = *indices.get_unchecked(i + 1);

            let w = f64x2::new([*weights.get_unchecked(i), *weights.get_unchecked(i + 1)]);
            let px = f64x2::new([*x.get_unchecked(idx0 * 2), *x.get_unchecked(idx1 * 2)]);
            let py = f64x2::new([
                *x.get_unchecked(idx0 * 2 + 1),
                *x.get_unchecked(idx1 * 2 + 1),
            ]);
            let y_vals = f64x2::new([*y.get_unchecked(idx0), *y.get_unchecked(idx1)]);

            let dx = px - qx;
            let dy = py - qy;

            let wdx = w * dx;
            let wdy = w * dy;

            s_w += w;
            s_dx += wdx;
            s_dy += wdy;
            s_dx2 += wdx * dx;
            s_dy2 += wdy * dy;
            s_dxdy += wdx * dy;
            s_wy += w * y_vals;
            s_wdxy += wdx * y_vals;
            s_wdyy += wdy * y_vals;

            i += 2;
        }
    }

    let mut a_w = s_w.reduce_add();
    let mut a_dx = s_dx.reduce_add();
    let mut a_dy = s_dy.reduce_add();
    let mut a_dx2 = s_dx2.reduce_add();
    let mut a_dy2 = s_dy2.reduce_add();
    let mut a_dxdy = s_dxdy.reduce_add();
    let mut a_wy = s_wy.reduce_add();
    let mut a_wdxy = s_wdxy.reduce_add();
    let mut a_wdyy = s_wdyy.reduce_add();

    unsafe {
        for k in i..n {
            let w = *weights.get_unchecked(k);
            if w <= f64::EPSILON {
                continue;
            }

            let idx = *indices.get_unchecked(k);
            let dx = *x.get_unchecked(idx * 2) - query_x;
            let dy = *x.get_unchecked(idx * 2 + 1) - query_y;
            let y_val = *y.get_unchecked(idx);

            let wdx = w * dx;
            let wdy = w * dy;

            a_w += w;
            a_dx += wdx;
            a_dy += wdy;
            a_dx2 += wdx * dx;
            a_dy2 += wdy * dy;
            a_dxdy += wdx * dy;
            a_wy += w * y_val;
            a_wdxy += wdx * y_val;
            a_wdyy += wdy * y_val;
        }
    }

    xtwx[0] = a_w;
    xtwx[1] = a_dx;
    xtwx[2] = a_dy;
    xtwx[3] = a_dx;
    xtwx[4] = a_dx2;
    xtwx[5] = a_dxdy;
    xtwx[6] = a_dy;
    xtwx[7] = a_dxdy;
    xtwx[8] = a_dy2;

    xtwy[0] = a_wy;
    xtwy[1] = a_wdxy;
    xtwy[2] = a_wdyy;
}

/// Optimized accumulation for 2D Quadratic Case (Scalar).
#[allow(clippy::too_many_arguments)]
pub fn accumulate_2d_quadratic_scalar<T: Float>(
    x: &[T],
    y: &[T],
    indices: &[usize],
    weights: &[T],
    query_x: T,
    query_y: T,
    xtwx: &mut [T; 36],
    xtwy: &mut [T; 6],
) {
    let n = indices.len();
    let mut s_w = T::zero();
    let mut s_dx = T::zero();
    let mut s_dy = T::zero();
    let mut s_dx2 = T::zero();
    let mut s_dxdy = T::zero();
    let mut s_dy2 = T::zero();
    let mut s_dx3 = T::zero();
    let mut s_dx2dy = T::zero();
    let mut s_dxdy2 = T::zero();
    let mut s_dy3 = T::zero();
    let mut s_dx4 = T::zero();
    let mut s_dx3dy = T::zero();
    let mut s_dx2dy2 = T::zero();
    let mut s_dxdy3 = T::zero();
    let mut s_dy4 = T::zero();
    let mut s_wy = T::zero();
    let mut s_wyx = T::zero();
    let mut s_wyy = T::zero();
    let mut s_wyx2 = T::zero();
    let mut s_wyxy = T::zero();
    let mut s_wyy2 = T::zero();

    // SAFETY: Invariants guaranteed by KD-tree.
    unsafe {
        for i in 0..n {
            let w = *weights.get_unchecked(i);
            if w <= T::epsilon() {
                continue;
            }
            let idx = *indices.get_unchecked(i);
            let dx = *x.get_unchecked(idx * 2) - query_x;
            let dy = *x.get_unchecked(idx * 2 + 1) - query_y;
            let y_val = *y.get_unchecked(idx);
            let dx2 = dx * dx;
            let dy2 = dy * dy;
            let dxdy = dx * dy;
            let wdx = w * dx;
            let wdy = w * dy;
            let wdx2 = w * dx2;
            let wdy2 = w * dy2;
            let wdxdy = w * dxdy;

            s_w = s_w + w;
            s_dx = s_dx + wdx;
            s_dy = s_dy + wdy;
            s_dx2 = s_dx2 + wdx2;
            s_dxdy = s_dxdy + wdxdy;
            s_dy2 = s_dy2 + wdy2;
            s_dx3 = s_dx3 + wdx2 * dx;
            s_dx2dy = s_dx2dy + wdx2 * dy;
            s_dxdy2 = s_dxdy2 + wdy2 * dx;
            s_dy3 = s_dy3 + wdy2 * dy;
            s_dx4 = s_dx4 + wdx2 * dx2;
            s_dx3dy = s_dx3dy + wdx2 * dxdy;
            s_dx2dy2 = s_dx2dy2 + dx2 * wdy2;
            s_dxdy3 = s_dxdy3 + dy2 * wdxdy;
            s_dy4 = s_dy4 + wdy2 * dy2;
            s_wy = s_wy + w * y_val;
            s_wyx = s_wyx + wdx * y_val;
            s_wyy = s_wyy + wdy * y_val;
            s_wyx2 = s_wyx2 + wdx2 * y_val;
            s_wyxy = s_wyxy + wdxdy * y_val;
            s_wyy2 = s_wyy2 + wdy2 * y_val;
        }
    }

    // Terms: 1, x, y, x^2, xy, y^2
    // Row 0
    xtwx[0] = s_w;
    xtwx[1] = s_dx;
    xtwx[2] = s_dy;
    xtwx[3] = s_dx2;
    xtwx[4] = s_dxdy;
    xtwx[5] = s_dy2;
    // Row 1
    xtwx[6] = s_dx;
    xtwx[7] = s_dx2;
    xtwx[8] = s_dxdy;
    xtwx[9] = s_dx3;
    xtwx[10] = s_dx2dy;
    xtwx[11] = s_dxdy2;
    // Row 2
    xtwx[12] = s_dy;
    xtwx[13] = s_dxdy;
    xtwx[14] = s_dy2;
    xtwx[15] = s_dx2dy;
    xtwx[16] = s_dxdy2;
    xtwx[17] = s_dy3;
    // Row 3
    xtwx[18] = s_dx2;
    xtwx[19] = s_dx3;
    xtwx[20] = s_dx2dy;
    xtwx[21] = s_dx4;
    xtwx[22] = s_dx3dy;
    xtwx[23] = s_dx2dy2;
    // Row 4
    xtwx[24] = s_dxdy;
    xtwx[25] = s_dx2dy;
    xtwx[26] = s_dxdy2;
    xtwx[27] = s_dx3dy;
    xtwx[28] = s_dx2dy2;
    xtwx[29] = s_dxdy3;
    // Row 5
    xtwx[30] = s_dy2;
    xtwx[31] = s_dxdy2;
    xtwx[32] = s_dy3;
    xtwx[33] = s_dx2dy2;
    xtwx[34] = s_dxdy3;
    xtwx[35] = s_dy4;

    xtwy[0] = s_wy;
    xtwy[1] = s_wyx;
    xtwy[2] = s_wyy;
    xtwy[3] = s_wyx2;
    xtwy[4] = s_wyxy;
    xtwy[5] = s_wyy2;
}

/// Optimized accumulation for 2D Quadratic Case using SIMD.
#[allow(clippy::too_many_arguments)]
pub fn accumulate_2d_quadratic_simd(
    x: &[f64],
    y: &[f64],
    indices: &[usize],
    weights: &[f64],
    query_x: f64,
    query_y: f64,
    xtwx: &mut [f64; 36],
    xtwy: &mut [f64; 6],
) {
    let n = indices.len();
    let mut i = 0;

    // SIMD accumulators
    let mut s_w = f64x2::splat(0.0);
    let mut s_dx = f64x2::splat(0.0);
    let mut s_dy = f64x2::splat(0.0);
    let mut s_dx2 = f64x2::splat(0.0);
    let mut s_dxdy = f64x2::splat(0.0);
    let mut s_dy2 = f64x2::splat(0.0);
    let mut s_dx3 = f64x2::splat(0.0);
    let mut s_dx2dy = f64x2::splat(0.0);
    let mut s_dxdy2 = f64x2::splat(0.0);
    let mut s_dy3 = f64x2::splat(0.0);
    let mut s_dx4 = f64x2::splat(0.0);
    let mut s_dx3dy = f64x2::splat(0.0);
    let mut s_dx2dy2 = f64x2::splat(0.0);
    let mut s_dxdy3 = f64x2::splat(0.0);
    let mut s_dy4 = f64x2::splat(0.0);
    let mut s_wy = f64x2::splat(0.0);
    let mut s_wyx = f64x2::splat(0.0);
    let mut s_wyy = f64x2::splat(0.0);
    let mut s_wyx2 = f64x2::splat(0.0);
    let mut s_wyxy = f64x2::splat(0.0);
    let mut s_wyy2 = f64x2::splat(0.0);

    let qx = f64x2::splat(query_x);
    let qy = f64x2::splat(query_y);

    // Process 2 elements at a time (f64x2)
    // SAFETY: Use unchecked access for performance. Invariants guaranteed by KD-tree.
    unsafe {
        while i + 2 <= n {
            let idx0 = *indices.get_unchecked(i);
            let idx1 = *indices.get_unchecked(i + 1);

            // Gather weights
            let w = f64x2::new([*weights.get_unchecked(i), *weights.get_unchecked(i + 1)]);

            // Gather x coordinates (requires generic gather or manual construction)
            // Since we don't have direct gather for f64x2 from &[f64] with indices in 'wide' easily,
            // we construct manually.
            let px = f64x2::new([*x.get_unchecked(idx0 * 2), *x.get_unchecked(idx1 * 2)]);
            let py = f64x2::new([
                *x.get_unchecked(idx0 * 2 + 1),
                *x.get_unchecked(idx1 * 2 + 1),
            ]);
            let y_vals = f64x2::new([*y.get_unchecked(idx0), *y.get_unchecked(idx1)]);

            let dx = px - qx;
            let dy = py - qy;

            // Calculations
            let dx2 = dx * dx;
            let dy2 = dy * dy;
            let dxdy = dx * dy;

            let wdx = w * dx;
            let wdy = w * dy;
            let wdx2 = w * dx2;
            let wdy2 = w * dy2;
            let wdxdy = w * dxdy;

            s_w += w;
            s_dx += wdx;
            s_dy += wdy;
            s_dx2 += wdx2;
            s_dxdy += wdxdy;
            s_dy2 += wdy2;

            // Optimization: dx4 = dx2 * dx2

            s_dx3 += wdx2 * dx;
            s_dx2dy += wdx2 * dy;
            s_dxdy2 += wdy2 * dx;
            s_dy3 += wdy2 * dy;

            s_dx4 += wdx2 * dx2;
            s_dx3dy += wdx2 * dxdy; // w * dx^2 * dx * dy = w * dx^3 * dy
            s_dx2dy2 += wdy2 * dx2;
            s_dxdy3 += wdy2 * dxdy;
            s_dy4 += wdy2 * dy2;

            s_wy += w * y_vals;
            s_wyx += wdx * y_vals;
            s_wyy += wdy * y_vals;
            s_wyx2 += wdx2 * y_vals;
            s_wyxy += wdxdy * y_vals;
            s_wyy2 += wdy2 * y_vals;

            i += 2;
        }
    }

    // Reduce SIMD accumulators
    let mut a_w = s_w.reduce_add();
    let mut a_dx = s_dx.reduce_add();
    let mut a_dy = s_dy.reduce_add();
    let mut a_dx2 = s_dx2.reduce_add();
    let mut a_dxdy = s_dxdy.reduce_add();
    let mut a_dy2 = s_dy2.reduce_add();
    let mut a_dx3 = s_dx3.reduce_add();
    let mut a_dx2dy = s_dx2dy.reduce_add();
    let mut a_dxdy2 = s_dxdy2.reduce_add();
    let mut a_dy3 = s_dy3.reduce_add();
    let mut a_dx4 = s_dx4.reduce_add();
    let mut a_dx3dy = s_dx3dy.reduce_add();
    let mut a_dx2dy2 = s_dx2dy2.reduce_add();
    let mut a_dxdy3 = s_dxdy3.reduce_add();
    let mut a_dy4 = s_dy4.reduce_add();
    let mut a_wy = s_wy.reduce_add();
    let mut a_wyx = s_wyx.reduce_add();
    let mut a_wyy = s_wyy.reduce_add();
    let mut a_wyx2 = s_wyx2.reduce_add();
    let mut a_wyxy = s_wyxy.reduce_add();
    let mut a_wyy2 = s_wyy2.reduce_add();

    // Handle tail
    // Handle tail
    unsafe {
        for k in i..n {
            let w = *weights.get_unchecked(k);
            if w <= f64::EPSILON {
                continue;
            }

            let idx = *indices.get_unchecked(k);
            let dx = *x.get_unchecked(idx * 2) - query_x;
            let dy = *x.get_unchecked(idx * 2 + 1) - query_y;
            let y_val = *y.get_unchecked(idx);

            let dx2 = dx * dx;
            let dy2 = dy * dy;
            let dxdy = dx * dy;

            let wdx = w * dx;
            let wdy = w * dy;
            let wdx2 = w * dx2;
            let wdy2 = w * dy2;
            let wdxdy = w * dxdy;

            a_w += w;
            a_dx += wdx;
            a_dy += wdy;
            a_dx2 += wdx2;
            a_dxdy += wdxdy;
            a_dy2 += wdy2;

            a_dx3 += wdx2 * dx;
            a_dx2dy += wdx2 * dy;
            a_dxdy2 += wdy2 * dx;
            a_dy3 += wdy2 * dy;

            a_dx4 += wdx2 * dx2;
            a_dx3dy += wdx2 * dxdy;
            a_dx2dy2 += wdy2 * dx2; // Corrected: dx^2 * dy^2
            a_dxdy3 += wdy2 * dxdy;
            a_dy4 += wdy2 * dy2;

            a_wy += w * y_val;
            a_wyx += wdx * y_val;
            a_wyy += wdy * y_val;
            a_wyx2 += wdx2 * y_val;
            a_wyxy += wdxdy * y_val;
            a_wyy2 += wdy2 * y_val;
        }
    }

    // Write to output buffers
    xtwx[0] = a_w;
    xtwx[1] = a_dx;
    xtwx[2] = a_dy;
    xtwx[3] = a_dx2;
    xtwx[4] = a_dxdy;
    xtwx[5] = a_dy2;

    xtwx[6] = a_dx;
    xtwx[7] = a_dx2;
    xtwx[8] = a_dxdy;
    xtwx[9] = a_dx3;
    xtwx[10] = a_dx2dy;
    xtwx[11] = a_dxdy2;

    xtwx[12] = a_dy;
    xtwx[13] = a_dxdy;
    xtwx[14] = a_dy2;
    xtwx[15] = a_dx2dy;
    xtwx[16] = a_dxdy2;
    xtwx[17] = a_dy3;

    xtwx[18] = a_dx2;
    xtwx[19] = a_dx3;
    xtwx[20] = a_dx2dy;
    xtwx[21] = a_dx4;
    xtwx[22] = a_dx3dy;
    xtwx[23] = a_dx2dy2;

    xtwx[24] = a_dxdy;
    xtwx[25] = a_dx2dy;
    xtwx[26] = a_dxdy2;
    xtwx[27] = a_dx3dy;
    xtwx[28] = a_dx2dy2;
    xtwx[29] = a_dxdy3;

    xtwx[30] = a_dy2;
    xtwx[31] = a_dxdy2;
    xtwx[32] = a_dy3;
    xtwx[33] = a_dx2dy2;
    xtwx[34] = a_dxdy3;
    xtwx[35] = a_dy4;

    xtwy[0] = a_wy;
    xtwy[1] = a_wyx;
    xtwy[2] = a_wyy;
    xtwy[3] = a_wyx2;
    xtwy[4] = a_wyxy;
    xtwy[5] = a_wyy2;
}

/// Optimized accumulation for 2D Cubic Case (Scalar).
///
/// Matrix size 10x10.
/// Terms: 1, x, y, x2, xy, y2, x3, x2y, xy2, y3
#[allow(clippy::too_many_arguments)]
#[allow(clippy::cognitive_complexity)]
pub fn accumulate_2d_cubic_scalar<T: Float>(
    x: &[T],
    y: &[T],
    indices: &[usize],
    weights: &[T],
    query_x: T,
    query_y: T,
    xtwx: &mut [T; 100],
    xtwy: &mut [T; 10],
) {
    let n = indices.len();

    // Sums
    let mut s_w = T::zero();

    // Order 1
    let mut s_x = T::zero();
    let mut s_y = T::zero();

    // Order 2
    let mut s_x2 = T::zero();
    let mut s_xy = T::zero();
    let mut s_y2 = T::zero();

    // Order 3
    let mut s_x3 = T::zero();
    let mut s_x2y = T::zero();
    let mut s_xy2 = T::zero();
    let mut s_y3 = T::zero();

    // Order 4
    let mut s_x4 = T::zero();
    let mut s_x3y = T::zero();
    let mut s_x2y2 = T::zero();
    let mut s_xy3 = T::zero();
    let mut s_y4 = T::zero();

    // Order 5
    let mut s_x5 = T::zero();
    let mut s_x4y = T::zero();
    let mut s_x3y2 = T::zero();
    let mut s_x2y3 = T::zero();
    let mut s_xy4 = T::zero();
    let mut s_y5 = T::zero();

    // Order 6
    let mut s_x6 = T::zero();
    let mut s_x5y = T::zero();
    let mut s_x4y2 = T::zero();
    let mut s_x3y3 = T::zero();
    let mut s_x2y4 = T::zero();
    let mut s_xy5 = T::zero();
    let mut s_y6 = T::zero();

    // RHS sums
    let mut s_wy = T::zero();
    // Order 1 * y
    let mut s_wy_x = T::zero();
    let mut s_wy_y = T::zero();
    // Order 2 * y
    let mut s_wy_x2 = T::zero();
    let mut s_wy_xy = T::zero();
    let mut s_wy_y2 = T::zero();
    // Order 3 * y
    let mut s_wy_x3 = T::zero();
    let mut s_wy_x2y = T::zero();
    let mut s_wy_xy2 = T::zero();
    let mut s_wy_y3 = T::zero();

    // SAFETY: Invariants guaranteed by KD-tree.
    unsafe {
        for i in 0..n {
            let w = *weights.get_unchecked(i);
            if w <= T::epsilon() {
                continue;
            }
            let idx = *indices.get_unchecked(i);
            let dx = *x.get_unchecked(idx * 2) - query_x;
            let dy = *x.get_unchecked(idx * 2 + 1) - query_y;
            let y_val = *y.get_unchecked(idx);

            // Powers
            let dx2 = dx * dx;
            let dy2 = dy * dy;
            let dxy = dx * dy;

            let dx3 = dx2 * dx;
            let dy3 = dy2 * dy;
            let dx2y = dx2 * dy;
            let dxy2 = dx * dy2;

            let wdx = w * dx;
            let wdy = w * dy;

            let wdx2 = w * dx2;
            let wdxy = w * dxy;
            let wdy2 = w * dy2;

            // Accumulate Weighted Powers

            // Order 0
            s_w = s_w + w;

            // Order 1
            s_x = s_x + wdx;
            s_y = s_y + wdy;

            // Order 2
            s_x2 = s_x2 + wdx2;
            s_xy = s_xy + wdxy;
            s_y2 = s_y2 + wdy2;

            // Order 3
            let wdx3 = w * dx3;
            let wdx2y = w * dx2y;
            let wdxy2 = w * dxy2;
            let wdy3 = w * dy3;

            s_x3 = s_x3 + wdx3;
            s_x2y = s_x2y + wdx2y;
            s_xy2 = s_xy2 + wdxy2;
            s_y3 = s_y3 + wdy3;

            // Order 4
            s_x4 = s_x4 + wdx3 * dx;
            s_x3y = s_x3y + wdx3 * dy;
            s_x2y2 = s_x2y2 + wdx2y * dy;
            s_xy3 = s_xy3 + wdxy2 * dy;
            s_y4 = s_y4 + wdy3 * dy;

            // Order 5
            s_x5 = s_x5 + wdx3 * dx2;
            s_x4y = s_x4y + wdx3 * dxy;
            s_x3y2 = s_x3y2 + wdx3 * dy2;
            s_x2y3 = s_x2y3 + wdx2y * dy2;
            s_xy4 = s_xy4 + wdxy2 * dy2;
            s_y5 = s_y5 + wdy3 * dy2;

            // Order 6
            s_x6 = s_x6 + wdx3 * dx3;
            s_x5y = s_x5y + wdx3 * dx2y;
            s_x4y2 = s_x4y2 + wdx3 * dxy2;
            s_x3y3 = s_x3y3 + wdx3 * dy3;
            s_x2y4 = s_x2y4 + wdx2y * dy3;
            s_xy5 = s_xy5 + wdxy2 * dy3;
            s_y6 = s_y6 + wdy3 * dy3;

            // RHS
            let wy = w * y_val;
            s_wy = s_wy + wy;
            s_wy_x = s_wy_x + wdx * y_val;
            s_wy_y = s_wy_y + wdy * y_val;
            s_wy_x2 = s_wy_x2 + wdx2 * y_val;
            s_wy_xy = s_wy_xy + wdxy * y_val;
            s_wy_y2 = s_wy_y2 + wdy2 * y_val;
            s_wy_x3 = s_wy_x3 + wdx3 * y_val;
            s_wy_x2y = s_wy_x2y + wdx2y * y_val;
            s_wy_xy2 = s_wy_xy2 + wdxy2 * y_val;
            s_wy_y3 = s_wy_y3 + wdy3 * y_val;
        }
    }

    // Fill Matrix
    // Mapping:
    // 0: 1
    // 1: x
    // 2: y
    // 3: x2
    // 4: xy
    // 5: y2
    // 6: x3
    // 7: x2y
    // 8: xy2
    // 9: y3

    // Row 0 (1): 1, x, y, x2, xy, y2, x3, x2y, xy2, y3
    xtwx[0] = s_w;
    xtwx[1] = s_x;
    xtwx[2] = s_y;
    xtwx[3] = s_x2;
    xtwx[4] = s_xy;
    xtwx[5] = s_y2;
    xtwx[6] = s_x3;
    xtwx[7] = s_x2y;
    xtwx[8] = s_xy2;
    xtwx[9] = s_y3;

    // Row 1 (x): x, x2, xy, x3, x2y, xy2, x4, x3y, x2y2, xy3
    xtwx[10] = s_x;
    xtwx[11] = s_x2;
    xtwx[12] = s_xy;
    xtwx[13] = s_x3;
    xtwx[14] = s_x2y;
    xtwx[15] = s_xy2;
    xtwx[16] = s_x4;
    xtwx[17] = s_x3y;
    xtwx[18] = s_x2y2;
    xtwx[19] = s_xy3;

    // Row 2 (y): y, xy, y2, x2y, xy2, y3, x3y, x2y2, xy3, y4
    xtwx[20] = s_y;
    xtwx[21] = s_xy;
    xtwx[22] = s_y2;
    xtwx[23] = s_x2y;
    xtwx[24] = s_xy2;
    xtwx[25] = s_y3;
    xtwx[26] = s_x3y;
    xtwx[27] = s_x2y2;
    xtwx[28] = s_xy3;
    xtwx[29] = s_y4;

    // Row 3 (x2): x2, x3, x2y, x4, x3y, x2y2, x5, x4y, x3y2, x2y3
    xtwx[30] = s_x2;
    xtwx[31] = s_x3;
    xtwx[32] = s_x2y;
    xtwx[33] = s_x4;
    xtwx[34] = s_x3y;
    xtwx[35] = s_x2y2;
    xtwx[36] = s_x5;
    xtwx[37] = s_x4y;
    xtwx[38] = s_x3y2;
    xtwx[39] = s_x2y3;

    // Row 4 (xy): xy, x2y, xy2, x3y, x2y2, xy3, x4y, x3y2, x2y3, xy4
    xtwx[40] = s_xy;
    xtwx[41] = s_x2y;
    xtwx[42] = s_xy2;
    xtwx[43] = s_x3y;
    xtwx[44] = s_x2y2;
    xtwx[45] = s_xy3;
    xtwx[46] = s_x4y;
    xtwx[47] = s_x3y2;
    xtwx[48] = s_x2y3;
    xtwx[49] = s_xy4;

    // Row 5 (y2): y2, xy2, y3, x2y2, xy3, y4, x3y2, x2y3, xy4, y5
    xtwx[50] = s_y2;
    xtwx[51] = s_xy2;
    xtwx[52] = s_y3;
    xtwx[53] = s_x2y2;
    xtwx[54] = s_xy3;
    xtwx[55] = s_y4;
    xtwx[56] = s_x3y2;
    xtwx[57] = s_x2y3;
    xtwx[58] = s_xy4;
    xtwx[59] = s_y5;

    // Row 6 (x3): x3, x4, x3y, x5, x4y, x3y2, x6, x5y, x4y2, x3y3
    xtwx[60] = s_x3;
    xtwx[61] = s_x4;
    xtwx[62] = s_x3y;
    xtwx[63] = s_x5;
    xtwx[64] = s_x4y;
    xtwx[65] = s_x3y2;
    xtwx[66] = s_x6;
    xtwx[67] = s_x5y;
    xtwx[68] = s_x4y2;
    xtwx[69] = s_x3y3;

    // Row 7 (x2y): x2y, x3y, x2y2, x4y, x3y2, x2y3, x5y, x4y2, x3y3, x2y4
    xtwx[70] = s_x2y;
    xtwx[71] = s_x3y;
    xtwx[72] = s_x2y2;
    xtwx[73] = s_x4y;
    xtwx[74] = s_x3y2;
    xtwx[75] = s_x2y3;
    xtwx[76] = s_x5y;
    xtwx[77] = s_x4y2;
    xtwx[78] = s_x3y3;
    xtwx[79] = s_x2y4;

    // Row 8 (xy2): xy2, x2y2, xy3, x3y2, x2y3, xy4, x4y2, x3y3, x2y4, xy5
    xtwx[80] = s_xy2;
    xtwx[81] = s_x2y2;
    xtwx[82] = s_xy3;
    xtwx[83] = s_x3y2;
    xtwx[84] = s_x2y3;
    xtwx[85] = s_xy4;
    xtwx[86] = s_x4y2;
    xtwx[87] = s_x3y3;
    xtwx[88] = s_x2y4;
    xtwx[89] = s_xy5;

    // Row 9 (y3): y3, xy3, y4, x2y3, xy4, y5, x3y3, x2y4, xy5, y6
    xtwx[90] = s_y3;
    xtwx[91] = s_xy3;
    xtwx[92] = s_y4;
    xtwx[93] = s_x2y3;
    xtwx[94] = s_xy4;
    xtwx[95] = s_y5;
    xtwx[96] = s_x3y3;
    xtwx[97] = s_x2y4;
    xtwx[98] = s_xy5;
    xtwx[99] = s_y6;

    // RHS (10)
    xtwy[0] = s_wy;
    xtwy[1] = s_wy_x;
    xtwy[2] = s_wy_y;
    xtwy[3] = s_wy_x2;
    xtwy[4] = s_wy_xy;
    xtwy[5] = s_wy_y2;
    xtwy[6] = s_wy_x3;
    xtwy[7] = s_wy_x2y;
    xtwy[8] = s_wy_xy2;
    xtwy[9] = s_wy_y3;
}

/// Optimized accumulation for 2D Cubic Case using SIMD.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::cognitive_complexity)]
pub fn accumulate_2d_cubic_simd(
    x: &[f64],
    y: &[f64],
    indices: &[usize],
    weights: &[f64],
    query_x: f64,
    query_y: f64,
    xtwx: &mut [f64; 100],
    xtwy: &mut [f64; 10],
) {
    let n = indices.len();
    let mut i = 0;

    // Scalar sums for fallback / reduction
    // Using SIMD accumulators
    let mut s_w = f64x2::splat(0.0);
    // 1
    let mut s_x = f64x2::splat(0.0);
    let mut s_y = f64x2::splat(0.0);
    // 2
    let mut s_x2 = f64x2::splat(0.0);
    let mut s_xy = f64x2::splat(0.0);
    let mut s_y2 = f64x2::splat(0.0);
    // 3
    let mut s_x3 = f64x2::splat(0.0);
    let mut s_x2y = f64x2::splat(0.0);
    let mut s_xy2 = f64x2::splat(0.0);
    let mut s_y3 = f64x2::splat(0.0);
    // 4
    let mut s_x4 = f64x2::splat(0.0);
    let mut s_x3y = f64x2::splat(0.0);
    let mut s_x2y2 = f64x2::splat(0.0);
    let mut s_xy3 = f64x2::splat(0.0);
    let mut s_y4 = f64x2::splat(0.0);
    // 5
    let mut s_x5 = f64x2::splat(0.0);
    let mut s_x4y = f64x2::splat(0.0);
    let mut s_x3y2 = f64x2::splat(0.0);
    let mut s_x2y3 = f64x2::splat(0.0);
    let mut s_xy4 = f64x2::splat(0.0);
    let mut s_y5 = f64x2::splat(0.0);
    // 6
    let mut s_x6 = f64x2::splat(0.0);
    let mut s_x5y = f64x2::splat(0.0);
    let mut s_x4y2 = f64x2::splat(0.0);
    let mut s_x3y3 = f64x2::splat(0.0);
    let mut s_x2y4 = f64x2::splat(0.0);
    let mut s_xy5 = f64x2::splat(0.0);
    let mut s_y6 = f64x2::splat(0.0);

    // RHS
    let mut s_wy = f64x2::splat(0.0);
    let mut s_wy_x = f64x2::splat(0.0);
    let mut s_wy_y = f64x2::splat(0.0);
    let mut s_wy_x2 = f64x2::splat(0.0);
    let mut s_wy_xy = f64x2::splat(0.0);
    let mut s_wy_y2 = f64x2::splat(0.0);
    let mut s_wy_x3 = f64x2::splat(0.0);
    let mut s_wy_x2y = f64x2::splat(0.0);
    let mut s_wy_xy2 = f64x2::splat(0.0);
    let mut s_wy_y3 = f64x2::splat(0.0);

    let qx = f64x2::splat(query_x);
    let qy = f64x2::splat(query_y);

    // SAFETY: Use unchecked access for performance. Invariants guaranteed by KD-tree.
    unsafe {
        while i + 2 <= n {
            let idx0 = *indices.get_unchecked(i);
            let idx1 = *indices.get_unchecked(i + 1);

            let w = f64x2::new([*weights.get_unchecked(i), *weights.get_unchecked(i + 1)]);
            let px = f64x2::new([*x.get_unchecked(idx0 * 2), *x.get_unchecked(idx1 * 2)]);
            let py = f64x2::new([
                *x.get_unchecked(idx0 * 2 + 1),
                *x.get_unchecked(idx1 * 2 + 1),
            ]);
            let y_vals = f64x2::new([*y.get_unchecked(idx0), *y.get_unchecked(idx1)]);

            let dx = px - qx;
            let dy = py - qy;

            let dx2 = dx * dx;
            let dy2 = dy * dy;
            let dxy = dx * dy;

            let dx3 = dx2 * dx;
            let dy3 = dy2 * dy;
            let dx2y = dx2 * dy;
            let dxy2 = dx * dy2;

            let wdx = w * dx;
            let wdy = w * dy;

            let wdx2 = w * dx2;
            let wdxy = w * dxy;
            let wdy2 = w * dy2;

            let wdx3 = w * dx3;
            let wdx2y = w * dx2y;
            let wdxy2 = w * dxy2;
            let wdy3 = w * dy3;

            s_w += w;

            s_x += wdx;
            s_y += wdy;

            s_x2 += wdx2;
            s_xy += wdxy;
            s_y2 += wdy2;

            s_x3 += wdx3;
            s_x2y += wdx2y;
            s_xy2 += wdxy2;
            s_y3 += wdy3;

            s_x4 += wdx3 * dx;
            s_x3y += wdx3 * dy;
            s_x2y2 += wdx2y * dy;
            s_xy3 += wdxy2 * dy;
            s_y4 += wdy3 * dy;

            s_x5 += wdx3 * dx2;
            s_x4y += wdx3 * dxy;
            s_x3y2 += wdx3 * dy2;
            s_x2y3 += wdx2y * dy2;
            s_xy4 += wdxy2 * dy2;
            s_y5 += wdy3 * dy2;

            s_x6 += wdx3 * dx3;
            s_x5y += wdx3 * dx2y;
            s_x4y2 += wdx3 * dxy2;
            s_x3y3 += wdx3 * dy3;
            s_x2y4 += wdx2y * dy3;
            s_xy5 += wdxy2 * dy3;
            s_y6 += wdy3 * dy3;

            let r_y = w * y_vals;
            s_wy += r_y;
            s_wy_x += wdx * y_vals;
            s_wy_y += wdy * y_vals;
            s_wy_x2 += wdx2 * y_vals;
            s_wy_xy += wdxy * y_vals;
            s_wy_y2 += wdy2 * y_vals;
            s_wy_x3 += wdx3 * y_vals;
            s_wy_x2y += wdx2y * y_vals;
            s_wy_xy2 += wdxy2 * y_vals;
            s_wy_y3 += wdy3 * y_vals;

            i += 2;
        }
    }

    // Reduce
    let mut a_w = s_w.reduce_add();
    let mut a_x = s_x.reduce_add();
    let mut a_y = s_y.reduce_add();
    let mut a_x2 = s_x2.reduce_add();
    let mut a_xy = s_xy.reduce_add();
    let mut a_y2 = s_y2.reduce_add();
    let mut a_x3 = s_x3.reduce_add();
    let mut a_x2y = s_x2y.reduce_add();
    let mut a_xy2 = s_xy2.reduce_add();
    let mut a_y3 = s_y3.reduce_add();
    let mut a_x4 = s_x4.reduce_add();
    let mut a_x3y = s_x3y.reduce_add();
    let mut a_x2y2 = s_x2y2.reduce_add();
    let mut a_xy3 = s_xy3.reduce_add();
    let mut a_y4 = s_y4.reduce_add();
    let mut a_x5 = s_x5.reduce_add();
    let mut a_x4y = s_x4y.reduce_add();
    let mut a_x3y2 = s_x3y2.reduce_add();
    let mut a_x2y3 = s_x2y3.reduce_add();
    let mut a_xy4 = s_xy4.reduce_add();
    let mut a_y5 = s_y5.reduce_add();
    let mut a_x6 = s_x6.reduce_add();
    let mut a_x5y = s_x5y.reduce_add();
    let mut a_x4y2 = s_x4y2.reduce_add();
    let mut a_x3y3 = s_x3y3.reduce_add();
    let mut a_x2y4 = s_x2y4.reduce_add();
    let mut a_xy5 = s_xy5.reduce_add();
    let mut a_y6 = s_y6.reduce_add();

    let mut a_wy = s_wy.reduce_add();
    let mut a_wy_x = s_wy_x.reduce_add();
    let mut a_wy_y = s_wy_y.reduce_add();
    let mut a_wy_x2 = s_wy_x2.reduce_add();
    let mut a_wy_xy = s_wy_xy.reduce_add();
    let mut a_wy_y2 = s_wy_y2.reduce_add();
    let mut a_wy_x3 = s_wy_x3.reduce_add();
    let mut a_wy_x2y = s_wy_x2y.reduce_add();
    let mut a_wy_xy2 = s_wy_xy2.reduce_add();
    let mut a_wy_y3 = s_wy_y3.reduce_add();

    // Tail
    // Tail
    unsafe {
        for k in i..n {
            let w = *weights.get_unchecked(k);
            if w <= f64::EPSILON {
                continue;
            }

            let idx = *indices.get_unchecked(k);
            let dx = *x.get_unchecked(idx * 2) - query_x;
            let dy = *x.get_unchecked(idx * 2 + 1) - query_y;
            let y_val = *y.get_unchecked(idx);

            let dx2 = dx * dx;
            let dy2 = dy * dy;
            let dxy = dx * dy;

            let dx3 = dx2 * dx;
            let dy3 = dy2 * dy;
            let dx2y = dx2 * dy;
            let dxy2 = dx * dy2;

            let wdx = w * dx;
            let wdy = w * dy;

            let wdx2 = w * dx2;
            let wdxy = w * dxy;
            let wdy2 = w * dy2;

            let wdx3 = w * dx3;
            let wdx2y = w * dx2y;
            let wdxy2 = w * dxy2;
            let wdy3 = w * dy3;

            a_w += w;
            a_x += wdx;
            a_y += wdy;
            a_x2 += wdx2;
            a_xy += wdxy;
            a_y2 += wdy2;
            a_x3 += wdx3;
            a_x2y += wdx2y;
            a_xy2 += wdxy2;
            a_y3 += wdy3;

            a_x4 += wdx3 * dx;
            a_x3y += wdx3 * dy;
            a_x2y2 += wdx2y * dy;
            a_xy3 += wdxy2 * dy;
            a_y4 += wdy3 * dy;

            a_x5 += wdx3 * dx2;
            a_x4y += wdx3 * dxy;
            a_x3y2 += wdx3 * dy2;
            a_x2y3 += wdx2y * dy2;
            a_xy4 += wdxy2 * dy2;
            a_y5 += wdy3 * dy2;

            a_x6 += wdx3 * dx3;
            a_x5y += wdx3 * dx2y;
            a_x4y2 += wdx3 * dxy2;
            a_x3y3 += wdx3 * dy3;
            a_x2y4 += wdx2y * dy3;
            a_xy5 += wdxy2 * dy3;
            a_y6 += wdy3 * dy3;

            let wy = w * y_val;
            a_wy += wy;
            a_wy_x += wdx * y_val;
            a_wy_y += wdy * y_val;
            a_wy_x2 += wdx2 * y_val;
            a_wy_xy += wdxy * y_val;
            a_wy_y2 += wdy2 * y_val;
            a_wy_x3 += wdx3 * y_val;
            a_wy_x2y += wdx2y * y_val;
            a_wy_xy2 += wdxy2 * y_val;
            a_wy_y3 += wdy3 * y_val;
        }
    }

    xtwx[0] = a_w;
    xtwx[1] = a_x;
    xtwx[2] = a_y;
    xtwx[3] = a_x2;
    xtwx[4] = a_xy;
    xtwx[5] = a_y2;
    xtwx[6] = a_x3;
    xtwx[7] = a_x2y;
    xtwx[8] = a_xy2;
    xtwx[9] = a_y3;

    xtwx[10] = a_x;
    xtwx[11] = a_x2;
    xtwx[12] = a_xy;
    xtwx[13] = a_x3;
    xtwx[14] = a_x2y;
    xtwx[15] = a_xy2;
    xtwx[16] = a_x4;
    xtwx[17] = a_x3y;
    xtwx[18] = a_x2y2;
    xtwx[19] = a_xy3;

    xtwx[20] = a_y;
    xtwx[21] = a_xy;
    xtwx[22] = a_y2;
    xtwx[23] = a_x2y;
    xtwx[24] = a_xy2;
    xtwx[25] = a_y3;
    xtwx[26] = a_x3y;
    xtwx[27] = a_x2y2;
    xtwx[28] = a_xy3;
    xtwx[29] = a_y4;

    xtwx[30] = a_x2;
    xtwx[31] = a_x3;
    xtwx[32] = a_x2y;
    xtwx[33] = a_x4;
    xtwx[34] = a_x3y;
    xtwx[35] = a_x2y2;
    xtwx[36] = a_x5;
    xtwx[37] = a_x4y;
    xtwx[38] = a_x3y2;
    xtwx[39] = a_x2y3;

    xtwx[40] = a_xy;
    xtwx[41] = a_x2y;
    xtwx[42] = a_xy2;
    xtwx[43] = a_x3y;
    xtwx[44] = a_x2y2;
    xtwx[45] = a_xy3;
    xtwx[46] = a_x4y;
    xtwx[47] = a_x3y2;
    xtwx[48] = a_x2y3;
    xtwx[49] = a_xy4;

    xtwx[50] = a_y2;
    xtwx[51] = a_xy2;
    xtwx[52] = a_y3;
    xtwx[53] = a_x2y2;
    xtwx[54] = a_xy3;
    xtwx[55] = a_y4;
    xtwx[56] = a_x3y2;
    xtwx[57] = a_x2y3;
    xtwx[58] = a_xy4;
    xtwx[59] = a_y5;

    xtwx[60] = a_x3;
    xtwx[61] = a_x4;
    xtwx[62] = a_x3y;
    xtwx[63] = a_x5;
    xtwx[64] = a_x4y;
    xtwx[65] = a_x3y2;
    xtwx[66] = a_x6;
    xtwx[67] = a_x5y;
    xtwx[68] = a_x4y2;
    xtwx[69] = a_x3y3;

    xtwx[70] = a_x2y;
    xtwx[71] = a_x3y;
    xtwx[72] = a_x2y2;
    xtwx[73] = a_x4y;
    xtwx[74] = a_x3y2;
    xtwx[75] = a_x2y3;
    xtwx[76] = a_x5y;
    xtwx[77] = a_x4y2;
    xtwx[78] = a_x3y3;
    xtwx[79] = a_x2y4;

    xtwx[80] = a_xy2;
    xtwx[81] = a_x2y2;
    xtwx[82] = a_xy3;
    xtwx[83] = a_x3y2;
    xtwx[84] = a_x2y3;
    xtwx[85] = a_xy4;
    xtwx[86] = a_x4y2;
    xtwx[87] = a_x3y3;
    xtwx[88] = a_x2y4;
    xtwx[89] = a_xy5;

    xtwx[90] = a_y3;
    xtwx[91] = a_xy3;
    xtwx[92] = a_y4;
    xtwx[93] = a_x2y3;
    xtwx[94] = a_xy4;
    xtwx[95] = a_y5;
    xtwx[96] = a_x3y3;
    xtwx[97] = a_x2y4;
    xtwx[98] = a_xy5;
    xtwx[99] = a_y6;

    xtwy[0] = a_wy;
    xtwy[1] = a_wy_x;
    xtwy[2] = a_wy_y;
    xtwy[3] = a_wy_x2;
    xtwy[4] = a_wy_xy;
    xtwy[5] = a_wy_y2;
    xtwy[6] = a_wy_x3;
    xtwy[7] = a_wy_x2y;
    xtwy[8] = a_wy_xy2;
    xtwy[9] = a_wy_y3;
}
