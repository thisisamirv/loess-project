//! Specialized 3D Accumulators
//!
//! ## Purpose
//!
//! This module provides optimized scalar and SIMD accumulation functions for building Normal Equations matrices in 3D linear, quadratic, and cubic regression.

// External dependencies
use num_traits::Float;
use wide::f64x2;

// ============================================================================
// Specialized Accumulation Functions
// ============================================================================

/// Optimized accumulation for 3D Linear Case (Scalar).
#[allow(clippy::too_many_arguments)]
pub fn accumulate_3d_linear_scalar<T: Float>(
    x: &[T],
    y: &[T],
    indices: &[usize],
    weights: &[T],
    query_x: T,
    query_y: T,
    query_z: T,
    xtwx: &mut [T; 16],
    xtwy: &mut [T; 4],
) {
    let n = indices.len();
    let mut s_w = T::zero();
    let mut s_x = T::zero();
    let mut s_y = T::zero();
    let mut s_z = T::zero();
    let mut s_x2 = T::zero();
    let mut s_y2 = T::zero();
    let mut s_z2 = T::zero();
    let mut s_xy = T::zero();
    let mut s_xz = T::zero();
    let mut s_yz = T::zero();
    let mut s_wy = T::zero();
    let mut s_wxy = T::zero();
    let mut s_wyy = T::zero();
    let mut s_wzy = T::zero();

    // SAFETY: Invariants guaranteed by KD-tree.
    unsafe {
        for i in 0..n {
            let w = *weights.get_unchecked(i);
            if w <= T::epsilon() {
                continue;
            }
            let idx = *indices.get_unchecked(i);
            let dx = *x.get_unchecked(idx * 3) - query_x;
            let dy = *x.get_unchecked(idx * 3 + 1) - query_y;
            let dz = *x.get_unchecked(idx * 3 + 2) - query_z;
            let val_y = *y.get_unchecked(idx);

            let wdx = w * dx;
            let wdy = w * dy;
            let wdz = w * dz;

            s_w = s_w + w;
            s_x = s_x + wdx;
            s_y = s_y + wdy;
            s_z = s_z + wdz;
            s_x2 = s_x2 + wdx * dx;
            s_y2 = s_y2 + wdy * dy;
            s_z2 = s_z2 + wdz * dz;
            s_xy = s_xy + wdx * dy;
            s_xz = s_xz + wdx * dz;
            s_yz = s_yz + wdy * dz;

            s_wy = s_wy + w * val_y;
            s_wxy = s_wxy + wdx * val_y;
            s_wyy = s_wyy + wdy * val_y;
            s_wzy = s_wzy + wdz * val_y;
        }
    }

    xtwx[0] = s_w;
    xtwx[1] = s_x;
    xtwx[2] = s_y;
    xtwx[3] = s_z;
    xtwx[4] = s_x;
    xtwx[5] = s_x2;
    xtwx[6] = s_xy;
    xtwx[7] = s_xz;
    xtwx[8] = s_y;
    xtwx[9] = s_xy;
    xtwx[10] = s_y2;
    xtwx[11] = s_yz;
    xtwx[12] = s_z;
    xtwx[13] = s_xz;
    xtwx[14] = s_yz;
    xtwx[15] = s_z2;

    xtwy[0] = s_wy;
    xtwy[1] = s_wxy;
    xtwy[2] = s_wyy;
    xtwy[3] = s_wzy;
}

/// Optimized accumulation for 3D Linear Case using SIMD.
#[allow(clippy::too_many_arguments)]
pub fn accumulate_3d_linear_simd(
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
    let n = indices.len();

    let mut s_w = f64x2::splat(0.0);
    let mut s_x = f64x2::splat(0.0);
    let mut s_y = f64x2::splat(0.0);
    let mut s_z = f64x2::splat(0.0);
    let mut s_x2 = f64x2::splat(0.0);
    let mut s_y2 = f64x2::splat(0.0);
    let mut s_z2 = f64x2::splat(0.0);
    let mut s_xy = f64x2::splat(0.0);
    let mut s_xz = f64x2::splat(0.0);
    let mut s_yz = f64x2::splat(0.0);
    let mut s_wy = f64x2::splat(0.0);
    let mut s_wxy = f64x2::splat(0.0);
    let mut s_wyy = f64x2::splat(0.0);
    let mut s_wzy = f64x2::splat(0.0);

    let qx = f64x2::splat(query_x);
    let qy = f64x2::splat(query_y);
    let qz = f64x2::splat(query_z);

    let mut i = 0;
    // SAFETY: Use unchecked access for performance. Invariants guaranteed by KD-tree.
    unsafe {
        while i + 2 <= n {
            let idx0 = *indices.get_unchecked(i);
            let idx1 = *indices.get_unchecked(i + 1);

            let w = f64x2::new([*weights.get_unchecked(i), *weights.get_unchecked(i + 1)]);
            let px = f64x2::new([*x.get_unchecked(idx0 * 3), *x.get_unchecked(idx1 * 3)]);
            let py = f64x2::new([
                *x.get_unchecked(idx0 * 3 + 1),
                *x.get_unchecked(idx1 * 3 + 1),
            ]);
            let pz = f64x2::new([
                *x.get_unchecked(idx0 * 3 + 2),
                *x.get_unchecked(idx1 * 3 + 2),
            ]);
            let vy = f64x2::new([*y.get_unchecked(idx0), *y.get_unchecked(idx1)]);

            let dx = px - qx;
            let dy = py - qy;
            let dz = pz - qz;

            let wdx = w * dx;
            let wdy = w * dy;
            let wdz = w * dz;

            s_w += w;
            s_x += wdx;
            s_y += wdy;
            s_z += wdz;
            s_x2 += wdx * dx;
            s_y2 += wdy * dy;
            s_z2 += wdz * dz;
            s_xy += wdx * dy;
            s_xz += wdx * dz;
            s_yz += wdy * dz;

            s_wy += w * vy;
            s_wxy += wdx * vy;
            s_wyy += wdy * vy;
            s_wzy += wdz * vy;

            i += 2;
        }
    }

    let mut a_w = s_w.reduce_add();
    let mut a_x = s_x.reduce_add();
    let mut a_y = s_y.reduce_add();
    let mut a_z = s_z.reduce_add();
    let mut a_x2 = s_x2.reduce_add();
    let mut a_y2 = s_y2.reduce_add();
    let mut a_z2 = s_z2.reduce_add();
    let mut a_xy = s_xy.reduce_add();
    let mut a_xz = s_xz.reduce_add();
    let mut a_yz = s_yz.reduce_add();
    let mut a_wy = s_wy.reduce_add();
    let mut a_wxy = s_wxy.reduce_add();
    let mut a_wyy = s_wyy.reduce_add();
    let mut a_wzy = s_wzy.reduce_add();

    unsafe {
        for k in i..n {
            let w = *weights.get_unchecked(k);
            if w <= f64::EPSILON {
                continue;
            }

            let idx = *indices.get_unchecked(k);
            let dx = *x.get_unchecked(idx * 3) - query_x;
            let dy = *x.get_unchecked(idx * 3 + 1) - query_y;
            let dz = *x.get_unchecked(idx * 3 + 2) - query_z;
            let val_y = *y.get_unchecked(idx);

            let wdx = w * dx;
            let wdy = w * dy;
            let wdz = w * dz;

            a_w += w;
            a_x += wdx;
            a_y += wdy;
            a_z += wdz;
            a_x2 += wdx * dx;
            a_y2 += wdy * dy;
            a_z2 += wdz * dz;
            a_xy += wdx * dy;
            a_xz += wdx * dz;
            a_yz += wdy * dz;

            a_wy += w * val_y;
            a_wxy += wdx * val_y;
            a_wyy += wdy * val_y;
            a_wzy += wdz * val_y;
        }
    }

    xtwx[0] = a_w;
    xtwx[1] = a_x;
    xtwx[2] = a_y;
    xtwx[3] = a_z;
    xtwx[4] = a_x;
    xtwx[5] = a_x2;
    xtwx[6] = a_xy;
    xtwx[7] = a_xz;
    xtwx[8] = a_y;
    xtwx[9] = a_xy;
    xtwx[10] = a_y2;
    xtwx[11] = a_yz;
    xtwx[12] = a_z;
    xtwx[13] = a_xz;
    xtwx[14] = a_yz;
    xtwx[15] = a_z2;

    xtwy[0] = a_wy;
    xtwy[1] = a_wxy;
    xtwy[2] = a_wyy;
    xtwy[3] = a_wzy;
}

/// Optimized accumulation for 3D Quadratic Case (Scalar).
///
/// Matrix size 10x10.
/// Terms: 1, x, y, z, x2, xy, xz, y2, yz, z2.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::cognitive_complexity)]
pub fn accumulate_3d_quadratic_scalar<T: Float>(
    x: &[T],
    y: &[T],
    indices: &[usize],
    weights: &[T],
    query_x: T,
    query_y: T,
    query_z: T,
    xtwx: &mut [T; 100],
    xtwy: &mut [T; 10],
) {
    let n = indices.len();

    // Accumulators for moments up to order 4
    let mut s_w = T::zero();
    let mut s_x = T::zero();
    let mut s_y = T::zero();
    let mut s_z = T::zero();
    let mut s_x2 = T::zero();
    let mut s_y2 = T::zero();
    let mut s_z2 = T::zero();
    let mut s_xy = T::zero();
    let mut s_xz = T::zero();
    let mut s_yz = T::zero();
    let mut s_x3 = T::zero();
    let mut s_y3 = T::zero();
    let mut s_z3 = T::zero();
    let mut s_x2y = T::zero();
    let mut s_x2z = T::zero();
    let mut s_xy2 = T::zero();
    let mut s_xz2 = T::zero();
    let mut s_y2z = T::zero();
    let mut s_yz2 = T::zero();
    let mut s_xyz = T::zero();
    let mut s_x4 = T::zero();
    let mut s_y4 = T::zero();
    let mut s_z4 = T::zero();
    let mut s_x3y = T::zero();
    let mut s_x3z = T::zero();
    let mut s_xy3 = T::zero();
    let mut s_xz3 = T::zero();
    let mut s_yz3 = T::zero();
    let mut s_y3z = T::zero();
    let mut s_x2y2 = T::zero();
    let mut s_x2z2 = T::zero();
    let mut s_y2z2 = T::zero();
    let mut s_x2yz = T::zero();
    let mut s_xy2z = T::zero();
    let mut s_xyz2 = T::zero();

    // Accumulators for RHS (XT W y)
    let mut s_wy = T::zero();
    let mut s_wxy = T::zero();
    let mut s_wyy = T::zero();
    let mut s_wzy = T::zero();
    let mut s_wx2y = T::zero();
    let mut s_wxyy = T::zero();
    let mut s_wxzy = T::zero();
    let mut s_wy2y = T::zero();
    let mut s_wyzy = T::zero();
    let mut s_wz2y = T::zero();

    // SAFETY: Invariants guaranteed by KD-tree.
    unsafe {
        for i in 0..n {
            let w = *weights.get_unchecked(i);
            if w <= T::epsilon() {
                continue;
            }

            let idx = *indices.get_unchecked(i);
            let dx = *x.get_unchecked(idx * 3) - query_x;
            let dy = *x.get_unchecked(idx * 3 + 1) - query_y;
            let dz = *x.get_unchecked(idx * 3 + 2) - query_z;
            let val = *y.get_unchecked(idx);

            let wdx = w * dx;
            let wdy = w * dy;
            let wdz = w * dz;

            let dy2 = dy * dy;
            let dz2 = dz * dz;

            // Order 0, 1
            s_w = s_w + w;
            s_x = s_x + wdx;
            s_y = s_y + wdy;
            s_z = s_z + wdz;

            // Order 2
            s_x2 = s_x2 + wdx * dx;
            s_y2 = s_y2 + wdy * dy;
            s_z2 = s_z2 + wdz * dz;
            s_xy = s_xy + wdx * dy;
            s_xz = s_xz + wdx * dz;
            s_yz = s_yz + wdy * dz;

            // Order 3
            let wdx2 = wdx * dx;
            let wdy2 = wdy * dy;
            let wdz2 = wdz * dz;

            s_x3 = s_x3 + wdx2 * dx;
            s_y3 = s_y3 + wdy2 * dy;
            s_z3 = s_z3 + wdz2 * dz;
            s_x2y = s_x2y + wdx2 * dy;
            s_x2z = s_x2z + wdx2 * dz;
            s_xy2 = s_xy2 + wdy2 * dx;
            s_xz2 = s_xz2 + wdz2 * dx;
            s_y2z = s_y2z + wdy2 * dz;
            s_yz2 = s_yz2 + wdz2 * dy;
            s_xyz = s_xyz + wdx * dy * dz;

            // Order 4
            let wdx3 = wdx2 * dx;
            let wdy3 = wdy2 * dy;
            let wdz3 = wdz2 * dz;

            s_x4 = s_x4 + wdx3 * dx;
            s_y4 = s_y4 + wdy3 * dy;
            s_z4 = s_z4 + wdz3 * dz;
            s_x3y = s_x3y + wdx3 * dy;
            s_x3z = s_x3z + wdx3 * dz;
            s_xy3 = s_xy3 + wdy3 * dx;
            s_xz3 = s_xz3 + wdz3 * dx;
            s_yz3 = s_yz3 + wdz3 * dy;
            s_y3z = s_y3z + wdy3 * dz;
            s_x2y2 = s_x2y2 + wdx2 * dy2;
            s_x2z2 = s_x2z2 + wdx2 * dz2;
            s_y2z2 = s_y2z2 + wdy2 * dz2;
            s_x2yz = s_x2yz + wdx2 * dy * dz;
            s_xy2z = s_xy2z + wdy2 * dx * dz;
            s_xyz2 = s_xyz2 + wdz2 * dx * dy;

            // RHS
            s_wy = s_wy + w * val;
            s_wxy = s_wxy + wdx * val;
            s_wyy = s_wyy + wdy * val;
            s_wzy = s_wzy + wdz * val;

            s_wx2y = s_wx2y + wdx2 * val;
            s_wxyy = s_wxyy + wdx * dy * val;
            s_wxzy = s_wxzy + wdx * dz * val;
            s_wy2y = s_wy2y + wdy2 * val;
            s_wyzy = s_wyzy + wdy * dz * val;
            s_wz2y = s_wz2y + wdz2 * val;
        }
    }

    // Fill Matrix (10x10)
    // 0: 1, 1: x, 2: y, 3: z, 4: x2, 5: xy, 6: xz, 7: y2, 8: yz, 9: z2

    // Row 0 (1)
    xtwx[0] = s_w;
    xtwx[1] = s_x;
    xtwx[2] = s_y;
    xtwx[3] = s_z;
    xtwx[4] = s_x2;
    xtwx[5] = s_xy;
    xtwx[6] = s_xz;
    xtwx[7] = s_y2;
    xtwx[8] = s_yz;
    xtwx[9] = s_z2;

    // Row 1 (x)
    xtwx[10] = s_x;
    xtwx[11] = s_x2;
    xtwx[12] = s_xy;
    xtwx[13] = s_xz;
    xtwx[14] = s_x3;
    xtwx[15] = s_x2y;
    xtwx[16] = s_x2z;
    xtwx[17] = s_xy2;
    xtwx[18] = s_xyz;
    xtwx[19] = s_xz2;

    // Row 2 (y)
    xtwx[20] = s_y;
    xtwx[21] = s_xy;
    xtwx[22] = s_y2;
    xtwx[23] = s_yz;
    xtwx[24] = s_x2y;
    xtwx[25] = s_xy2;
    xtwx[26] = s_xyz;
    xtwx[27] = s_y3;
    xtwx[28] = s_y2z;
    xtwx[29] = s_yz2;

    // Row 3 (z)
    xtwx[30] = s_z;
    xtwx[31] = s_xz;
    xtwx[32] = s_yz;
    xtwx[33] = s_z2;
    xtwx[34] = s_x2z;
    xtwx[35] = s_xyz;
    xtwx[36] = s_xz2;
    xtwx[37] = s_y2z;
    xtwx[38] = s_yz2;
    xtwx[39] = s_z3;

    // Row 4 (x2)
    xtwx[40] = s_x2;
    xtwx[41] = s_x3;
    xtwx[42] = s_x2y;
    xtwx[43] = s_x2z;
    xtwx[44] = s_x4;
    xtwx[45] = s_x3y;
    xtwx[46] = s_x3z;
    xtwx[47] = s_x2y2;
    xtwx[48] = s_x2yz;
    xtwx[49] = s_x2z2;

    // Row 5 (xy)
    xtwx[50] = s_xy;
    xtwx[51] = s_x2y;
    xtwx[52] = s_xy2;
    xtwx[53] = s_xyz;
    xtwx[54] = s_x3y;
    xtwx[55] = s_x2y2;
    xtwx[56] = s_x2yz;
    xtwx[57] = s_xy3;
    xtwx[58] = s_xy2z;
    xtwx[59] = s_xyz2;

    // Row 6 (xz)
    xtwx[60] = s_xz;
    xtwx[61] = s_x2z;
    xtwx[62] = s_xyz;
    xtwx[63] = s_xz2;
    xtwx[64] = s_x3z;
    xtwx[65] = s_x2yz;
    xtwx[66] = s_x2z2;
    xtwx[67] = s_xy2z;
    xtwx[68] = s_xyz2;
    xtwx[69] = s_xz3;

    // Row 7 (y2)
    xtwx[70] = s_y2;
    xtwx[71] = s_xy2;
    xtwx[72] = s_y3;
    xtwx[73] = s_y2z;
    xtwx[74] = s_x2y2;
    xtwx[75] = s_xy3;
    xtwx[76] = s_xy2z;
    xtwx[77] = s_y4;
    xtwx[78] = s_y3z;
    xtwx[79] = s_y2z2;

    // Row 8 (yz)
    xtwx[80] = s_yz;
    xtwx[81] = s_xyz;
    xtwx[82] = s_y2z;
    xtwx[83] = s_yz2;
    xtwx[84] = s_x2yz;
    xtwx[85] = s_xy2z;
    xtwx[86] = s_xyz2;
    xtwx[87] = s_y3z;
    xtwx[88] = s_y2z2;
    xtwx[89] = s_yz3;

    // Row 9 (z2)
    xtwx[90] = s_z2;
    xtwx[91] = s_xz2;
    xtwx[92] = s_yz2;
    xtwx[93] = s_z3;
    xtwx[94] = s_x2z2;
    xtwx[95] = s_xyz2;
    xtwx[96] = s_xz3;
    xtwx[97] = s_y2z2;
    xtwx[98] = s_yz3;
    xtwx[99] = s_z4;

    // Fill RHS
    xtwy[0] = s_wy;
    xtwy[1] = s_wxy;
    xtwy[2] = s_wyy;
    xtwy[3] = s_wzy;
    xtwy[4] = s_wx2y;
    xtwy[5] = s_wxyy;
    xtwy[6] = s_wxzy;
    xtwy[7] = s_wy2y;
    xtwy[8] = s_wyzy;
    xtwy[9] = s_wz2y;
}

/// Optimized accumulation for 3D Quadratic Case using SIMD.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::cognitive_complexity)]
pub fn accumulate_3d_quadratic_simd(
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
    let n = indices.len();

    let mut s_w = f64x2::splat(0.0);
    let mut s_x = f64x2::splat(0.0);
    let mut s_y = f64x2::splat(0.0);
    let mut s_z = f64x2::splat(0.0);
    let mut s_x2 = f64x2::splat(0.0);
    let mut s_y2 = f64x2::splat(0.0);
    let mut s_z2 = f64x2::splat(0.0);
    let mut s_xy = f64x2::splat(0.0);
    let mut s_xz = f64x2::splat(0.0);
    let mut s_yz = f64x2::splat(0.0);
    let mut s_x3 = f64x2::splat(0.0);
    let mut s_y3 = f64x2::splat(0.0);
    let mut s_z3 = f64x2::splat(0.0);
    let mut s_x2y = f64x2::splat(0.0);
    let mut s_x2z = f64x2::splat(0.0);
    let mut s_xy2 = f64x2::splat(0.0);
    let mut s_xz2 = f64x2::splat(0.0);
    let mut s_y2z = f64x2::splat(0.0);
    let mut s_yz2 = f64x2::splat(0.0);
    let mut s_xyz = f64x2::splat(0.0);
    let mut s_x4 = f64x2::splat(0.0);
    let mut s_y4 = f64x2::splat(0.0);
    let mut s_z4 = f64x2::splat(0.0);
    let mut s_x3y = f64x2::splat(0.0);
    let mut s_x3z = f64x2::splat(0.0);
    let mut s_xy3 = f64x2::splat(0.0);
    let mut s_xz3 = f64x2::splat(0.0);
    let mut s_yz3 = f64x2::splat(0.0);
    let mut s_y3z = f64x2::splat(0.0);
    let mut s_x2y2 = f64x2::splat(0.0);
    let mut s_x2z2 = f64x2::splat(0.0);
    let mut s_y2z2 = f64x2::splat(0.0);
    let mut s_x2yz = f64x2::splat(0.0);
    let mut s_xy2z = f64x2::splat(0.0);
    let mut s_xyz2 = f64x2::splat(0.0);

    let mut s_wy = f64x2::splat(0.0);
    let mut s_wxy = f64x2::splat(0.0);
    let mut s_wyy = f64x2::splat(0.0);
    let mut s_wzy = f64x2::splat(0.0);
    let mut s_wx2y = f64x2::splat(0.0);
    let mut s_wxyy = f64x2::splat(0.0);
    let mut s_wxzy = f64x2::splat(0.0);
    let mut s_wy2y = f64x2::splat(0.0);
    let mut s_wyzy = f64x2::splat(0.0);
    let mut s_wz2y = f64x2::splat(0.0);

    let qx = f64x2::splat(query_x);
    let qy = f64x2::splat(query_y);
    let qz = f64x2::splat(query_z);

    let mut i = 0;
    // SAFETY: Use unchecked access for performance. Invariants guaranteed by KD-tree.
    unsafe {
        while i + 2 <= n {
            let idx0 = *indices.get_unchecked(i);
            let idx1 = *indices.get_unchecked(i + 1);

            let w = f64x2::new([*weights.get_unchecked(i), *weights.get_unchecked(i + 1)]);
            let px = f64x2::new([*x.get_unchecked(idx0 * 3), *x.get_unchecked(idx1 * 3)]);
            let py = f64x2::new([
                *x.get_unchecked(idx0 * 3 + 1),
                *x.get_unchecked(idx1 * 3 + 1),
            ]);
            let pz = f64x2::new([
                *x.get_unchecked(idx0 * 3 + 2),
                *x.get_unchecked(idx1 * 3 + 2),
            ]);
            let val = f64x2::new([*y.get_unchecked(idx0), *y.get_unchecked(idx1)]);

            let dx = px - qx;
            let dy = py - qy;
            let dz = pz - qz;

            let wdx = w * dx;
            let wdy = w * dy;
            let wdz = w * dz;

            let dy2 = dy * dy;
            let dz2 = dz * dz;

            s_w += w;
            s_x += wdx;
            s_y += wdy;
            s_z += wdz;

            s_x2 += wdx * dx;
            s_y2 += wdy * dy;
            s_z2 += wdz * dz;
            s_xy += wdx * dy;
            s_xz += wdx * dz;
            s_yz += wdy * dz;

            let wdx2 = wdx * dx;
            let wdy2 = wdy * dy;
            let wdz2 = wdz * dz;

            s_x3 += wdx2 * dx;
            s_y3 += wdy2 * dy;
            s_z3 += wdz2 * dz;
            s_x2y += wdx2 * dy;
            s_x2z += wdx2 * dz;
            s_xy2 += wdy2 * dx;
            s_xz2 += wdz2 * dx;
            s_y2z += wdy2 * dz;
            s_yz2 += wdz2 * dy;
            s_xyz += wdx * dy * dz;

            let wdx3 = wdx2 * dx;
            let wdy3 = wdy2 * dy;
            let wdz3 = wdz2 * dz;

            s_x4 += wdx3 * dx;
            s_y4 += wdy3 * dy;
            s_z4 += wdz3 * dz;
            s_x3y += wdx3 * dy;
            s_x3z += wdx3 * dz;
            s_xy3 += wdy3 * dx;
            s_xz3 += wdz3 * dx;
            s_yz3 += wdz3 * dy;
            s_y3z += wdy3 * dz;
            s_x2y2 += wdx2 * dy2;
            s_x2z2 += wdx2 * dz2;
            s_y2z2 += wdy2 * dz2;
            s_x2yz += wdx2 * dy * dz;
            s_xy2z += wdy2 * dx * dz;
            s_xyz2 += wdz2 * dx * dy;

            s_wy += w * val;
            s_wxy += wdx * val;
            s_wyy += wdy * val;
            s_wzy += wdz * val;

            s_wx2y += wdx2 * val;
            s_wxyy += wdx * dy * val;
            s_wxzy += wdx * dz * val;
            s_wy2y += wdy2 * val;
            s_wyzy += wdy * dz * val;
            s_wz2y += wdz2 * val;

            i += 2;
        }
    }

    let mut a_w = s_w.reduce_add();
    let mut a_x = s_x.reduce_add();
    let mut a_y = s_y.reduce_add();
    let mut a_z = s_z.reduce_add();
    let mut a_x2 = s_x2.reduce_add();
    let mut a_y2 = s_y2.reduce_add();
    let mut a_z2 = s_z2.reduce_add();
    let mut a_xy = s_xy.reduce_add();
    let mut a_xz = s_xz.reduce_add();
    let mut a_yz = s_yz.reduce_add();
    let mut a_x3 = s_x3.reduce_add();
    let mut a_y3 = s_y3.reduce_add();
    let mut a_z3 = s_z3.reduce_add();
    let mut a_x2y = s_x2y.reduce_add();
    let mut a_x2z = s_x2z.reduce_add();
    let mut a_xy2 = s_xy2.reduce_add();
    let mut a_xz2 = s_xz2.reduce_add();
    let mut a_y2z = s_y2z.reduce_add();
    let mut a_yz2 = s_yz2.reduce_add();
    let mut a_xyz = s_xyz.reduce_add();
    let mut a_x4 = s_x4.reduce_add();
    let mut a_y4 = s_y4.reduce_add();
    let mut a_z4 = s_z4.reduce_add();
    let mut a_x3y = s_x3y.reduce_add();
    let mut a_x3z = s_x3z.reduce_add();
    let mut a_xy3 = s_xy3.reduce_add();
    let mut a_xz3 = s_xz3.reduce_add();
    let mut a_yz3 = s_yz3.reduce_add();
    let mut a_y3z = s_y3z.reduce_add();
    let mut a_x2y2 = s_x2y2.reduce_add();
    let mut a_x2z2 = s_x2z2.reduce_add();
    let mut a_y2z2 = s_y2z2.reduce_add();
    let mut a_x2yz = s_x2yz.reduce_add();
    let mut a_xy2z = s_xy2z.reduce_add();
    let mut a_xyz2 = s_xyz2.reduce_add();

    let mut a_wy = s_wy.reduce_add();
    let mut a_wxy = s_wxy.reduce_add();
    let mut a_wyy = s_wyy.reduce_add();
    let mut a_wzy = s_wzy.reduce_add();
    let mut a_wx2y = s_wx2y.reduce_add();
    let mut a_wxyy = s_wxyy.reduce_add();
    let mut a_wxzy = s_wxzy.reduce_add();
    let mut a_wy2y = s_wy2y.reduce_add();
    let mut a_wyzy = s_wyzy.reduce_add();
    let mut a_wz2y = s_wz2y.reduce_add();

    unsafe {
        for k in i..n {
            let w = *weights.get_unchecked(k);
            if w <= f64::EPSILON {
                continue;
            }

            let idx = *indices.get_unchecked(k);
            let dx = *x.get_unchecked(idx * 3) - query_x;
            let dy = *x.get_unchecked(idx * 3 + 1) - query_y;
            let dz = *x.get_unchecked(idx * 3 + 2) - query_z;
            let val = *y.get_unchecked(idx);

            let wdx = w * dx;
            let wdy = w * dy;
            let wdz = w * dz;

            let dy2 = dy * dy;
            let dz2 = dz * dz;

            a_w += w;
            a_x += wdx;
            a_y += wdy;
            a_z += wdz;

            a_x2 += wdx * dx;
            a_y2 += wdy * dy;
            a_z2 += wdz * dz;
            a_xy += wdx * dy;
            a_xz += wdx * dz;
            a_yz += wdy * dz;

            let wdx2 = wdx * dx;
            let wdy2 = wdy * dy;
            let wdz2 = wdz * dz;

            a_x3 += wdx2 * dx;
            a_y3 += wdy2 * dy;
            a_z3 += wdz2 * dz;
            a_x2y += wdx2 * dy;
            a_x2z += wdx2 * dz;
            a_xy2 += wdy2 * dx;
            a_xz2 += wdz2 * dx;
            a_y2z += wdy2 * dz;
            a_yz2 += wdz2 * dy;
            a_xyz += wdx * dy * dz;

            let wdx3 = wdx2 * dx;
            let wdy3 = wdy2 * dy;
            let wdz3 = wdz2 * dz;

            a_x4 += wdx3 * dx;
            a_y4 += wdy3 * dy;
            a_z4 += wdz3 * dz;
            a_x3y += wdx3 * dy;
            a_x3z += wdx3 * dz;
            a_xy3 += wdy3 * dx;
            a_xz3 += wdz3 * dx;
            a_yz3 += wdz3 * dy;
            a_y3z += wdy3 * dz;
            a_x2y2 += wdx2 * dy2;
            a_x2z2 += wdx2 * dz2;
            a_y2z2 += wdy2 * dz2;
            a_x2yz += wdx2 * dy * dz;
            a_xy2z += wdy2 * dx * dz;
            a_xyz2 += wdz2 * dx * dy;

            a_wy += w * val;
            a_wxy += wdx * val;
            a_wyy += wdy * val;
            a_wzy += wdz * val;

            a_wx2y += wdx2 * val;
            a_wxyy += wdx * dy * val;
            a_wxzy += wdx * dz * val;
            a_wy2y += wdy2 * val;
            a_wyzy += wdy * dz * val;
            a_wz2y += wdz2 * val;
        }
    }

    xtwx[0] = a_w;
    xtwx[1] = a_x;
    xtwx[2] = a_y;
    xtwx[3] = a_z;
    xtwx[4] = a_x2;
    xtwx[5] = a_xy;
    xtwx[6] = a_xz;
    xtwx[7] = a_y2;
    xtwx[8] = a_yz;
    xtwx[9] = a_z2;

    xtwx[10] = a_x;
    xtwx[11] = a_x2;
    xtwx[12] = a_xy;
    xtwx[13] = a_xz;
    xtwx[14] = a_x3;
    xtwx[15] = a_x2y;
    xtwx[16] = a_x2z;
    xtwx[17] = a_xy2;
    xtwx[18] = a_xyz;
    xtwx[19] = a_xz2;

    xtwx[20] = a_y;
    xtwx[21] = a_xy;
    xtwx[22] = a_y2;
    xtwx[23] = a_yz;
    xtwx[24] = a_x2y;
    xtwx[25] = a_xy2;
    xtwx[26] = a_xyz;
    xtwx[27] = a_y3;
    xtwx[28] = a_y2z;
    xtwx[29] = a_yz2;

    xtwx[30] = a_z;
    xtwx[31] = a_xz;
    xtwx[32] = a_yz;
    xtwx[33] = a_z2;
    xtwx[34] = a_x2z;
    xtwx[35] = a_xyz;
    xtwx[36] = a_xz2;
    xtwx[37] = a_y2z;
    xtwx[38] = a_yz2;
    xtwx[39] = a_z3;

    xtwx[40] = a_x2;
    xtwx[41] = a_x3;
    xtwx[42] = a_x2y;
    xtwx[43] = a_x2z;
    xtwx[44] = a_x4;
    xtwx[45] = a_x3y;
    xtwx[46] = a_x3z;
    xtwx[47] = a_x2y2;
    xtwx[48] = a_x2yz;
    xtwx[49] = a_x2z2;

    xtwx[50] = a_xy;
    xtwx[51] = a_x2y;
    xtwx[52] = a_xy2;
    xtwx[53] = a_xyz;
    xtwx[54] = a_x3y;
    xtwx[55] = a_x2y2;
    xtwx[56] = a_x2yz;
    xtwx[57] = a_xy3;
    xtwx[58] = a_xy2z;
    xtwx[59] = a_xyz2;

    xtwx[60] = a_xz;
    xtwx[61] = a_x2z;
    xtwx[62] = a_xyz;
    xtwx[63] = a_xz2;
    xtwx[64] = a_x3z;
    xtwx[65] = a_x2yz;
    xtwx[66] = a_x2z2;
    xtwx[67] = a_xy2z;
    xtwx[68] = a_xyz2;
    xtwx[69] = a_xz3;

    xtwx[70] = a_y2;
    xtwx[71] = a_xy2;
    xtwx[72] = a_y3;
    xtwx[73] = a_y2z;
    xtwx[74] = a_x2y2;
    xtwx[75] = a_xy3;
    xtwx[76] = a_xy2z;
    xtwx[77] = a_y4;
    xtwx[78] = a_y3z;
    xtwx[79] = a_y2z2;

    xtwx[80] = a_yz;
    xtwx[81] = a_xyz;
    xtwx[82] = a_y2z;
    xtwx[83] = a_yz2;
    xtwx[84] = a_x2yz;
    xtwx[85] = a_xy2z;
    xtwx[86] = a_xyz2;
    xtwx[87] = a_y3z;
    xtwx[88] = a_y2z2;
    xtwx[89] = a_yz3;

    xtwx[90] = a_z2;
    xtwx[91] = a_xz2;
    xtwx[92] = a_yz2;
    xtwx[93] = a_z3;
    xtwx[94] = a_x2z2;
    xtwx[95] = a_xyz2;
    xtwx[96] = a_xz3;
    xtwx[97] = a_y2z2;
    xtwx[98] = a_yz3;
    xtwx[99] = a_z4;

    xtwy[0] = a_wy;
    xtwy[1] = a_wxy;
    xtwy[2] = a_wyy;
    xtwy[3] = a_wzy;
    xtwy[4] = a_wx2y;
    xtwy[5] = a_wxyy;
    xtwy[6] = a_wxzy;
    xtwy[7] = a_wy2y;
    xtwy[8] = a_wyzy;
    xtwy[9] = a_wz2y;
}
