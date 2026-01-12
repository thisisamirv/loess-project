//! Distance metrics for nD LOESS.
//!
//! ## Purpose
//!
//! This module provides distance computation for multivariate (nD) LOESS.
//! In 1D LOWESS, distance is simply |x - x_i|. For nD LOESS, we need proper
//! distance metrics that work in higher-dimensional spaces.
//!
//! ## Design notes
//!
//! * **Decoupling**: Distance calculation is separated from kernel evaluation.
//! * **Normalization**: Supports normalizing dimensions to handle differing scales.
//! * **SIMD**: Uses vectorized operations via `DistanceLinalg` for f64/f32.
//!
//! ## Key concepts
//!
//! * **Metric**: Defines how "closeness" is measured (Euclidean, Manhattan, etc.).
//! * **Normalization**: Rescaling dimensions to [0, 1] to ensure equal influence.
//!
//! ## Invariants
//!
//! * Distance is always non-negative.
//! * Distance is zero if and only if points are identical (for metrics satisfying identity).
//!
//! ## Non-goals
//!
//! * This module does not handle the kernel weighting (bandwidth/smoothing).

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use num_traits::Float;
use wide::{f32x8, f64x4};

// ============================================================================
// DistanceLinalg Trait
// ============================================================================

/// Trait for SIMD-optimized distance computations.
pub trait DistanceLinalg: Float + 'static {
    /// Compute Euclidean distance between two points.
    fn euclidean_distance(a: &[Self], b: &[Self]) -> Self;

    /// Compute normalized Euclidean distance with per-dimension scales.
    fn normalized_distance(a: &[Self], b: &[Self], scales: &[Self]) -> Self;

    /// Compute weighted Euclidean distance.
    fn weighted_distance(a: &[Self], b: &[Self], weights: &[Self]) -> Self;

    /// Compute Manhattan distance (L1 norm): Σ|xᵢ - yᵢ|
    fn manhattan_distance(a: &[Self], b: &[Self]) -> Self;

    /// Compute Chebyshev distance (L∞ norm): max|xᵢ - yᵢ|
    fn chebyshev_distance(a: &[Self], b: &[Self]) -> Self;

    /// Compute Minkowski distance (Lp norm): (Σ|xᵢ - yᵢ|^p)^(1/p)
    fn minkowski_distance(a: &[Self], b: &[Self], p: Self) -> Self;

    /// Compute squared Euclidean distance (avoids sqrt).
    fn euclidean_distance_squared(a: &[Self], b: &[Self]) -> Self;

    /// Compute squared normalized Euclidean distance.
    fn normalized_distance_squared(a: &[Self], b: &[Self], scales: &[Self]) -> Self;

    /// Compute squared weighted Euclidean distance.
    fn weighted_distance_squared(a: &[Self], b: &[Self], weights: &[Self]) -> Self;

    /// Compute squared Manhattan distance.
    fn manhattan_distance_squared(a: &[Self], b: &[Self]) -> Self;

    /// Compute squared Chebyshev distance.
    fn chebyshev_distance_squared(a: &[Self], b: &[Self]) -> Self;

    /// Compute squared Minkowski distance.
    fn minkowski_distance_squared(a: &[Self], b: &[Self], p: Self) -> Self;
}

impl DistanceLinalg for f64 {
    #[inline]
    fn euclidean_distance(a: &[Self], b: &[Self]) -> Self {
        simd_distance::euclidean_f64(a, b)
    }
    #[inline]
    fn normalized_distance(a: &[Self], b: &[Self], scales: &[Self]) -> Self {
        simd_distance::normalized_f64(a, b, scales)
    }
    #[inline]
    fn weighted_distance(a: &[Self], b: &[Self], weights: &[Self]) -> Self {
        simd_distance::weighted_f64(a, b, weights)
    }
    #[inline]
    fn manhattan_distance(a: &[Self], b: &[Self]) -> Self {
        simd_distance::manhattan_f64(a, b)
    }
    #[inline]
    fn chebyshev_distance(a: &[Self], b: &[Self]) -> Self {
        simd_distance::chebyshev_f64(a, b)
    }
    #[inline]
    fn minkowski_distance(a: &[Self], b: &[Self], p: Self) -> Self {
        simd_distance::minkowski_f64(a, b, p)
    }
    #[inline]
    fn euclidean_distance_squared(a: &[Self], b: &[Self]) -> Self {
        simd_distance::euclidean_sq_f64(a, b)
    }
    #[inline]
    fn normalized_distance_squared(a: &[Self], b: &[Self], scales: &[Self]) -> Self {
        simd_distance::normalized_sq_f64(a, b, scales)
    }
    #[inline]
    fn weighted_distance_squared(a: &[Self], b: &[Self], weights: &[Self]) -> Self {
        simd_distance::weighted_sq_f64(a, b, weights)
    }
    #[inline]
    fn manhattan_distance_squared(a: &[Self], b: &[Self]) -> Self {
        let d = simd_distance::manhattan_f64(a, b);
        d * d
    }
    #[inline]
    fn chebyshev_distance_squared(a: &[Self], b: &[Self]) -> Self {
        let d = simd_distance::chebyshev_f64(a, b);
        d * d
    }
    #[inline]
    fn minkowski_distance_squared(a: &[Self], b: &[Self], p: Self) -> Self {
        let d = simd_distance::minkowski_f64(a, b, p);
        d * d
    }
}

impl DistanceLinalg for f32 {
    #[inline]
    fn euclidean_distance(a: &[Self], b: &[Self]) -> Self {
        simd_distance::euclidean_f32(a, b)
    }
    #[inline]
    fn normalized_distance(a: &[Self], b: &[Self], scales: &[Self]) -> Self {
        simd_distance::normalized_f32(a, b, scales)
    }
    #[inline]
    fn weighted_distance(a: &[Self], b: &[Self], weights: &[Self]) -> Self {
        simd_distance::weighted_f32(a, b, weights)
    }
    #[inline]
    fn manhattan_distance(a: &[Self], b: &[Self]) -> Self {
        simd_distance::manhattan_f32(a, b)
    }
    #[inline]
    fn chebyshev_distance(a: &[Self], b: &[Self]) -> Self {
        simd_distance::chebyshev_f32(a, b)
    }
    #[inline]
    fn minkowski_distance(a: &[Self], b: &[Self], p: Self) -> Self {
        simd_distance::minkowski_f32(a, b, p)
    }
    #[inline]
    fn euclidean_distance_squared(a: &[Self], b: &[Self]) -> Self {
        simd_distance::euclidean_sq_f32(a, b)
    }
    #[inline]
    fn normalized_distance_squared(a: &[Self], b: &[Self], scales: &[Self]) -> Self {
        simd_distance::normalized_sq_f32(a, b, scales)
    }
    #[inline]
    fn weighted_distance_squared(a: &[Self], b: &[Self], weights: &[Self]) -> Self {
        simd_distance::weighted_sq_f32(a, b, weights)
    }
    #[inline]
    fn manhattan_distance_squared(a: &[Self], b: &[Self]) -> Self {
        let d = simd_distance::manhattan_f32(a, b);
        d * d
    }
    #[inline]
    fn chebyshev_distance_squared(a: &[Self], b: &[Self]) -> Self {
        let d = simd_distance::chebyshev_f32(a, b);
        d * d
    }
    #[inline]
    fn minkowski_distance_squared(a: &[Self], b: &[Self], p: Self) -> Self {
        let d = simd_distance::minkowski_f32(a, b, p);
        d * d
    }
}

// ============================================================================
// Distance Metric Enum
// ============================================================================

/// Distance metric for nD LOESS neighborhood computation.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum DistanceMetric<T> {
    /// Standard Euclidean distance: √(Σ(xᵢ - yᵢ)²)
    Euclidean,

    /// Normalized Euclidean distance.
    #[default]
    Normalized,

    /// Manhattan distance (L1 norm): Σ|xᵢ - yᵢ|
    Manhattan,

    /// Chebyshev distance (L∞ norm): max|xᵢ - yᵢ|
    Chebyshev,

    /// Minkowski distance (Lp norm): (Σ|xᵢ - yᵢ|^p)^(1/p)
    Minkowski(T),

    /// Weighted Euclidean distance: √(Σ wᵢ(xᵢ - yᵢ)²)
    Weighted(Vec<T>),
}

// ============================================================================
// Distance Computation Functions
// ============================================================================

impl<T: DistanceLinalg> DistanceMetric<T> {
    /// Compute Euclidean distance between two nD points.
    #[inline]
    pub fn euclidean(a: &[T], b: &[T]) -> T {
        debug_assert_eq!(a.len(), b.len(), "Points must have same dimension");
        T::euclidean_distance(a, b)
    }

    /// Compute normalized distance between two nD points.
    #[inline]
    pub fn normalized(a: &[T], b: &[T], scales: &[T]) -> T {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), scales.len());
        T::normalized_distance(a, b, scales)
    }

    /// Compute Manhattan distance (L1 norm).
    #[inline]
    pub fn manhattan(a: &[T], b: &[T]) -> T {
        debug_assert_eq!(a.len(), b.len());
        T::manhattan_distance(a, b)
    }

    /// Compute Chebyshev distance (L-inf norm).
    #[inline]
    pub fn chebyshev(a: &[T], b: &[T]) -> T {
        debug_assert_eq!(a.len(), b.len());
        T::chebyshev_distance(a, b)
    }

    /// Compute Minkowski distance (Lp norm).
    #[inline]
    pub fn minkowski(a: &[T], b: &[T], p: T) -> T {
        debug_assert_eq!(a.len(), b.len());
        T::minkowski_distance(a, b, p)
    }

    /// Compute Weighted Euclidean distance.
    #[inline]
    pub fn weighted(a: &[T], b: &[T], weights: &[T]) -> T {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), weights.len());
        T::weighted_distance(a, b, weights)
    }

    /// Compute Squared Euclidean distance.
    #[inline]
    pub fn euclidean_squared(a: &[T], b: &[T]) -> T {
        debug_assert_eq!(a.len(), b.len());
        T::euclidean_distance_squared(a, b)
    }

    /// Compute Squared Normalized Euclidean distance.
    #[inline]
    pub fn normalized_squared(a: &[T], b: &[T], scales: &[T]) -> T {
        debug_assert_eq!(a.len(), b.len());
        T::normalized_distance_squared(a, b, scales)
    }

    /// Compute Squared Weighted Euclidean distance.
    #[inline]
    pub fn weighted_squared(a: &[T], b: &[T], weights: &[T]) -> T {
        debug_assert_eq!(a.len(), b.len());
        T::weighted_distance_squared(a, b, weights)
    }

    /// Compute Squared Manhattan distance.
    #[inline]
    pub fn manhattan_squared(a: &[T], b: &[T]) -> T {
        debug_assert_eq!(a.len(), b.len());
        T::manhattan_distance_squared(a, b)
    }

    /// Compute Squared Chebyshev distance.
    #[inline]
    pub fn chebyshev_squared(a: &[T], b: &[T]) -> T {
        debug_assert_eq!(a.len(), b.len());
        T::chebyshev_distance_squared(a, b)
    }

    /// Compute Squared Minkowski distance.
    #[inline]
    pub fn minkowski_squared(a: &[T], b: &[T], p: T) -> T {
        debug_assert_eq!(a.len(), b.len());
        T::minkowski_distance_squared(a, b, p)
    }
}

// ============================================================================
// SIMD Distance Implementation
// ============================================================================

/// SIMD-optimized distance calculations using the `wide` crate.
pub mod simd_distance {
    use super::*;

    // ========================================================================
    // Euclidean Distance
    // ========================================================================

    /// SIMD-optimized Euclidean distance for f64 slices.
    /// Processes 4 elements at a time using AVX/SSE2 instructions.
    #[inline]
    pub fn euclidean_f64(a: &[f64], b: &[f64]) -> f64 {
        euclidean_sq_f64(a, b).sqrt()
    }

    /// SIMD-optimized squared Euclidean distance for f64 slices.
    #[inline]
    pub fn euclidean_sq_f64(a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len(), "Points must have same dimension");

        let n = a.len();

        // For small dimensions (1-3), use scalar path - no SIMD overhead
        if n < 4 {
            return euclidean_sq_scalar(a, b);
        }

        let chunks = n / 4;
        let remainder = n % 4;

        let mut sum = f64x4::ZERO;

        // Process 4 elements at a time
        for i in 0..chunks {
            let base = i * 4;
            let va = f64x4::new([a[base], a[base + 1], a[base + 2], a[base + 3]]);
            let vb = f64x4::new([b[base], b[base + 1], b[base + 2], b[base + 3]]);
            let diff = va - vb;
            sum += diff * diff;
        }

        // Horizontal sum of SIMD vector
        let arr = sum.to_array();
        let mut total = arr[0] + arr[1] + arr[2] + arr[3];

        // Handle remainder elements
        let base = chunks * 4;
        for i in 0..remainder {
            let diff = a[base + i] - b[base + i];
            total += diff * diff;
        }

        total
    }

    /// SIMD-optimized squared Euclidean distance for f32 slices.
    #[inline]
    pub fn euclidean_sq_f32(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Points must have same dimension");

        let n = a.len();

        // For small dimensions, use scalar path
        if n < 8 {
            return euclidean_sq_scalar(a, b);
        }

        let chunks = n / 8;
        let remainder = n % 8;

        let mut sum = f32x8::ZERO;

        // Process 8 elements at a time
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
            let diff = va - vb;
            sum += diff * diff;
        }

        // Horizontal sum of SIMD vector
        let arr = sum.to_array();
        let mut total = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];

        // Handle remainder elements
        let base = chunks * 8;
        for i in 0..remainder {
            let diff = a[base + i] - b[base + i];
            total += diff * diff;
        }

        total
    }

    /// SIMD-optimized Euclidean distance for f32 slices.
    /// Processes 8 elements at a time using AVX/SSE2 instructions.
    #[inline]
    pub fn euclidean_f32(a: &[f32], b: &[f32]) -> f32 {
        euclidean_sq_f32(a, b).sqrt()
    }

    // ========================================================================
    // Normalized Distance
    // ========================================================================

    /// SIMD-optimized normalized Euclidean distance for f64 slices.
    #[inline]
    pub fn normalized_f64(a: &[f64], b: &[f64], scales: &[f64]) -> f64 {
        normalized_sq_f64(a, b, scales).sqrt()
    }

    /// SIMD-optimized squared normalized Euclidean distance for f64 slices.
    #[inline]
    pub fn normalized_sq_f64(a: &[f64], b: &[f64], scales: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), scales.len());

        let n = a.len();

        if n < 4 {
            return normalized_sq_scalar(a, b, scales);
        }

        let chunks = n / 4;
        let remainder = n % 4;

        let mut sum = f64x4::ZERO;

        for i in 0..chunks {
            let base = i * 4;
            let va = f64x4::new([a[base], a[base + 1], a[base + 2], a[base + 3]]);
            let vb = f64x4::new([b[base], b[base + 1], b[base + 2], b[base + 3]]);
            let vs = f64x4::new([
                scales[base],
                scales[base + 1],
                scales[base + 2],
                scales[base + 3],
            ]);
            let diff = (va - vb) * vs;
            sum += diff * diff;
        }

        let arr = sum.to_array();
        let mut total = arr[0] + arr[1] + arr[2] + arr[3];

        let base = chunks * 4;
        for i in 0..remainder {
            let diff = (a[base + i] - b[base + i]) * scales[base + i];
            total += diff * diff;
        }

        total
    }

    /// SIMD-optimized normalized Euclidean distance for f32 slices.
    #[inline]
    pub fn normalized_f32(a: &[f32], b: &[f32], scales: &[f32]) -> f32 {
        normalized_sq_f32(a, b, scales).sqrt()
    }

    /// SIMD-optimized squared normalized Euclidean distance for f32 slices.
    #[inline]
    pub fn normalized_sq_f32(a: &[f32], b: &[f32], scales: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), scales.len());

        let n = a.len();

        if n < 8 {
            return normalized_sq_scalar(a, b, scales);
        }

        let chunks = n / 8;
        let remainder = n % 8;

        let mut sum = f32x8::ZERO;

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
            let vs = f32x8::new([
                scales[base],
                scales[base + 1],
                scales[base + 2],
                scales[base + 3],
                scales[base + 4],
                scales[base + 5],
                scales[base + 6],
                scales[base + 7],
            ]);
            let diff = (va - vb) * vs;
            sum += diff * diff;
        }

        let arr = sum.to_array();
        let mut total = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];

        let base = chunks * 8;
        for i in 0..remainder {
            let diff = (a[base + i] - b[base + i]) * scales[base + i];
            total += diff * diff;
        }

        total
    }

    // ========================================================================
    // Weighted Distance
    // ========================================================================

    /// SIMD-optimized weighted Euclidean distance for f64 slices.
    #[inline]
    pub fn weighted_f64(a: &[f64], b: &[f64], weights: &[f64]) -> f64 {
        weighted_sq_f64(a, b, weights).sqrt()
    }

    /// SIMD-optimized squared weighted Euclidean distance for f64 slices.
    #[inline]
    pub fn weighted_sq_f64(a: &[f64], b: &[f64], weights: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), weights.len());

        let n = a.len();

        if n < 4 {
            return weighted_sq_scalar(a, b, weights);
        }

        let chunks = n / 4;
        let remainder = n % 4;

        let mut sum = f64x4::ZERO;

        for i in 0..chunks {
            let base = i * 4;
            let va = f64x4::new([a[base], a[base + 1], a[base + 2], a[base + 3]]);
            let vb = f64x4::new([b[base], b[base + 1], b[base + 2], b[base + 3]]);
            let vw = f64x4::new([
                weights[base],
                weights[base + 1],
                weights[base + 2],
                weights[base + 3],
            ]);
            let diff = va - vb;
            sum += vw * diff * diff;
        }

        let arr = sum.to_array();
        let mut total = arr[0] + arr[1] + arr[2] + arr[3];

        let base = chunks * 4;
        for i in 0..remainder {
            let diff = a[base + i] - b[base + i];
            total += weights[base + i] * diff * diff;
        }

        total
    }

    /// SIMD-optimized squared weighted Euclidean distance for f32 slices.
    #[inline]
    pub fn weighted_sq_f32(a: &[f32], b: &[f32], weights: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), weights.len());

        let n = a.len();

        if n < 8 {
            return weighted_sq_scalar(a, b, weights);
        }

        let chunks = n / 8;
        let remainder = n % 8;

        let mut sum = f32x8::ZERO;

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
            let vw = f32x8::new([
                weights[base],
                weights[base + 1],
                weights[base + 2],
                weights[base + 3],
                weights[base + 4],
                weights[base + 5],
                weights[base + 6],
                weights[base + 7],
            ]);
            let diff = va - vb;
            sum += vw * diff * diff;
        }

        let arr = sum.to_array();
        let mut total = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];

        let base = chunks * 8;
        for i in 0..remainder {
            let diff = a[base + i] - b[base + i];
            total += weights[base + i] * diff * diff;
        }

        total
    }

    /// SIMD-optimized weighted Euclidean distance for f32 slices.
    #[inline]
    pub fn weighted_f32(a: &[f32], b: &[f32], weights: &[f32]) -> f32 {
        weighted_sq_f32(a, b, weights).sqrt()
    }

    // ========================================================================
    // Manhattan Distance (L1 norm)
    // ========================================================================

    /// SIMD-optimized Manhattan distance for f64 slices.
    #[inline]
    pub fn manhattan_f64(a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len(), "Points must have same dimension");

        let n = a.len();

        if n < 4 {
            return manhattan_scalar(a, b);
        }

        let chunks = n / 4;
        let remainder = n % 4;

        let mut sum = f64x4::ZERO;

        for i in 0..chunks {
            let base = i * 4;
            let va = f64x4::new([a[base], a[base + 1], a[base + 2], a[base + 3]]);
            let vb = f64x4::new([b[base], b[base + 1], b[base + 2], b[base + 3]]);
            let diff = va - vb;
            sum += diff.abs();
        }

        let arr = sum.to_array();
        let mut total = arr[0] + arr[1] + arr[2] + arr[3];

        let base = chunks * 4;
        for i in 0..remainder {
            total += (a[base + i] - b[base + i]).abs();
        }

        total
    }

    /// SIMD-optimized Manhattan distance for f32 slices.
    #[inline]
    pub fn manhattan_f32(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Points must have same dimension");

        let n = a.len();

        if n < 8 {
            return manhattan_scalar(a, b);
        }

        let chunks = n / 8;
        let remainder = n % 8;

        let mut sum = f32x8::ZERO;

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
            let diff = va - vb;
            sum += diff.abs();
        }

        let arr = sum.to_array();
        let mut total = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];

        let base = chunks * 8;
        for i in 0..remainder {
            total += (a[base + i] - b[base + i]).abs();
        }

        total
    }

    // ========================================================================
    // Chebyshev Distance (L-infinity norm)
    // ========================================================================

    /// SIMD-optimized Chebyshev distance for f64 slices.
    #[inline]
    pub fn chebyshev_f64(a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len(), "Points must have same dimension");

        let n = a.len();

        if n < 4 {
            return chebyshev_scalar(a, b);
        }

        let chunks = n / 4;
        let remainder = n % 4;

        let mut max_vec = f64x4::ZERO;

        for i in 0..chunks {
            let base = i * 4;
            let va = f64x4::new([a[base], a[base + 1], a[base + 2], a[base + 3]]);
            let vb = f64x4::new([b[base], b[base + 1], b[base + 2], b[base + 3]]);
            let diff = (va - vb).abs();
            max_vec = max_vec.max(diff);
        }

        let arr = max_vec.to_array();
        let mut total = arr[0].max(arr[1]).max(arr[2]).max(arr[3]);

        let base = chunks * 4;
        for i in 0..remainder {
            total = total.max((a[base + i] - b[base + i]).abs());
        }

        total
    }

    /// SIMD-optimized Chebyshev distance for f32 slices.
    #[inline]
    pub fn chebyshev_f32(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Points must have same dimension");

        let n = a.len();

        if n < 8 {
            return chebyshev_scalar(a, b);
        }

        let chunks = n / 8;
        let remainder = n % 8;

        let mut max_vec = f32x8::ZERO;

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
            max_vec = max_vec.max(diff);
        }

        let arr = max_vec.to_array();
        let mut total = arr[0]
            .max(arr[1])
            .max(arr[2])
            .max(arr[3])
            .max(arr[4])
            .max(arr[5])
            .max(arr[6])
            .max(arr[7]);

        let base = chunks * 8;
        for i in 0..remainder {
            total = total.max((a[base + i] - b[base + i]).abs());
        }

        total
    }

    // ========================================================================
    // Minkowski Distance (Lp norm)
    // ========================================================================

    /// Minkowski distance for f64 slices.
    #[inline]
    pub fn minkowski_f64(a: &[f64], b: &[f64], p: f64) -> f64 {
        debug_assert_eq!(a.len(), b.len(), "Points must have same dimension");

        // Special cases that are already optimized
        if (p - 1.0).abs() < f64::EPSILON {
            return manhattan_f64(a, b);
        }
        if (p - 2.0).abs() < f64::EPSILON {
            return euclidean_f64(a, b);
        }

        // General case - scalar implementation (powf not easily vectorized)
        let sum_pow: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).abs().powf(p))
            .sum();
        sum_pow.powf(1.0 / p)
    }

    /// Minkowski distance for f32 slices.
    #[inline]
    pub fn minkowski_f32(a: &[f32], b: &[f32], p: f32) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Points must have same dimension");

        // Special cases that are already optimized
        if (p - 1.0).abs() < f32::EPSILON {
            return manhattan_f32(a, b);
        }
        if (p - 2.0).abs() < f32::EPSILON {
            return euclidean_f32(a, b);
        }

        // General case - scalar implementation (powf not easily vectorized)
        let sum_pow: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).abs().powf(p))
            .sum();
        sum_pow.powf(1.0 / p)
    }

    // ========================================================================
    // Scalar Fallbacks
    // ========================================================================

    #[inline]
    fn euclidean_sq_scalar<T: Float>(a: &[T], b: &[T]) -> T {
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| {
                let diff = ai - bi;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x)
    }

    #[inline]
    fn normalized_sq_scalar<T: Float>(a: &[T], b: &[T], scales: &[T]) -> T {
        a.iter()
            .zip(b.iter())
            .zip(scales.iter())
            .map(|((&ai, &bi), &scale)| {
                let diff = (ai - bi) * scale;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x)
    }

    #[inline]
    fn weighted_sq_scalar<T: Float>(a: &[T], b: &[T], weights: &[T]) -> T {
        a.iter()
            .zip(b.iter())
            .zip(weights.iter())
            .map(|((&ai, &bi), &w)| {
                let diff = ai - bi;
                w * diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x)
    }

    #[inline]
    fn manhattan_scalar<T: Float>(a: &[T], b: &[T]) -> T {
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).abs())
            .fold(T::zero(), |acc, x| acc + x)
    }

    #[inline]
    fn chebyshev_scalar<T: Float>(a: &[T], b: &[T]) -> T {
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).abs())
            .fold(T::zero(), T::max)
    }
}
