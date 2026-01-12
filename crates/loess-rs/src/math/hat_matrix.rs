//! Hat matrix computation for LOESS inference.
//!
//! ## Purpose
//!
//! This module provides functions for computing the hat matrix (smoother matrix)
//! and its derived quantities needed for proper statistical inference in LOESS:
//! - Leverage values (diagonal of L)
//! - Trace of L (Equivalent Number of Parameters)
//! - Delta parameters for proper standard error computation
//!
//! ## Background
//!
//! In LOESS, the smoothed values can be written as ŷ = L * y where L is the
//! "hat" or "smoother" matrix. The trace of L gives the Equivalent Number of
//! Parameters (ENP), which measures model complexity.
//!
//! For proper confidence intervals, we need:
//! - delta1 = tr((I-L)(I-L)') = n - 2*tr(L) + tr(L * L')
//! - delta2 = tr(((I-L)(I-L)')²)
//!
//! The residual scale is estimated as: sigma = sqrt(RSS / delta1)

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use num_traits::Float;

// ============================================================================
// Hat Matrix Statistics
// ============================================================================

/// Statistics derived from the hat (smoother) matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct HatMatrixStats<T> {
    /// Leverage values (diagonal of L) at each point.
    pub leverage: Vec<T>,

    /// Trace of L = sum of leverage values = ENP.
    pub trace: T,

    /// Delta1 = tr((I-L)(I-L)') for residual scale estimation.
    pub delta1: T,

    /// Delta2 = tr(((I-L)(I-L)')²) for SE computation.
    pub delta2: T,
}

impl<T: Float> HatMatrixStats<T> {
    /// Create stats from leverage values only (approximation).
    ///
    /// This provides an approximation of delta1 and delta2 when the full
    /// hat matrix is not available. Uses the approximation:
    /// - delta1 ≈ n - 2*tr(L) + tr(L²) ≈ n - 2*tr(L) + tr(L)²/n
    /// - delta2 ≈ delta1² / n
    pub fn from_leverage(leverage: Vec<T>) -> Self {
        let n = T::from(leverage.len()).unwrap();
        let trace = leverage.iter().fold(T::zero(), |acc, &l| acc + l);

        // Approximate tr(L*L') ≈ sum(l_ii²) (assuming L is approximately diagonal)
        let trace_l_sq = leverage.iter().fold(T::zero(), |acc, &l| acc + l * l);

        // delta1 = n - 2*tr(L) + tr(L*L')
        let delta1 = n - T::from(2.0).unwrap() * trace + trace_l_sq;

        // delta2 approximation (Cleveland et al. 1988)
        let delta2 = delta1 * delta1 / n;

        Self {
            leverage,
            trace,
            delta1,
            delta2,
        }
    }

    /// Compute residual scale estimate.
    ///
    /// sigma = sqrt(RSS / delta1)
    pub fn compute_residual_scale(&self, rss: T) -> T {
        if self.delta1 > T::zero() {
            (rss / self.delta1).sqrt()
        } else {
            T::zero()
        }
    }
}
