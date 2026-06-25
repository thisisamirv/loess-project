//! Hat matrix statistics for LOESS inference.
//!
//! This module provides leverage values and delta parameters derived from the
//! hat (smoother) matrix L, where ŷ = L * y. Used for confidence intervals
//! and effective degrees of freedom estimation.
//!
//! ## srrstats Compliance
//!
//! @srrstats {RE5.0} Hat matrix diagonal (leverage) and delta parameters for SE computation.
//! @srrstats {G1.0} Method follows Cleveland & Devlin (1988) delta1/delta2 formulation.

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

// Statistics derived from the hat (smoother) matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct HatMatrixStats<T> {
    // Leverage values (diagonal of L) at each point.
    pub leverage: Vec<T>,

    // Trace of L = sum of leverage values = ENP.
    pub trace: T,

    // Delta1 = tr((I-L)(I-L)') for residual scale estimation.
    pub delta1: T,

    // Delta2 = tr(((I-L)(I-L)')²) for SE computation.
    pub delta2: T,
}

impl<T: Float> HatMatrixStats<T> {
    // Create stats from leverage values only (approximation).
    //
    // This provides an approximation of delta1 and delta2 when the full
    // hat matrix is not available. Uses the approximation:
    // - delta1 ≈ n - 2*tr(L) + tr(L²) ≈ n - 2*tr(L) + tr(L)²/n
    // - delta2 ≈ delta1² / n
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

    // Compute residual scale estimate.
    //
    // sigma = sqrt(RSS / delta1)
    pub fn compute_residual_scale(&self, rss: T) -> T {
        if self.delta1 > T::zero() {
            (rss / self.delta1).sqrt()
        } else {
            T::zero()
        }
    }
}
