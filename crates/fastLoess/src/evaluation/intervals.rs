//! Parallel interval estimation for LOESS smoothing.
//!
//! ## Purpose
//!
//! This module provides parallel computation of standard errors for
//! confidence and prediction intervals. The computation involves
//! fitting local regressions at each point to estimate leverage,
//! which is naturally parallelizable.
//!
//! ## Design notes
//!
//! * **Parallelism**: Uses `rayon` to compute standard errors in parallel.
//! * **Integration**: Plugs into the `loess-rs` executor via the `IntervalPassFn` hook.
//! * **Generics**: Generic over `Float` types.
//!
//! ## Key concepts
//!
//! * **Standard Errors**: Computed in parallel for each query point.
//! * **Leverage**: Hat matrix diagonal elements used for uncertainty estimation.
//! * **Parallel Computation**: Independent per-point calculations for high throughput.
//!
//! ## Invariants
//!
//! * Input arrays x and y must have the same length.
//! * Window size must be sufficient for degrees of freedom.
//! * Standard errors are non-negative.
//!
//! ## Non-goals
//!
//! * This module does not compute the intervals itself (only standard errors).
//! * This module does not handle T-distribution approximations (delegated to loess-rs).

// Feature-gated imports
#[cfg(feature = "cpu")]
use rayon::prelude::*;

// External dependencies
use num_traits::Float;
use std::cmp::Ordering::Equal;
use std::fmt::Debug;
use std::vec::Vec;

// Export dependencies from loess-rs crate
use loess_rs::internals::algorithms::regression::{
    PolynomialDegree, RegressionContext, SolverLinalg, ZeroWeightFallback,
};
use loess_rs::internals::evaluation::intervals::IntervalMethod;
use loess_rs::internals::math::distance::{DistanceLinalg, DistanceMetric};
use loess_rs::internals::math::kernel::WeightFunction;
use loess_rs::internals::math::linalg::FloatLinalg;
use loess_rs::internals::math::neighborhood::{KDTree, Neighborhood, NodeDistance};
use loess_rs::internals::primitives::buffer::{FittingBuffer, NeighborhoodSearchBuffer};

use crate::engine::executor::LoessDistanceCalculator;

// ============================================================================
// Parallel Interval Estimation
// ============================================================================

/// Compute standard errors in parallel for interval estimation.
///
/// This function computes the leverage values (hat matrix diagonal) for each
/// point in parallel, then uses them to estimate standard errors.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cpu")]
pub fn interval_pass_parallel<T>(
    x: &[T],
    y: &[T],
    x_search: &[T], // Augmented data
    y_search: &[T], // Augmented values
    y_smooth: &[T],
    dims: usize,
    window_size: usize,
    robustness_weights: &[T],
    weight_function: WeightFunction,
    _interval_method: &IntervalMethod<T>,
    polynomial_degree: PolynomialDegree,
    distance_metric: &DistanceMetric<T>,
    scales: &[T],
) -> Vec<T>
where
    T: FloatLinalg + DistanceLinalg + SolverLinalg + Float + Debug + Send + Sync + 'static,
{
    let n = y.len();
    if n == 0 {
        return Vec::new();
    }

    // Build KD-Tree on AUGMENTED data
    let kdtree = KDTree::new(x_search, dims);

    // Compute residuals for sigma estimation
    let mut residuals: Vec<T> = y
        .iter()
        .zip(y_smooth.iter())
        .map(|(&yi, &si)| (yi - si).abs())
        .collect();

    // Compute sigma using MAD
    let median_idx = n / 2;
    if median_idx < residuals.len() {
        residuals.select_nth_unstable_by(median_idx, |a, b| a.partial_cmp(b).unwrap_or(Equal));
    }
    let median_residual = residuals.get(median_idx).copied().unwrap_or(T::zero());
    let sigma = median_residual * T::from(1.4826).unwrap_or(T::one());

    // Compute leverage values in parallel
    let leverages: Vec<T> = (0..n)
        .into_par_iter()
        .map_init(
            || {
                (
                    NeighborhoodSearchBuffer::<NodeDistance<T>>::new(window_size),
                    Neighborhood::<T>::new(),
                    FittingBuffer::new(window_size, dims),
                )
            },
            |(search_buffer, neighborhood, fitting_buffer), i| {
                let dist_calc = LoessDistanceCalculator {
                    metric: distance_metric,
                    scales,
                };

                let query_offset = i * dims;
                let query_point = &x[query_offset..query_offset + dims];

                // Find k-nearest neighbors in AUGMENTED data
                kdtree.find_k_nearest(
                    query_point,
                    window_size,
                    &dist_calc,
                    None,
                    search_buffer,
                    neighborhood,
                );

                // Create regression context with leverage computation enabled using AUGMENTED data
                let mut context = RegressionContext::new(
                    x_search,
                    dims,
                    y_search,
                    i,
                    Some(query_point),
                    neighborhood,
                    !robustness_weights.is_empty(),
                    robustness_weights,
                    weight_function,
                    ZeroWeightFallback::UseLocalMean,
                    polynomial_degree,
                    true, // compute_leverage
                    Some(fitting_buffer),
                );

                if let Some((_, leverage)) = context.fit() {
                    leverage
                } else {
                    T::zero()
                }
            },
        )
        .collect();

    // Compute standard errors: SE = sigma * sqrt(leverage)
    leverages
        .iter()
        .map(|&lev| {
            if lev > T::zero() {
                sigma * lev.sqrt()
            } else {
                sigma * T::from(0.1).unwrap_or(T::zero())
            }
        })
        .collect()
}

// Sequential fallback
#[cfg(not(feature = "cpu"))]
pub fn interval_pass_parallel<T>(
    _x: &[T],
    y: &[T],
    y_smooth: &[T],
    _dims: usize,
    _window_size: usize,
    _robustness_weights: &[T],
    _weight_function: WeightFunction,
    _interval_method: &IntervalMethod<T>,
    _polynomial_degree: PolynomialDegree,
    _distance_metric: &DistanceMetric<T>,
    _scales: &[T],
) -> Vec<T>
where
    T: Float + Send + Sync,
{
    // Return approximate standard errors based on residuals
    let n = y.len();
    let mut residuals: Vec<T> = y
        .iter()
        .zip(y_smooth.iter())
        .map(|(&yi, &si)| (yi - si).abs())
        .collect();

    if n == 0 {
        return Vec::new();
    }

    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = residuals.get(n / 2).copied().unwrap_or(T::zero());
    let sigma = median * T::from(1.4826).unwrap_or(T::one());

    vec![sigma; n]
}
