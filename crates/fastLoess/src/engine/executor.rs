//! Parallel execution engine for LOESS smoothing operations.
//!
//! ## Purpose
//!
//! This module provides parallel smoothing functions that are injected into
//! the `loess-rs` crate's execution engine. It enables multi-threaded execution
//! of the local regression fits, significantly speeding up LOESS smoothing
//! for large datasets by utilizing all available CPU cores.
//!
//! ## Design notes
//!
//! * **Implementation**: Provides drop-in replacement for the sequential smoothing pass.
//! * **Parallelism**: Uses `rayon` for data-parallel execution across CPU cores.
//! * **Optimization**: Reuses buffers per thread to minimize allocations.
//! * **Multi-dimensional**: Fully supports N-dimensional LOESS with configurable distance metrics.
//! * **Generics**: Generic over `Float` types.
//!
//! ## Key concepts
//!
//! * **Parallel Fitting**: Distributes points across CPU cores independently.
//! * **Buffer Reuse**: Thread-local scratch buffers to avoid allocation overhead.
//! * **Integration**: Plugs into the `loess-rs` executor via the `SmoothPassFn` hook.
//!
//! ## Invariants
//!
//! * Input slices must have matching lengths.
//! * All values must be finite.
//! * Window size is at least 1 and at most n.
//!
//! ## Non-goals
//!
//! * This module does not handle the iteration loop (handled by `loess-rs::executor`).
//! * This module does not validate input data (handled by `validator`).

// Feature-gated imports
#[cfg(feature = "cpu")]
use rayon::prelude::*;

// External dependencies
use num_traits::Float;
use std::fmt::Debug;

// Export dependencies from loess-rs crate
use loess_rs::internals::algorithms::regression::{
    PolynomialDegree, RegressionContext, SolverLinalg, ZeroWeightFallback,
};

use loess_rs::internals::math::distance::{DistanceLinalg, DistanceMetric};
use loess_rs::internals::math::kernel::WeightFunction;
use loess_rs::internals::math::linalg::FloatLinalg;
use loess_rs::internals::math::neighborhood::{KDTree, Neighborhood, NodeDistance, PointDistance};
use loess_rs::internals::primitives::buffer::{
    CachedNeighborhood, FittingBuffer, NeighborhoodSearchBuffer,
};

// ============================================================================
// LOESS Distance Calculator
// ============================================================================

/// Standard LOESS distance calculator for neighbor finding.
pub struct LoessDistanceCalculator<'a, T: FloatLinalg + DistanceLinalg + SolverLinalg> {
    /// The distance metric to use.
    pub metric: &'a DistanceMetric<T>,
    /// Normalization scales for each dimension.
    pub scales: &'a [T],
}

impl<'a, T: FloatLinalg + DistanceLinalg + SolverLinalg> PointDistance<T>
    for LoessDistanceCalculator<'a, T>
{
    fn split_distance(&self, dim: usize, split_val: T, query_val: T) -> T {
        let diff = (query_val - split_val).abs();
        match self.metric {
            DistanceMetric::Normalized => diff * self.scales[dim],
            DistanceMetric::Euclidean => diff,
            DistanceMetric::Manhattan => diff,
            DistanceMetric::Chebyshev => diff,
            DistanceMetric::Minkowski(_) => diff,
            DistanceMetric::Weighted(w) => diff * w[dim].sqrt(),
        }
    }

    fn distance_squared(&self, a: &[T], b: &[T]) -> T {
        match self.metric {
            DistanceMetric::Normalized => DistanceMetric::normalized_squared(a, b, self.scales),
            DistanceMetric::Euclidean => DistanceMetric::euclidean_squared(a, b),
            DistanceMetric::Weighted(w) => DistanceMetric::weighted_squared(a, b, w),
            DistanceMetric::Manhattan => DistanceMetric::manhattan_squared(a, b),
            DistanceMetric::Chebyshev => DistanceMetric::chebyshev_squared(a, b),
            DistanceMetric::Minkowski(p) => DistanceMetric::minkowski_squared(a, b, *p),
        }
    }

    fn split_distance_squared(&self, dim: usize, split_val: T, query_val: T) -> T {
        let diff = query_val - split_val;
        match self.metric {
            DistanceMetric::Normalized => {
                let d = diff * self.scales[dim];
                d * d
            }
            DistanceMetric::Euclidean => diff * diff,
            DistanceMetric::Weighted(w) => diff * diff * w[dim],
            _ => {
                let d = self.split_distance(dim, split_val, query_val);
                d * d
            }
        }
    }

    fn post_process_distance(&self, d: T) -> T {
        d.sqrt()
    }
}

// ============================================================================
// Parallel Smoothing Function
// ============================================================================

/// Perform a single smoothing pass over all points in parallel.
///
/// This function is designed to be injected via the `SmoothPassFn` hook.
/// It parallelizes the local regression fitting process using rayon.
///
/// # Parameters
///
/// * `x` - Input x-values (flattened, row-major for multi-dimensional)
/// * `y` - Input y-values
/// * `dims` - Number of dimensions
/// * `window_size` - Number of neighbors to use
/// * `use_robustness` - Whether to apply robustness weights
/// * `robustness_weights` - Weights from robustness iteration
/// * `y_smooth` - Output buffer for smoothed values
/// * `weight_function` - Kernel weight function
/// * `zero_weight_fallback` - Fallback strategy for zero weights
/// * `polynomial_degree` - Degree of local polynomial (Linear, Quadratic)
/// * `distance_metric` - Distance metric for neighbor finding
/// * `scales` - Normalization scales per dimension
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cpu")]
pub fn smooth_pass_parallel<T>(
    x: &[T],
    y: &[T],
    x_search: &[T], // Augmented data
    y_search: &[T], // Augmented values
    dims: usize,
    window_size: usize,
    use_robustness: bool,
    robustness_weights: &[T],
    y_smooth: &mut [T],
    weight_function: WeightFunction,
    zero_weight_fallback: ZeroWeightFallback,
    polynomial_degree: PolynomialDegree,
    distance_metric: &DistanceMetric<T>,
    scales: &[T],
) where
    T: FloatLinalg + DistanceLinalg + SolverLinalg + Float + Debug + Send + Sync + 'static,
{
    let n = y.len();
    if n == 0 {
        return;
    }

    // Build KD-Tree for efficient neighbor finding on AUGMENTED data
    let kdtree = KDTree::new(x_search, dims);

    // Parallel iteration over all points
    let smoothed_values: Vec<T> = (0..n)
        .into_par_iter()
        .map_init(
            || {
                // Thread-local buffers
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

                // Create regression context and fit using AUGMENTED data
                let mut context = RegressionContext::new(
                    x_search,
                    dims,
                    y_search,
                    i,
                    Some(query_point),
                    neighborhood,
                    use_robustness,
                    robustness_weights,
                    weight_function,
                    zero_weight_fallback,
                    polynomial_degree,
                    false, // compute_leverage
                    Some(fitting_buffer),
                );

                if let Some((val, _)) = context.fit() {
                    val
                } else {
                    y[i]
                }
            },
        )
        .collect();

    // Copy results to output
    y_smooth[..n].copy_from_slice(&smoothed_values);
}

/// Perform a parallel vertex pass for interpolation mode.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cpu")]
pub fn vertex_pass_parallel<T>(
    x: &[T], // augmented x
    y: &[T], // augmented y
    dims: usize,
    vertices: &[T],
    window_size: usize,
    use_robustness: bool,
    robustness_weights: &[T],
    vertex_data_out: &mut [T],
    existing_neighborhoods: Option<&[CachedNeighborhood<T>]>,
    output_neighborhoods: &mut Vec<CachedNeighborhood<T>>,
    weight_function: WeightFunction,
    zero_weight_fallback: ZeroWeightFallback,
    polynomial_degree: PolynomialDegree,
    distance_metric: &DistanceMetric<T>,
    scales: &[T],
    boundary_degree_fallback: bool,
) where
    T: FloatLinalg + DistanceLinalg + SolverLinalg + Float + Debug + Send + Sync + 'static,
{
    let n_vertices = vertices.len() / dims;
    let stride = dims + 1;
    let n_data = y.len();

    // Compute tight bounds for Boundary Degree Fallback
    let mut tight_lower = vec![T::infinity(); dims];
    let mut tight_upper = vec![T::neg_infinity(); dims];
    let n_points = x.len() / dims;

    for i in 0..n_points {
        for d in 0..dims {
            let val = x[i * dims + d];
            if val < tight_lower[d] {
                tight_lower[d] = val;
            }
            if val > tight_upper[d] {
                tight_upper[d] = val;
            }
        }
    }

    // KD-Tree is only needed if we don't have existing neighborhoods
    let kdtree_opt = if existing_neighborhoods.is_none() {
        Some(KDTree::new(x, dims))
    } else {
        None
    };

    let results: Vec<(Vec<T>, Option<CachedNeighborhood<T>>)> = (0..n_vertices)
        .into_par_iter()
        .map_init(
            || {
                (
                    NeighborhoodSearchBuffer::<NodeDistance<T>>::new(window_size),
                    Neighborhood::<T>::new(),
                    FittingBuffer::new(window_size, dims),
                )
            },
            |(search_buffer, neighborhood, fitting_buffer), v_idx| {
                let v_start = v_idx * dims;
                let vertex = &vertices[v_start..v_start + dims];

                // Boundary Degree Fallback logic
                let is_outside = (0..dims).any(|d| {
                    vertex[d] < tight_lower[d] - T::epsilon()
                        || vertex[d] > tight_upper[d] + T::epsilon()
                });

                let effective_degree =
                    if boundary_degree_fallback && is_outside && polynomial_degree.value() > 1 {
                        PolynomialDegree::Linear
                    } else {
                        polynomial_degree
                    };

                let mut cached_opt = None;

                if let Some(existing) = existing_neighborhoods {
                    let cached = &existing[v_idx];
                    neighborhood.indices.clear();
                    neighborhood.indices.extend_from_slice(&cached.indices);
                    neighborhood.distances.clear();
                    neighborhood.distances.extend_from_slice(&cached.distances);
                    neighborhood.max_distance = cached.max_distance;
                } else if let Some(ref kdtree) = kdtree_opt {
                    let dist_calc = LoessDistanceCalculator {
                        metric: distance_metric,
                        scales,
                    };
                    kdtree.find_k_nearest(
                        vertex,
                        window_size,
                        &dist_calc,
                        None,
                        search_buffer,
                        neighborhood,
                    );

                    cached_opt = Some(CachedNeighborhood {
                        indices: neighborhood.indices.clone(),
                        distances: neighborhood.distances.clone(),
                        max_distance: neighborhood.max_distance,
                    });
                }

                if neighborhood.is_empty() {
                    let mean =
                        y.iter().copied().fold(T::zero(), |a, b| a + b) / T::from(n_data).unwrap();
                    let mut coeffs = vec![T::zero(); stride];
                    coeffs[0] = mean;
                    return (coeffs, cached_opt);
                }

                let mut context = RegressionContext::new(
                    x,
                    dims,
                    y,
                    0, // query_idx unused
                    Some(vertex),
                    neighborhood,
                    use_robustness,
                    robustness_weights,
                    weight_function,
                    zero_weight_fallback,
                    effective_degree,
                    false, // compute_leverage
                    Some(fitting_buffer),
                );

                let coeffs = context.fit_with_coefficients().unwrap_or_else(|| {
                    let mean =
                        y.iter().copied().fold(T::zero(), |a, b| a + b) / T::from(n_data).unwrap();
                    let mut c = vec![T::zero(); stride];
                    c[0] = mean;
                    c
                });

                (coeffs, cached_opt)
            },
        )
        .collect();

    // Collect results sequentially to preserve order
    for (v_idx, (coeffs, cached_opt)) in results.into_iter().enumerate() {
        let base_idx = v_idx * stride;
        for (i, &c) in coeffs.iter().take(stride).enumerate() {
            vertex_data_out[base_idx + i] = c;
        }
        if let Some(cached) = cached_opt {
            output_neighborhoods.push(cached);
        }
    }
}

// Sequential fallback (when cpu feature is not enabled)
#[cfg(not(feature = "cpu"))]
pub fn vertex_pass_parallel<T>(
    _x: &[T],
    _y: &[T],
    _dims: usize,
    _vertices: &[T],
    _window_size: usize,
    _use_robustness: bool,
    _robustness_weights: &[T],
    _vertex_data_out: &mut [T],
    _existing_neighborhoods: Option<&[CachedNeighborhood<T>]>,
    _output_neighborhoods: &mut Vec<CachedNeighborhood<T>>,
    _weight_function: WeightFunction,
    _zero_weight_fallback: ZeroWeightFallback,
    _polynomial_degree: PolynomialDegree,
    _distance_metric: &DistanceMetric<T>,
    _scales: &[T],
    _boundary_degree_fallback: bool,
) where
    T: Float + Send + Sync,
{
}
