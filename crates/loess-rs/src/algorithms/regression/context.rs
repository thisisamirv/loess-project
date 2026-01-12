//! Regression Context and Fitting Logic
//!
//! ## Purpose
//!
//! This module defines the `RegressionContext` which captures all state needed
//! for a single local fit, and implements the high-level orchestration of the solver.

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::marker::PhantomData;

// Internal dependencies
use crate::math::kernel::WeightFunction;
use crate::math::linalg::FloatLinalg;
use crate::math::neighborhood::Neighborhood;
use crate::primitives::buffer::FittingBuffer;

// Module dependencies
use super::generic::{self, GenericTermGenerator};
use super::specialized::SolverLinalg;
use super::types::{PolynomialDegree, ZeroWeightFallback};

/// Context containing all data needed to fit a single point (unified 1D/nD).
pub struct RegressionContext<'a, T: FloatLinalg + SolverLinalg> {
    /// Independent variables (flattened if row-major)
    pub x: &'a [T],
    /// Dimensions of independent variables
    pub dimensions: usize,
    /// Dependent variables
    pub y: &'a [T],
    /// Index of the current query point (if using a point from x)
    pub query_idx: usize,
    /// Explicit query point (if not using query_idx)
    pub query_point: Option<&'a [T]>,
    /// Neighborhood of points to consider
    pub neighborhood: &'a Neighborhood<T>,
    /// Whether to use robustness weights
    pub use_robustness: bool,
    /// Robustness weights for the neighborhood
    pub robustness_weights: &'a [T],
    /// Weight function (kernel)
    pub weight_function: WeightFunction,
    /// Fallback strategy for zero-weight cases
    pub zero_weight_fallback: ZeroWeightFallback,
    /// Degree of polynomial to fit
    pub polynomial_degree: PolynomialDegree,
    /// Whether to compute leverage (diagonal of hat matrix)
    pub compute_leverage: bool,
    /// Persistent buffer for reuse
    pub buffer: Option<&'a mut FittingBuffer<T>>,
    _phantom: PhantomData<T>,
}

impl<'a, T: FloatLinalg + SolverLinalg> RegressionContext<'a, T> {
    /// Create a new regression context for a specific query point.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        x: &'a [T],
        dimensions: usize,
        y: &'a [T],
        query_idx: usize,
        query_point: Option<&'a [T]>,
        neighborhood: &'a Neighborhood<T>,
        use_robustness: bool,
        robustness_weights: &'a [T],
        weight_function: WeightFunction,
        zero_weight_fallback: ZeroWeightFallback,
        polynomial_degree: PolynomialDegree,
        compute_leverage: bool,
        buffer: Option<&'a mut FittingBuffer<T>>,
    ) -> Self {
        Self {
            x,
            dimensions,
            y,
            query_idx,
            query_point,
            neighborhood,
            use_robustness,
            robustness_weights,
            weight_function,
            zero_weight_fallback,
            polynomial_degree,
            compute_leverage,
            buffer,
            _phantom: PhantomData,
        }
    }

    /// Returns the (predicted value, leverage) at the query point.
    pub fn fit(&mut self) -> Option<(T, T)> {
        let n_neighbors = self.neighborhood.len();
        if n_neighbors == 0 {
            return None;
        }

        let d = self.dimensions;
        let n_coeffs = self.polynomial_degree.num_coefficients_nd(d);
        let max_distance = self.neighborhood.max_distance;

        // Handle zero bandwidth case (all neighbors at same location)
        if max_distance <= T::epsilon() {
            let (val, sum_w) = self.weighted_mean_and_sum();
            let leverage = if sum_w > T::epsilon() {
                T::one() / sum_w
            } else {
                T::zero()
            };
            return Some((val, leverage));
        }

        // Get query point coordinates
        let query_point = if let Some(qp) = self.query_point {
            qp
        } else {
            let query_offset = self.query_idx * d;
            &self.x[query_offset..query_offset + d]
        };

        // For constant degree, just compute weighted mean
        if self.polynomial_degree == PolynomialDegree::Constant {
            let (val, sum_w) = self.weighted_mean_and_sum();
            let leverage = if sum_w > T::epsilon() {
                T::one() / sum_w
            } else {
                T::zero()
            };
            return Some((val, leverage));
        }

        let mut buffer = self.buffer.take();
        let result = if let Some(ref mut buf) = buffer {
            buf.weights.clear();
            let weights = &mut buf.weights;

            for i in 0..n_neighbors {
                let neighbor_idx = self.neighborhood.indices[i];
                let dist = self.neighborhood.distances[i];
                let u = dist / max_distance;
                let kernel_w = self.weight_function.compute_weight(u);
                let w = if self.use_robustness {
                    kernel_w * self.robustness_weights[neighbor_idx]
                } else {
                    kernel_w
                };
                weights.push(w);
            }

            // Check numerical stability of weights
            let weight_sum: T = weights.iter().copied().fold(T::zero(), |a, b| a + b);
            if weight_sum <= T::epsilon() {
                self.buffer = buffer;
                return self.handle_zero_weights_fit();
            }

            buf.xtw_x.clear();
            buf.xtw_x.resize(n_coeffs * n_coeffs, T::zero());
            buf.xtw_y.clear();
            buf.xtw_y.resize(n_coeffs, T::zero());

            self.fit_polynomial_wls_internal(
                weights,
                query_point,
                n_coeffs,
                &mut buf.xtw_x,
                &mut buf.xtw_y,
            )
        } else {
            let mut weights = Vec::with_capacity(n_neighbors);
            for i in 0..n_neighbors {
                let neighbor_idx = self.neighborhood.indices[i];
                let dist = self.neighborhood.distances[i];
                let u = dist / max_distance;
                let kernel_w = self.weight_function.compute_weight(u);
                let w = if self.use_robustness {
                    kernel_w * self.robustness_weights[neighbor_idx]
                } else {
                    kernel_w
                };
                weights.push(w);
            }

            // Check numerical stability of weights
            let weight_sum: T = weights.iter().copied().fold(T::zero(), |a, b| a + b);
            if weight_sum <= T::epsilon() {
                self.buffer = buffer;
                return self.handle_zero_weights_fit();
            }

            let mut xtw_x = vec![T::zero(); n_coeffs * n_coeffs];
            let mut xtw_y = vec![T::zero(); n_coeffs];

            self.fit_polynomial_wls_internal(
                &weights,
                query_point,
                n_coeffs,
                &mut xtw_x,
                &mut xtw_y,
            )
        };
        self.buffer = buffer;
        result
    }

    /// Returns all polynomial coefficients [value, d/dx1, d/dx2, ..., d/dxd] at the query point.
    pub fn fit_with_coefficients(&mut self) -> Option<Vec<T>> {
        let n_neighbors = self.neighborhood.len();
        if n_neighbors == 0 {
            return None;
        }

        let d = self.dimensions;
        let n_coeffs = self.polynomial_degree.num_coefficients_nd(d);
        let max_distance = self.neighborhood.max_distance;

        // Handle zero bandwidth or constant degree - return [value, 0, 0, ...]
        if max_distance <= T::epsilon() || self.polynomial_degree == PolynomialDegree::Constant {
            let (val, _) = self.weighted_mean_and_sum();
            let mut coeffs = vec![T::zero(); d + 1];
            coeffs[0] = val;
            return Some(coeffs);
        }

        let query_point = if let Some(qp) = self.query_point {
            qp
        } else {
            let query_offset = self.query_idx * d;
            &self.x[query_offset..query_offset + d]
        };

        // Build normal equations and solve
        let mut buffer = self.buffer.take();
        let result = if let Some(ref mut buf) = buffer {
            buf.weights.clear();
            let weights = &mut buf.weights;

            for i in 0..n_neighbors {
                let neighbor_idx = self.neighborhood.indices[i];
                let dist = self.neighborhood.distances[i];
                let u = dist / max_distance;
                let kernel_w = self.weight_function.compute_weight(u);
                let w = if self.use_robustness {
                    kernel_w * self.robustness_weights[neighbor_idx]
                } else {
                    kernel_w
                };
                weights.push(w);
            }

            buf.xtw_x.clear();
            buf.xtw_x.resize(n_coeffs * n_coeffs, T::zero());
            buf.xtw_y.clear();
            buf.xtw_y.resize(n_coeffs, T::zero());

            // Accumulate normal equations
            match (self.dimensions, self.polynomial_degree) {
                (1, PolynomialDegree::Linear) => {
                    let mut a = [T::zero(); 4];
                    let mut b = [T::zero(); 2];
                    T::accumulate_1d_linear(
                        self.x,
                        self.y,
                        &self.neighborhood.indices,
                        weights,
                        query_point[0],
                        &mut a,
                        &mut b,
                    );
                    buf.xtw_x[..4].copy_from_slice(&a);
                    buf.xtw_y[..2].copy_from_slice(&b);
                }
                (1, PolynomialDegree::Quadratic) => {
                    let mut a = [T::zero(); 9];
                    let mut b = [T::zero(); 3];
                    T::accumulate_1d_quadratic(
                        self.x,
                        self.y,
                        &self.neighborhood.indices,
                        weights,
                        query_point[0],
                        &mut a,
                        &mut b,
                    );
                    buf.xtw_x[..9].copy_from_slice(&a);
                    buf.xtw_y[..3].copy_from_slice(&b);
                }
                (1, PolynomialDegree::Cubic) => {
                    let mut a = [T::zero(); 16];
                    let mut b = [T::zero(); 4];
                    T::accumulate_1d_cubic(
                        self.x,
                        self.y,
                        &self.neighborhood.indices,
                        weights,
                        query_point[0],
                        &mut a,
                        &mut b,
                    );
                    buf.xtw_x[..16].copy_from_slice(&a);
                    buf.xtw_y[..4].copy_from_slice(&b);
                }
                (2, PolynomialDegree::Linear) => {
                    let mut a = [T::zero(); 9];
                    let mut b = [T::zero(); 3];
                    T::accumulate_2d_linear(
                        self.x,
                        self.y,
                        &self.neighborhood.indices,
                        weights,
                        query_point[0],
                        query_point[1],
                        &mut a,
                        &mut b,
                    );
                    buf.xtw_x[..9].copy_from_slice(&a);
                    buf.xtw_y[..3].copy_from_slice(&b);
                }
                (2, PolynomialDegree::Quadratic) => {
                    let mut a = [T::zero(); 36];
                    let mut b = [T::zero(); 6];
                    T::accumulate_2d_quadratic(
                        self.x,
                        self.y,
                        &self.neighborhood.indices,
                        weights,
                        query_point[0],
                        query_point[1],
                        &mut a,
                        &mut b,
                    );
                    buf.xtw_x[..36].copy_from_slice(&a);
                    buf.xtw_y[..6].copy_from_slice(&b);
                }
                (2, PolynomialDegree::Cubic) => {
                    // 10 coefficients -> 100 elements in XT W X, 10 in XT W y
                    let mut a = [T::zero(); 100];
                    let mut b = [T::zero(); 10];
                    T::accumulate_2d_cubic(
                        self.x,
                        self.y,
                        &self.neighborhood.indices,
                        weights,
                        query_point[0],
                        query_point[1],
                        &mut a,
                        &mut b,
                    );
                    buf.xtw_x[..100].copy_from_slice(&a);
                    buf.xtw_y[..10].copy_from_slice(&b);
                }
                (3, PolynomialDegree::Linear) => {
                    let mut a = [T::zero(); 16];
                    let mut b = [T::zero(); 4];
                    T::accumulate_3d_linear(
                        self.x,
                        self.y,
                        &self.neighborhood.indices,
                        weights,
                        query_point[0],
                        query_point[1],
                        query_point[2],
                        &mut a,
                        &mut b,
                    );
                    buf.xtw_x[..16].copy_from_slice(&a);
                    buf.xtw_y[..4].copy_from_slice(&b);
                }
                (3, PolynomialDegree::Quadratic) => {
                    let mut a = [T::zero(); 100];
                    let mut b = [T::zero(); 10];
                    T::accumulate_3d_quadratic(
                        self.x,
                        self.y,
                        &self.neighborhood.indices,
                        weights,
                        query_point[0],
                        query_point[1],
                        query_point[2],
                        &mut a,
                        &mut b,
                    );
                    buf.xtw_x[..100].copy_from_slice(&a);
                    buf.xtw_y[..10].copy_from_slice(&b);
                }
                _ => {
                    let term_gen = GenericTermGenerator::new(
                        self.polynomial_degree,
                        self.dimensions,
                        n_coeffs,
                    );
                    generic::accumulate_normal_equations(
                        self.x,
                        self.y,
                        &self.neighborhood.indices,
                        weights,
                        term_gen,
                        query_point,
                        &mut buf.xtw_x,
                        &mut buf.xtw_y,
                    );
                }
            }

            // Column equilibration
            buf.col_norms.resize(n_coeffs, T::one());
            let col_norms = &mut buf.col_norms;
            for (j, norm) in col_norms.iter_mut().enumerate() {
                let diag = buf.xtw_x[j * n_coeffs + j];
                if diag > T::epsilon() {
                    *norm = diag.sqrt();
                } else {
                    *norm = T::one();
                }
            }
            for i in 0..n_coeffs {
                for j in 0..n_coeffs {
                    let idx = i * n_coeffs + j;
                    buf.xtw_x[idx] = buf.xtw_x[idx] / (col_norms[i] * col_norms[j]);
                }
                buf.xtw_y[i] = buf.xtw_y[i] / col_norms[i];
            }

            // Solve
            let beta = T::solve_normal(&buf.xtw_x, &buf.xtw_y, n_coeffs);

            if let Some(mut coeffs) = beta {
                // Undo equilibration
                for j in 0..n_coeffs {
                    coeffs[j] = coeffs[j] / col_norms[j];
                }
                let mut res_vec = vec![T::zero(); d + 1];
                for (i, &c) in coeffs.iter().take(d + 1).enumerate() {
                    res_vec[i] = c;
                }
                Some(res_vec)
            } else {
                None
            }
        } else {
            // Non-buffered version...
            None
        };

        self.buffer = buffer;
        if result.is_some() {
            return result;
        }

        // Fallback: return [mean, 0, 0, ...]
        let (val, _) = self.weighted_mean_and_sum();
        let mut coeffs = vec![T::zero(); d + 1];
        coeffs[0] = val;
        Some(coeffs)
    }

    /// Handle zero weight cases using fallback policy.
    fn handle_zero_weights_fit(&self) -> Option<(T, T)> {
        match self.zero_weight_fallback {
            ZeroWeightFallback::UseLocalMean => {
                let n_f = T::from(self.neighborhood.len()).unwrap_or_else(|| T::one());
                let sum_y = self
                    .neighborhood
                    .indices
                    .iter()
                    .map(|&i| self.y[i])
                    .fold(T::zero(), |a, b| a + b);
                Some((sum_y / n_f, T::zero()))
            }
            ZeroWeightFallback::ReturnOriginal => Some((self.y[self.query_idx], T::zero())),
            ZeroWeightFallback::ReturnNone => None,
        }
    }

    /// Compute weighted mean and sum of weights.
    fn weighted_mean_and_sum(&self) -> (T, T) {
        let mut sum_wy = T::zero();
        let mut sum_w = T::zero();
        let max_dist = self.neighborhood.max_distance;
        let bandwidth = if max_dist > T::epsilon() {
            max_dist
        } else {
            T::one()
        };

        for i in 0..self.neighborhood.len() {
            let idx = self.neighborhood.indices[i];
            let dist = self.neighborhood.distances[i];
            let u = dist / bandwidth;
            let kernel_w = self.weight_function.compute_weight(u);
            let w = if self.use_robustness {
                kernel_w * self.robustness_weights[idx]
            } else {
                kernel_w
            };
            sum_wy = sum_wy + w * self.y[idx];
            sum_w = sum_w + w;
        }

        let val = if sum_w > T::epsilon() {
            sum_wy / sum_w
        } else {
            let n_f = T::from(self.neighborhood.len()).unwrap_or_else(|| T::one());
            self.neighborhood
                .indices
                .iter()
                .map(|&i| self.y[i])
                .fold(T::zero(), |a, b| a + b)
                / n_f
        };
        (val, sum_w)
    }

    /// Internal WLS solver.
    fn fit_polynomial_wls_internal(
        &self,
        weights: &[T],
        query_point: &[T],
        n_coeffs: usize,
        xtw_x: &mut [T],
        xtw_y: &mut [T],
    ) -> Option<(T, T)> {
        let n_neighbors = self.neighborhood.len();
        if n_neighbors < n_coeffs {
            let (val, sum_w) = self.weighted_mean_and_sum();
            let leverage = if sum_w > T::epsilon() {
                T::one() / sum_w
            } else {
                T::zero()
            };
            return Some((val, leverage));
        }

        // 1. Accumulate Normal Equations
        match (self.dimensions, self.polynomial_degree) {
            (1, PolynomialDegree::Linear) => {
                let mut a = [T::zero(); 4];
                let mut b = [T::zero(); 2];
                T::accumulate_1d_linear(
                    self.x,
                    self.y,
                    &self.neighborhood.indices,
                    weights,
                    query_point[0],
                    &mut a,
                    &mut b,
                );
                xtw_x[..4].copy_from_slice(&a);
                xtw_y[..2].copy_from_slice(&b);
            }
            (1, PolynomialDegree::Quadratic) => {
                let mut a = [T::zero(); 9];
                let mut b = [T::zero(); 3];
                T::accumulate_1d_quadratic(
                    self.x,
                    self.y,
                    &self.neighborhood.indices,
                    weights,
                    query_point[0],
                    &mut a,
                    &mut b,
                );
                xtw_x[..9].copy_from_slice(&a);
                xtw_y[..3].copy_from_slice(&b);
            }
            (1, PolynomialDegree::Cubic) => {
                let mut a = [T::zero(); 16];
                let mut b = [T::zero(); 4];
                T::accumulate_1d_cubic(
                    self.x,
                    self.y,
                    &self.neighborhood.indices,
                    weights,
                    query_point[0],
                    &mut a,
                    &mut b,
                );
                xtw_x[..16].copy_from_slice(&a);
                xtw_y[..4].copy_from_slice(&b);
            }
            (2, PolynomialDegree::Linear) => {
                let mut a = [T::zero(); 9];
                let mut b = [T::zero(); 3];
                T::accumulate_2d_linear(
                    self.x,
                    self.y,
                    &self.neighborhood.indices,
                    weights,
                    query_point[0],
                    query_point[1],
                    &mut a,
                    &mut b,
                );
                xtw_x[..9].copy_from_slice(&a);
                xtw_y[..3].copy_from_slice(&b);
            }
            (2, PolynomialDegree::Quadratic) => {
                let mut a = [T::zero(); 36];
                let mut b = [T::zero(); 6];
                T::accumulate_2d_quadratic(
                    self.x,
                    self.y,
                    &self.neighborhood.indices,
                    weights,
                    query_point[0],
                    query_point[1],
                    &mut a,
                    &mut b,
                );
                xtw_x[..36].copy_from_slice(&a);
                xtw_y[..6].copy_from_slice(&b);
            }
            (2, PolynomialDegree::Cubic) => {
                let mut a = [T::zero(); 100];
                let mut b = [T::zero(); 10];
                T::accumulate_2d_cubic(
                    self.x,
                    self.y,
                    &self.neighborhood.indices,
                    weights,
                    query_point[0],
                    query_point[1],
                    &mut a,
                    &mut b,
                );
                xtw_x[..100].copy_from_slice(&a);
                xtw_y[..10].copy_from_slice(&b);
            }
            (3, PolynomialDegree::Linear) => {
                let mut a = [T::zero(); 16];
                let mut b = [T::zero(); 4];
                T::accumulate_3d_linear(
                    self.x,
                    self.y,
                    &self.neighborhood.indices,
                    weights,
                    query_point[0],
                    query_point[1],
                    query_point[2],
                    &mut a,
                    &mut b,
                );
                xtw_x[..16].copy_from_slice(&a);
                xtw_y[..4].copy_from_slice(&b);
            }
            (3, PolynomialDegree::Quadratic) => {
                let mut a = [T::zero(); 100];
                let mut b = [T::zero(); 10];
                T::accumulate_3d_quadratic(
                    self.x,
                    self.y,
                    &self.neighborhood.indices,
                    weights,
                    query_point[0],
                    query_point[1],
                    query_point[2],
                    &mut a,
                    &mut b,
                );
                xtw_x[..100].copy_from_slice(&a);
                xtw_y[..10].copy_from_slice(&b);
            }
            _ => {
                let term_gen =
                    GenericTermGenerator::new(self.polynomial_degree, self.dimensions, n_coeffs);
                generic::accumulate_normal_equations(
                    self.x,
                    self.y,
                    &self.neighborhood.indices,
                    weights,
                    term_gen,
                    query_point,
                    xtw_x,
                    xtw_y,
                );
            }
        }

        // Unified solvent using solve_normal with equilibration
        let mut col_norms = vec![T::one(); n_coeffs];
        for j in 0..n_coeffs {
            let diag = xtw_x[j * n_coeffs + j];
            if diag > T::epsilon() {
                col_norms[j] = diag.sqrt();
            }
        }

        // Apply equilibration: scale rows and columns of X'WX
        for i in 0..n_coeffs {
            for j in 0..n_coeffs {
                let idx = i * n_coeffs + j;
                xtw_x[idx] = xtw_x[idx] / (col_norms[i] * col_norms[j]);
            }
            // Also scale X'Wy
            xtw_y[i] = xtw_y[i] / col_norms[i];
        }

        // Solve the equilibrated system
        let result = T::solve_normal(xtw_x, xtw_y, n_coeffs).map(|mut coeffs| {
            // Undo equilibration on the solution
            for j in 0..n_coeffs {
                coeffs[j] = coeffs[j] / col_norms[j];
            }
            let leverage = if self.compute_leverage {
                // Solve A' * x' = e1 to get (A'^-1)_00
                // Leverage = (A^-1)_00 = (A'^-1)_00 / col_norms[0]^2
                let mut e1 = vec![T::zero(); n_coeffs];
                e1[0] = T::one();
                T::solve_normal(xtw_x, &e1, n_coeffs)
                    .map(|x| x[0] / (col_norms[0] * col_norms[0]))
                    .unwrap_or(T::zero())
            } else {
                T::zero()
            };
            (coeffs[0], leverage)
        });

        if let Some(res) = result {
            return Some(res);
        }

        // Fallback or Singularity Handling
        let (val, sum_w) = self.weighted_mean_and_sum();
        if sum_w > T::epsilon() {
            Some((val, T::one() / sum_w))
        } else {
            self.handle_zero_weights_fit()
        }
    }
}
