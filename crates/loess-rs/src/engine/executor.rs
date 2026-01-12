//! Execution engine for LOESS smoothing operations.
//!
//! ## Purpose
//!
//! This module provides the core execution engine that orchestrates LOESS
//! smoothing operations. It handles the iteration loop, robustness weight
//! updates, convergence checking, cross-validation, and variance estimation.
//! The executor is the central component that coordinates all lower-level
//! algorithms to produce smoothed results.
//!
//! ## Design notes
//!
//! * Provides both configuration-based and parameter-based entry points.
//! * Handles cross-validation for automatic fraction selection.
//! * Supports auto-convergence for adaptive iteration counts.
//! * Manages working buffers efficiently to minimize allocations.
//! * Separates concerns: fitting, interpolation, robustness, convergence.
//! * Generic over `Float` types to support f32 and f64.
//!
//! ## Key concepts
//!
//! * Execution loop: The central iteration cycle (Fit -> Residuals -> Weights -> Repeat).
//! * Auto-convergence: Dynamically stopping iterations when parameters stabilize.
//!
//! ## Invariants
//!
//! * Input x-values are assumed to be monotonically increasing (sorted).
//! * All working buffers have the same length as input data.
//! * Robustness weights are always in [0, 1].
//! * Window size is at least 2 and at most n.
//! * Iteration count is non-negative.
//!
//! ## Non-goals
//!
//! * This module does not validate input data (handled by `validator`).
//! * This module does not sort input data (caller's responsibility).
//! * This module does not provide public-facing result formatting.

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::cmp::Ordering::Equal;
use core::fmt::Debug;
use num_traits::Float;

// Internal dependencies
use crate::algorithms::interpolation::InterpolationSurface;
use crate::algorithms::regression::{
    PolynomialDegree, RegressionContext, SolverLinalg, ZeroWeightFallback,
};
use crate::algorithms::robustness::RobustnessMethod;
use crate::evaluation::cv::CVKind;
use crate::evaluation::intervals::IntervalMethod;
use crate::math::boundary::BoundaryPolicy;
use crate::math::distance::{DistanceLinalg, DistanceMetric};
use crate::math::kernel::WeightFunction;
use crate::math::linalg::FloatLinalg;
use crate::math::neighborhood::{KDTree, Neighborhood, NodeDistance, PointDistance};
use crate::math::scaling::ScalingMethod;
use crate::primitives::backend::Backend;
use crate::primitives::buffer::{
    CachedNeighborhood, FittingBuffer, LoessBuffer, NeighborhoodSearchBuffer,
};
use crate::primitives::window::Window;

/// Standard LOESS distance calculator.
///
/// Implements `PointDistance` using either Euclidean or Normalized Euclidean metrics.
pub struct LoessDistanceCalculator<'a, T: FloatLinalg + DistanceLinalg + SolverLinalg> {
    /// The distance metric to use (Euclidean or Normalized).
    pub metric: DistanceMetric<T>,
    /// Normalization scales for each dimension (used if metric is Normalized).
    pub scales: &'a [T],
}

impl<'a, T: FloatLinalg + DistanceLinalg + SolverLinalg> PointDistance<T>
    for LoessDistanceCalculator<'a, T>
{
    fn split_distance(&self, dim: usize, split_val: T, query_val: T) -> T {
        let diff = (query_val - split_val).abs();
        match &self.metric {
            DistanceMetric::Normalized => diff * self.scales[dim],
            DistanceMetric::Euclidean => diff,
            DistanceMetric::Manhattan => diff,
            DistanceMetric::Chebyshev => diff,
            DistanceMetric::Minkowski(_) => diff,
            DistanceMetric::Weighted(w) => diff * w[dim].sqrt(),
        }
    }

    fn distance_squared(&self, a: &[T], b: &[T]) -> T {
        match &self.metric {
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
        match &self.metric {
            DistanceMetric::Normalized => {
                let d = diff * self.scales[dim];
                d * d
            }
            DistanceMetric::Euclidean => diff * diff,
            DistanceMetric::Weighted(w) => diff * diff * w[dim],
            // Squaring the linear split distance |diff| is valid for comparison
            // against squared total distances due to monotonicity.
            _ => {
                let d = self.split_distance(dim, split_val, query_val);
                d * d
            }
        }
    }

    fn post_process_distance(&self, d: T) -> T {
        // Get the linear distance from the squared distance
        d.sqrt()
    }
}

// ============================================================================
// Surface Mode
// ============================================================================

/// Mode for surface evaluation.
///
/// Controls whether to use interpolation surface (faster, less accurate) or
/// direct per-point fitting (slower, more accurate).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SurfaceMode {
    /// Use interpolation surface for faster evaluation.
    #[default]
    Interpolation,

    /// Use direct per-point fitting for maximum accuracy.
    Direct,
}

// ============================================================================
// Type Definitions
// ============================================================================

/// Signature for custom smooth pass function
#[doc(hidden)]
pub type SmoothPassFn<T> = fn(
    &[T],               // x (query points)
    &[T],               // y (associated values)
    &[T],               // x_search (augmented data for neighbor search)
    &[T],               // y_search (augmented data for fitting)
    usize,              // dimensions
    usize,              // window_size
    bool,               // use_robustness
    &[T],               // robustness_weights
    &mut [T],           // output (y_smooth)
    WeightFunction,     // weight_function
    ZeroWeightFallback, // zero_weight_fallback
    PolynomialDegree,   // polynomial_degree
    &DistanceMetric<T>, // distance_metric
    &[T],               // scales (normalization scales per dimension)
);

/// Signature for custom cross-validation pass function
#[doc(hidden)]
pub type CVPassFn<T> = fn(
    &[T],            // x
    &[T],            // y
    &[T],            // candidate fractions
    CVKind,          // CV strategy
    &LoessConfig<T>, // Config for internal fits
) -> (T, Vec<T>); // (best_fraction, scores)

/// Signature for custom interval estimation pass function
#[doc(hidden)]
pub type IntervalPassFn<T> = fn(
    &[T],               // x (query points)
    &[T],               // y (associated values)
    &[T],               // x_search (augmented data for neighbor search)
    &[T],               // y_search (augmented data for fitting)
    &[T],               // y_smooth
    usize,              // dimensions
    usize,              // window_size
    &[T],               // robustness_weights
    WeightFunction,     // weight_function
    &IntervalMethod<T>, // interval configuration
    PolynomialDegree,   // polynomial_degree
    &DistanceMetric<T>, // distance_metric
    &[T],               // scales (normalization scales per dimension)
) -> Vec<T>; // standard errors

/// Signature for custom iteration batch pass function (GPU acceleration).
#[doc(hidden)]
pub type FitPassFn<T> = fn(
    &[T],            // x
    &[T],            // y
    &LoessConfig<T>, // full configuration
) -> (
    Vec<T>,         // smoothed
    Option<Vec<T>>, // std_errors
    usize,          // iterations
    Vec<T>,         // robustness_weights
);

/// Signature for custom vertex pass function (Interpolation mode).
#[doc(hidden)]
pub type VertexPassFn<T> = fn(
    &[T],                             // x (augmented)
    &[T],                             // y (augmented)
    usize,                            // dimensions
    &[T],                             // vertices
    usize,                            // window_size
    bool,                             // use_robustness
    &[T],                             // robustness_weights
    &mut [T],                         // vertex_data (output: value + derivatives)
    Option<&[CachedNeighborhood<T>]>, // existing neighborhoods (if refitting)
    &mut Vec<CachedNeighborhood<T>>,  // output neighborhoods (if building)
    WeightFunction,
    ZeroWeightFallback,
    PolynomialDegree,
    &DistanceMetric<T>,
    &[T], // scales
    bool, // boundary_degree_fallback
);

/// Signature for custom KD-tree builder function.
#[doc(hidden)]
pub type KDTreeBuilderFn<T> = fn(points: &[T], dims: usize) -> KDTree<T>;

/// Output from LOESS execution.
#[derive(Debug, Clone)]
pub struct ExecutorOutput<T: FloatLinalg> {
    /// Smoothed y-values.
    pub smoothed: Vec<T>,

    /// Standard errors (if SE estimation or intervals were requested).
    pub std_errors: Option<Vec<T>>,

    /// Number of iterations performed (if auto-convergence was active).
    pub iterations: Option<usize>,

    /// Smoothing fraction used (selected by CV or configured).
    pub used_fraction: T,

    /// RMSE scores for each tested fraction (if CV was performed).
    pub cv_scores: Option<Vec<T>>,

    /// Final robustness weights from iterative refinement.
    pub robustness_weights: Vec<T>,

    /// Leverage values (hat matrix diagonal) for each point.
    /// Only computed when intervals are requested.
    pub leverage: Option<Vec<T>>,
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for LOESS execution.
#[derive(Debug, Clone)]
pub struct LoessConfig<T: FloatLinalg + SolverLinalg> {
    /// Smoothing fraction (0, 1].
    /// If `None` and `cv_fractions` are provided, bandwidth selection is performed.
    pub fraction: Option<T>,

    /// Number of robustness iterations (0 means initial fit only).
    pub iterations: usize,

    /// Kernel weight function used for local regression.
    pub weight_function: WeightFunction,

    /// Zero-weight fallback policy.
    pub zero_weight_fallback: ZeroWeightFallback,

    /// Robustness weighting method for outlier downweighting.
    pub robustness_method: RobustnessMethod,

    /// Residual scaling method (MAR or MAD).
    pub scaling_method: ScalingMethod,

    /// Candidate fractions to evaluate during cross-validation.
    pub cv_fractions: Option<Vec<T>>,

    /// Cross-validation strategy (e.g., K-Fold or LOOCV).
    pub cv_kind: Option<CVKind>,

    /// Seed for random number generation in cross-validation.
    pub cv_seed: Option<u64>,

    /// Convergence tolerance for early stopping of robustness iterations.
    pub auto_converge: Option<T>,

    /// Configuration for standard errors and intervals.
    pub return_variance: Option<IntervalMethod<T>>,

    /// Boundary handling policy.
    pub boundary_policy: BoundaryPolicy,

    /// Polynomial degree for local regression (0=constant, 1=linear, 2=quadratic).
    pub polynomial_degree: PolynomialDegree,

    /// Number of predictor dimensions (default: 1).
    pub dimensions: usize,

    /// Distance metric for nD neighborhood computation.
    pub distance_metric: DistanceMetric<T>,

    /// Surface evaluation mode (Interpolation or Direct).
    pub surface_mode: SurfaceMode,

    /// Maximum number of vertices for the interpolation surface.
    pub interpolation_vertices: Option<usize>,

    /// Cell size as a fraction of the smoothing span (default: 0.2).
    /// Used to determine subdivision when `surface_mode` is `Interpolation`.
    pub cell: Option<f64>,

    /// Whether to reduce polynomial degree to Linear at boundary vertices during interpolation.
    /// When `true` (default), vertices outside the tight data bounds use Linear fits to avoid
    /// unstable extrapolation. Set to `false` to match R's loess behavior exactly.
    pub boundary_degree_fallback: bool,

    // ++++++++++++++++++++++++++++++++++++++
    // +               DEV                  +
    // ++++++++++++++++++++++++++++++++++++++
    /// Custom smooth pass function (enables parallel execution).
    #[doc(hidden)]
    pub custom_smooth_pass: Option<SmoothPassFn<T>>,

    /// Custom cross-validation pass function.
    #[doc(hidden)]
    pub custom_cv_pass: Option<CVPassFn<T>>,

    /// Custom interval estimation pass function.
    #[doc(hidden)]
    pub custom_interval_pass: Option<IntervalPassFn<T>>,

    /// Custom iteration batch pass function for GPU acceleration.
    #[doc(hidden)]
    pub custom_fit_pass: Option<FitPassFn<T>>,

    /// Custom vertex pass function (Interpolation mode).
    #[doc(hidden)]
    pub custom_vertex_pass: Option<VertexPassFn<T>>,

    /// Custom KD-tree builder function.
    #[doc(hidden)]
    pub custom_kdtree_builder: Option<KDTreeBuilderFn<T>>,

    /// Execution backend hint for extension crates.
    #[doc(hidden)]
    pub backend: Option<Backend>,

    /// Whether to use parallel execution
    #[doc(hidden)]
    pub parallel: bool,
}

impl<T: FloatLinalg + DistanceLinalg + Debug + Send + Sync + SolverLinalg> Default
    for LoessConfig<T>
{
    fn default() -> Self {
        Self {
            fraction: None,
            iterations: 3,
            weight_function: WeightFunction::default(),
            zero_weight_fallback: ZeroWeightFallback::default(),
            robustness_method: RobustnessMethod::default(),
            scaling_method: ScalingMethod::default(),
            cv_fractions: None,
            cv_kind: None,
            cv_seed: None,
            auto_converge: None,
            return_variance: None,
            boundary_policy: BoundaryPolicy::default(),
            polynomial_degree: PolynomialDegree::default(),
            dimensions: 1,
            distance_metric: DistanceMetric::default(),
            surface_mode: SurfaceMode::default(),
            interpolation_vertices: None,
            cell: None,
            boundary_degree_fallback: true,
            custom_smooth_pass: None,
            custom_cv_pass: None,
            custom_interval_pass: None,
            custom_fit_pass: None,
            custom_vertex_pass: None,
            custom_kdtree_builder: None,
            parallel: false,
            backend: None,
        }
    }
}

/// Unified executor for LOESS smoothing operations.
#[derive(Debug, Clone)]
pub struct LoessExecutor<T: FloatLinalg + SolverLinalg> {
    /// Smoothing fraction (0, 1].
    pub fraction: T,

    /// Number of robustness iterations.
    pub iterations: usize,

    /// Kernel weight function.
    pub weight_function: WeightFunction,

    /// Zero weight fallback flag.
    pub zero_weight_fallback: ZeroWeightFallback,

    /// Robustness method for iterative refinement.
    pub robustness_method: RobustnessMethod,

    /// Residual scaling method.
    pub scaling_method: ScalingMethod,

    /// Boundary handling policy.
    pub boundary_policy: BoundaryPolicy,

    /// Polynomial degree for local regression.
    pub polynomial_degree: PolynomialDegree,

    /// Number of predictor dimensions.
    pub dimensions: usize,

    /// Distance metric for nD neighborhood computation.
    pub distance_metric: DistanceMetric<T>,

    /// Surface evaluation mode (Interpolation or Direct).
    pub surface_mode: SurfaceMode,

    /// Maximum number of vertices for interpolation.
    pub interpolation_vertices: Option<usize>,

    /// Cell size for interpolation subdivision.
    pub cell: Option<f64>,

    /// Whether to reduce polynomial degree to Linear at boundary vertices during interpolation.
    pub boundary_degree_fallback: bool,

    // ++++++++++++++++++++++++++++++++++++++
    // +               DEV                  +
    // ++++++++++++++++++++++++++++++++++++++
    /// Custom smooth pass function (e.g., for parallel execution).
    #[doc(hidden)]
    pub custom_smooth_pass: Option<SmoothPassFn<T>>,

    /// Custom cross-validation pass function.
    #[doc(hidden)]
    pub custom_cv_pass: Option<CVPassFn<T>>,

    /// Custom interval estimation pass function.
    #[doc(hidden)]
    pub custom_interval_pass: Option<IntervalPassFn<T>>,

    /// Custom iteration batch pass function for GPU acceleration.
    #[doc(hidden)]
    pub custom_fit_pass: Option<FitPassFn<T>>,

    /// Custom vertex pass function (Interpolation mode).
    #[doc(hidden)]
    pub custom_vertex_pass: Option<VertexPassFn<T>>,

    /// Custom KD-tree builder function.
    #[doc(hidden)]
    pub custom_kdtree_builder: Option<KDTreeBuilderFn<T>>,

    /// Execution backend hint for extension crates.
    #[doc(hidden)]
    pub backend: Option<Backend>,

    /// Whether to use parallel execution
    #[doc(hidden)]
    pub parallel: bool,
}

impl<T: FloatLinalg + DistanceLinalg + Debug + Send + Sync + 'static + SolverLinalg> Default
    for LoessExecutor<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatLinalg + DistanceLinalg + Debug + Send + Sync + 'static + SolverLinalg>
    LoessExecutor<T>
{
    // ========================================================================
    // Constructor and Builder Methods
    // ========================================================================

    /// Create a new executor with default parameters.
    pub fn new() -> Self {
        Self {
            fraction: T::from(0.67).unwrap_or_else(|| T::from(0.5).unwrap()),
            iterations: 3,
            weight_function: WeightFunction::default(),
            zero_weight_fallback: ZeroWeightFallback::default(),
            robustness_method: RobustnessMethod::default(),
            scaling_method: ScalingMethod::default(),
            boundary_policy: BoundaryPolicy::default(),
            polynomial_degree: PolynomialDegree::default(),
            dimensions: 1,
            distance_metric: DistanceMetric::default(),
            surface_mode: SurfaceMode::default(),
            interpolation_vertices: None,
            cell: None,
            boundary_degree_fallback: true,
            custom_smooth_pass: None,
            custom_cv_pass: None,
            custom_interval_pass: None,
            custom_fit_pass: None,
            custom_vertex_pass: None,
            custom_kdtree_builder: None,
            parallel: false,
            backend: None,
        }
    }

    /// Create a new executor from a `LoessConfig`.
    pub fn from_config(config: &LoessConfig<T>) -> Self {
        let default_frac = T::from(0.67).unwrap_or_else(|| T::from(0.5).unwrap());
        Self::new()
            .fraction(config.fraction.unwrap_or(default_frac))
            .iterations(config.iterations)
            .weight_function(config.weight_function)
            .zero_weight_fallback(config.zero_weight_fallback)
            .robustness_method(config.robustness_method)
            .scaling_method(config.scaling_method)
            .boundary_policy(config.boundary_policy)
            .polynomial_degree(config.polynomial_degree)
            .dimensions(config.dimensions)
            .distance_metric(config.distance_metric.clone())
            .surface_mode(config.surface_mode)
            .interpolation_vertices(config.interpolation_vertices)
            .cell(config.cell)
            .boundary_degree_fallback(config.boundary_degree_fallback)
            // ++++++++++++++++++++++++++++++++++++++
            // +               DEV                  +
            // ++++++++++++++++++++++++++++++++++++++
            .custom_smooth_pass(config.custom_smooth_pass)
            .custom_cv_pass(config.custom_cv_pass)
            .custom_interval_pass(config.custom_interval_pass)
            .custom_fit_pass(config.custom_fit_pass)
            .custom_vertex_pass(config.custom_vertex_pass)
            .custom_kdtree_builder(config.custom_kdtree_builder)
            .parallel(config.parallel)
            .backend(config.backend)
    }

    /// Set the smoothing fraction (bandwidth).
    pub fn fraction(mut self, frac: T) -> Self {
        self.fraction = frac;
        self
    }

    /// Set the number of robustness iterations.
    pub fn iterations(mut self, niter: usize) -> Self {
        self.iterations = niter;
        self
    }

    /// Set the kernel weight function.
    pub fn weight_function(mut self, wf: WeightFunction) -> Self {
        self.weight_function = wf;
        self
    }

    /// Set the zero weight fallback policy flag.
    pub fn zero_weight_fallback(mut self, flag: ZeroWeightFallback) -> Self {
        self.zero_weight_fallback = flag;
        self
    }

    /// Set the robustness method for iterative refinement.
    pub fn robustness_method(mut self, method: RobustnessMethod) -> Self {
        self.robustness_method = method;
        self
    }

    /// Set the residual scaling method (MAR/MAD).
    pub fn scaling_method(mut self, method: ScalingMethod) -> Self {
        self.scaling_method = method;
        self
    }

    /// Set the boundary handling policy.
    pub fn boundary_policy(mut self, policy: BoundaryPolicy) -> Self {
        self.boundary_policy = policy;
        self
    }

    /// Set the polynomial degree for local regression.
    pub fn polynomial_degree(mut self, degree: PolynomialDegree) -> Self {
        self.polynomial_degree = degree;
        self
    }

    /// Set the number of predictor dimensions.
    pub fn dimensions(mut self, dims: usize) -> Self {
        self.dimensions = dims;
        self
    }

    /// Set the distance metric for nD neighborhood computation.
    pub fn distance_metric(mut self, metric: DistanceMetric<T>) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Set the surface evaluation mode (Interpolation or Direct).
    pub fn surface_mode(mut self, mode: SurfaceMode) -> Self {
        self.surface_mode = mode;
        self
    }

    /// Set the maximum number of vertices for interpolation.
    pub fn interpolation_vertices(mut self, vertices: Option<usize>) -> Self {
        self.interpolation_vertices = vertices;
        self
    }

    /// Set the interpolation cell size.
    pub fn cell(mut self, cell: Option<f64>) -> Self {
        self.cell = cell;
        self
    }

    /// Set whether to reduce polynomial degree at boundary vertices.
    pub fn boundary_degree_fallback(mut self, enabled: bool) -> Self {
        self.boundary_degree_fallback = enabled;
        self
    }

    // ++++++++++++++++++++++++++++++++++++++
    // +               DEV                  +
    // ++++++++++++++++++++++++++++++++++++++

    /// Set a custom smooth pass function (e.g., for parallelization).
    #[doc(hidden)]
    pub fn custom_smooth_pass(mut self, smooth_pass_fn: Option<SmoothPassFn<T>>) -> Self {
        self.custom_smooth_pass = smooth_pass_fn;
        self
    }

    /// Set a custom cross-validation pass function.
    #[doc(hidden)]
    pub fn custom_cv_pass(mut self, cv_pass_fn: Option<CVPassFn<T>>) -> Self {
        self.custom_cv_pass = cv_pass_fn;
        self
    }

    /// Set a custom interval estimation pass function.
    #[doc(hidden)]
    pub fn custom_interval_pass(mut self, interval_pass_fn: Option<IntervalPassFn<T>>) -> Self {
        self.custom_interval_pass = interval_pass_fn;
        self
    }

    /// Set a custom vertex pass function (Interpolation mode).
    #[doc(hidden)]
    pub fn custom_vertex_pass(mut self, vertex_pass_fn: Option<VertexPassFn<T>>) -> Self {
        self.custom_vertex_pass = vertex_pass_fn;
        self
    }

    /// Set a custom iteration batch pass function for GPU acceleration.
    #[doc(hidden)]
    pub fn custom_fit_pass(mut self, fit_pass_fn: Option<FitPassFn<T>>) -> Self {
        self.custom_fit_pass = fit_pass_fn;
        self
    }

    /// Set a custom KD-tree builder function.
    #[doc(hidden)]
    pub fn custom_kdtree_builder(mut self, kdtree_builder_fn: Option<KDTreeBuilderFn<T>>) -> Self {
        self.custom_kdtree_builder = kdtree_builder_fn;
        self
    }

    /// Set whether to use parallel execution.
    #[doc(hidden)]
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Set the execution backend hint.
    #[doc(hidden)]
    pub fn backend(mut self, backend: Option<Backend>) -> Self {
        self.backend = backend;
        self
    }

    // ========================================================================
    // Main Entry Points
    // ========================================================================

    /// Smooth data using a `LoessConfig` payload.
    pub fn run_with_config(x: &[T], y: &[T], config: LoessConfig<T>) -> ExecutorOutput<T>
    where
        T: Float + Debug + Send + Sync + 'static,
    {
        let executor = LoessExecutor::from_config(&config);
        let dims = executor.dimensions;
        let n = x.len() / dims;
        let eff_fraction = config.fraction.unwrap_or(executor.fraction);
        let window_size = Window::calculate_span(n, eff_fraction);
        let n_coeffs = executor.polynomial_degree.num_coefficients_nd(dims);

        // Create a workspace to be reused across CV and final fit
        let mut workspace =
            LoessBuffer::<T, NodeDistance<T>, Neighborhood<T>>::new(n, dims, window_size, n_coeffs);

        // Handle cross-validation if configured
        if let Some(ref cv_fracs) = config.cv_fractions {
            let cv_kind = config.cv_kind.unwrap_or(CVKind::KFold(5));

            // Run CV to find best fraction
            let (best_frac, scores) = if let Some(callback) = config.custom_cv_pass {
                callback(x, y, cv_fracs, cv_kind, &config)
            } else {
                let predictor = if dims > 1 {
                    Some(|train_x: &[T], train_y: &[T], test_x: &[T], f: T| {
                        let n_train = train_y.len();
                        let window_size = Window::calculate_span(n_train, f);

                        // Use Min-Max scaling consistent with main run()
                        let mut scales = vec![T::one(); dims];
                        if n_train > 0 {
                            let mut mins = train_x[..dims].to_vec();
                            let mut maxs = train_x[..dims].to_vec();

                            for i in 1..n_train {
                                for d in 0..dims {
                                    let val = train_x[i * dims + d];
                                    if val < mins[d] {
                                        mins[d] = val;
                                    }
                                    if val > maxs[d] {
                                        maxs[d] = val;
                                    }
                                }
                            }

                            for d in 0..dims {
                                let range = maxs[d] - mins[d];
                                if range > T::zero() {
                                    scales[d] = T::one() / range;
                                }
                            }
                        }
                        let kdtree = if let Some(builder) = executor.custom_kdtree_builder {
                            builder(train_x, dims)
                        } else {
                            KDTree::new(train_x, dims)
                        };

                        executor.predict(
                            train_x,
                            train_y,
                            &vec![T::one(); n_train],
                            test_x,
                            window_size,
                            &scales,
                            &kdtree,
                        )
                    })
                } else {
                    None
                };

                let LoessBuffer {
                    ref mut cv_buffer, ..
                } = workspace;

                cv_kind.run(
                    x,
                    y,
                    dims,
                    cv_fracs,
                    config.cv_seed,
                    |tx, ty, f| {
                        executor
                            .run(tx, ty, Some(f), None, None, None, None)
                            .smoothed
                    },
                    predictor,
                    cv_buffer,
                )
            };

            // Run final pass with best fraction
            let mut output = executor.run(
                x,
                y,
                Some(best_frac),
                Some(config.iterations),
                config.auto_converge,
                config.return_variance.as_ref(),
                Some(&mut workspace),
            );
            output.cv_scores = Some(scores);
            output.used_fraction = best_frac;
            output
        } else {
            // Direct run (no CV)
            executor.run(
                x,
                y,
                config.fraction,
                Some(config.iterations),
                config.auto_converge,
                config.return_variance.as_ref(),
                Some(&mut workspace),
            )
        }
    }

    /// Execute smoothing with explicit overrides for specific parameters.
    ///
    /// Uses interpolation surface for efficient evaluation - fits only at
    /// cell vertices and interpolates for all other points.
    ///
    /// # Special Cases
    ///
    /// * **Insufficient data** (n < 2): Returns original y-values.
    /// * **Global regression** (fraction >= 1.0): Performs OLS on the entire dataset.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        x: &[T],
        y: &[T],
        fraction: Option<T>,
        max_iter: Option<usize>,
        tolerance: Option<T>,
        confidence_method: Option<&IntervalMethod<T>>,
        workspace: Option<&mut LoessBuffer<T, NodeDistance<T>, Neighborhood<T>>>,
    ) -> ExecutorOutput<T>
    where
        T: Float + Debug + Send + Sync + 'static,
    {
        let dims = self.dimensions;
        let n = x.len() / dims;
        let eff_fraction = fraction.unwrap_or(self.fraction);

        // Calculate window size
        let window_size = Window::calculate_span(n, eff_fraction);
        let target_iterations = max_iter.unwrap_or(self.iterations);
        let n_coeffs = self.polynomial_degree.num_coefficients_nd(dims);

        // Apply boundary policy (unified)
        let (ax, ay, mapping) = self.boundary_policy.apply(x, y, dims, window_size);
        let n_total = ay.len();
        let is_augmented = n_total > n;

        let mut new_workspace;
        let workspace = if let Some(ws) = workspace {
            ws.ensure_capacity(n_total, dims, window_size, n_coeffs);
            ws
        } else {
            new_workspace = LoessBuffer::<T, NodeDistance<T>, Neighborhood<T>>::new(
                n_total,
                dims,
                window_size,
                n_coeffs,
            );
            &mut new_workspace
        };

        // Compute normalization scales using the full (augmented) range
        workspace.executor_buffer.ensure_capacity(n_total, dims);
        let mins = &mut workspace.executor_buffer.mins;
        let maxs = &mut workspace.executor_buffer.maxs;
        mins.resize(dims, T::zero());
        maxs.resize(dims, T::zero());

        // Initialize mins/maxs with first point
        mins[..dims].copy_from_slice(&ax[..dims]);
        maxs[..dims].copy_from_slice(&ax[..dims]);
        for i in 1..n_total {
            for d in 0..dims {
                let val = ax[i * dims + d];
                if val < mins[d] {
                    mins[d] = val;
                }
                if val > maxs[d] {
                    maxs[d] = val;
                }
            }
        }

        workspace.executor_buffer.scales.resize(dims, T::one());
        let scales_ref = &mut workspace.executor_buffer.scales;
        for d in 0..dims {
            let range = maxs[d] - mins[d];
            if range > T::zero() {
                scales_ref[d] = T::one() / range;
            } else {
                scales_ref[d] = T::one();
            }
        }

        // Copy scales locally for distance calculator to avoid borrowing workspace
        let scales_local = workspace.executor_buffer.scales.clone();

        // Build KD-Tree for efficient kNN
        let kdtree = if let Some(builder) = self.custom_kdtree_builder {
            builder(&ax, dims)
        } else {
            KDTree::new(&ax, dims)
        };

        // Define distance calculator
        let dist_calc = LoessDistanceCalculator {
            metric: self.distance_metric.clone(),
            scales: &scales_local,
        };

        // Resolution First: no default limit unless explicitly provided
        let max_vertices = self.interpolation_vertices.unwrap_or(usize::MAX);

        let n_coeffs = self.polynomial_degree.num_coefficients_nd(dims);
        workspace.ensure_capacity(n_total, dims, window_size, n_coeffs);

        let mut y_smooth = vec![T::zero(); n];
        workspace
            .executor_buffer
            .robustness_weights
            .resize(n_total, T::one());
        workspace.executor_buffer.residuals.resize(n, T::zero());

        // For interpolation mode: build surface once before iterations
        let mut _surface_opt: Option<InterpolationSurface<T>> = None;
        if self.surface_mode == SurfaceMode::Interpolation {
            let fitter = |vertex: &[T],
                          neighborhood: &Neighborhood<T>,
                          fb: &mut FittingBuffer<T>,
                          degree: PolynomialDegree| {
                let mut context = RegressionContext::new(
                    &ax,
                    dims,
                    &ay,
                    0, // query_idx is not used when query_point is Some
                    Some(vertex),
                    neighborhood,
                    false, // use_robustness
                    &workspace.executor_buffer.robustness_weights,
                    self.weight_function,
                    self.zero_weight_fallback,
                    degree,
                    false, // compute_leverage
                    Some(fb),
                );
                context.fit_with_coefficients()
            };

            let cell_fraction =
                T::from(self.cell.unwrap_or(0.2)).unwrap_or_else(|| T::from(0.2).unwrap());

            let surface = InterpolationSurface::build(
                &ax,
                &ay,
                dims,
                eff_fraction,
                window_size,
                &dist_calc,
                &kdtree,
                max_vertices,
                fitter,
                &mut workspace.search_buffer,
                &mut workspace.neighborhood,
                &mut workspace.fitting_buffer,
                cell_fraction,
                self.custom_vertex_pass,
                &scales_local,
                self.weight_function,
                self.zero_weight_fallback,
                self.polynomial_degree,
                &self.distance_metric,
                self.boundary_degree_fallback,
            );
            _surface_opt = Some(surface);

            // Re-evaluate surface at all original points
            let surface = _surface_opt.as_ref().unwrap();
            for (i, val) in y_smooth.iter_mut().enumerate().take(n) {
                let query_offset = i * dims;
                // Use original x (not augmented) for query points
                let query_point = &x[query_offset..query_offset + dims];
                *val = surface.evaluate(query_point);
            }
        } else {
            // Direct mode: initial fit - populate neighborhood cache
            workspace.executor_buffer.neighborhood_cache.entries.clear();

            // Check for custom smooth pass callback
            if let Some(callback) = self.custom_smooth_pass {
                // Use custom parallel/accelerated implementation
                callback(
                    x,
                    y,
                    &ax,
                    &ay,
                    dims,
                    window_size,
                    false, // use_robustness (first pass)
                    &workspace.executor_buffer.robustness_weights,
                    &mut y_smooth,
                    self.weight_function,
                    self.zero_weight_fallback,
                    self.polynomial_degree,
                    &self.distance_metric,
                    &scales_local,
                );
                // Mark cache as invalid since we bypassed the internal method
                workspace.executor_buffer.neighborhood_cache.is_valid = false;
            } else {
                self.smooth_pass(
                    &ax,
                    &ay,
                    x, // x_query
                    y, // y_query
                    window_size,
                    &workspace.executor_buffer.robustness_weights,
                    false,
                    &scales_local,
                    &mut y_smooth,
                    n,
                    &kdtree,
                    &mut workspace.search_buffer,
                    &mut workspace.neighborhood,
                    &mut workspace.fitting_buffer,
                    None, // No leverage collection during initial fit
                    Some(&mut workspace.executor_buffer.neighborhood_cache.entries), // Populate cache
                    None, // Not using cache yet
                );
                workspace.executor_buffer.neighborhood_cache.is_valid = true;
            }
        }

        let mut iterations_performed = 1;

        // Robustness iteration loop
        for iter in 1..target_iterations {
            iterations_performed = iter + 1;

            // Update robustness weights based on residuals from previous pass
            T::batch_abs_residuals(
                &y[..n],
                &y_smooth[..n],
                &mut workspace.executor_buffer.residuals[..n],
            );

            // 1. Compute new weights using the unified robustness method
            // This handles scaling (MAR/MAD) and weight function application
            let n_res = n; // length of residuals
            let mut new_weights = vec![T::zero(); n];

            // Re-use sorted_residuals as scratch space for median computation
            workspace
                .executor_buffer
                .sorted_residuals
                .resize(n_res, T::zero());

            self.robustness_method.apply_robustness_weights(
                &workspace.executor_buffer.residuals[..n_res],
                &mut new_weights,
                self.scaling_method,
                &mut workspace.executor_buffer.sorted_residuals,
            );

            let tolerance_val = tolerance.unwrap_or_else(|| T::from(1e-6).unwrap());

            // 2. Sync to robustness_weights and check convergence
            let mut max_change = T::zero();

            if is_augmented {
                for (aug_idx, &orig_idx) in mapping.iter().enumerate() {
                    let old_w = workspace.executor_buffer.robustness_weights[aug_idx];
                    let new_w = new_weights[orig_idx];
                    workspace.executor_buffer.robustness_weights[aug_idx] = new_w;

                    let change = (new_w - old_w).abs();
                    if change > max_change {
                        max_change = change;
                    }
                }
            } else {
                for (i, &new_w) in new_weights.iter().enumerate().take(n) {
                    let old_w = workspace.executor_buffer.robustness_weights[i];
                    workspace.executor_buffer.robustness_weights[i] = new_w;

                    let change = (new_w - old_w).abs();
                    if change > max_change {
                        max_change = change;
                    }
                }
            }

            if max_change < tolerance_val {
                break;
            }

            // Re-fit with new robustness weights
            match self.surface_mode {
                SurfaceMode::Interpolation => {
                    // Refit vertex values using existing surface structure
                    if let Some(ref mut surface) = _surface_opt {
                        let fitter =
                            |vertex: &[T],
                             neighborhood: &Neighborhood<T>,
                             fb: &mut FittingBuffer<T>,
                             degree: PolynomialDegree| {
                                let mut context = RegressionContext::new(
                                    &ax,
                                    dims,
                                    &ay,
                                    0, // query_idx is not used when query_point is Some
                                    Some(vertex),
                                    neighborhood,
                                    true, // use_robustness
                                    &workspace.executor_buffer.robustness_weights,
                                    self.weight_function,
                                    self.zero_weight_fallback,
                                    degree,
                                    false, // compute_leverage
                                    Some(fb),
                                );
                                context.fit_with_coefficients()
                            };

                        surface.refit_values(
                            &ax,
                            &ay,
                            fitter,
                            &mut workspace.neighborhood,
                            &mut workspace.fitting_buffer,
                            self.custom_vertex_pass,
                            self.weight_function,
                            self.zero_weight_fallback,
                            self.polynomial_degree,
                            &self.distance_metric,
                            &scales_local,
                            &workspace.executor_buffer.robustness_weights,
                            self.boundary_degree_fallback,
                        );

                        // Re-evaluate at data points using original x (not augmented)
                        for (i, val) in y_smooth.iter_mut().enumerate().take(n) {
                            let query_offset = i * dims;
                            let query_point = &x[query_offset..query_offset + dims];
                            *val = surface.evaluate(query_point);
                        }
                    }
                }
                SurfaceMode::Direct => {
                    // Check for custom smooth pass callback
                    if let Some(callback) = self.custom_smooth_pass {
                        // Use custom parallel/accelerated implementation
                        callback(
                            x,
                            y,
                            &ax,
                            &ay,
                            dims,
                            window_size,
                            true, // use_robustness
                            &workspace.executor_buffer.robustness_weights,
                            &mut y_smooth,
                            self.weight_function,
                            self.zero_weight_fallback,
                            self.polynomial_degree,
                            &self.distance_metric,
                            &scales_local,
                        );
                    } else {
                        // Use cached neighborhoods to skip KD-tree searches
                        let cache_ref = if workspace.executor_buffer.neighborhood_cache.is_valid {
                            Some(
                                workspace
                                    .executor_buffer
                                    .neighborhood_cache
                                    .entries
                                    .as_slice(),
                            )
                        } else {
                            None
                        };
                        self.smooth_pass(
                            &ax,
                            &ay,
                            x, // x_query
                            y, // y_query
                            window_size,
                            &workspace.executor_buffer.robustness_weights,
                            true,
                            &scales_local,
                            &mut y_smooth,
                            n,
                            &kdtree,
                            &mut workspace.search_buffer,
                            &mut workspace.neighborhood,
                            &mut workspace.fitting_buffer,
                            None,      // No leverage collection during robustness iterations
                            None,      // Not populating cache
                            cache_ref, // Use cached neighborhoods
                        );
                    }
                }
            }
        }

        // Collect leverage values when intervals are requested
        let leverage_values =
            if confidence_method.is_some() && self.surface_mode == SurfaceMode::Direct {
                let mut leverages = Vec::with_capacity(n);
                // Final pass with leverage collection
                // Use cached neighborhoods for leverage pass
                let cache_ref = if workspace.executor_buffer.neighborhood_cache.is_valid {
                    Some(
                        workspace
                            .executor_buffer
                            .neighborhood_cache
                            .entries
                            .as_slice(),
                    )
                } else {
                    None
                };
                self.smooth_pass(
                    &ax,
                    &ay,
                    x, // x_query
                    y, // y_query
                    window_size,
                    &workspace.executor_buffer.robustness_weights,
                    true,
                    &scales_local,
                    &mut y_smooth,
                    n,
                    &kdtree,
                    &mut workspace.search_buffer,
                    &mut workspace.neighborhood,
                    &mut workspace.fitting_buffer,
                    Some(&mut leverages),
                    None,      // Not populating cache
                    cache_ref, // Use cached neighborhoods
                );
                Some(leverages)
            } else {
                None
            };

        // Standard errors (now using actual leverage if available)
        let se = if let Some(interval_method) = confidence_method {
            // Check for custom interval pass callback
            if let Some(callback) = self.custom_interval_pass {
                // Use custom parallel/accelerated implementation
                Some(callback(
                    x,
                    y,
                    &ax,
                    &ay,
                    &y_smooth,
                    dims,
                    window_size,
                    &workspace.executor_buffer.robustness_weights,
                    self.weight_function,
                    interval_method,
                    self.polynomial_degree,
                    &self.distance_metric,
                    &scales_local,
                ))
            } else if let Some(ref lev) = leverage_values {
                // Use actual leverage values
                T::batch_abs_residuals(
                    &y[..n],
                    &y_smooth[..n],
                    &mut workspace.executor_buffer.residuals[..n],
                );
                let mut sorted_residuals = workspace.executor_buffer.residuals.clone();
                let median_idx = n / 2;
                if median_idx < sorted_residuals.len() {
                    sorted_residuals.select_nth_unstable_by(median_idx, |a: &T, b| {
                        a.partial_cmp(b).unwrap_or(Equal)
                    });
                }
                let median_residual = sorted_residuals[median_idx];
                let sigma = median_residual * T::from(1.4826).unwrap();

                // SE = sigma * sqrt(leverage)
                let mut se_vec = vec![T::zero(); n];
                T::batch_sqrt_scale(lev, sigma, &mut se_vec);
                Some(se_vec)
            } else {
                // Fallback to approximate leverage (for Interpolation mode).
                // NOTE: This is a coarse heuristic based on the smoothing fraction.
                // It assumes uniform leverage across all points, which is rarely true
                // but provides a stable baseline for standard errors when local
                // hat matrix diagonals are not available.
                T::batch_abs_residuals(
                    &y[..n],
                    &y_smooth[..n],
                    &mut workspace.executor_buffer.residuals[..n],
                );
                let mut sorted_residuals = workspace.executor_buffer.residuals.clone();
                let median_idx = n / 2;
                if median_idx < sorted_residuals.len() {
                    sorted_residuals.select_nth_unstable_by(median_idx, |a: &T, b| {
                        a.partial_cmp(b).unwrap_or(Equal)
                    });
                }
                let median_residual = sorted_residuals[median_idx];
                let sigma = median_residual * T::from(1.4826).unwrap();

                let approx_leverage = eff_fraction / T::from(n).unwrap();
                let se_vec: Vec<T> = (0..n).map(|_| sigma * approx_leverage.sqrt()).collect();
                Some(se_vec)
            }
        } else {
            None
        };

        // Extract robustness weights for original points only
        let final_robustness_weights = if is_augmented {
            let mut rw = vec![T::one(); n];
            for (i, &idx) in mapping.iter().enumerate().take(n_total) {
                if idx < n {
                    rw[idx] = workspace.executor_buffer.robustness_weights[i];
                }
            }
            rw
        } else {
            workspace.executor_buffer.robustness_weights[..n].to_vec()
        };

        ExecutorOutput {
            smoothed: y_smooth,
            std_errors: se,
            iterations: Some(iterations_performed),
            used_fraction: eff_fraction,
            cv_scores: None,
            robustness_weights: final_robustness_weights,
            leverage: leverage_values,
        }
    }

    // ========================================================================
    // Main Algorithmic Logic
    // ========================================================================

    /// Predict values at arbitrary points using the provided training data.
    ///
    /// This is used for out-of-sample prediction, specifically during cross-validation.
    #[allow(clippy::too_many_arguments)]
    pub fn predict(
        &self,
        x_train: &[T],
        y_train: &[T],
        robustness_weights: &[T],
        x_query: &[T],
        window_size: usize,
        scales: &[T],
        kdtree: &KDTree<T>,
    ) -> Vec<T>
    where
        T: Float + Debug + Send + Sync + 'static,
    {
        let dims = self.dimensions;
        let n_query = x_query.len() / dims;
        let mut y_pred = vec![T::zero(); n_query];

        let dist_calc = LoessDistanceCalculator {
            metric: self.distance_metric.clone(),
            scales,
        };

        let n_points_train = x_train.len() / dims;
        let n_coeffs = self.polynomial_degree.num_coefficients_nd(dims);
        let mut workspace = LoessBuffer::<T, NodeDistance<T>, Neighborhood<T>>::new(
            n_points_train,
            dims,
            window_size,
            n_coeffs,
        );

        for (i, pred) in y_pred.iter_mut().enumerate() {
            let query_offset = i * dims;
            let query_point = &x_query[query_offset..query_offset + dims];

            // Find neighbors in training data (KD-tree is always available)
            kdtree.find_k_nearest(
                query_point,
                window_size,
                &dist_calc,
                None,
                &mut workspace.search_buffer,
                &mut workspace.neighborhood,
            );
            let neighborhood = &workspace.neighborhood;

            // Fit local polynomial
            let mut context = RegressionContext::new(
                x_train,
                dims,
                y_train,
                0, // query_idx is not used when query_point is Some
                Some(query_point),
                neighborhood,
                true, // use_robustness
                robustness_weights,
                self.weight_function,
                ZeroWeightFallback::UseLocalMean, // CV fallback
                self.polynomial_degree,
                false, // compute_leverage
                Some(&mut workspace.fitting_buffer),
            );

            if let Some((val, _)) = context.fit() {
                *pred = val;
            } else {
                // If fitting fails, fallback to something or zero
                // For CV, zero is better than crashing, but we could try to find a global mean
                *pred = T::zero();
            }
        }
        y_pred
    }

    /// Perform a single smoothing pass over all nD points (Direct mode).
    ///
    /// # Caching Behavior
    ///
    /// * `populate_cache`: If `Some(&mut cache)`, populate the cache with neighborhoods during this pass.
    /// * `cached_neighborhoods`: If `Some(&cache)`, use cached neighborhoods instead of calling `find_k_nearest`.
    ///
    /// Typically, the first pass populates the cache, and subsequent robustness iterations use it.
    #[allow(clippy::too_many_arguments)]
    fn smooth_pass(
        &self,
        x_context: &[T],
        y_context: &[T],
        x_query: &[T],
        y_query: &[T],
        window_size: usize,
        robustness_weights: &[T],
        use_robustness: bool,
        scales: &[T],
        y_smooth: &mut [T],
        original_n: usize,
        kdtree: &KDTree<T>,
        search_buffer: &mut NeighborhoodSearchBuffer<NodeDistance<T>>,
        neighborhood: &mut Neighborhood<T>,
        fitting_buffer: &mut FittingBuffer<T>,
        mut leverage_out: Option<&mut Vec<T>>,
        mut populate_cache: Option<&mut Vec<CachedNeighborhood<T>>>,
        cached_neighborhoods: Option<&[CachedNeighborhood<T>]>,
    ) where
        T: Float + Debug + Send + Sync + 'static,
    {
        let dims = self.dimensions;
        let dist_calc = LoessDistanceCalculator {
            metric: self.distance_metric.clone(),
            scales,
        };
        let compute_leverage = leverage_out.is_some();

        // Prepare cache for population if requested
        if let Some(ref mut cache) = populate_cache.as_ref() {
            // Pre-allocate capacity but will push during iteration
            let _ = cache; // just to check it's mutable
        }

        for i in 0..original_n {
            let query_offset = i * dims;
            let query_point = &x_query[query_offset..query_offset + dims];

            // Either use cached neighborhood or compute fresh
            if let Some(cache) = cached_neighborhoods {
                // Use cached neighborhood
                let cached = &cache[i];
                neighborhood.indices.clear();
                neighborhood.indices.extend_from_slice(&cached.indices);
                neighborhood.distances.clear();
                neighborhood.distances.extend_from_slice(&cached.distances);
                neighborhood.max_distance = cached.max_distance;
            } else {
                // Compute neighborhood via KD-tree
                kdtree.find_k_nearest(
                    query_point,
                    window_size,
                    &dist_calc,
                    None,
                    search_buffer,
                    neighborhood,
                );

                // Populate cache if requested
                if let Some(ref mut cache) = populate_cache {
                    cache.push(CachedNeighborhood {
                        indices: neighborhood.indices.clone(),
                        distances: neighborhood.distances.clone(),
                        max_distance: neighborhood.max_distance,
                    });
                }
            }

            let neighborhood_ref = &*neighborhood;

            let mut context = RegressionContext::new(
                x_context,
                dims,
                y_context,
                i,
                Some(query_point),
                neighborhood_ref,
                use_robustness,
                robustness_weights,
                self.weight_function,
                self.zero_weight_fallback,
                self.polynomial_degree,
                compute_leverage,
                Some(fitting_buffer),
            );

            if let Some((val, lev)) = context.fit() {
                y_smooth[i] = val;
                if let Some(ref mut lev_vec) = leverage_out {
                    if lev_vec.len() <= i {
                        lev_vec.resize(i + 1, T::zero());
                    }
                    lev_vec[i] = lev;
                }
            } else {
                y_smooth[i] = y_query[i];
                if let Some(ref mut lev_vec) = leverage_out {
                    if lev_vec.len() <= i {
                        lev_vec.resize(i + 1, T::zero());
                    }
                    lev_vec[i] = T::zero();
                }
            }
        }
    }
}
