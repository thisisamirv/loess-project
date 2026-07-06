//! High-level API for LOESS smoothing with parallel execution support.
//!
//! This module provides the primary user-facing entry point for LOESS with
//! heavy-duty parallel execution capabilities. It extends the loess API
//! with adapters that utilize all available CPU cores.

// Imports
use crate::adapters::batch::{ParallelBatchLoess, ParallelBatchLoessBuilder};
use crate::adapters::online::{ParallelOnlineLoess, ParallelOnlineLoessBuilder};
use crate::adapters::streaming::{ParallelStreamingLoess, ParallelStreamingLoessBuilder};

// External dependencies
use std::fmt::Debug;

// Import base marker types for delegation
use loess_rs::internals::api::Batch as BaseBatch;
use loess_rs::internals::api::Online as BaseOnline;
use loess_rs::internals::api::Streaming as BaseStreaming;

// Linear algebra imports
use loess_rs::internals::algorithms::regression::SolverLinalg;
use loess_rs::internals::math::distance::DistanceLinalg;
use loess_rs::internals::math::linalg::FloatLinalg;

// Publicly re-exported types
pub use loess_rs::internals::adapters::online::UpdateMode;
pub use loess_rs::internals::adapters::streaming::MergeStrategy;
pub use loess_rs::internals::algorithms::regression::{PolynomialDegree, ZeroWeightFallback};
pub use loess_rs::internals::algorithms::robustness::RobustnessMethod;
pub use loess_rs::internals::api::{LoessAdapter, LoessBuilder};
pub use loess_rs::internals::engine::executor::SurfaceMode;
pub use loess_rs::internals::engine::output::LoessResult;
pub use loess_rs::internals::math::boundary::BoundaryPolicy;
pub use loess_rs::internals::math::distance::DistanceMetric;
pub use loess_rs::internals::math::kernel::WeightFunction;
pub use loess_rs::internals::math::scaling::ScalingMethod;
pub use loess_rs::internals::primitives::backend::Backend;
pub use loess_rs::internals::primitives::errors::LoessError;

// Adapter selection namespace.
#[allow(non_snake_case)]
pub mod Adapter {
    pub use super::{Batch, Online, Streaming};
}
#[derive(Debug, Clone, Copy)]
pub struct Batch;

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync + 'static> LoessAdapter<T>
    for Batch
{
    type Output = ParallelBatchLoessBuilder<T>;

    fn convert<Mode>(builder: LoessBuilder<T, Mode>) -> Self::Output {
        // Determine parallel mode: user choice OR default to true for fastLoess Batch
        let parallel = builder.parallel.unwrap_or(true);
        // Extract fastLoess-specific fields before the base convert consumes the builder
        let weighted_metric_weights = builder.weighted_metric_weights.clone();
        let cv_method_str = builder.cv_method_str.clone();
        let cv_k_val = builder.cv_k_val;

        // Delegate to base implementation to create base builder
        let mut base = <BaseBatch as LoessAdapter<T>>::convert(builder);
        base.parallel = Some(parallel);

        // Wrap with extension fields
        ParallelBatchLoessBuilder {
            base,
            parse_errors: Vec::new(),
            weighted_metric_weights,
            cv_method_str,
            cv_k_val,
        }
    }
}

// Marker for parallel chunked streaming processing.
#[derive(Debug, Clone, Copy)]
pub struct Streaming;

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync + 'static> LoessAdapter<T>
    for Streaming
{
    type Output = ParallelStreamingLoessBuilder<T>;

    fn convert<Mode>(builder: LoessBuilder<T, Mode>) -> Self::Output {
        // Determine parallel mode: user choice OR default to true for fastLoess Streaming
        let parallel = builder.parallel.unwrap_or(true);
        // Extract fastLoess-specific fields before the base convert consumes the builder
        let weighted_metric_weights = builder.weighted_metric_weights.clone();

        // Delegate to base implementation to create base builder
        let mut base = <BaseStreaming as LoessAdapter<T>>::convert(builder);
        base.parallel = Some(parallel);

        // Wrap with extension fields
        ParallelStreamingLoessBuilder {
            base,
            parse_errors: Vec::new(),
            weighted_metric_weights,
        }
    }
}

// Marker for incremental online processing with parallel support.
#[derive(Debug, Clone, Copy)]
pub struct Online;

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync + 'static> LoessAdapter<T>
    for Online
{
    type Output = ParallelOnlineLoessBuilder<T>;

    fn convert<Mode>(builder: LoessBuilder<T, Mode>) -> Self::Output {
        // Determine parallel mode: user choice OR default to false for fastLoess Online
        let parallel = builder.parallel.unwrap_or(false);
        // Extract fastLoess-specific fields before the base convert consumes the builder
        let weighted_metric_weights = builder.weighted_metric_weights.clone();

        // Delegate to base implementation to create base builder
        let mut base = <BaseOnline as LoessAdapter<T>>::convert(builder);
        base.parallel = Some(parallel);

        // Wrap with extension fields
        ParallelOnlineLoessBuilder {
            base,
            parse_errors: Vec::new(),
            weighted_metric_weights,
        }
    }
}

// Entry-point wrapper types: Loess / StreamingLoess / OnlineLoess
// Mirror the bindings API while defaulting to parallel execution.

// Macro: generate method-forwarding impls common to all three entry-point types.
macro_rules! impl_common_builder {
    ($t:ty) => {
        impl $t {
            pub fn new() -> Self {
                Self(LoessBuilder::new())
            }
            // string enum options
            pub fn weight_function(mut self, s: &str) -> Self {
                self.0 = self.0.weight_function(s);
                self
            }
            pub fn robustness_method(mut self, s: &str) -> Self {
                self.0 = self.0.robustness_method(s);
                self
            }
            pub fn scaling_method(mut self, s: &str) -> Self {
                self.0 = self.0.scaling_method(s);
                self
            }
            pub fn zero_weight_fallback(mut self, s: &str) -> Self {
                self.0 = self.0.zero_weight_fallback(s);
                self
            }
            pub fn boundary_policy(mut self, s: &str) -> Self {
                self.0 = self.0.boundary_policy(s);
                self
            }
            pub fn degree(mut self, s: &str) -> Self {
                self.0 = self.0.degree(s);
                self
            }
            pub fn distance_metric(mut self, s: &str) -> Self {
                self.0 = self.0.distance_metric(s);
                self
            }
            pub fn surface_mode(mut self, s: &str) -> Self {
                self.0 = self.0.surface_mode(s);
                self
            }
            // numeric / bool options
            pub fn fraction(mut self, f: f64) -> Self {
                self.0 = self.0.fraction(f);
                self
            }
            pub fn iterations(mut self, i: usize) -> Self {
                self.0 = self.0.iterations(i);
                self
            }
            pub fn confidence_intervals(mut self, level: f64) -> Self {
                self.0 = self.0.confidence_intervals(level);
                self
            }
            pub fn prediction_intervals(mut self, level: f64) -> Self {
                self.0 = self.0.prediction_intervals(level);
                self
            }
            pub fn auto_converge(mut self, tol: f64) -> Self {
                self.0 = self.0.auto_converge(tol);
                self
            }
            pub fn dimensions(mut self, d: usize) -> Self {
                self.0 = self.0.dimensions(d);
                self
            }
            pub fn cell(mut self, c: f64) -> Self {
                self.0 = self.0.cell(c);
                self
            }
            pub fn interpolation_vertices(mut self, v: usize) -> Self {
                self.0 = self.0.interpolation_vertices(v);
                self
            }
            pub fn boundary_degree_fallback(mut self, b: bool) -> Self {
                self.0 = self.0.boundary_degree_fallback(b);
                self
            }
            pub fn weighted_metric_weights(mut self, w: Vec<f64>) -> Self {
                self.0 = self.0.weighted_metric_weights(w);
                self
            }
            // flag options (no argument)
            pub fn return_se(mut self) -> Self {
                self.0 = self.0.return_se();
                self
            }
            pub fn return_diagnostics(mut self) -> Self {
                self.0 = self.0.return_diagnostics();
                self
            }
            pub fn return_residuals(mut self) -> Self {
                self.0 = self.0.return_residuals();
                self
            }
            pub fn return_robustness_weights(mut self) -> Self {
                self.0 = self.0.return_robustness_weights();
                self
            }
            // dev options
            #[doc(hidden)]
            pub fn parallel(mut self, p: bool) -> Self {
                self.0 = self.0.parallel(p);
                self
            }
            #[doc(hidden)]
            pub fn backend(mut self, b: Backend) -> Self {
                self.0 = self.0.backend(b);
                self
            }
        }
    };
}

// Parallel batch LOESS entry point.
pub struct Loess(LoessBuilder<f64>);
impl_common_builder!(Loess);
impl Loess {
    pub fn custom_weights(mut self, w: Vec<f64>) -> Self {
        self.0 = self.0.custom_weights(w);
        self
    }
    pub fn cv_method(mut self, m: &str) -> Self {
        self.0 = self.0.cv_method(m);
        self
    }
    pub fn cv_k(mut self, k: usize) -> Self {
        self.0 = self.0.cv_k(k);
        self
    }
    pub fn cv_fractions(mut self, f: Vec<f64>) -> Self {
        self.0 = self.0.cv_fractions(f);
        self
    }
    pub fn cv_seed(mut self, s: u64) -> Self {
        self.0 = self.0.cv_seed(s);
        self
    }

    pub fn build(self) -> Result<ParallelBatchLoess<f64>, LoessError> {
        Batch::convert(self.0).build()
    }
}

// Parallel streaming LOESS entry point.
pub struct StreamingLoess(LoessBuilder<f64>);
impl_common_builder!(StreamingLoess);
impl StreamingLoess {
    pub fn chunk_size(mut self, s: usize) -> Self {
        self.0 = self.0.chunk_size(s);
        self
    }
    pub fn overlap(mut self, o: usize) -> Self {
        self.0 = self.0.overlap(o);
        self
    }
    pub fn merge_strategy(mut self, s: &str) -> Self {
        self.0 = self.0.merge_strategy(s);
        self
    }

    pub fn build(self) -> Result<ParallelStreamingLoess<f64>, LoessError> {
        Streaming::convert(self.0).build()
    }
}

// Parallel online LOESS entry point.
pub struct OnlineLoess(LoessBuilder<f64>);
impl_common_builder!(OnlineLoess);
impl OnlineLoess {
    pub fn window_capacity(mut self, c: usize) -> Self {
        self.0 = self.0.window_capacity(c);
        self
    }
    pub fn min_points(mut self, m: usize) -> Self {
        self.0 = self.0.min_points(m);
        self
    }
    pub fn update_mode(mut self, s: &str) -> Self {
        self.0 = self.0.update_mode(s);
        self
    }

    pub fn build(self) -> Result<ParallelOnlineLoess<f64>, LoessError> {
        Online::convert(self.0).build()
    }
}
