//! High-level API for LOESS smoothing.
//!
//! ## Purpose
//!
//! This module provides the primary user-facing entry point for LOESS. It
//! implements a fluent builder pattern for configuring regression parameters
//! and choosing an execution adapter (Batch, Streaming, or Online).
//!
//! ## Design notes
//!
//! * **Ergonomic**: Fluent builder with sensible defaults for all parameters.
//! * **Polymorphic**: Uses marker types to transition to specialized adapter builders.
//! * **Validated**: Core parameters are validated during adapter construction.
//! * **Type-Safe**: Generic over `Float` types for flexible precision.
//!
//! ## Key concepts
//!
//! * **Execution Adapters**: Batch, Streaming, and Online modes.
//! * **Configuration Flow**: Builder pattern ending in `.adapter(Adapter::Type)`.
//! * **Validation**: Parameters are validated when `.build()` is called on the adapter.
//!
//! ### Configuration Flow
//!
//! 1. Create a [`LoessBuilder`](crate::api::LoessBuilder) via `Loess::new()`.
//! 2. Chain configuration methods (`.fraction()`, `.iterations()`, etc.).
//! 3. Select an adapter via `.adapter(Adapter::Batch)` to get an execution builder.

// Feature-gated imports
#[cfg(feature = "cpu")]
use crate::adapters::batch::ParallelBatchLoessBuilder;
#[cfg(feature = "cpu")]
use crate::adapters::online::ParallelOnlineLoessBuilder;
#[cfg(feature = "cpu")]
use crate::adapters::streaming::ParallelStreamingLoessBuilder;

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
pub use loess_rs::internals::evaluation::cv::{KFold, LOOCV};
pub use loess_rs::internals::math::boundary::BoundaryPolicy;
pub use loess_rs::internals::math::distance::DistanceMetric;
pub use loess_rs::internals::math::kernel::WeightFunction;
pub use loess_rs::internals::math::scaling::ScalingMethod;
pub use loess_rs::internals::primitives::backend::Backend;
pub use loess_rs::internals::primitives::errors::LoessError;

// ============================================================================
// Adapter Module
// ============================================================================

/// Adapter selection namespace.
#[allow(non_snake_case)]
pub mod Adapter {
    pub use super::{Batch, Online, Streaming};
}

// ============================================================================
// Adapter Marker Types
// ============================================================================

/// Marker for parallel in-memory batch processing.
#[derive(Debug, Clone, Copy)]
pub struct Batch;

#[cfg(feature = "cpu")]
impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync + 'static> LoessAdapter<T>
    for Batch
{
    type Output = ParallelBatchLoessBuilder<T>;

    fn convert(builder: LoessBuilder<T>) -> Self::Output {
        // Determine parallel mode: user choice OR default to true for fastLoess Batch
        let parallel = builder.parallel.unwrap_or(true);

        // Delegate to base implementation to create base builder
        let mut base = <BaseBatch as LoessAdapter<T>>::convert(builder);
        base.parallel = Some(parallel);

        // Wrap with extension fields
        ParallelBatchLoessBuilder { base }
    }
}

/// Marker for parallel chunked streaming processing.
#[derive(Debug, Clone, Copy)]
pub struct Streaming;

#[cfg(feature = "cpu")]
impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync + 'static> LoessAdapter<T>
    for Streaming
{
    type Output = ParallelStreamingLoessBuilder<T>;

    fn convert(builder: LoessBuilder<T>) -> Self::Output {
        // Determine parallel mode: user choice OR default to true for fastLoess Streaming
        let parallel = builder.parallel.unwrap_or(true);

        // Delegate to base implementation to create base builder
        let mut base = <BaseStreaming as LoessAdapter<T>>::convert(builder);
        base.parallel = Some(parallel);

        // Wrap with extension fields
        ParallelStreamingLoessBuilder { base }
    }
}

/// Marker for incremental online processing with parallel support.
#[derive(Debug, Clone, Copy)]
pub struct Online;

#[cfg(feature = "cpu")]
impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync + 'static> LoessAdapter<T>
    for Online
{
    type Output = ParallelOnlineLoessBuilder<T>;

    fn convert(builder: LoessBuilder<T>) -> Self::Output {
        // Determine parallel mode: user choice OR default to false for fastLoess Online
        let parallel = builder.parallel.unwrap_or(false);

        // Delegate to base implementation to create base builder
        let mut base = <BaseOnline as LoessAdapter<T>>::convert(builder);
        base.parallel = Some(parallel);

        // Wrap with extension fields
        ParallelOnlineLoessBuilder { base }
    }
}
