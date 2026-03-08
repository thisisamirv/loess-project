//! High-level API for LOESS smoothing with parallel execution support.
//!
//! This module provides the primary user-facing entry point for LOESS with
//! heavy-duty parallel execution capabilities. It extends the `loess` API
//! with adapters that utilize all available CPU cores or GPU hardware.

// Internal dependencies
use crate::adapters::batch::ParallelBatchLoessBuilder;
use crate::adapters::online::ParallelOnlineLoessBuilder;
use crate::adapters::streaming::ParallelStreamingLoessBuilder;

// External dependencies
use num_traits::Float;

// Import base marker types for delegation
use loess_rs::internals::api::Batch as BaseBatch;
use loess_rs::internals::api::Online as BaseOnline;
use loess_rs::internals::api::Streaming as BaseStreaming;

// Publicly re-exported types
pub use loess_rs::internals::adapters::online::UpdateMode;
pub use loess_rs::internals::adapters::streaming::MergeStrategy;
pub use loess_rs::internals::algorithms::regression::ZeroWeightFallback;
pub use loess_rs::internals::algorithms::robustness::RobustnessMethod;
pub use loess_rs::internals::api::{LoessAdapter, LoessBuilder};
pub use loess_rs::internals::engine::output::LoessResult;
pub use loess_rs::internals::evaluation::cv::{KFold, LOOCV};
pub use loess_rs::internals::math::boundary::BoundaryPolicy;
pub use loess_rs::internals::math::kernel::WeightFunction;
pub use loess_rs::internals::math::scaling::ScalingMethod;
pub use loess_rs::internals::primitives::backend::Backend;
pub use loess_rs::internals::primitives::errors::LoessError;

// Adapter selection namespace.
#[allow(non_snake_case)]
pub mod Adapter {
    pub use super::{Batch, Online, Streaming};
}

// Marker for parallel in-memory batch processing.
#[derive(Debug, Clone, Copy)]
pub struct Batch;

impl<T: Float> LoessAdapter<T> for Batch {
    type Output = ParallelBatchLoessBuilder<T>;

    fn convert(builder: LoessBuilder<T>) -> Self::Output {
        // Determine parallel mode: user choice OR default to true for fastLoess Batch
        let parallel = builder.parallel.unwrap_or(true);

        // Delegate to base implementation to create base builder
        let mut base = <BaseBatch as LoessAdapter<T>>::convert(builder);
        base = base.parallel(parallel);

        // Wrap with extension fields
        ParallelBatchLoessBuilder { base }
    }
}

// Marker for parallel chunked streaming processing.
#[derive(Debug, Clone, Copy)]
pub struct Streaming;

impl<T: Float> LoessAdapter<T> for Streaming {
    type Output = ParallelStreamingLoessBuilder<T>;

    fn convert(builder: LoessBuilder<T>) -> Self::Output {
        // Determine parallel mode: user choice OR default to true for fastLoess Streaming
        let parallel = builder.parallel.unwrap_or(true);

        // Delegate to base implementation to create base builder
        let mut base = <BaseStreaming as LoessAdapter<T>>::convert(builder);
        base = base.parallel(parallel);

        // Wrap with extension fields
        ParallelStreamingLoessBuilder { base }
    }
}

// Marker for incremental online processing with parallel support.
#[derive(Debug, Clone, Copy)]
pub struct Online;

impl<T: Float> LoessAdapter<T> for Online {
    type Output = ParallelOnlineLoessBuilder<T>;

    fn convert(builder: LoessBuilder<T>) -> Self::Output {
        // Determine parallel mode: user choice OR default to false for fastLoess Online
        let parallel = builder.parallel.unwrap_or(false);

        // Delegate to base implementation to create base builder
        let mut base = <BaseOnline as LoessAdapter<T>>::convert(builder);
        base = base.parallel(parallel);

        // Wrap with extension fields
        ParallelOnlineLoessBuilder { base }
    }
}
