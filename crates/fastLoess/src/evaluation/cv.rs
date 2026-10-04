//! Parallel cross-validation for LOESS bandwidth selection.
//!
//! This module provides the parallel cross-validation logic for selecting the
//! optimal smoothing fraction. It utilizes all available CPU cores to evaluate
//! multiple candidate fractions concurrently.
// ## srrstats Compliance
//
// @srrstats {RE6.0} Parallel CV: fraction candidates evaluated concurrently.
// @srrstats {G3.0} Rayon-based parallelization for bandwidth grid search.

// Imports
use rayon::prelude::*;

// External dependencies
use num_traits::Float;
use std::cmp::Ordering::Equal;
use std::fmt::Debug;
use std::vec::Vec;

// Export dependencies from loess-rs crate
use loess_rs::internals::algorithms::regression::specialized::SolverLinalg;
use loess_rs::internals::engine::executor::{CVRunOptions, LoessConfig, LoessExecutor};
use loess_rs::internals::evaluation::cv::CVKind;
use loess_rs::internals::math::distance::DistanceLinalg;
use loess_rs::internals::math::linalg::FloatLinalg;
use loess_rs::internals::primitives::buffer::CVBuffer;

// Perform parallel cross-validation to select optimal LOESS bandwidth.
//
// This function evaluates candidate fractions in parallel to find the
// one that minimizes the cross-validation error.
pub fn cv_pass_parallel<T>(
    x: &[T],
    y: &[T],
    fractions: &[T],
    cv_kind: CVKind,
    config: &LoessConfig<T>,
) -> (T, Vec<T>)
where
    T: FloatLinalg + DistanceLinalg + SolverLinalg + Float + Debug + Send + Sync + 'static,
{
    // Evaluate each fraction in parallel
    let scores: Vec<T> = fractions
        .par_iter()
        .map(|&frac| evaluate_fraction_cv(x, y, frac, cv_kind, config))
        .collect();

    // Find the fraction with minimum CV score (RMSE)
    let best_idx = scores
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    let best_fraction = fractions
        .get(best_idx)
        .copied()
        .unwrap_or_else(|| T::from(0.67).unwrap());

    (best_fraction, scores)
}

// Evaluate a single fraction using cross-validation.
fn evaluate_fraction_cv<T>(
    x: &[T],
    y: &[T],
    fraction: T,
    cv_kind: CVKind,
    config: &LoessConfig<T>,
) -> T
where
    T: FloatLinalg + DistanceLinalg + SolverLinalg + Float + Debug + Send + Sync + 'static,
{
    let executor = LoessExecutor::from_config(config);
    let mut buffer = CVBuffer::new(y.len(), config.dimensions);
    let (_, scores) = executor.cross_validate_with_options(
        x,
        y,
        &[fraction],
        CVRunOptions {
            kind: cv_kind,
            seed: config.cv_seed,
            tolerance: config.auto_converge,
        },
        &mut buffer,
    );
    scores.first().copied().unwrap_or_else(T::infinity)
}
