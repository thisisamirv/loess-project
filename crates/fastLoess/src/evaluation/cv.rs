//! Parallel cross-validation for LOESS bandwidth selection.
//!
//! ## Purpose  
//!
//! This module provides parallel cross-validation functions for optimal
//! bandwidth (fraction) selection. Cross-validation is computationally
//! expensive as it requires fitting the model multiple times, making it
//! an ideal candidate for parallelization.
//!
//! ## Design notes
//!
//! * **Parallelism**: Uses `rayon` to evaluate candidate fractions in parallel.
//! * **Integration**: Plugs into the `loess-rs` executor via the `CVPassFn` hook.
//! * **Generics**: Generic over `Float` types.
//!
//! ## Key concepts
//!
//! * **Parallel Evaluation**: Evaluates all candidate fractions simultaneously.
//! * **Fraction Selection**: Selects bandwidth minimizing Cross-Validation (CV) error.
//! * **Integration**: Seamlessly integrates with serial/parallel executors.
//!
//! ## Invariants
//!
//! * Input arrays x and y must have the same length.
//! * Fractions must be in (0, 1].
//! * At least 2 data points are required.
//!
//! ## Non-goals
//!
//! * This module does not implement the CV strategy logic (delegated to loess-rs).
//! * This module does not handle partial parallelism (all-or-nothing).

// Feature-gated imports
#[cfg(feature = "cpu")]
use rayon::prelude::*;

// External dependencies
use num_traits::Float;
use std::fmt::Debug;
use std::vec::Vec;

// Export dependencies from loess-rs crate
use loess_rs::internals::algorithms::regression::SolverLinalg;
use loess_rs::internals::engine::executor::{LoessConfig, LoessExecutor};
use loess_rs::internals::evaluation::cv::CVKind;
use loess_rs::internals::math::distance::DistanceLinalg;
use loess_rs::internals::math::linalg::FloatLinalg;
use loess_rs::internals::math::neighborhood::KDTree;
use loess_rs::internals::primitives::window::Window;

// ============================================================================
// Parallel Cross-Validation
// ============================================================================

/// Perform parallel cross-validation to select optimal LOESS bandwidth.
///
/// This function evaluates candidate fractions in parallel to find the
/// one that minimizes the cross-validation error.
#[cfg(feature = "cpu")]
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
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    let best_fraction = fractions
        .get(best_idx)
        .copied()
        .unwrap_or_else(|| T::from(0.67).unwrap());

    (best_fraction, scores)
}

/// Evaluate a single fraction using cross-validation.
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
    let dims = config.dimensions;
    let n = y.len();

    // Create a modified config with the test fraction
    let mut cv_config = config.clone();
    cv_config.fraction = Some(fraction);
    cv_config.cv_fractions = None; // Don't recurse
    cv_config.return_variance = None; // Speed up CV
    cv_config.return_variance = None; // Speed up CV

    match cv_kind {
        CVKind::LOOCV => {
            // Leave-one-out cross-validation
            // For efficiency, we just fit once and compute predictions
            let _window_size = Window::calculate_span(n, fraction);

            // Fit the model once
            let result = LoessExecutor::run_with_config(x, y, cv_config.clone());

            // Compute RMSE (approximate LOOCV using residuals)
            let mut sse = T::zero();
            for (i, &y_val) in y.iter().enumerate().take(n) {
                let residual = y_val - result.smoothed[i];
                sse = sse + residual * residual;
            }
            (sse / T::from(n).unwrap()).sqrt()
        }
        CVKind::KFold(k) => {
            if k < 2 {
                return T::infinity();
            }

            // K-fold cross-validation
            let fold_size = n / k;
            if fold_size < 2 {
                return T::infinity();
            }

            let mut fold_rmses = Vec::with_capacity(k);
            let executor = LoessExecutor::from_config(&cv_config);
            let window_size = Window::calculate_span(n - fold_size, fraction); // Approx n_train

            for fold in 0..k {
                let test_start = fold * fold_size;
                let test_end = if fold == k - 1 {
                    n
                } else {
                    (fold + 1) * fold_size
                };
                let test_size = test_end - test_start;

                // Build training and test data
                let train_n = n - test_size;
                let mut train_x = Vec::with_capacity(train_n * dims);
                let mut train_y = Vec::with_capacity(train_n);
                let mut test_x = Vec::with_capacity(test_size * dims);

                for i in 0..n {
                    if i < test_start || i >= test_end {
                        for d in 0..dims {
                            train_x.push(x[i * dims + d]);
                        }
                        train_y.push(y[i]);
                    } else {
                        for d in 0..dims {
                            test_x.push(x[i * dims + d]);
                        }
                    }
                }

                if train_y.len() < 3 {
                    continue;
                }

                let mut fold_sse = T::zero();
                let mut fold_count = 0usize;

                // 1D Case: Use Interpolation to match Sequential behavior
                if dims == 1 {
                    // Fit on training data
                    let train_result =
                        LoessExecutor::run_with_config(&train_x, &train_y, cv_config.clone());

                    // Sort (x, smooth) for interpolation
                    let mut train_data: Vec<(T, T)> = train_x
                        .iter()
                        .zip(train_result.smoothed.iter())
                        .map(|(&xi, &yi)| (xi, yi))
                        .collect();

                    train_data
                        .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                    let (sorted_tx, sorted_smooth): (Vec<T>, Vec<T>) =
                        train_data.into_iter().unzip();

                    let mut preds = vec![T::zero(); test_x.len()];
                    CVKind::interpolate_prediction_batch(
                        &sorted_tx,
                        &sorted_smooth,
                        &test_x,
                        &mut preds,
                    );

                    for (i, &pred) in preds.iter().enumerate() {
                        let actual_y = y[test_start + i];
                        let residual = actual_y - pred;
                        fold_sse = fold_sse + residual * residual;
                        fold_count += 1;
                    }
                } else {
                    // nD Case: Use Direct Prediction (standard LOESS)

                    // Compute scales (Min-Max)
                    let mut scales = vec![T::one(); dims];
                    if train_n > 0 {
                        let mut mins = vec![train_x[0]; dims];
                        let mut maxs = vec![train_x[0]; dims];

                        for i in 0..train_n {
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
                            if range > T::epsilon() {
                                scales[d] = T::one() / range;
                            }
                        }
                    }

                    // Build KD-Tree on Training Data
                    let kdtree = KDTree::new(&train_x, dims);

                    let robustness_weights = vec![T::one(); train_n];

                    let predictions = executor.predict(
                        &train_x,
                        &train_y,
                        &robustness_weights,
                        &test_x,
                        window_size,
                        &scales,
                        &kdtree,
                    );

                    // Accumulate Error
                    for (i, &pred) in predictions.iter().enumerate() {
                        let actual_y = y[test_start + i];
                        let residual = actual_y - pred;
                        fold_sse = fold_sse + residual * residual;
                        fold_count += 1;
                    }
                }

                if fold_count > 0 {
                    fold_rmses.push((fold_sse / T::from(fold_count).unwrap()).sqrt());
                }
            }

            if fold_rmses.is_empty() {
                T::infinity()
            } else {
                let sum: T = fold_rmses.iter().copied().fold(T::zero(), |a, b| a + b);
                sum / T::from(fold_rmses.len()).unwrap()
            }
        }
    }
}

// Sequential fallback
#[cfg(not(feature = "cpu"))]
pub fn cv_pass_parallel<T>(
    _x: &[T],
    _y: &[T],
    fractions: &[T],
    _cv_kind: CVKind,
    _config: &LoessConfig<T>,
) -> (T, Vec<T>)
where
    T: Float + Send + Sync,
{
    // Return first fraction as default if parallel CV not available
    let best = fractions
        .first()
        .copied()
        .unwrap_or_else(|| T::from(0.67).unwrap());
    let scores = vec![T::infinity(); fractions.len()];
    (best, scores)
}
