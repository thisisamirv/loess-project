//! Shared typed policies and configuration for LOESS execution.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingPolicy {
    #[default]
    Error,
    Drop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UpdateMode {
    Full,
    #[default]
    Incremental,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MergeStrategy {
    Average,
    #[default]
    WeightedAverage,
    TakeFirst,
    TakeLast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoundaryPolicy {
    // Linearly extrapolate x-values and replicate y-values to provide context.
    #[default]
    Extend,

    // Mirror values across the boundary.
    Reflect,

    // Use zero padding beyond data boundaries.
    Zero,

    // No boundary padding (standard LOESS behavior).
    NoBoundary,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum DistanceMetric<T> {
    // Standard Euclidean distance: sqrt(sum((x_i - y_i)^2))
    Euclidean,

    // Normalized Euclidean distance.
    #[default]
    Normalized,

    // Manhattan distance (L1 norm): sum(|x_i - y_i|)
    Manhattan,

    // Chebyshev distance (L-infinity norm): max|x_i - y_i|
    Chebyshev,

    // Minkowski distance (Lp norm): (sum(|x_i - y_i|^p))^(1/p)
    Minkowski(T),

    // Weighted Euclidean distance: sqrt(sum(w_i(x_i - y_i)^2))
    Weighted(Vec<T>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum WeightFunction {
    // Cosine kernel: K(u) = cos(pi * u / 2) for |u| < 1.
    Cosine,

    // Epanechnikov kernel: K(u) = (1 - u^2) for |u| < 1.
    Epanechnikov,

    // Gaussian kernel: K(u) = exp(-u^2 / 2).
    Gaussian,

    // Biweight (quartic) kernel: K(u) = (1 - u^2)^2 for |u| < 1.
    Biweight,

    // Triangular (linear) kernel: K(u) = (1 - |u|) for |u| < 1.
    Triangle,

    // Tricube kernel: K(u) = (1 - |u|^3)^3 for |u| < 1.
    //
    // This is the default and recommended kernel choice.
    #[default]
    Tricube,

    // Uniform (rectangular) kernel: K(u) = 1 for |u| < 1.
    Uniform,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ScalingMethod {
    // Median Absolute Residual: `median(|r|)`.
    MAR,

    // Median Absolute Deviation: `median(|r - median(r)|)`.
    #[default]
    MAD,

    // Mean Absolute Residual: `mean(|r|)`.
    Mean,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PolynomialDegree {
    // Degree 0: Local constant (weighted mean)
    Constant,

    // Degree 1: Local linear regression (default)
    #[default]
    Linear,

    // Degree 2: Local quadratic regression
    Quadratic,

    // Degree 3: Local cubic regression
    Cubic,

    // Degree 4: Local quartic regression
    Quartic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZeroWeightFallback {
    // Use local mean (default).
    #[default]
    UseLocalMean,

    // Return the original y-value.
    ReturnOriginal,

    // Return None (propagate failure).
    ReturnNone,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RobustnessMethod {
    // Bisquare (Tukey's biweight) - default and most common.
    #[default]
    Bisquare,

    // Huber weights - less aggressive downweighting.
    Huber,

    // Talwar (hard threshold) - most aggressive.
    Talwar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SurfaceMode {
    // Use interpolation surface for faster evaluation.
    #[default]
    Interpolation,

    // Use direct per-point fitting for maximum accuracy.
    Direct,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExtrapolationPolicy {
    // Clamp each out-of-range dimension to the nearest training boundary (default;
    // matches `predict()`'s original behavior).
    #[default]
    Clamp,

    // Linearly extrapolate from the boundary point's local fit and gradient (first-order
    // Taylor expansion from the clamped point).
    Linear,

    // Fail the whole `predict()` call with `LoessError::PredictOutOfRange` if any query
    // point falls outside the training range on any dimension.
    Error,
}
