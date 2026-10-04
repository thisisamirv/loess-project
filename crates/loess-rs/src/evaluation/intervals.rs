//! Confidence and prediction intervals for LOESS smoothing.
//!
//! This module provides tools for quantifying uncertainty in LOESS smoothing
//! through standard errors, confidence intervals, and prediction intervals.
// ## srrstats Compliance
//
// @srrstats {RE5.0} Confidence intervals for the mean smoothed function.
// Prediction intervals for new observations.
// Acklams rational approximation for inverse normal CDF (z-scores).

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use num_traits::Float;

// Internal dependencies
use crate::evaluation::defaults::DEFAULT_INTERVAL_LEVEL;
use crate::math::scaling::ScalingMethod;
use crate::primitives::errors::LoessError;
use crate::primitives::window::Window;

// Configuration for computing confidence/prediction intervals and standard errors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntervalMethod<T> {
    // Desired probability coverage (e.g., 0.95 for 95% intervals).
    pub level: T,

    // Whether to compute confidence intervals for the mean function.
    pub confidence: bool,

    // Whether to compute prediction intervals for new observations.
    pub prediction: bool,

    // Whether to return estimated standard errors for fitted values.
    pub se: bool,
}

impl<T: Float> Default for IntervalMethod<T> {
    fn default() -> Self {
        Self::none()
    }
}

impl<T: Float> IntervalMethod<T> {
    // No intervals or standard errors.
    fn none() -> Self {
        Self {
            level: T::from(DEFAULT_INTERVAL_LEVEL).unwrap(),
            confidence: false,
            prediction: false,
            se: false,
        }
    }

    // Confidence intervals only at the specified level.
    pub fn confidence(level: T) -> Self {
        Self {
            level,
            confidence: true,
            prediction: false,
            se: true,
        }
    }

    // Prediction intervals only at the specified level.
    pub fn prediction(level: T) -> Self {
        Self {
            level,
            confidence: false,
            prediction: true,
            se: true,
        }
    }

    // Standard errors only (no intervals).
    pub fn se() -> Self {
        Self {
            level: T::from(0.95).unwrap(),
            confidence: false,
            prediction: false,
            se: true,
        }
    }
}

impl<T: Float> IntervalMethod<T> {
    // Constant to convert MAD to an unbiased estimate of sigma for normal data.
    //
    // For normally distributed data, MAD × 1.4826 ≈ standard deviation.
    const MAD_TO_STD_FACTOR: f64 = 1.4826;

    // Minimum tuned-scale absolute epsilon to avoid division by zero.
    const MIN_TUNED_SCALE: f64 = 1e-12;

    // Number of parameters in local linear regression (intercept + slope).
    const LINEAR_PARAMS: f64 = 2.0;

    // Estimate the residual standard deviation using a robust method or delta1.
    // - If delta1 is provided: sigma = sqrt(RSS / delta1)
    // - Fallback: sigma_hat = 1.4826 * MAD(residuals).
    //
    // `pub(crate)` so `Predict::call()` can compute a residual scale for
    // out-of-sample prediction intervals regardless of whether the original `fit()`
    // call requested intervals.
    pub(crate) fn calculate_residual_sd(residuals: &[T], delta1: Option<T>) -> T {
        if let Some(d1) = delta1
            && d1 > T::zero()
        {
            let rss = residuals.iter().fold(T::zero(), |acc, &r| acc + r * r);
            return (rss / d1).sqrt();
        }

        let n = residuals.len();
        let scale_const = T::from(Self::MAD_TO_STD_FACTOR).unwrap();

        if n == 1 {
            return residuals[0].abs() * scale_const;
        }

        let mut vals = residuals.to_vec();
        let mad = ScalingMethod::MAD.compute(&mut vals);
        if mad > T::zero() {
            mad * scale_const
        } else {
            // Apply minimum scale to avoid division by zero
            let min_eps = T::from(Self::MIN_TUNED_SCALE).unwrap();
            min_eps * scale_const
        }
    }

    // Core mathematical function for computing standard error of a local
    // linear fit from centered design moments. The leverage is the squared
    // norm of the equivalent-kernel row, and the residual degrees of freedom
    // corrects for the weighted design.
    pub fn compute_se(sum_w: T, sum_w_r2: T, s1: T, s2: T, t0: T, t1: T, t2: T) -> T {
        if sum_w <= T::zero() {
            return T::zero();
        }

        let two = T::from(Self::LINEAR_PARAMS).unwrap();
        let det = sum_w * s2 - s1 * s1;
        if det <= T::zero() {
            return T::zero();
        }

        // Squared norm of the local-linear equivalent-kernel row.
        let leverage = (s2 * s2 * t0 - two * s1 * s2 * t1 + s1 * s1 * t2) / (det * det);
        if leverage <= T::zero() {
            return T::zero();
        }

        // Kernel-corrected residual degrees of freedom.
        let df = sum_w - two + t0 / sum_w;
        if df <= T::zero() {
            return T::zero();
        }

        let variance = sum_w_r2 / df;
        (variance * leverage).sqrt()
    }

    // Classical simple-linear-regression standard errors for a global 1D fit.
    // This is the same se.fit formula used by stats::lm:
    // sigma_hat * sqrt(1/n + (x0 - x_mean)^2 / Sxx).
    pub fn compute_global_ols_se(x: &[T], y: &[T], y_smooth: &[T]) -> Vec<T> {
        let n = x.len();
        if n == 0 {
            return Vec::new();
        }

        let n_t = T::from(n).unwrap_or(T::one());
        let x_mean = x.iter().fold(T::zero(), |sum, &value| sum + value) / n_t;
        let mut sse = T::zero();
        let mut sxx = T::zero();
        for ((&xi, &yi), &fit) in x.iter().zip(y.iter()).zip(y_smooth.iter()) {
            let dx = xi - x_mean;
            sxx = sxx + dx * dx;
            let residual = yi - fit;
            sse = sse + residual * residual;
        }

        let two = T::from(Self::LINEAR_PARAMS).unwrap();
        let df = n_t - two;
        if df <= T::zero() {
            return vec![T::zero(); n];
        }

        let sigma = (sse / df).sqrt();
        let tol = T::epsilon() * x.iter().fold(T::zero(), |sum, &value| sum + value * value);
        if sxx <= tol {
            let se = sigma * (T::one() / n_t).sqrt();
            return vec![se; n];
        }

        x.iter()
            .map(|&xi| sigma * (T::one() / n_t + (xi - x_mean) * (xi - x_mean) / sxx).sqrt())
            .collect()
    }

    // Compute standard errors for all points in a smoothed series.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_window_se<F>(
        &self,
        x: &[T],
        y: &[T],
        y_smooth: &[T],
        window_size: usize,
        robustness_weights: &[T],
        std_errors: &mut [T],
        weight_fn: &F,
    ) where
        F: Fn(T) -> T,
    {
        // Early exit if no intervals or SE requested
        if !self.se && !self.confidence && !self.prediction {
            return;
        }

        let n = x.len();

        for (i, se) in std_errors.iter_mut().enumerate().take(n) {
            // Initialize and center window
            let mut window = Window::initialize(i, window_size, n);
            window.recenter(x, i, n);

            let idx = i;
            let left = window.left;
            let right = window.right;

            // Compute bandwidth
            let x_current = x[idx];
            let bandwidth_left = x_current - x[left];
            let bandwidth_right = x[right] - x_current;
            let bandwidth = T::max(bandwidth_left, bandwidth_right);

            if bandwidth <= T::zero() {
                *se = T::zero();
                continue;
            }

            // Compute weight for current point (distance = 0)
            let u_idx = T::zero();
            let kernel_val = weight_fn(u_idx);
            let w_idx = kernel_val * robustness_weights[idx];

            // Accumulate weighted residual variance
            let mut sum_w_r2 = T::zero();
            let mut sum_w = T::zero();
            let mut s1 = T::zero();
            let mut s2 = T::zero();
            let mut t0 = T::zero();
            let mut t1 = T::zero();
            let mut t2 = T::zero();

            for j in left..=right {
                let dist = (x[j] - x_current).abs();
                let u = dist / bandwidth;
                let w = if j == idx {
                    w_idx
                } else {
                    weight_fn(u) * robustness_weights[j]
                };

                let r = y[j] - y_smooth[j];
                sum_w_r2 = sum_w_r2 + w * r * r;
                sum_w = sum_w + w;
                s1 = s1 + w * (x[j] - x_current);
                s2 = s2 + w * (x[j] - x_current) * (x[j] - x_current);
                t0 = t0 + w * w;
                t1 = t1 + w * w * (x[j] - x_current);
                t2 = t2 + w * w * (x[j] - x_current) * (x[j] - x_current);
            }

            *se = Self::compute_se(sum_w, sum_w_r2, s1, s2, t0, t1, t2);
        }
    }

    // Compute requested intervals (confidence and/or prediction).
    #[allow(clippy::type_complexity)]
    pub fn compute_intervals(
        &self,
        y_smooth: &[T],
        std_errors: &[T],
        residuals: &[T],
        delta1: Option<T>,
        delta2: Option<T>,
    ) -> Result<
        (
            Option<Vec<T>>, // confidence lower
            Option<Vec<T>>, // confidence upper
            Option<Vec<T>>, // prediction lower
            Option<Vec<T>>, // prediction upper
        ),
        LoessError,
    > {
        // Effective degrees of freedom: df = delta1^2 / delta2
        let df = if let (Some(d1), Some(d2)) = (delta1, delta2) {
            if d2 > T::zero() {
                Some(d1 * d1 / d2)
            } else {
                None
            }
        } else {
            None
        };

        // Compute confidence intervals if requested
        let (mut conf_lower, mut conf_upper) = if self.confidence {
            let (lower, upper) = self
                .compute_confidence_intervals_impl(y_smooth, std_errors, df)
                .map_err(|_| LoessError::InvalidIntervals(self.level.to_f64().unwrap_or(0.0)))?;
            (Some(lower), Some(upper))
        } else {
            (None, None)
        };

        // Compute prediction intervals if requested
        let (mut pred_lower, mut pred_upper) = if self.prediction {
            let residual_sd = Self::calculate_residual_sd(residuals, delta1);
            let (lower, upper) = self
                .compute_prediction_intervals_impl(y_smooth, std_errors, residual_sd, df)
                .map_err(|_| LoessError::InvalidIntervals(self.level.to_f64().unwrap_or(0.0)))?;
            (Some(lower), Some(upper))
        } else {
            (None, None)
        };

        // Guard against degenerate intervals
        let residual_sd = Self::calculate_residual_sd(residuals, delta1);
        let any_std_nonzero = std_errors.iter().any(|&s| s > T::zero());

        if residual_sd > T::zero() || any_std_nonzero {
            let eps = T::from(1e-12).unwrap_or_else(|| T::from(1e-6).unwrap());

            // Fix degenerate confidence intervals
            if let (Some(lo), Some(hi)) = (&mut conf_lower, &mut conf_upper) {
                for (l, h) in lo.iter_mut().zip(hi.iter_mut()) {
                    let width = *h - *l;
                    if !width.is_finite() || width <= T::zero() {
                        *h = *l + eps;
                    }
                }
            }

            // Fix degenerate prediction intervals
            if let (Some(lo), Some(hi)) = (&mut pred_lower, &mut pred_upper) {
                for (l, h) in lo.iter_mut().zip(hi.iter_mut()) {
                    let width = *h - *l;
                    if !width.is_finite() || width <= T::zero() {
                        *h = *l + eps;
                    }
                }
            }
        }

        Ok((conf_lower, conf_upper, pred_lower, pred_upper))
    }

    fn compute_confidence_intervals_impl(
        &self,
        y_smooth: &[T],
        std_errors: &[T],
        df: Option<T>,
    ) -> Result<(Vec<T>, Vec<T>), &'static str> {
        let z = if let Some(df_val) = df {
            Self::approximate_t_score(self.level, df_val)?
        } else {
            Self::approximate_z_score(self.level)?
        };

        let lower: Vec<T> = y_smooth
            .iter()
            .zip(std_errors.iter())
            .map(|(&ys, &se)| ys - z * se)
            .collect();

        let upper: Vec<T> = y_smooth
            .iter()
            .zip(std_errors.iter())
            .map(|(&ys, &se)| ys + z * se)
            .collect();

        Ok((lower, upper))
    }

    fn compute_prediction_intervals_impl(
        &self,
        y_smooth: &[T],
        std_errors: &[T],
        residual_sd: T,
        df: Option<T>,
    ) -> Result<(Vec<T>, Vec<T>), &'static str> {
        let z = if let Some(df_val) = df {
            Self::approximate_t_score(self.level, df_val)?
        } else {
            Self::approximate_z_score(self.level)?
        };
        let rsd_sq = residual_sd * residual_sd;

        let lower: Vec<T> = y_smooth
            .iter()
            .zip(std_errors.iter())
            .map(|(&ys, &se)| {
                let pred_se = (se * se + rsd_sq).sqrt();
                ys - z * pred_se
            })
            .collect();

        let upper: Vec<T> = y_smooth
            .iter()
            .zip(std_errors.iter())
            .map(|(&ys, &se)| {
                let pred_se = (se * se + rsd_sq).sqrt();
                ys + z * pred_se
            })
            .collect();

        Ok((lower, upper))
    }

    // Approximate the critical value (T-score) for a given confidence level and DOF.
    // For very large DOF, fallback to Z-score.
    pub fn approximate_t_score(confidence_level: T, df: T) -> Result<T, &'static str> {
        let df_f = df.to_f64().unwrap_or(2.0);

        // Approximation for T-distribution: Z * sqrt(df / (df - 2))
        // This is only valid for df > 2 and is an approximation for the variance.
        // A better approach for small df would be appreciated, but for LOESS,
        // df is usually reasonable.

        let z = Self::approximate_z_score(confidence_level)?;
        let z_f = z.to_f64().unwrap_or(1.96);

        let t_f = if df_f > 2.0 {
            z_f * (df_f / (df_f - 2.0)).sqrt()
        } else {
            // Very small DOF fallback: increase Z significantly or use a fixed high value
            z_f * 1.5
        };

        Ok(T::from(t_f).unwrap_or(z))
    }

    // Approximate the critical value (Z-score) for a given confidence level.
    // z = Phi^-1((1 + p) / 2) where Phi^-1 is the inverse standard normal CDF.
    pub fn approximate_z_score(confidence_level: T) -> Result<T, &'static str> {
        let cl_f = confidence_level.to_f64().unwrap_or(0.95);

        // Convert confidence level to cumulative probability
        let p = (1.0 + cl_f) / 2.0;

        // Fast paths for common confidence levels
        let z = if (cl_f - 0.99).abs() < 1e-6 {
            2.576
        } else if (cl_f - 0.95).abs() < 1e-6 {
            1.960
        } else if (cl_f - 0.90).abs() < 1e-6 {
            1.645
        } else {
            // Use Acklam's algorithm for other values
            Self::acklam_inverse_cdf(p)
        };

        Ok(T::from(z).unwrap_or_else(|| T::one()))
    }

    // Rational approximation of the inverse standard normal CDF.
    fn acklam_inverse_cdf(p: f64) -> f64 {
        if p <= 0.0 || p >= 1.0 {
            return 0.0;
        }

        // Coefficients for central region
        const A: [f64; 6] = [
            -3.969_683_028_665_376e1,
            2.209_460_984_245_205e2,
            -2.759_285_104_469_687e2,
            1.383_577_518_672_69e2,
            -3.066_479_806_614_716e1,
            2.506_628_277_459_239e0,
        ];
        const B: [f64; 5] = [
            -5.447_609_879_822_406e1,
            1.615_858_368_580_409e2,
            -1.556_989_798_598_866e2,
            6.680_131_188_771_972e1,
            -1.328_068_155_288_572e1,
        ];

        // Coefficients for tail regions
        const C: [f64; 6] = [
            -7.784_894_002_430_293e-3,
            -3.223_964_580_411_365e-1,
            -2.400_758_277_161_838e0,
            -2.549_732_539_343_734e0,
            4.374_664_141_464_968e0,
            2.938_163_982_698_783e0,
        ];
        const D: [f64; 4] = [
            7.784_695_709_041_462e-3,
            3.224_671_290_700_398e-1,
            2.445_134_137_142_996e0,
            3.754_408_661_907_416e0,
        ];

        const P_LOW: f64 = 0.02425;
        const P_HIGH: f64 = 0.97575;

        if p < P_LOW {
            // Lower tail
            let q = (-2.0 * p.ln()).sqrt();
            let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
            let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
            num / den
        } else if p > P_HIGH {
            // Upper tail
            let q = (-2.0 * (1.0 - p).ln()).sqrt();
            let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
            let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
            -(num / den)
        } else {
            // Central region
            let q = p - 0.5;
            let r = q * q;
            let num = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q;
            let den = ((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0;
            num / den
        }
    }
}

pub const DEFAULT_BOOTSTRAP_SEED: u64 = 0x5EED_B007;
pub const MIN_BOOTSTRAP_SAMPLES: usize = 2;
pub const BOOTSTRAP_BATCH_SIZE: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BootstrapConfig {
    pub n_boot: usize,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BootstrapOutput<T> {
    pub std_errors: Vec<T>,
    pub confidence_lower: Option<Vec<T>>,
    pub confidence_upper: Option<Vec<T>>,
    pub prediction_lower: Option<Vec<T>>,
    pub prediction_upper: Option<Vec<T>>,
}

impl BootstrapConfig {
    pub fn compute<T, F>(
        &self,
        method: &IntervalMethod<T>,
        y_smooth: &[T],
        residuals: &[T],
        refit: F,
    ) -> Result<BootstrapOutput<T>, LoessError>
    where
        T: Float,
        F: FnMut(&[Vec<T>]) -> Result<Vec<Vec<T>>, LoessError>,
    {
        self.compute_at(method, y_smooth, residuals, y_smooth.len(), refit)
    }

    pub fn compute_at<T, F>(
        &self,
        method: &IntervalMethod<T>,
        y_smooth: &[T],
        residuals: &[T],
        n_output: usize,
        refit: F,
    ) -> Result<BootstrapOutput<T>, LoessError>
    where
        T: Float,
        F: FnMut(&[Vec<T>]) -> Result<Vec<Vec<T>>, LoessError>,
    {
        self.compute_at_levels(method, method.level, y_smooth, residuals, n_output, refit)
    }

    pub(crate) fn compute_at_levels<T, F>(
        &self,
        method: &IntervalMethod<T>,
        prediction_level: T,
        y_smooth: &[T],
        residuals: &[T],
        n_output: usize,
        mut refit: F,
    ) -> Result<BootstrapOutput<T>, LoessError>
    where
        T: Float,
        F: FnMut(&[Vec<T>]) -> Result<Vec<Vec<T>>, LoessError>,
    {
        if self.n_boot < MIN_BOOTSTRAP_SAMPLES {
            return Err(LoessError::InvalidBootstrapSamples(self.n_boot));
        }
        if y_smooth.is_empty() {
            return Err(LoessError::EmptyInput);
        }
        if y_smooth.len() != residuals.len() {
            return Err(LoessError::InvalidInput(
                "Bootstrap residual length mismatch".into(),
            ));
        }
        if y_smooth
            .iter()
            .chain(residuals)
            .any(|value| !value.is_finite())
        {
            return Err(LoessError::InvalidNumericValue(
                "Bootstrap inputs must be finite".into(),
            ));
        }
        if (method.confidence || method.prediction)
            && (!method.level.is_finite() || method.level <= T::zero() || method.level >= T::one())
        {
            return Err(LoessError::InvalidIntervals(
                method.level.to_f64().unwrap_or(f64::NAN),
            ));
        }
        let capacity = n_output
            .checked_mul(self.n_boot)
            .ok_or_else(|| LoessError::InvalidInput("Bootstrap output size overflow".into()))?;
        let mut fits = vec![T::zero(); capacity];
        let mean_residual = residuals
            .iter()
            .copied()
            .fold(T::zero(), |sum, value| sum + value)
            / T::from(residuals.len()).unwrap();
        let centered: Vec<T> = residuals
            .iter()
            .map(|&value| value - mean_residual)
            .collect();
        let mut state = self.seed.unwrap_or(DEFAULT_BOOTSTRAP_SEED);
        let mut draw = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((state >> 32) * centered.len() as u64) >> 32) as usize
        };
        let mut done = 0;
        while done < self.n_boot {
            let count = BOOTSTRAP_BATCH_SIZE.min(self.n_boot - done);
            let batch: Vec<Vec<T>> = (0..count)
                .map(|_| {
                    y_smooth
                        .iter()
                        .map(|&value| value + centered[draw()])
                        .collect()
                })
                .collect();
            let results = refit(&batch)?;
            if results.len() != count
                || results
                    .iter()
                    .any(|fit| fit.len() != n_output || fit.iter().any(|value| !value.is_finite()))
            {
                return Err(LoessError::RuntimeError(
                    "Bootstrap refit returned invalid results".into(),
                ));
            }
            for (replicate, fit) in results.iter().enumerate() {
                for (point, &value) in fit.iter().enumerate() {
                    fits[point * self.n_boot + done + replicate] = value;
                }
            }
            done += count;
        }
        let tail = (T::one() - method.level) / T::from(2).unwrap();
        let prediction_tail = (T::one() - prediction_level) / T::from(2).unwrap();
        let mut output = BootstrapOutput {
            std_errors: Vec::with_capacity(n_output),
            confidence_lower: method.confidence.then(Vec::new),
            confidence_upper: method.confidence.then(Vec::new),
            prediction_lower: method.prediction.then(Vec::new),
            prediction_upper: method.prediction.then(Vec::new),
        };
        for point in 0..n_output {
            let samples = &mut fits[point * self.n_boot..(point + 1) * self.n_boot];
            let mean = samples
                .iter()
                .copied()
                .fold(T::zero(), |sum, value| sum + value)
                / T::from(self.n_boot).unwrap();
            let variance = samples.iter().fold(T::zero(), |sum, &value| {
                sum + (value - mean) * (value - mean)
            }) / T::from(self.n_boot - 1).unwrap();
            output.std_errors.push(variance.sqrt());
            if method.prediction {
                let mut predictions: Vec<T> = samples
                    .iter()
                    .map(|&value| value + centered[draw()])
                    .collect();
                predictions.sort_unstable_by(|left, right| {
                    left.partial_cmp(right)
                        .unwrap_or(core::cmp::Ordering::Equal)
                });
                output
                    .prediction_lower
                    .as_mut()
                    .unwrap()
                    .push(bootstrap_quantile(&predictions, prediction_tail));
                output
                    .prediction_upper
                    .as_mut()
                    .unwrap()
                    .push(bootstrap_quantile(&predictions, T::one() - prediction_tail));
            }
            if method.confidence {
                samples.sort_unstable_by(|left, right| {
                    left.partial_cmp(right)
                        .unwrap_or(core::cmp::Ordering::Equal)
                });
                output
                    .confidence_lower
                    .as_mut()
                    .unwrap()
                    .push(bootstrap_quantile(samples, tail));
                output
                    .confidence_upper
                    .as_mut()
                    .unwrap()
                    .push(bootstrap_quantile(samples, T::one() - tail));
            }
        }
        Ok(output)
    }
}

fn bootstrap_quantile<T: Float>(samples: &[T], probability: T) -> T {
    let position = probability * T::from(samples.len() - 1).unwrap();
    let lower = position.floor().to_usize().unwrap();
    let upper = (lower + 1).min(samples.len() - 1);
    let fraction = position - T::from(lower).unwrap();
    samples[lower] + fraction * (samples[upper] - samples[lower])
}
