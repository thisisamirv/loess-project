//! Robustness weight computation for outlier downweighting.
//!
//! This module implements iterative reweighted least squares (IRLS) for robust
//! LOESS smoothing. After an initial fit, residuals are computed and used to
//! downweight outliers in subsequent iterations.
//!
//! ## srrstats Compliance
//!
//! @srrstats {RE4.0} Iterative reweighted least squares (IRLS) for robust regression.
//! @srrstats {RE4.1} Multiple robustness methods: Bisquare (Tukey), Huber, Talwar.
//! @srrstats {G2.4} Tuning constants documented (Bisquare=6.0, Huber=1.345, Talwar=2.5).
//! Robust scale estimation via MAD with fallback to MAR.

// External dependencies
use core::str::FromStr;
use num_traits::Float;

// Internal dependencies
use crate::math::scaling::ScalingMethod;
use crate::primitives::errors::LoessError;

// Robustness weighting method for outlier downweighting.
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

impl FromStr for RobustnessMethod {
    type Err = LoessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "bisquare" | "biweight" => Ok(RobustnessMethod::Bisquare),
            "huber" => Ok(RobustnessMethod::Huber),
            "talwar" => Ok(RobustnessMethod::Talwar),
            _ => Err(LoessError::InvalidOption {
                option: "robustness_method",
                value: s.to_string(),
                valid: "bisquare, huber, talwar",
            }),
        }
    }
}

impl RobustnessMethod {
    // Default tuning constant for bisquare robustness weights.
    //
    // Value of 6.0 follows Cleveland (1979) and is applied to the raw MAD.
    const DEFAULT_BISQUARE_C: f64 = 6.0;

    // Default tuning constant for Huber weights.
    //
    // Value of 1.345 is the standard threshold for 95% efficiency.
    // Note: This is applied directly to the MAD-scaled residuals.
    const DEFAULT_HUBER_C: f64 = 1.345;

    // Default tuning constant for Talwar weights.
    //
    // Value of 2.5 provides aggressive outlier rejection.
    const DEFAULT_TALWAR_C: f64 = 2.5;

    // Minimum scale threshold relative to mean absolute residual.
    //
    // If MAD < SCALE_THRESHOLD × MAR, use MAR instead of MAD.
    const SCALE_THRESHOLD: f64 = 1e-7;

    // Minimum tuned-scale absolute epsilon to avoid division by zero.
    const MIN_TUNED_SCALE: f64 = 1e-12;

    // Apply robustness weights using the configured method.
    pub fn apply_robustness_weights<T: Float>(
        &self,
        residuals: &[T],
        weights: &mut [T],
        scaling_method: ScalingMethod,
        scratch: &mut [T],
    ) {
        if residuals.is_empty() {
            return;
        }

        let base_scale = self.compute_scale(residuals, scaling_method, scratch);

        let (method_type, tuning_constant) = match self {
            Self::Bisquare => (0, Self::DEFAULT_BISQUARE_C),
            Self::Huber => (1, Self::DEFAULT_HUBER_C),
            Self::Talwar => (2, Self::DEFAULT_TALWAR_C),
        };

        let c_t = T::from(tuning_constant).unwrap();

        for (i, &r) in residuals.iter().enumerate() {
            weights[i] = match method_type {
                0 => Self::bisquare_weight(r, base_scale, c_t),
                1 => Self::huber_weight(r, base_scale, c_t),
                _ => Self::talwar_weight(r, base_scale, c_t),
            };
        }
    }

    // Compute robust scale estimate with zero-scale safety fallback.
    //
    // If the robust (Median-based) scale is zero or extremely small, this method
    // falls back to the Mean Absolute Error (MAE) to ensure numerical stability.
    fn compute_scale<T: Float>(
        &self,
        residuals: &[T],
        scaling_method: ScalingMethod,
        scratch: &mut [T],
    ) -> T {
        // Step 1: Compute Mean Absolute Error (MAE).
        // This is O(N) and defines our scale threshold.
        let n = residuals.len();
        if n == 0 {
            return T::zero();
        }

        let mut sum_abs = T::zero();
        for &r in residuals {
            sum_abs = sum_abs + r.abs();
        }
        let mae = sum_abs / T::from(n).unwrap();

        // Safety: If Mean is 0, Median is also 0. Exit early.
        if mae.is_zero() {
            return T::zero();
        }

        // Step 2: Establish the safety threshold.
        // We use either a relative threshold (portion of MAE) or an absolute floor.
        let relative_threshold = T::from(Self::SCALE_THRESHOLD).unwrap() * mae;
        let absolute_threshold = T::from(Self::MIN_TUNED_SCALE).unwrap();
        let scale_threshold = relative_threshold.max(absolute_threshold);

        // Step 3: Compute robust scale using selected method (Median-based).
        // This is usually the more expensive operation (O(N) or O(N log N)).
        scratch.copy_from_slice(residuals);
        let scale_val = scaling_method.compute(scratch);

        // Step 4: Final decision.
        // If the robust Median-based scale is too small, fallback to MAE.
        if scale_val <= scale_threshold {
            // Use MAE as fallback (it's less robust but more stable near zero)
            mae.max(scale_val)
        } else {
            // Robust scale is healthy
            scale_val
        }
    }

    // Compute bisquare weight.
    //
    // # Formula
    //
    // u = |r| / (c * s)
    //
    // w(u) = (1 - u^2)^2  if 0.001 < u < 0.999
    //
    // w(u) = 1            if u <= 0.001
    //
    // w(u) = 0            if u >= 0.999
    #[inline]
    pub(crate) fn bisquare_weight<T: Float>(residual: T, scale: T, c: T) -> T {
        if scale <= T::zero() {
            return T::one();
        }

        let min_eps = T::from(Self::MIN_TUNED_SCALE).unwrap();
        let c_clamped = c.max(min_eps);
        let tuned_scale = (scale * c_clamped).max(min_eps);

        let u = (residual / tuned_scale).abs();

        // Thresholds (0.001 and 0.999)
        let low_threshold = T::from(0.001).unwrap();
        let high_threshold = T::from(0.999).unwrap();

        if u >= high_threshold {
            T::zero()
        } else if u <= low_threshold {
            T::one()
        } else {
            let tmp = T::one() - u * u;
            tmp * tmp
        }
    }

    // Compute Huber weight.
    //
    // # Formula
    //
    // u = |r| / s
    //
    // w(u) = 1      if u <= c
    //
    // w(u) = c / u  if u > c
    #[inline]
    pub(crate) fn huber_weight<T: Float>(residual: T, scale: T, c: T) -> T {
        if scale <= T::zero() {
            return T::one();
        }

        let u = (residual / scale).abs();
        if u <= c { T::one() } else { c / u }
    }

    // Compute Talwar weight.
    //
    // # Formula
    //
    // u = |r| / s
    //
    // w(u) = 1  if u <= c
    //
    // w(u) = 0  if u > c
    #[inline]
    pub(crate) fn talwar_weight<T: Float>(residual: T, scale: T, c: T) -> T {
        if scale <= T::zero() {
            return T::one();
        }

        let u = (residual / scale).abs();
        if u <= c { T::one() } else { T::zero() }
    }
}
