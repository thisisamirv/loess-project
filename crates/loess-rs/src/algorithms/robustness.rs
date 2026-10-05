//! Robustness weight computation for outlier downweighting.
//!
//! This module implements iterative reweighted least squares (IRLS) for robust
//! LOESS smoothing. After an initial fit, residuals are computed and used to
//! downweight outliers in subsequent iterations.
// ## srrstats Compliance
//
// @srrstats {RE4.0} Iterative reweighted least squares (IRLS) for robust regression.
// @srrstats {RE4.1} Multiple robustness methods: Bisquare (Tukey), Huber, Talwar.
// @srrstats {G2.4} Tuning constants documented (Bisquare=6.0, Huber=1.345, Talwar=2.5).
// Robust scale estimation via MAD with fallback to MAR.

// External dependencies
use num_traits::Float;

// Internal dependencies
use crate::primitives::policies::ScalingMethod;

// Robustness weighting method for outlier downweighting.
use crate::primitives::policies::RobustnessMethod;

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

    // Minimum tuned-scale absolute epsilon to avoid division by zero.
    const MIN_TUNED_SCALE: f64 = 1e-12;

    // Apply robustness weights using the configured method. Returns true when
    // the MAR scale is effectively zero and the caller should stop iterating.
    pub fn apply_robustness_weights<T: Float>(
        &self,
        residuals: &[T],
        weights: &mut [T],
        scaling_method: ScalingMethod,
        scratch: &mut [T],
    ) -> bool {
        if residuals.is_empty() || residuals.iter().any(|residual| !residual.is_finite()) {
            return false;
        }

        let (method_type, tuning_constant) = match self {
            Self::Bisquare => (0, Self::DEFAULT_BISQUARE_C),
            Self::Huber => (1, Self::DEFAULT_HUBER_C),
            Self::Talwar => (2, Self::DEFAULT_TALWAR_C),
        };

        let c_t = T::from(tuning_constant).unwrap_or(T::one());
        let mut bisquare_scale_factor = c_t;
        let (base_scale, tuned_scale) = if matches!(self, Self::Bisquare)
            && matches!(scaling_method, ScalingMethod::MAR)
        {
            for (value, residual) in scratch.iter_mut().zip(residuals) {
                *value = residual.abs();
            }
            let middle_index = residuals.len() / 2;
            let (lower, middle, _) = scratch.select_nth_unstable_by(middle_index, |left, right| {
                left.partial_cmp(right)
                    .unwrap_or(core::cmp::Ordering::Equal)
            });
            let middle_value = *middle;
            if residuals.len().is_multiple_of(2) {
                let lower_value = lower
                    .iter()
                    .fold(T::zero(), |largest, &value| largest.max(value));
                let base = lower_value + (middle_value - lower_value) / T::from(2.0).unwrap();
                if base > T::zero() {
                    bisquare_scale_factor = T::from(3.0).unwrap_or(T::one())
                        * (lower_value / base + middle_value / base);
                }
                (
                    base,
                    T::from(3.0).unwrap_or(T::one()) * (lower_value + middle_value),
                )
            } else {
                (middle_value, c_t * middle_value)
            }
        } else {
            let base = self.compute_scale(residuals, scaling_method, scratch);
            (base, base * c_t)
        };

        if matches!(scaling_method, ScalingMethod::MAR) && tuned_scale < T::min_positive_value() {
            return true;
        }

        for (i, &r) in residuals.iter().enumerate() {
            weights[i] = match method_type {
                0 if tuned_scale.is_finite() => Self::bisquare_weight(r, tuned_scale),
                0 => Self::bisquare_weight_from_factor(r, base_scale, bisquare_scale_factor),
                1 => Self::huber_weight(r, base_scale, c_t),
                _ => Self::talwar_weight(r, base_scale, c_t),
            };
        }
        false
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

        let mean_abs_scale = residuals
            .iter()
            .fold(T::zero(), |scale, residual| scale.max(residual.abs()));
        if mean_abs_scale == T::zero() {
            return T::zero();
        }
        let scaled_sum = residuals.iter().fold(T::zero(), |sum, residual| {
            sum + residual.abs() / mean_abs_scale
        });
        let mean_abs = mean_abs_scale * (scaled_sum / T::from(n).unwrap());

        // Compute robust scale using the selected method (median-based).
        // This is usually the more expensive operation (O(N) or O(N log N)).
        scratch.copy_from_slice(residuals);
        let scale_val = scaling_method.compute(scratch);

        // Centered MAD can collapse to zero on tied residuals. Keep its fallback
        // separate from MAR, whose near-zero scale is an iteration stop signal.
        if matches!(scaling_method, ScalingMethod::MAD)
            && scale_val <= T::from(Self::MIN_TUNED_SCALE).unwrap_or_else(T::epsilon)
        {
            mean_abs.max(scale_val)
        } else {
            scale_val
        }
    }

    // Compute bisquare weight.
    //
    // # Formula
    //
    // u = |r| / tuned_scale, where tuned_scale = c * s
    //
    // w(u) = (1 - u^2)^2  if 0.001 < u <= 0.999
    //
    // w(u) = 1            if u <= 0.001
    //
    // w(u) = 0            if u >= 0.999
    #[inline]
    pub(crate) fn bisquare_weight<T: Float>(residual: T, tuned_scale: T) -> T {
        if tuned_scale <= T::zero() {
            return T::one();
        }
        let abs_residual = residual.abs();
        let low_threshold = T::from(0.001).unwrap();
        let high_threshold = T::from(0.999).unwrap();

        if abs_residual > tuned_scale * high_threshold {
            T::zero()
        } else if abs_residual <= tuned_scale * low_threshold {
            T::one()
        } else {
            let u = abs_residual / tuned_scale;
            let tmp = T::one() - u * u;
            tmp * tmp
        }
    }

    // Evaluate bisquare weights without constructing an overflowing tuned scale.
    #[inline]
    fn bisquare_weight_from_factor<T: Float>(residual: T, base_scale: T, factor: T) -> T {
        if base_scale <= T::zero() || factor <= T::zero() {
            return T::one();
        }

        let normalized = (residual.abs() / factor) / base_scale;
        let low_threshold = T::from(0.001).unwrap();
        let high_threshold = T::from(0.999).unwrap();
        if normalized <= low_threshold {
            T::one()
        } else if normalized <= high_threshold {
            let tmp = T::one() - normalized * normalized;
            tmp * tmp
        } else {
            T::zero()
        }
    }

    // Compute Huber weight.
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
