//! Centralised string-alias maps for all option enums.
//!
//! All `impl FromStr` blocks for option types live here so that accepted
//! aliases and their canonical spellings stay in one place.  Every binding
//! frontend (`fastLoess::binding_support`) delegates to these impls instead
//! of maintaining its own duplicated match arms.
//!
//! ## Canonical names
//!
//! The first alias listed in each match arm is the canonical (round-trip)
//! name.  The `*_str` helpers in `fastLoess::binding_support` always return
//! the canonical name so that `parse → str → parse` round-trips correctly.

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::string::ToString;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// External dependencies
use core::str::FromStr;
use num_traits::Float;

// Internal dependencies
use crate::adapters::online::UpdateMode;
use crate::adapters::streaming::MergeStrategy;
use crate::algorithms::regression::{PolynomialDegree, ZeroWeightFallback};
use crate::algorithms::robustness::RobustnessMethod;
use crate::engine::executor::SurfaceMode;
use crate::math::boundary::BoundaryPolicy;
use crate::math::distance::DistanceMetric;
use crate::math::kernel::WeightFunction;
use crate::math::scaling::ScalingMethod;
use crate::primitives::errors::LoessError;

// ── WeightFunction ────────────────────────────────────────────────────────────

impl FromStr for WeightFunction {
    type Err = LoessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cosine" => Ok(WeightFunction::Cosine),
            "epanechnikov" => Ok(WeightFunction::Epanechnikov),
            "gaussian" => Ok(WeightFunction::Gaussian),
            "biweight" | "bisquare" => Ok(WeightFunction::Biweight),
            "triangle" | "triangular" => Ok(WeightFunction::Triangle),
            "tricube" => Ok(WeightFunction::Tricube),
            "uniform" | "boxcar" => Ok(WeightFunction::Uniform),
            _ => Err(LoessError::InvalidOption {
                option: "weight_function",
                value: s.to_string(),
                valid: "tricube, epanechnikov, gaussian, uniform, biweight, triangle, cosine",
            }),
        }
    }
}

// ── BoundaryPolicy ────────────────────────────────────────────────────────────

impl FromStr for BoundaryPolicy {
    type Err = LoessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "extend" | "pad" => Ok(BoundaryPolicy::Extend),
            "reflect" | "mirror" => Ok(BoundaryPolicy::Reflect),
            "zero" => Ok(BoundaryPolicy::Zero),
            "noboundary" | "none" => Ok(BoundaryPolicy::NoBoundary),
            _ => Err(LoessError::InvalidOption {
                option: "boundary_policy",
                value: s.to_string(),
                valid: "extend, reflect, zero, noboundary",
            }),
        }
    }
}

// ── DistanceMetric<T> ─────────────────────────────────────────────────────────

impl<T> FromStr for DistanceMetric<T>
where
    T: Float + FromStr,
{
    type Err = LoessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_lowercase();
        if let Some(p_str) = lower.strip_prefix("minkowski:") {
            let p: T = p_str.parse().map_err(|_| LoessError::InvalidOption {
                option: "distance_metric",
                value: s.to_string(),
                valid: "normalized, euclidean, manhattan, chebyshev, minkowski, minkowski:<p>, weighted",
            })?;
            return Ok(DistanceMetric::Minkowski(p));
        }
        match lower.as_str() {
            "normalized" | "norm" => Ok(DistanceMetric::Normalized),
            "euclidean" | "euclid" => Ok(DistanceMetric::Euclidean),
            "manhattan" | "l1" => Ok(DistanceMetric::Manhattan),
            "chebyshev" | "linf" => Ok(DistanceMetric::Chebyshev),
            "minkowski" => Ok(DistanceMetric::Minkowski(T::from(2.0).unwrap())),
            "weighted" | "weighted_euclidean" => Ok(DistanceMetric::Weighted(Vec::new())),
            _ => Err(LoessError::InvalidOption {
                option: "distance_metric",
                value: s.to_string(),
                valid: "normalized, euclidean, manhattan, chebyshev, minkowski, minkowski:<p>, weighted",
            }),
        }
    }
}

// ── ScalingMethod ─────────────────────────────────────────────────────────────

impl FromStr for ScalingMethod {
    type Err = LoessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "mar" | "median_absolute_residual" => Ok(ScalingMethod::MAR),
            "mad" | "median_absolute_deviation" => Ok(ScalingMethod::MAD),
            "mean" | "mean_absolute_residual" => Ok(ScalingMethod::Mean),
            _ => Err(LoessError::InvalidOption {
                option: "scaling_method",
                value: s.to_string(),
                valid: "mad, mar, mean",
            }),
        }
    }
}

// ── RobustnessMethod ──────────────────────────────────────────────────────────

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

// ── SurfaceMode ───────────────────────────────────────────────────────────────

impl FromStr for SurfaceMode {
    type Err = LoessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "interpolation" | "interp" | "interpolate" => Ok(SurfaceMode::Interpolation),
            "direct" => Ok(SurfaceMode::Direct),
            _ => Err(LoessError::InvalidOption {
                option: "surface_mode",
                value: s.to_string(),
                valid: "interpolation, direct",
            }),
        }
    }
}

// ── PolynomialDegree ──────────────────────────────────────────────────────────

impl FromStr for PolynomialDegree {
    type Err = LoessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "constant" | "0" => Ok(PolynomialDegree::Constant),
            "linear" | "1" => Ok(PolynomialDegree::Linear),
            "quadratic" | "2" => Ok(PolynomialDegree::Quadratic),
            "cubic" | "3" => Ok(PolynomialDegree::Cubic),
            "quartic" | "4" => Ok(PolynomialDegree::Quartic),
            _ => Err(LoessError::InvalidOption {
                option: "degree",
                value: s.to_string(),
                valid: "constant (0), linear (1), quadratic (2), cubic (3), quartic (4)",
            }),
        }
    }
}

// ── ZeroWeightFallback ────────────────────────────────────────────────────────

impl FromStr for ZeroWeightFallback {
    type Err = LoessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "use_local_mean" | "local_mean" | "mean" => Ok(ZeroWeightFallback::UseLocalMean),
            "return_original" | "original" => Ok(ZeroWeightFallback::ReturnOriginal),
            "return_none" | "none" | "nan" => Ok(ZeroWeightFallback::ReturnNone),
            _ => Err(LoessError::InvalidOption {
                option: "zero_weight_fallback",
                value: s.to_string(),
                valid: "use_local_mean, return_original, return_none",
            }),
        }
    }
}

// ── MergeStrategy ─────────────────────────────────────────────────────────────

impl FromStr for MergeStrategy {
    type Err = LoessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "average" | "mean" => Ok(MergeStrategy::Average),
            "weighted_average" | "weighted" | "weightedaverage" => {
                Ok(MergeStrategy::WeightedAverage)
            }
            "take_first" | "first" | "takefirst" | "left" => Ok(MergeStrategy::TakeFirst),
            "take_last" | "last" | "takelast" | "right" => Ok(MergeStrategy::TakeLast),
            _ => Err(LoessError::InvalidOption {
                option: "merge_strategy",
                value: s.to_string(),
                valid: "average, weighted_average, take_first, take_last",
            }),
        }
    }
}

// ── UpdateMode ────────────────────────────────────────────────────────────────

impl FromStr for UpdateMode {
    type Err = LoessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "full" | "resmooth" => Ok(UpdateMode::Full),
            "incremental" | "single" => Ok(UpdateMode::Incremental),
            _ => Err(LoessError::InvalidOption {
                option: "update_mode",
                value: s.to_string(),
                valid: "full, incremental",
            }),
        }
    }
}

// ─── Binding helpers (only with the `dev` feature) ───────────────────────────
//
// Parse and canonical-name wrappers used by the binding layer.  Only compiled
// when the `dev` feature is active; re-exported through
// `loess_rs::internals::alias`.

#[cfg(feature = "dev")]
pub mod helpers {
    use super::{
        BoundaryPolicy, DistanceMetric, LoessError, MergeStrategy, PolynomialDegree,
        RobustnessMethod, ScalingMethod, SurfaceMode, UpdateMode, WeightFunction,
        ZeroWeightFallback,
    };

    // ─── Parse helpers ────────────────────────────────────────────────────────

    pub fn parse_weight_function(s: &str) -> Result<WeightFunction, LoessError> {
        s.parse()
    }

    pub fn parse_robustness_method(s: &str) -> Result<RobustnessMethod, LoessError> {
        s.parse()
    }

    pub fn parse_zero_weight_fallback(s: &str) -> Result<ZeroWeightFallback, LoessError> {
        s.parse()
    }

    pub fn parse_boundary_policy(s: &str) -> Result<BoundaryPolicy, LoessError> {
        s.parse()
    }

    pub fn parse_scaling_method(s: &str) -> Result<ScalingMethod, LoessError> {
        s.parse()
    }

    pub fn parse_polynomial_degree(s: &str) -> Result<PolynomialDegree, LoessError> {
        s.parse()
    }

    pub fn parse_distance_metric(s: &str) -> Result<DistanceMetric<f64>, LoessError> {
        s.parse()
    }

    pub fn parse_surface_mode(s: &str) -> Result<SurfaceMode, LoessError> {
        s.parse()
    }

    pub fn parse_update_mode(s: &str) -> Result<UpdateMode, LoessError> {
        s.parse()
    }

    pub fn parse_merge_strategy(s: &str) -> Result<MergeStrategy, LoessError> {
        s.parse()
    }

    // ─── Canonical-name helpers ───────────────────────────────────────────────
    //
    // Round-trip guarantee: `X_str(v).parse::<X>().unwrap() == v` for all `v`.
    // `DistanceMetric` is excluded because `Minkowski(p)` requires a formatted
    // string; use `distance_metric_components` in the binding layer instead.

    pub fn weight_function_str(v: WeightFunction) -> &'static str {
        match v {
            WeightFunction::Tricube => "tricube",
            WeightFunction::Epanechnikov => "epanechnikov",
            WeightFunction::Gaussian => "gaussian",
            WeightFunction::Uniform => "uniform",
            WeightFunction::Biweight => "biweight",
            WeightFunction::Triangle => "triangle",
            WeightFunction::Cosine => "cosine",
        }
    }

    pub fn robustness_method_str(v: RobustnessMethod) -> &'static str {
        match v {
            RobustnessMethod::Bisquare => "bisquare",
            RobustnessMethod::Huber => "huber",
            RobustnessMethod::Talwar => "talwar",
        }
    }

    pub fn scaling_method_str(v: ScalingMethod) -> &'static str {
        match v {
            ScalingMethod::MAD => "mad",
            ScalingMethod::MAR => "mar",
            ScalingMethod::Mean => "mean",
        }
    }

    pub fn zero_weight_fallback_str(v: ZeroWeightFallback) -> &'static str {
        match v {
            ZeroWeightFallback::UseLocalMean => "use_local_mean",
            ZeroWeightFallback::ReturnOriginal => "return_original",
            ZeroWeightFallback::ReturnNone => "return_none",
        }
    }

    pub fn boundary_policy_str(v: BoundaryPolicy) -> &'static str {
        match v {
            BoundaryPolicy::Extend => "extend",
            BoundaryPolicy::Reflect => "reflect",
            BoundaryPolicy::Zero => "zero",
            BoundaryPolicy::NoBoundary => "noboundary",
        }
    }

    pub fn polynomial_degree_str(v: PolynomialDegree) -> &'static str {
        match v {
            PolynomialDegree::Constant => "constant",
            PolynomialDegree::Linear => "linear",
            PolynomialDegree::Quadratic => "quadratic",
            PolynomialDegree::Cubic => "cubic",
            PolynomialDegree::Quartic => "quartic",
        }
    }

    pub fn surface_mode_str(v: SurfaceMode) -> &'static str {
        match v {
            SurfaceMode::Interpolation => "interpolation",
            SurfaceMode::Direct => "direct",
        }
    }

    pub fn update_mode_str(v: UpdateMode) -> &'static str {
        match v {
            UpdateMode::Full => "full",
            UpdateMode::Incremental => "incremental",
        }
    }

    pub fn merge_strategy_str(v: MergeStrategy) -> &'static str {
        match v {
            MergeStrategy::Average => "average",
            MergeStrategy::WeightedAverage => "weighted_average",
            MergeStrategy::TakeFirst => "take_first",
            MergeStrategy::TakeLast => "take_last",
        }
    }
}
