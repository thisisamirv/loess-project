// Default values for math module types (kernel, scaling, boundary, distance).

use crate::math::boundary::BoundaryPolicy;
use crate::math::distance::DistanceMetric;
use crate::math::kernel::WeightFunction;
use crate::math::scaling::ScalingMethod;

// Default kernel weight function.
pub const DEFAULT_WEIGHT_FUNCTION_ENUM: WeightFunction = WeightFunction::Tricube;
#[cfg(feature = "dev")]
pub const DEFAULT_WEIGHT_FUNCTION: &str = "tricube";

// Default robust scale estimator.
pub const DEFAULT_SCALING_METHOD_ENUM: ScalingMethod = ScalingMethod::MAD;
#[cfg(feature = "dev")]
pub const DEFAULT_SCALING_METHOD: &str = "mad";

// Default boundary padding policy.
pub const DEFAULT_BOUNDARY_POLICY_ENUM: BoundaryPolicy = BoundaryPolicy::Extend;
#[cfg(feature = "dev")]
pub const DEFAULT_BOUNDARY_POLICY: &str = "extend";

// Default distance metric: normalized Euclidean.
pub const fn default_distance_metric<T>() -> DistanceMetric<T> {
    DistanceMetric::Normalized
}
#[cfg(feature = "dev")]
pub const DEFAULT_DISTANCE_METRIC: &str = "normalized";
