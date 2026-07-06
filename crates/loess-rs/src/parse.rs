// String input compatibility for builder methods.
//
// Defines [`IntoEnum`], a sealed conversion trait that allows builder methods
// to accept either a typed enum value or a string literal/`String`, mirroring
// the case-insensitive string parsing used by all language bindings.
//
// Invalid strings do not panic; errors accumulate in the builder and are
// returned together as [`LoessError::ParseErrors`] by `build()`.

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(feature = "std")]
use std::string::String;

// External dependencies
use core::str::FromStr;

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

// Converts a value into a typed enum, either infallibly (enum variant) or
// via case-insensitive string parsing (string literal / `String`).
pub(crate) trait IntoEnum<E> {
    fn into_enum(self) -> Result<E, LoessError>;
}

// Generate IntoEnum impls for a concrete (non-generic) enum type.
macro_rules! impl_into_enum_for {
    ($ty:ty) => {
        impl IntoEnum<$ty> for $ty {
            #[inline]
            fn into_enum(self) -> Result<$ty, LoessError> {
                Ok(self)
            }
        }

        impl IntoEnum<$ty> for &str {
            #[inline]
            fn into_enum(self) -> Result<$ty, LoessError> {
                self.parse()
            }
        }

        impl IntoEnum<$ty> for String {
            #[inline]
            fn into_enum(self) -> Result<$ty, LoessError> {
                self.as_str().parse()
            }
        }
    };
}

impl_into_enum_for!(BoundaryPolicy);
impl_into_enum_for!(MergeStrategy);
impl_into_enum_for!(PolynomialDegree);
impl_into_enum_for!(RobustnessMethod);
impl_into_enum_for!(ScalingMethod);
impl_into_enum_for!(SurfaceMode);
impl_into_enum_for!(UpdateMode);
impl_into_enum_for!(WeightFunction);
impl_into_enum_for!(ZeroWeightFallback);

// IntoEnum for DistanceMetric<T>.
//
// Accepting a string requires `T: Float + FromStr` to parse the Minkowski `p`
// parameter from `"minkowski:3.0"` syntax. Passing an enum variant directly
// works unconditionally.
impl<T> IntoEnum<DistanceMetric<T>> for DistanceMetric<T> {
    #[inline]
    fn into_enum(self) -> Result<DistanceMetric<T>, LoessError> {
        Ok(self)
    }
}

impl<T> IntoEnum<DistanceMetric<T>> for &str
where
    T: num_traits::Float + FromStr,
{
    #[inline]
    fn into_enum(self) -> Result<DistanceMetric<T>, LoessError> {
        self.parse()
    }
}

impl<T> IntoEnum<DistanceMetric<T>> for String
where
    T: num_traits::Float + FromStr,
{
    #[inline]
    fn into_enum(self) -> Result<DistanceMetric<T>, LoessError> {
        self.as_str().parse()
    }
}
