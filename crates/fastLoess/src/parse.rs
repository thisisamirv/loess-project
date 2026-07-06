// String input compatibility for fastLoess parallel builder methods.
//
// Mirrors the `parse` module in loess-rs. Defines [`IntoEnum`] so that
// parallel builder methods can accept either typed enum values or strings,
// exactly as the underlying LoessBuilder from loess-rs does.

// External dependencies
use core::str::FromStr;
use std::string::String;

// Export dependencies from loess-rs crate
use loess_rs::internals::adapters::online::UpdateMode;
use loess_rs::internals::adapters::streaming::MergeStrategy;
use loess_rs::internals::algorithms::regression::{PolynomialDegree, ZeroWeightFallback};
use loess_rs::internals::algorithms::robustness::RobustnessMethod;
use loess_rs::internals::engine::executor::SurfaceMode;
use loess_rs::internals::math::boundary::BoundaryPolicy;
use loess_rs::internals::math::distance::DistanceMetric;
use loess_rs::internals::math::kernel::WeightFunction;
use loess_rs::internals::math::scaling::ScalingMethod;
use loess_rs::internals::primitives::errors::LoessError;

// Converts a value into a typed enum, either infallibly (enum variant) or
// via case-insensitive string parsing (string literal / `String`).
//
// This is a sealed trait: impls for enum types, `&str`, and `String` are
// pre-generated via the macro below. External code cannot name this trait.
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
// String form requires `T: Float + FromStr` to parse Minkowski p.
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
