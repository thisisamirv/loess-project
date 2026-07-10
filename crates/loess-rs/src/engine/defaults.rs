// Default values for engine configuration (surface evaluation, geometry).

use crate::engine::executor::SurfaceMode;

// Default number of predictor dimensions.
pub const DEFAULT_DIMENSIONS: usize = 1;

// Default surface evaluation mode.
pub const DEFAULT_SURFACE_MODE_ENUM: SurfaceMode = SurfaceMode::Interpolation;
#[cfg(feature = "dev")]
pub const DEFAULT_SURFACE_MODE: &str = "interpolation";

// Default: reduce polynomial degree at boundary vertices during interpolation.
pub const DEFAULT_BOUNDARY_DEGREE_FALLBACK: bool = true;
