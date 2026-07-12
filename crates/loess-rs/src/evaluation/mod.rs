//! Layer 4: Evaluation
//!
//! This layer calculates high-level statistical metrics based on the smoothing results:
//! - Cross-validation for parameter selection
//! - Diagnostic metrics for fit quality
//! - Confidence and prediction intervals

// Default configuration values for the evaluation layer.
pub mod defaults;

// Cross-validation for bandwidth selection.
pub mod cv;

// Diagnostic metrics for fit quality assessment.
pub mod diagnostics;

// Confidence and prediction interval computation.
pub mod intervals;
