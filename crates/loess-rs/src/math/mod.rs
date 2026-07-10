//! Layer 2: Math
//!
//! This layer provides pure mathematical functions used throughout LOESS:
//! - Kernel functions for distance-based weighting
//! - Robust statistics (MAD/MAR)

// Kernel (weight) functions for distance-based weighting.
pub mod kernel;

// Default values for math module types.
pub mod defaults;

// Robust scale estimation (MAR/MAD).
pub mod scaling;

// Boundary padding utilities.
pub mod boundary;

// Distance metrics for nD LOESS.
pub mod distance;

// nD neighborhood search (KD-Tree implementation).
pub mod neighborhood;

// Linear algebra backend abstraction.
pub mod linalg;

// Hat matrix and delta parameter computation.
pub mod hat_matrix;
