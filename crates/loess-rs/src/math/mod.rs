//! Layer 2: Math
//!
//! # Purpose
//!
//! This layer provides pure mathematical functions used throughout LOESS:
//! - Kernel functions for distance-based weighting
//! - Robust statistics (MAD)
//! - Linear algebra backends
//! - Hat matrix computation
//!
//! These are reusable mathematical building blocks with no algorithm-specific logic.
//!
//! # Architecture
//!
//! ```text
//! Layer 7: API
//!   ↓
//! Layer 6: Adapters
//!   ↓
//! Layer 5: Engine
//!   ↓
//! Layer 4: Evaluation
//!   ↓
//! Layer 3: Algorithms
//!   ↓
//! Layer 2: Math ← You are here
//!   ↓
//! Layer 1: Primitives
//! ```

/// Kernel (weight) functions for distance-based weighting.
pub mod kernel;

/// Robust scale estimation (MAR/MAD).
pub mod scaling;

/// Boundary padding utilities.
pub mod boundary;

/// Distance metrics for nD LOESS.
pub mod distance;

/// nD neighborhood search (KD-Tree implementation).
pub mod neighborhood;

/// Linear algebra backend abstraction.
pub mod linalg;

/// Hat matrix and delta parameter computation.
pub mod hat_matrix;
