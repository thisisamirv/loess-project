//! Layer 5: Engine
//!
//! This layer provides the parallel execution engine for LOESS smoothing.
//! It handles the distribution of compute tasks across CPU cores or GPU hardware.

// Parallel execution engine for LOESS smoothing operations.
pub mod executor;
