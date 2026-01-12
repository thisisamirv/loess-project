//! Layer 4: Evaluation
//!
//! ## Purpose
//!
//! This layer provides parallel implementations of evaluation functions,
//! including cross-validation and interval estimation.
//!
//! ## Architecture
//!
//! ```text
//! Layer 7: API
//!   ↓
//! Layer 6: Adapters
//!   ↓
//! Layer 5: Engine
//!   ↓
//! Layer 4: Evaluation ← You are here
//!   ↓
//! Layer 3: Algorithms (at loess-rs)
//!   ↓
//! Layer 2: Math
//!   ↓
//! loess-rs
//! ```

/// Parallel cross-validation implementation.
pub mod cv;

/// Parallel interval estimation implementation.
pub mod intervals;
