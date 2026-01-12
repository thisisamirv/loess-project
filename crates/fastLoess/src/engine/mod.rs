//! Layer 5: Engine
//!
//! ## Purpose
//!
//! This layer provides the parallel execution engine for LOESS smoothing.
//! It implements rayon-based parallel processing that can be injected into
//! the loess-rs executor via callback hooks.
//!
//! ## Architecture
//!
//! ```text
//! Layer 7: API
//!   ↓
//! Layer 6: Adapters
//!   ↓
//! Layer 5: Engine ← You are here
//!   ↓
//! Layer 4: Evaluation
//!   ↓
//! Layer 3: Algorithms (at loess-rs)
//!   ↓
//! Layer 2: Math
//!   ↓
//! loess-rs
//! ```

/// Parallel execution engine for LOESS smoothing operations.
pub mod executor;
