//! Layer 2: Math
//!
//! ## Purpose
//!
//! This layer provides parallel implementations of mathematical utilities,
//! primarily the KD-tree construction for neighbor search.
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
//! Layer 4: Evaluation
//!   ↓
//! Layer 3: Algorithms (at loess-rs)
//!   ↓
//! Layer 2: Math ← You are here
//!   ↓
//! loess-rs
//! ```
//!

/// Parallel nD neighborhood search (KD-Tree implementation).
pub mod neighborhood;
