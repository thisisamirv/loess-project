//! Regression Module
//!
//! This module provides the core functionality for local regression fitting:
//! context, types, generic solvers, and specialized low-dimensional accumulators.
// ## srrstats Compliance
//
// @srrstats {RE2.0} Local polynomial fitting with distance-based kernel weighting.
// @srrstats {G2.0} Input validation at regression boundaries; empty/zero-weight fallbacks.

// Regression Context
pub mod context;

// Generic Regression
mod generic;

// Specialized Regression
pub mod specialized;

// Regression Types
mod types;
