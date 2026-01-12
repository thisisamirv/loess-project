//! Memory management and buffer recycling for LOESS operations.
//!
//! ## Purpose
//!
//! This module provides a centralized, reusable workspace (`LoessBuffer`) to minimize
//! dynamic memory allocations during repeated LOESS fitting. By allocating buffers once
//! and recycling them across multiple query points or cross-validation folds, we significantly
//! reduce allocator pressure and improve cache locality.
//!
//! ## Design notes
//!
//! * **Centralized Ownership**: The `LoessBuffer` struct holds all necessary scratch space
//!   for the entire pipeline (neighborhood search, regression weights, solver matrices, etc.).
//! * **Lazy Expansion**: Buffers are grown on demand via `ensure_capacity` but never shrunk,
//!   stabilizing at the maximum required size for the dataset.
//! * **Generic injection**: The workspace is generic over the `Neighborhood` storage to
//!   allow decoupling from specific spatial index implementations.
//!
//! ## Key concepts
//!
//! * **LoessBuffer**: The top-level struct passed through the executor pipeline.
//! * **NeighborhoodSearchBuffer**: Reusable heap and vector for K-nearest neighbor searches.
//! * **FittingBuffer**: Matrices and vectors for the weighted least squares solver.
//! * **ExecutorBuffer**: Buffers for global operations like robustness iteration and normalization.
//! * **CVBuffer**: Scratch space specifically for cross-validation data subsets.
//!
//! ## Invariants
//!
//! * Buffers are only logically cleared (e.g., `vec.clear()`), not deallocated, between iterations.
//! * Capability is monotonically increasing; `ensure_capacity` only reallocates if current capacity is insufficient.
//!
//! ## Non-goals
//!
//! * Thread-local automatic caching (buffers are explicitly passed to allow parallel execution with one buffer per thread).
//! * Dynamic shrinking or aggressive memory reclamation (performance is prioritized over minimal footprint).

// Feature-gated dependencies
#[cfg(not(feature = "std"))]
use alloc::collections::BinaryHeap;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::collections::BinaryHeap;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::fmt::Debug;
use core::ops::{Deref, DerefMut};
use num_traits::Float;

// ============================================================================
// Slot - Unified Vector Abstraction
// ============================================================================

/// A reusable vector slot with automatic capacity management.
#[derive(Debug, Clone)]
pub struct Slot<T>(Vec<T>);

impl<T> Slot<T> {
    /// Create a new slot with the given initial capacity.
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    /// Ensure the slot has at least the given capacity.
    /// Grows the underlying vector if needed; never shrinks.
    #[inline]
    pub fn ensure_capacity(&mut self, capacity: usize) {
        if self.0.capacity() < capacity {
            self.0.reserve(capacity - self.0.capacity());
        }
    }

    /// Clear the slot (sets length to 0, preserves capacity).
    #[inline]
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// Get a reference to the underlying vector.
    #[inline]
    pub fn as_vec(&self) -> &Vec<T> {
        &self.0
    }

    /// Get a mutable reference to the underlying vector.
    #[inline]
    pub fn as_vec_mut(&mut self) -> &mut Vec<T> {
        &mut self.0
    }

    /// Consume the slot and return the underlying vector.
    #[inline]
    pub fn into_inner(self) -> Vec<T> {
        self.0
    }
}

impl<T> Default for Slot<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<T> Deref for Slot<T> {
    type Target = Vec<T>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Slot<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> From<Vec<T>> for Slot<T> {
    fn from(v: Vec<T>) -> Self {
        Self(v)
    }
}

// ============================================================================
// Main Workspace
// ============================================================================

/// A struct containing pre-allocated buffers for LOESS operations.
pub struct LoessBuffer<T: Float, N, NH> {
    /// Buffer for KD-tree search state.
    pub search_buffer: NeighborhoodSearchBuffer<N>,
    /// Buffer for neighbor indices and distances.
    pub neighborhood: NH,
    /// Buffer for regression fitting (WLS matrices).
    pub fitting_buffer: FittingBuffer<T>,
    /// Buffer for global executor state.
    pub executor_buffer: ExecutorBuffer<T>,
    /// Buffer for cross-validation.
    pub cv_buffer: CVBuffer<T>,
}

impl<T: Float + Debug + Send + Sync + 'static, N, NH> LoessBuffer<T, N, NH>
where
    N: Ord,
    NH: NeighborhoodStorage,
{
    /// Create a new workspace with capacities matching the expected fit parameters.
    ///
    /// - `n`: Total number of points.
    /// - `dims`: Predictor dimensions.
    /// - `k`: Expected number of neighbors (window size).
    /// - `n_coeffs`: Expected number of polynomial coefficients.
    pub fn new(n: usize, dims: usize, k: usize, n_coeffs: usize) -> Self {
        Self {
            search_buffer: NeighborhoodSearchBuffer::new(k),
            neighborhood: NH::with_capacity(k),
            fitting_buffer: FittingBuffer::<T>::new(k, n_coeffs),
            executor_buffer: ExecutorBuffer::<T>::new(n, dims),
            cv_buffer: CVBuffer::<T>::new(n, dims),
        }
    }

    /// Ensure all buffers have enough capacity for the given problem size.
    pub fn ensure_capacity(&mut self, n_total: usize, dims: usize, k: usize, n_coeffs: usize) {
        if self.neighborhood.capacity() < k {
            self.neighborhood = NH::with_capacity(k);
            self.search_buffer = NeighborhoodSearchBuffer::new(k);
        }
        if self.fitting_buffer.weights.capacity() < k
            || self.fitting_buffer.xtw_x.capacity() < n_coeffs * n_coeffs
        {
            self.fitting_buffer = FittingBuffer::new(k, n_coeffs);
        }

        self.executor_buffer.ensure_capacity(n_total, dims);
        self.cv_buffer.ensure_capacity(n_total, dims);
    }
}

// ============================================================================
// Traits
// ============================================================================

/// Trait for neighborhood storage that can be injected into the workspace.
pub trait NeighborhoodStorage {
    /// Create a new neighborhood storage with given capacity.
    fn with_capacity(k: usize) -> Self;
    /// Get the current capacity of the storage.
    fn capacity(&self) -> usize;
}

// ============================================================================
// Internal Buffers
// ============================================================================

/// Persistent buffers for KD-tree search to avoid allocations.
pub struct NeighborhoodSearchBuffer<N> {
    pub(crate) heap: BinaryHeap<N>,
    pub(crate) stack: Vec<usize>,
}

impl<N: Ord> NeighborhoodSearchBuffer<N> {
    /// Create a new search buffer with capacity k.
    pub fn new(k: usize) -> Self {
        // Stack depth is bounded by tree height, typically O(log n).
        // Pre-allocate for ~1M points (log2(1M) ≈ 20).
        Self {
            heap: BinaryHeap::with_capacity(k),
            stack: Vec::with_capacity(32),
        }
    }

    /// Clear all internal buffers for reuse.
    pub fn clear(&mut self) {
        self.heap.clear();
        self.stack.clear();
    }
}

/// Persistent buffers for local regression to avoid allocations.
pub struct FittingBuffer<T> {
    /// Weights for each neighbor.
    pub weights: Slot<T>,
    /// Normal matrix X'WX.
    pub xtw_x: Slot<T>,
    /// Normal vector X'WY.
    pub xtw_y: Slot<T>,
    /// Column norms for equilibration.
    pub col_norms: Slot<T>,
}

impl<T> FittingBuffer<T> {
    /// Create a new fitting buffer with estimated capacities.
    pub fn new(k: usize, n_coeffs: usize) -> Self {
        Self {
            weights: Slot::new(k),
            xtw_x: Slot::new(n_coeffs * n_coeffs),
            xtw_y: Slot::new(n_coeffs),
            col_norms: Slot::new(n_coeffs),
        }
    }
}

/// Cache of pre-computed neighborhoods for each query point.
///
/// Used to avoid repeated KD-tree searches during robustness iterations,
/// since neighborhoods are invariant (only regression weights change).
#[derive(Debug, Clone)]
pub struct NeighborhoodCache<T> {
    /// Cached neighborhoods: (indices, distances, max_distance) for each point.
    pub entries: Vec<CachedNeighborhood<T>>,
    /// Whether the cache is populated and valid.
    pub is_valid: bool,
}

/// A single cached neighborhood for one query point.
#[derive(Debug, Clone)]
pub struct CachedNeighborhood<T> {
    /// Indices of the k nearest neighbors.
    pub indices: Vec<usize>,
    /// Distances to each neighbor.
    pub distances: Vec<T>,
    /// Maximum distance in the neighborhood (bandwidth).
    pub max_distance: T,
}

impl<T: Float> NeighborhoodCache<T> {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            is_valid: false,
        }
    }

    /// Create a cache with pre-allocated capacity for n points.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            entries: Vec::with_capacity(n),
            is_valid: false,
        }
    }
}

impl<T: Float> Default for NeighborhoodCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Persistent buffers for global executor state.
pub struct ExecutorBuffer<T> {
    /// Minimum values for each dimension.
    pub mins: Slot<T>,
    /// Maximum values for each dimension.
    pub maxs: Slot<T>,
    /// Normalization scales for each dimension.
    pub scales: Slot<T>,
    /// Robustness weights for iterative refinement.
    pub robustness_weights: Slot<T>,
    /// Residuals for iterative refinement.
    pub residuals: Slot<T>,
    /// Sorted residuals for median computation.
    pub sorted_residuals: Slot<T>,
    /// Cached neighborhoods to avoid repeated KD-tree searches.
    pub neighborhood_cache: NeighborhoodCache<T>,
}

impl<T: Float> ExecutorBuffer<T> {
    /// Create a new executor buffer with given capacities.
    pub fn new(n: usize, dims: usize) -> Self {
        Self {
            mins: Slot::new(dims),
            maxs: Slot::new(dims),
            scales: Slot::new(dims),
            robustness_weights: Slot::new(n),
            residuals: Slot::new(n),
            sorted_residuals: Slot::new(n),
            neighborhood_cache: NeighborhoodCache::with_capacity(n),
        }
    }

    /// Ensure buffers have enough capacity for given dimensions and points.
    pub fn ensure_capacity(&mut self, n: usize, dims: usize) {
        self.mins.ensure_capacity(dims);
        self.maxs.ensure_capacity(dims);
        self.scales.ensure_capacity(dims);
        self.robustness_weights.ensure_capacity(n);
        self.residuals.ensure_capacity(n);
        self.sorted_residuals.ensure_capacity(n);
    }
}

/// Persistent buffers for cross-validation subsets.
pub struct CVBuffer<T> {
    /// Training subset x-values.
    pub train_x: Slot<T>,
    /// Training subset y-values.
    pub train_y: Slot<T>,
    /// Test subset x-values.
    pub test_x: Slot<T>,
    /// Test subset y-values.
    pub test_y: Slot<T>,
}

impl<T> CVBuffer<T> {
    /// Create a new CV buffer with given capacities.
    pub fn new(n: usize, dims: usize) -> Self {
        Self {
            train_x: Slot::new(n * dims),
            train_y: Slot::new(n),
            test_x: Slot::new(n * dims),
            test_y: Slot::new(n),
        }
    }

    /// Ensure buffers have enough capacity for given dimensions and points.
    pub fn ensure_capacity(&mut self, n: usize, dims: usize) {
        self.train_x.ensure_capacity(n * dims);
        self.train_y.ensure_capacity(n);
        self.test_x.ensure_capacity(n * dims);
        self.test_y.ensure_capacity(n);
    }
}
