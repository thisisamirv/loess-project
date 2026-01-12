//! KD-Tree for efficient k-nearest neighbor search in nD space.
//!
//! ## Purpose
//!
//! This module implements a KD-tree to optimize nD neighborhood searches.
//! By organizing points in a spatial hierarchy, we can reduce the search
//! time from O(n) to approximately O(log n) per query.
//!
//! ## Design notes
//!
//! * **Static Construction**: The tree is built once and then used for queries.
//! * **Eytzinger Layout**: Tree nodes are stored in a left-complete binary tree (array) layout for optimal cache locality.
//! * **Node Compression**: Nodes are compressed to 8 bytes, calculating split information on the fly to maximize cache density.
//! * **Trait-based Distance**: Supports generic distance metrics.
//!
//! ## Key concepts
//!
//! * **Splitting Plane**: The dimension and value used to split points at each node.
//! * **Implicit Navigation**: Child nodes are accessed via arithmetic ($2i+1$, $2i+2$) rather than pointers.
//! * **Pruning**: Skipping branches that cannot possibly contain nearer neighbors.
//!
//! ## Invariants
//!
//! * Tree depth is bound by O(log n).
//! * Queries always return the exact nearest neighbors (no approximation).
//!
//! ## Non-goals
//!
//! * This module does not support dynamic insertions or deletions.
//! * This module does not support approximate nearest neighbor search.

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::cmp::Ordering::{self, Equal};
use num_traits::Float;

// Internal dependencies
use crate::primitives::buffer::{NeighborhoodSearchBuffer, NeighborhoodStorage};

// ============================================================================
// Helper Types
// ============================================================================

/// Helper structure for max-heap in KD-tree search.
/// Orders by distance (the second field).
#[derive(Debug, Clone, Copy)]
pub struct NodeDistance<T>(pub usize, pub T);

impl<T: PartialEq> PartialEq for NodeDistance<T> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}
impl<T: PartialEq> Eq for NodeDistance<T> {}

impl<T: PartialOrd> PartialOrd for NodeDistance<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: PartialOrd> Ord for NodeDistance<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.1.partial_cmp(&other.1).unwrap_or(Equal)
    }
}

/// Trait for distance calculations used in KD-tree search.
pub trait PointDistance<T: Float> {
    /// Compute squared distance between two points (optimization to avoid sqrt).
    fn distance_squared(&self, a: &[T], b: &[T]) -> T;

    /// Compute distance along a single dimension (for pruning).
    fn split_distance(&self, dim: usize, split_val: T, query_val: T) -> T;

    /// Compute squared distance along a single dimension.
    fn split_distance_squared(&self, dim: usize, split_val: T, query_val: T) -> T;

    /// Convert a distance from the comparison space (e.g., squared) to the actual metric space.
    /// For Euclidean or Manhattan, this computes `sqrt` from the squared distance.
    fn post_process_distance(&self, d: T) -> T;
}

// ============================================================================
// Neighborhood Structure
// ============================================================================

/// Result container for k-nearest neighbor search.
#[derive(Debug, Clone)]
pub struct Neighborhood<T> {
    /// Indices of the k nearest neighbors (unordered).
    pub indices: Vec<usize>,
    /// Distances to each neighbor (corresponding to indices).
    pub distances: Vec<T>,
    /// Maximum distance in the neighborhood (bandwidth).
    pub max_distance: T,
}

impl<T: Float> Neighborhood<T> {
    /// Create a new empty neighborhood.
    pub fn new() -> Self {
        Self {
            indices: Vec::new(),
            distances: Vec::new(),
            max_distance: T::zero(),
        }
    }

    /// Pre-allocate buffers for a neighborhood of size k.
    pub fn with_capacity(k: usize) -> Self {
        Self {
            indices: Vec::with_capacity(k),
            distances: Vec::with_capacity(k),
            max_distance: T::zero(),
        }
    }

    /// Returns the number of neighbors currently stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns true if no neighbors are stored.
    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

impl<T: Float> NeighborhoodStorage for Neighborhood<T> {
    fn with_capacity(k: usize) -> Self {
        Self::with_capacity(k)
    }
    fn capacity(&self) -> usize {
        self.indices.capacity()
    }
}

impl<T: Float> Default for Neighborhood<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// KD-Tree Implementation
// ============================================================================

/// Compressed node structure for Eytzinger layout.
/// Reduced to 8 bytes to maximize cache line efficiency.
#[derive(Debug, Clone, Copy, Default)]
pub struct KDNode {
    /// Index of the point in the original flattened data array.
    pub index: usize,
}

/// KD-tree for spatial indexing of nD points.
#[derive(Debug, Clone)]
pub struct KDTree<T: Float> {
    /// The implicit Eytzinger tree nodes.
    nodes: Vec<KDNode>,
    /// Permuted points aligned with the nodes for cache locality.
    points: Vec<T>,
    /// Dimensionality of the data.
    dimensions: usize,
}

impl<T: Float> KDTree<T> {
    // ------------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------------

    /// Build a KD-tree from a flattened data array.
    ///
    /// The tree is built using a left-complete Eytzinger, reordering the
    /// input points into a layout optimized for cache locality during search.
    pub fn new(points: &[T], dimensions: usize) -> Self {
        let n = points.len() / dimensions;
        let mut indices: Vec<usize> = (0..n).collect();

        // Eytzinger layout requires the vector to be full size to allow access during build.
        let mut nodes = vec![KDNode::default(); n];
        let mut permuted_points = vec![T::zero(); points.len()];

        Self::build_recursive(
            points,
            dimensions,
            &mut indices,
            0,
            &mut nodes,
            &mut permuted_points,
            0,
        );

        Self {
            nodes,
            points: permuted_points,
            dimensions,
        }
    }

    /// Create a KD-tree from pre-built parts.
    ///
    /// This allows external builders (e.g., parallel ones) to construct the tree.
    /// The nodes and points must follow the Eytzinger layout convention.
    pub fn from_parts(nodes: Vec<KDNode>, points: Vec<T>, dimensions: usize) -> Self {
        Self {
            nodes,
            points,
            dimensions,
        }
    }

    /// Optimized search for k nearest neighbors.
    ///
    /// This method uses the provided buffer and neighborhood structure to avoid allocations.
    pub fn find_k_nearest<D: PointDistance<T>>(
        &self,
        query: &[T],
        k: usize,
        dist_calc: &D,
        exclude_self: Option<usize>,
        buffer: &mut NeighborhoodSearchBuffer<NodeDistance<T>>,
        neighborhood: &mut Neighborhood<T>,
    ) {
        if k == 0 || self.nodes.is_empty() {
            neighborhood.max_distance = T::zero();
            neighborhood.indices.clear();
            neighborhood.distances.clear();
            return;
        }

        buffer.clear();
        self.search_iterative(query, k, dist_calc, exclude_self, buffer);

        // Copy results from MaxHeap to Neighborhood output
        neighborhood.indices.clear();
        neighborhood.distances.clear();
        for &NodeDistance(idx, dist) in buffer.heap.iter() {
            neighborhood.indices.push(idx);
            neighborhood
                .distances
                .push(dist_calc.post_process_distance(dist));
        }
        let raw_max = buffer.heap.peek().map(|nd| nd.1).unwrap_or(T::zero());
        neighborhood.max_distance = dist_calc.post_process_distance(raw_max);
    }

    // ------------------------------------------------------------------------
    // Private Helpers & Algorithms
    // ------------------------------------------------------------------------

    /// Recursively builds the tree in Eytzinger layout.
    fn build_recursive(
        points: &[T],
        dims: usize,
        indices: &mut [usize],
        depth: usize,
        nodes: &mut [KDNode],
        permuted_points: &mut [T],
        curr_idx: usize,
    ) {
        if indices.is_empty() {
            return;
        }

        let axis = depth % dims;
        let n = indices.len();

        // Calculate pivot rank for Left-Complete Tree to maintain Eytzinger property
        let left_count = Self::calculate_left_subtree_size(n);
        let median_idx = left_count;

        // Partition finding the median
        if median_idx < n {
            indices.select_nth_unstable_by(median_idx, |&a, &b| {
                points[a * dims + axis]
                    .partial_cmp(&points[b * dims + axis])
                    .unwrap_or(Equal)
            });
        }

        let point_idx = indices[median_idx];

        // Place node in Eytzinger array
        nodes[curr_idx] = KDNode { index: point_idx };

        // COPY DATA: Place point data into permuted buffer at matching index
        let src_start = point_idx * dims;
        let dest_start = curr_idx * dims;
        permuted_points[dest_start..dest_start + dims]
            .copy_from_slice(&points[src_start..src_start + dims]);

        // Recurse left and right
        let (left_part, right_part_with_median) = indices.split_at_mut(median_idx);
        let right_part = &mut right_part_with_median[1..];

        Self::build_recursive(
            points,
            dims,
            left_part,
            depth + 1,
            nodes,
            permuted_points,
            2 * curr_idx + 1,
        );
        Self::build_recursive(
            points,
            dims,
            right_part,
            depth + 1,
            nodes,
            permuted_points,
            2 * curr_idx + 2,
        );
    }

    /// Iterative search using an explicit stack for traversal.
    ///
    /// **Optimization Note**: To minimize the stack footprint and avoid storing 24-byte
    /// `KDNode` structs, we use bit-packing to store `(node_idx, axis)` in a single `usize`.
    /// - `node_idx`: High bits (>> 8)
    /// - `axis`: Low 8 bits (& 0xFF)
    #[inline]
    fn search_iterative<D: PointDistance<T>>(
        &self,
        query: &[T],
        k: usize,
        dist_calc: &D,
        exclude_self: Option<usize>,
        buffer: &mut NeighborhoodSearchBuffer<NodeDistance<T>>,
    ) {
        let d = self.dimensions;
        let heap = &mut buffer.heap;
        let stack = &mut buffer.stack;
        let nodes_len = self.nodes.len();

        let mut heap_full = false;
        let mut max_dist = T::infinity();

        if nodes_len > 0 {
            // Push Root: Index 0, Axis 0 -> Packed 0
            stack.push(0);
        }

        while let Some(packed) = stack.pop() {
            // Unpack state
            let axis = packed & 0xFF; // Low 8 bits for axis
            let node_idx = packed >> 8; // High bits for node_idx

            // SAFETY: We strictly control pushes to be < nodes_len
            let node = unsafe { self.nodes.get_unchecked(node_idx) };

            // Access point data directly from valid Eytzinger-ordered buffer
            let offset = node_idx * d;
            // SAFETY: offsets are pre-calculated to be valid
            let node_point = unsafe { self.points.get_unchecked(offset..offset + d) };

            // 1. Process current node distance
            if exclude_self != Some(node.index) {
                // OPTIMIZATION: Use squared distance to avoid sqrt in hot loop
                let dist = dist_calc.distance_squared(query, node_point);

                if !heap_full {
                    heap.push(NodeDistance(node.index, dist));
                    if heap.len() == k {
                        heap_full = true;
                        max_dist = heap.peek().map(|nd| nd.1).unwrap_or(T::infinity());
                    }
                } else if dist < max_dist {
                    if let Some(mut top) = heap.peek_mut() {
                        *top = NodeDistance(node.index, dist);
                    }
                    max_dist = heap.peek().map(|nd| nd.1).unwrap_or(T::infinity());
                }
            }

            // 2. Determine Traversal Order
            // Leaf check: implicit left child is at 2*i + 1
            let left_child = 2 * node_idx + 1;
            if left_child >= nodes_len {
                continue;
            }

            // Split logic: Reuse `node_point` which is already in L1/Registers
            let split_dim = axis;
            let split_val = node_point[split_dim];
            let diff = query[split_dim] - split_val;

            let right_child = left_child + 1;
            let has_right = right_child < nodes_len;
            let next_axis = if split_dim + 1 == d { 0 } else { split_dim + 1 };

            // Pack children indices with next axis
            let packed_left = (left_child << 8) | next_axis;
            let packed_right = (right_child << 8) | next_axis;

            // Decide Near/Far
            // If diff <= 0, query is on left side -> Near=Left, Far=Right
            let (near_packed, far_packed, has_far) = if diff <= T::zero() {
                (packed_left, packed_right, has_right)
            } else {
                (packed_right, packed_left, true)
            };

            // 3. Pruning: Only push valid Far children if they can contain closer points
            if has_far {
                let dist_to_plane =
                    dist_calc.split_distance_squared(split_dim, split_val, query[split_dim]);
                if !heap_full || dist_to_plane < max_dist {
                    stack.push(far_packed);
                }
            }

            // 4. Always explore valid Near child
            // If near is right_child, we must check has_right
            let near_exists = if diff <= T::zero() {
                true // Left always exists if we passed leaf check
            } else {
                has_right
            };

            if near_exists {
                stack.push(near_packed);
            }
        }
    }

    /// Calculate number of nodes in the left subtree of a left-complete binary tree of size N.
    pub fn calculate_left_subtree_size(n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        // Height: H = floor(log2(n))
        let h = (usize::BITS - n.leading_zeros() - 1) as usize;
        if h == 0 {
            return 0;
        }

        // Max nodes in full tree of height h
        let max_leaf_capacity = 1 << h; // 2^h

        // Nodes in last level R = n - (nodes in full tree of height h-1)
        let total_nodes_above_leaf = max_leaf_capacity - 1;
        let r = n - total_nodes_above_leaf;

        // Left subtree gets the filled portion of the last level
        let left_part_leaves = r.min(max_leaf_capacity / 2);

        // Full left subtree excluding leaves
        let left_subtree_capacity_full = (max_leaf_capacity / 2) - 1;
        left_subtree_capacity_full + left_part_leaves
    }
}
