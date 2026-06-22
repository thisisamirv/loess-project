//! Parallel KD-tree builder using Rayon.
//!
//! ## Purpose
//!
//! This module provides a multi-threaded builder for the KD-tree used in
//! nearest neighbor searches. Construction is parallelized at the top levels
//! of recursion to speed up initialization for large datasets.
//!
//! ## Design notes
//!
//! * **Recursive Parallelism**: Uses `rayon::join` to parallelize the recursive build steps.
//! * **Depth Limit**: Parallelism typically targets the upper levels of the tree.
//! * **Unsafe Access**: Uses raw pointers for concurrent writes to disjoint array indices.
//!
//! ## Key concepts
//!
//! * **KD-Tree**: Spatial indexing structure for fast neighbor lookup.
//! * **Eytzinger Layout**: Cache-optimal array layout (left-complete binary tree).
//! * **Median Splitting**: Balanced tree construction via `select_nth_unstable`.
//!
//! ## Invariants
//!
//! * Parallel construction produces an identical tree to sequential construction.
//! * Thread safety is guaranteed by disjoint index access patterns.
//!
//! ## Non-goals
//!
//! * This module does not implement the search logic (delegated to loess-rs).
//! * This module does not support dynamic updates.
//!
use loess_rs::internals::algorithms::regression::SolverLinalg;
use loess_rs::internals::math::distance::DistanceLinalg;
use loess_rs::internals::math::linalg::FloatLinalg;
use loess_rs::internals::math::neighborhood::{KDNode, KDTree};
use num_traits::Float;

// Feature-gated imports
#[cfg(feature = "cpu")]
use rayon::join;

/// Parallel KD-tree builder using Rayon.
///
/// This provides a multi-threaded implementation of the Eytzinger-layout KD-tree
/// used in loess-rs.
#[cfg(feature = "cpu")]
pub fn build_kdtree_parallel<T>(points: &[T], dimensions: usize) -> KDTree<T>
where
    T: FloatLinalg + DistanceLinalg + SolverLinalg + Float + Send + Sync + 'static,
{
    let n = points.len() / dimensions;
    let mut nodes = vec![KDNode::default(); n];
    let mut permuted_points = vec![T::zero(); points.len()];
    let mut indices: Vec<usize> = (0..n).collect();

    if n > 0 {
        // SAFETY: We use raw pointers to allow concurrent writes to the nodes array.
        // The Eytzinger layout guarantees that 2*v and 2*v + 1 create disjoint paths,
        // so no two threads will ever write to the same index.
        let nodes_ptr = nodes.as_mut_ptr() as usize;
        let permuted_ptr = permuted_points.as_mut_ptr() as usize;

        build_recursive_parallel(
            points,
            &mut indices,
            dimensions,
            nodes_ptr,
            permuted_ptr,
            0,
            1,
        );
    }

    KDTree::from_parts(nodes, permuted_points, dimensions)
}

#[cfg(feature = "cpu")]
fn build_recursive_parallel<T>(
    points: &[T],
    indices: &mut [usize],
    dimensions: usize,
    nodes_ptr: usize,
    permuted_ptr: usize,
    depth: usize,
    v: usize,
) where
    T: FloatLinalg + DistanceLinalg + SolverLinalg + Float + Send + Sync + 'static,
{
    let n = indices.len();
    if n == 0 {
        return;
    }

    let axis = depth % dimensions;
    let left_size = KDTree::<T>::calculate_left_subtree_size(n);
    let mid = left_size;

    // Partition around the median for this axis
    indices.select_nth_unstable_by(mid, |&a, &b| {
        let val_a = points[a * dimensions + axis];
        let val_b = points[b * dimensions + axis];
        val_a
            .partial_cmp(&val_b)
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    // Write to the current node and permuted points
    // SAFETY: v is unique for each recursive call path.
    unsafe {
        let node_ref = &mut *(nodes_ptr as *mut KDNode).add(v - 1);
        node_ref.index = indices[mid];

        // Copy point data to permuted buffer for cache locality during search
        let dest_offset = (v - 1) * dimensions;
        let src_offset = indices[mid] * dimensions;
        let dest_ptr = (permuted_ptr as *mut T).add(dest_offset);
        let src_ptr = points.as_ptr().add(src_offset);
        std::ptr::copy_nonoverlapping(src_ptr, dest_ptr, dimensions);
    }

    let (left_indices, right_indices_with_mid) = indices.split_at_mut(mid);
    let right_indices = &mut right_indices_with_mid[1..];

    // Threshold for spawning new threads
    if n > 1024 {
        join(
            || {
                build_recursive_parallel(
                    points,
                    left_indices,
                    dimensions,
                    nodes_ptr,
                    permuted_ptr,
                    depth + 1,
                    2 * v,
                )
            },
            || {
                build_recursive_parallel(
                    points,
                    right_indices,
                    dimensions,
                    nodes_ptr,
                    permuted_ptr,
                    depth + 1,
                    2 * v + 1,
                )
            },
        );
    } else {
        build_recursive_sequential(
            points,
            left_indices,
            dimensions,
            nodes_ptr,
            permuted_ptr,
            depth + 1,
            2 * v,
        );
        build_recursive_sequential(
            points,
            right_indices,
            dimensions,
            nodes_ptr,
            permuted_ptr,
            depth + 1,
            2 * v + 1,
        );
    }
}

#[cfg(feature = "cpu")]
fn build_recursive_sequential<T>(
    points: &[T],
    indices: &mut [usize],
    dimensions: usize,
    nodes_ptr: usize,
    permuted_ptr: usize,
    depth: usize,
    v: usize,
) where
    T: FloatLinalg + DistanceLinalg + SolverLinalg + Float + Send + Sync + 'static,
{
    let n = indices.len();
    if n == 0 {
        return;
    }

    let axis = depth % dimensions;
    let left_size = KDTree::<T>::calculate_left_subtree_size(n);
    let mid = left_size;

    indices.select_nth_unstable_by(mid, |&a, &b| {
        let val_a = points[a * dimensions + axis];
        let val_b = points[b * dimensions + axis];
        val_a
            .partial_cmp(&val_b)
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    unsafe {
        let node_ref = &mut *(nodes_ptr as *mut KDNode).add(v - 1);
        node_ref.index = indices[mid];

        // Copy point data to permuted buffer
        let dest_offset = (v - 1) * dimensions;
        let src_offset = indices[mid] * dimensions;
        let dest_ptr = (permuted_ptr as *mut T).add(dest_offset);
        let src_ptr = points.as_ptr().add(src_offset);
        std::ptr::copy_nonoverlapping(src_ptr, dest_ptr, dimensions);
    }

    let (left_indices, right_indices_with_mid) = indices.split_at_mut(mid);
    let right_indices = &mut right_indices_with_mid[1..];

    build_recursive_sequential(
        points,
        left_indices,
        dimensions,
        nodes_ptr,
        permuted_ptr,
        depth + 1,
        2 * v,
    );
    build_recursive_sequential(
        points,
        right_indices,
        dimensions,
        nodes_ptr,
        permuted_ptr,
        depth + 1,
        2 * v + 1,
    );
}

/// Fallback for non-CPU targets.
#[cfg(not(feature = "cpu"))]
pub fn build_kdtree_parallel<T>(points: &[T], dimensions: usize) -> KDTree<T>
where
    T: FloatLinalg + DistanceLinalg + SolverLinalg + Float + Send + Sync + 'static,
{
    KDTree::new(points, dimensions)
}
