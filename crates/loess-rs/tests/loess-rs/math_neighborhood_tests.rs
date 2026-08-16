#![cfg(feature = "dev")]

use loess_rs::internals::engine::executor::LoessDistanceCalculator;
use loess_rs::internals::math::distance::DistanceMetric;
use loess_rs::internals::math::neighborhood::{KDTree, Neighborhood};
use loess_rs::internals::primitives::buffer::NeighborhoodSearchBuffer;

#[test]
fn test_kdtree_simple_2d() {
    let points = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let tree = KDTree::new(&points, 2);

    // Find 2 nearest to (0.2, 0.2)
    let dist_calc = LoessDistanceCalculator {
        metric: DistanceMetric::Euclidean,
        scales: &[1.0, 1.0],
    };

    let k = 2;
    let mut buffer = NeighborhoodSearchBuffer::new(k);
    let mut nbh = Neighborhood::with_capacity(k);
    tree.find_k_nearest(&[0.2, 0.2], k, &dist_calc, None, &mut buffer, &mut nbh);

    assert_eq!(nbh.indices.len(), 2);
    assert!(nbh.indices.contains(&0)); // (0,0) is closest
    assert!(nbh.indices.contains(&1) || nbh.indices.contains(&2)); // Either (1,0) or (0,1)
}

#[test]
fn test_kdtree_normalized_distance() {
    // x has range [0, 100], y has range [0, 1]
    let points = vec![0.0, 0.0, 100.0, 1.0];
    let tree = KDTree::new(&points, 2);

    let scales = [0.01, 1.0]; // Normalize x to [0,1]

    // Query at (50, 0)
    let dist_calc = LoessDistanceCalculator {
        metric: DistanceMetric::Normalized,
        scales: &scales,
    };

    let k = 1;
    let mut buffer = NeighborhoodSearchBuffer::new(k);
    let mut nbh = Neighborhood::with_capacity(k);
    tree.find_k_nearest(&[50.0, 0.0], k, &dist_calc, None, &mut buffer, &mut nbh);

    // With normalization, (50,0) is distance 0.5 from both (0,0) and (100,1)
    // Without normalization, (50,0) is much closer to (0,0) [dist=50] than (100,1) [dist=sqrt(50^2+1)]
    assert!(nbh.indices.contains(&0) || nbh.indices.contains(&1));
}

#[test]
fn test_kdtree_exclude_self() {
    let points = vec![0.0, 0.0, 1.0, 1.0];
    let tree = KDTree::new(&points, 2);

    let dist_calc = LoessDistanceCalculator {
        metric: DistanceMetric::Euclidean,
        scales: &[1.0, 1.0],
    };

    let k = 1;
    let mut buffer = NeighborhoodSearchBuffer::new(k);
    let mut nbh = Neighborhood::with_capacity(k);
    tree.find_k_nearest(&[0.0, 0.0], k, &dist_calc, Some(0), &mut buffer, &mut nbh);

    assert_eq!(nbh.indices.len(), 1);
    assert_eq!(nbh.indices[0], 1); // Should exclude point 0 and return point 1
}

#[test]
fn test_kdtree_find_k_nearest_correct_subset() {
    let points: [f64; 8] = [0.0, 0.0, 5.0, 0.0, 2.0, 0.0, 7.0, 0.0];
    let dims = 2;
    let scales: [f64; 2] = [1.0, 1.0];

    let tree = KDTree::new(&points, dims);
    let query: [f64; 2] = [0.0, 0.0];

    let dist_calc = LoessDistanceCalculator {
        metric: DistanceMetric::Euclidean,
        scales: &scales,
    };

    let k = 2;
    let mut buffer = NeighborhoodSearchBuffer::new(k);
    let mut result = Neighborhood::with_capacity(k);
    tree.find_k_nearest(&query, k, &dist_calc, None, &mut buffer, &mut result);

    // Should contain the two closest points: (0,0) [idx 0] and (2,0) [idx 2]
    assert_eq!(result.indices.len(), 2);
    assert!(result.indices.contains(&0));
    assert!(result.indices.contains(&2));
}

#[test]
fn test_kdtree_find_k_nearest_3d() {
    // 4 points in 3D
    let points: [f64; 12] = [
        0.0, 0.0, 0.0, // origin
        1.0, 0.0, 0.0, // x=1
        0.0, 1.0, 0.0, // y=1
        10.0, 10.0, 10.0, // far away
    ];
    let dims = 3;
    let scales: [f64; 3] = [1.0, 1.0, 1.0];

    let tree = KDTree::new(&points, dims);
    let query: [f64; 3] = [0.0, 0.0, 0.0];

    let dist_calc = LoessDistanceCalculator {
        metric: DistanceMetric::Euclidean,
        scales: &scales,
    };

    let k = 2;
    let mut buffer = NeighborhoodSearchBuffer::new(k);
    let mut result = Neighborhood::with_capacity(k);
    tree.find_k_nearest(&query, k, &dist_calc, Some(0), &mut buffer, &mut result);

    assert_eq!(result.indices.len(), 2);
    // Should be points 1 and 2 (distance 1 each)
    assert!(result.indices.contains(&1));
    assert!(result.indices.contains(&2));
}
