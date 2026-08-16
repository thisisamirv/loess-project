#![cfg(feature = "dev")]
use fastLoess::prelude::*;

/// With n > 1024 data points and `parallel(true)`, the parallel KD-tree
/// builder splits into sub-trees with ≤ 1024 nodes and delegates to
/// `build_recursive_sequential` (math/neighborhood.rs).
#[test]
fn test_parallel_kdtree_large_dataset() {
    let n = 2049; // > 1024 ensures rayon::join is used and children use sequential build
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| (xi / 200.0).sin()).collect();

    let res = Loess::new()
        .fraction(0.05)
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert_eq!(res.y.len(), n);
}

/// Running parallel `Interpolation` mode exercises `vertex_pass_parallel`
/// (engine/executor.rs) and also triggers `build_kdtree_parallel` via the
/// `custom_kdtree_builder` callback injected in the Batch adapter.
#[test]
fn test_parallel_interpolation_mode() {
    let n = 50;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();

    let res = Loess::new()
        .fraction(0.3)
        .surface_mode("interpolation")
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert_eq!(res.y.len(), n);
}

/// Interpolation mode with n > 1024 ensures `build_recursive_sequential` is
/// reached from within the parallel KD-tree builder (math/neighborhood.rs).
#[test]
fn test_parallel_interpolation_large_dataset() {
    let n = 2049;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| (xi / 200.0).sin()).collect();

    let res = Loess::new()
        .fraction(0.05)
        .surface_mode("interpolation")
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert_eq!(res.y.len(), n);
}
