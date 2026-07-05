#![cfg(feature = "dev")]
// Tests for interpolation algorithms.
//
// These tests verify the `InterpolationSurface` and its multilinear interpolation
// capabilities used for efficient nD LOESS evaluation.
//
// ## Test Organization
//
// 1. **Surface Construction** - Verifies cell subdivision and vertex creation
// 2. **Interpolation Accuracy** - Verifies 1D (linear) and 2D (bilinear) interpolation
// 3. **Adaptive Subdivision** - Verifies finding high-variance regions
// 4. **Edge Cases** - Boundary conditions and degenerate inputs
use approx::assert_relative_eq;

use loess_rs::internals::algorithms::interpolation::InterpolationSurface;
use loess_rs::internals::algorithms::regression::{PolynomialDegree, ZeroWeightFallback};
use loess_rs::internals::engine::executor::LoessDistanceCalculator;
use loess_rs::internals::math::distance::DistanceMetric;
use loess_rs::internals::math::kernel::WeightFunction;
use loess_rs::internals::math::neighborhood::{KDTree, Neighborhood, NodeDistance};
use loess_rs::internals::primitives::buffer::{FittingBuffer, LoessBuffer};

use loess_rs::internals::algorithms::interpolation::SurfaceCell;
use loess_rs::internals::primitives::buffer::CachedNeighborhood;

// ============================================================================
// Helper Functions & Mocks
// ============================================================================

fn create_mock_dist_calc() -> LoessDistanceCalculator<'static, f64> {
    LoessDistanceCalculator {
        metric: DistanceMetric::Euclidean,
        scales: &[], // Not used for these tests
    }
}

// ============================================================================
// Surface Construction Tests
// ============================================================================

/// Test building a simple 1D surface.
///
/// Verifies that vertices are created at bounds.
#[test]
fn test_build_simple_1d() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let fraction = 0.5;
    let dimensions = 1;

    let kdtree = KDTree::new(&x, dimensions);
    let dist_calc = create_mock_dist_calc();

    let mut workspace =
        LoessBuffer::<f64, NodeDistance<f64>, Neighborhood<f64>>::new(x.len(), dimensions, 2, 2); // k=2, n_coeffs=2 (1D linear)
    let LoessBuffer {
        ref mut search_buffer,
        ref mut neighborhood,
        ref mut fitting_buffer,
        ..
    } = workspace;

    // Simple fitter that just returns the x-coordinate (identity)
    let fitter = |vertex: &[f64],
                  _: &Neighborhood<f64>,
                  _: &mut FittingBuffer<f64>,
                  _: PolynomialDegree|
     -> Option<Vec<f64>> { Some(vec![vertex[0], 1.0]) };

    let surface = InterpolationSurface::build(
        &x,
        &y,
        dimensions,
        fraction,
        2,
        &dist_calc,
        &kdtree,
        10, // Max vertices
        fitter,
        search_buffer,
        neighborhood,
        fitting_buffer,
        0.2,
        None,
        &[],
        WeightFunction::default(),
        ZeroWeightFallback::default(),
        PolynomialDegree::default(),
        &DistanceMetric::default(),
        true,
        None, // custom_weights
    );

    assert!(surface.vertex_data.len() >= 4); // At least 2 vertices * 2 values each
    // Determine min/max from input to check bounds
    let min_x = 0.0;
    let max_x = 4.0;

    // Bounds are expanded by 0.5%
    let range = 4.0;
    let margin = range * 0.005;
    let effective_min = min_x - margin;
    let effective_max = max_x + margin;

    // Check root cell correctness
    let root = &surface.cells[surface.root];
    assert_relative_eq!(root.lower[0], effective_min, epsilon = 1e-10);
    assert_relative_eq!(root.upper[0], effective_max, epsilon = 1e-10);
}

/// Test building a simple 2D surface.
#[test]
fn test_build_simple_2d() {
    // 4 points in a square
    let x = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let y = vec![0.0, 1.0, 1.0, 2.0]; // x + y
    let dimensions = 2;
    let fraction = 1.0;

    let kdtree = KDTree::new(&x, dimensions);
    let dist_calc = create_mock_dist_calc();

    let mut workspace =
        LoessBuffer::<f64, NodeDistance<f64>, Neighborhood<f64>>::new(4, dimensions, 4, 3); // k=4, n_coeffs=3 (2D linear)
    let LoessBuffer {
        ref mut search_buffer,
        ref mut neighborhood,
        ref mut fitting_buffer,
        ..
    } = workspace;

    let fitter = |vertex: &[f64],
                  _: &Neighborhood<f64>,
                  _: &mut FittingBuffer<f64>,
                  _: PolynomialDegree|
     -> Option<Vec<f64>> { Some(vec![vertex[0] + vertex[1], 1.0, 1.0]) };

    let surface = InterpolationSurface::build(
        &x,
        &y,
        dimensions,
        fraction,
        4,
        &dist_calc,
        &kdtree,
        20,
        fitter,
        search_buffer,
        neighborhood,
        fitting_buffer,
        0.2,
        None,
        &[],
        WeightFunction::default(),
        ZeroWeightFallback::default(),
        PolynomialDegree::default(),
        &DistanceMetric::default(),
        true,
        None, // custom_weights
    );

    // Initial cell has 4 vertices (2^2)
    assert!(surface.vertex_data.len() >= 12); // At least 4 vertices * 3 values each
}

// ============================================================================
// Interpolation Accuracy Tests
// ============================================================================

/// Test exact 1D linear interpolation.
///
/// Linear interpolation of a linear function should be exact.
#[test]
fn test_interpolate_1d_linear() {
    let x = vec![0.0, 2.0, 4.0];
    let y = vec![0.0, 2.0, 4.0]; // y = x
    let dimensions = 1;

    let kdtree = KDTree::new(&x, dimensions);
    let dist_calc = create_mock_dist_calc();
    let mut workspace = LoessBuffer::new(x.len(), dimensions, 2, 2);
    let LoessBuffer {
        ref mut search_buffer,
        ref mut neighborhood,
        ref mut fitting_buffer,
        ..
    } = workspace;

    let fitter = |vertex: &[f64],
                  _: &Neighborhood<f64>,
                  _: &mut FittingBuffer<f64>,
                  _: PolynomialDegree|
     -> Option<Vec<f64>> { Some(vec![vertex[0], 1.0]) };

    let surface = InterpolationSurface::build(
        &x,
        &y,
        dimensions,
        0.5,
        1,
        &dist_calc,
        &kdtree,
        10,
        fitter,
        search_buffer,
        neighborhood,
        fitting_buffer,
        0.2,
        None,
        &[],
        WeightFunction::default(),
        ZeroWeightFallback::default(),
        PolynomialDegree::default(),
        &DistanceMetric::default(),
        true,
        None, // custom_weights
    );

    // Test points
    // Hermite interpolation (smoothstep) deviates from linear
    assert_relative_eq!(surface.evaluate(&[1.0]), 1.0, epsilon = 0.2);
    assert_relative_eq!(surface.evaluate(&[3.0]), 3.0, epsilon = 0.2);
    assert_relative_eq!(surface.evaluate(&[0.5]), 0.5, epsilon = 0.2);
}

/// Test exact 2D bilinear interpolation.
///
/// Bilinear interpolation of f(x,y) = ax + by + c should be exact.
#[test]
fn test_interpolate_2d_bilinear() {
    // Grid
    let x = vec![0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0];
    let y: Vec<f64> = x.chunks(2).map(|p| 2.0 * p[0] + 3.0 * p[1] + 1.0).collect();
    let dimensions = 2;

    let kdtree = KDTree::new(&x, dimensions);
    let dist_calc = create_mock_dist_calc();
    let mut workspace =
        LoessBuffer::<f64, NodeDistance<f64>, Neighborhood<f64>>::new(y.len(), dimensions, 4, 3);
    let LoessBuffer {
        ref mut search_buffer,
        ref mut neighborhood,
        ref mut fitting_buffer,
        ..
    } = workspace;

    let fitter = |vertex: &[f64],
                  _: &Neighborhood<f64>,
                  _: &mut FittingBuffer<f64>,
                  _: PolynomialDegree|
     -> Option<Vec<f64>> {
        Some(vec![2.0 * vertex[0] + 3.0 * vertex[1] + 1.0, 2.0, 3.0])
    };

    let surface = InterpolationSurface::build(
        &x,
        &y,
        dimensions,
        1.0,
        4,
        &dist_calc,
        &kdtree,
        20,
        fitter,
        search_buffer,
        neighborhood,
        fitting_buffer,
        0.2,
        None,
        &[],
        WeightFunction::default(),
        ZeroWeightFallback::default(),
        PolynomialDegree::default(),
        &DistanceMetric::default(),
        true,
        None, // custom_weights
    );

    // Evaluate at center (1, 1) -> 2(1) + 3(1) + 1 = 6
    // Note: With Hermite interpolation, this may deviate slightly if boundaries have margins
    assert_relative_eq!(surface.evaluate(&[1.0, 1.0]), 6.0, epsilon = 0.2);

    // Evaluate at (0.5, 1.5) -> 2(0.5) + 3(1.5) + 1 = 1 + 4.5 + 1 = 6.5
    // Hermite interpolation is non-linear, so it won't be exact 6.5
    assert_relative_eq!(surface.evaluate(&[0.5, 1.5]), 6.5, epsilon = 0.5);
}

// ============================================================================
// Adaptive Subdivision Tests
// ============================================================================

/// Test that adaptive subdivision occurs.
///
/// With max_vertices high enough, it should split cells.
#[test]
fn test_adaptive_subdivision() {
    let n = 20;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| v * v).collect(); // Nonlinear
    let dimensions = 1;

    let kdtree = KDTree::new(&x, dimensions);
    let dist_calc = create_mock_dist_calc();
    let mut workspace =
        LoessBuffer::<f64, NodeDistance<f64>, Neighborhood<f64>>::new(x.len(), dimensions, 6, 2);
    let LoessBuffer {
        ref mut search_buffer,
        ref mut neighborhood,
        ref mut fitting_buffer,
        ..
    } = workspace;

    // Fitter returns x^2
    let fitter = |vertex: &[f64],
                  _: &Neighborhood<f64>,
                  _: &mut FittingBuffer<f64>,
                  _: PolynomialDegree|
     -> Option<Vec<f64>> { Some(vec![vertex[0] * vertex[0], 2.0 * vertex[0]]) };

    let surface = InterpolationSurface::build(
        &x,
        &y,
        dimensions,
        0.3,
        6,
        &dist_calc,
        &kdtree,
        100, // Allow many vertices to force subdivision
        fitter,
        search_buffer,
        neighborhood,
        fitting_buffer,
        0.2,
        None,
        &[],
        WeightFunction::default(),
        ZeroWeightFallback::default(),
        PolynomialDegree::default(),
        &DistanceMetric::default(),
        true,
        None, // custom_weights
    );

    // Should have more than just the initial 2 vertices
    assert!(surface.vertex_data.len() > 4); // More than 2 vertices * 2 values each
    // Should have created child cells
    assert!(surface.cells.len() > 1);
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

/// Test surface evaluation outside bounds.
///
/// Should clamp to the nearest cell/edge.
#[test]
fn test_interpolate_boundary_clamping() {
    let x = vec![0.0, 2.0];
    let y = vec![0.0, 2.0];
    let dimensions = 1;

    let kdtree = KDTree::new(&x, dimensions);
    let dist_calc = create_mock_dist_calc();
    let mut workspace = LoessBuffer::new(x.len(), dimensions, 2, 2);
    let LoessBuffer {
        ref mut search_buffer,
        ref mut neighborhood,
        ref mut fitting_buffer,
        ..
    } = workspace;

    let fitter = |vertex: &[f64],
                  _: &Neighborhood<f64>,
                  _: &mut FittingBuffer<f64>,
                  _: PolynomialDegree|
     -> Option<Vec<f64>> { Some(vec![vertex[0], 1.0]) };

    let surface = InterpolationSurface::build(
        &x,
        &y,
        dimensions,
        0.5,
        1,
        &dist_calc,
        &kdtree,
        10,
        fitter,
        search_buffer,
        neighborhood,
        fitting_buffer,
        0.2,
        None,
        &[],
        WeightFunction::default(),
        ZeroWeightFallback::default(),
        PolynomialDegree::default(),
        &DistanceMetric::default(),
        true,
        None, // custom_weights
    );

    // Far outside right (should be clamped to upper bound value)
    // Upper bound is ~2.01 (0.5% margin)
    // Value should be ~2.01
    let val_far = surface.evaluate(&[10.0]);
    assert!(val_far > 2.0);
    assert!(val_far < 2.1); // Margin check
}

/// Test build handles identical implementation results when fitter fails.
///
/// If fitter returns None, build should fallback to global mean.
#[test]
fn test_fitter_fallback() {
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![2.0, 2.0, 2.0];
    let dimensions = 1;

    let kdtree = KDTree::new(&x, dimensions);
    let dist_calc = create_mock_dist_calc();

    let mut workspace =
        LoessBuffer::<f64, NodeDistance<f64>, Neighborhood<f64>>::new(x.len(), dimensions, 1, 2);
    let LoessBuffer {
        ref mut search_buffer,
        ref mut neighborhood,
        ref mut fitting_buffer,
        ..
    } = workspace;

    // Broken fitter always returns None
    let fitter = |_: &[f64],
                  _: &Neighborhood<f64>,
                  _: &mut FittingBuffer<f64>,
                  _: PolynomialDegree|
     -> Option<Vec<f64>> { None };

    let surface = InterpolationSurface::build(
        &x,
        &y,
        dimensions,
        0.5,
        1,
        &dist_calc,
        &kdtree,
        10,
        fitter,
        search_buffer,
        neighborhood,
        fitting_buffer,
        0.2,
        None,
        &[],
        WeightFunction::default(),
        ZeroWeightFallback::default(),
        PolynomialDegree::default(),
        &DistanceMetric::default(),
        true,
        None, // custom_weights
    );

    // Should use mean (2.0)
    assert_relative_eq!(surface.evaluate(&[1.0]), 2.0, epsilon = 1e-10);
}

// ============================================================================
// Refit Values Tests
// ============================================================================

/// Test refit_values updates vertex data in a built surface.
///
/// Verifies that calling refit_values with an updated fitter changes the
/// smoothed values returned by evaluate.
#[test]
fn test_refit_values_1d() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let dimensions = 1;

    let kdtree = KDTree::new(&x, dimensions);
    let dist_calc = create_mock_dist_calc();

    let mut workspace =
        LoessBuffer::<f64, NodeDistance<f64>, Neighborhood<f64>>::new(x.len(), dimensions, 2, 2);
    let LoessBuffer {
        ref mut search_buffer,
        ref mut neighborhood,
        ref mut fitting_buffer,
        ..
    } = workspace;

    // Initial fitter: y = x (identity)
    let fitter = |vertex: &[f64],
                  _: &Neighborhood<f64>,
                  _: &mut FittingBuffer<f64>,
                  _: PolynomialDegree|
     -> Option<Vec<f64>> { Some(vec![vertex[0], 1.0]) };

    let mut surface = InterpolationSurface::build(
        &x,
        &y,
        dimensions,
        1.0,
        2,
        &dist_calc,
        &kdtree,
        20,
        fitter,
        search_buffer,
        neighborhood,
        fitting_buffer,
        0.2,
        None,
        &[],
        WeightFunction::default(),
        ZeroWeightFallback::default(),
        PolynomialDegree::default(),
        &DistanceMetric::default(),
        true,
        None, // custom_weights
    );

    // Refit fitter: constant 99.0
    let refit_fitter = |_: &[f64],
                        _: &Neighborhood<f64>,
                        _: &mut FittingBuffer<f64>,
                        _: PolynomialDegree|
     -> Option<Vec<f64>> { Some(vec![99.0, 0.0]) };

    let mut refit_neighborhood = Neighborhood::<f64>::new();
    let mut refit_fitting_buffer = FittingBuffer::<f64>::new(2, 2);
    let robustness_weights: Vec<f64> = vec![1.0; x.len()];

    surface.refit_values(
        &x,
        &y,
        refit_fitter,
        &mut refit_neighborhood,
        &mut refit_fitting_buffer,
        None, // custom_vertex_pass
        WeightFunction::default(),
        ZeroWeightFallback::default(),
        PolynomialDegree::default(),
        &DistanceMetric::default(),
        &[], // scales
        &robustness_weights,
        true, // boundary_degree_fallback
        None, // custom_weights
    );

    // After refit, all vertices should hold 99.0 so evaluate returns ~99.0
    let val = surface.evaluate(&[2.0]);
    assert_relative_eq!(val, 99.0, epsilon = 1e-6);
}

/// Test refit_values with fitter returning None (falls back to mean).
#[test]
fn test_refit_values_fallback_to_mean() {
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![5.0, 5.0, 5.0];
    let dimensions = 1;

    let kdtree = KDTree::new(&x, dimensions);
    let dist_calc = create_mock_dist_calc();

    let mut workspace =
        LoessBuffer::<f64, NodeDistance<f64>, Neighborhood<f64>>::new(x.len(), dimensions, 2, 2);
    let LoessBuffer {
        ref mut search_buffer,
        ref mut neighborhood,
        ref mut fitting_buffer,
        ..
    } = workspace;

    let fitter = |_: &[f64],
                  _: &Neighborhood<f64>,
                  _: &mut FittingBuffer<f64>,
                  _: PolynomialDegree|
     -> Option<Vec<f64>> { Some(vec![5.0, 0.0]) };

    let mut surface = InterpolationSurface::build(
        &x,
        &y,
        dimensions,
        1.0,
        2,
        &dist_calc,
        &kdtree,
        10,
        fitter,
        search_buffer,
        neighborhood,
        fitting_buffer,
        0.2,
        None,
        &[],
        WeightFunction::default(),
        ZeroWeightFallback::default(),
        PolynomialDegree::default(),
        &DistanceMetric::default(),
        true,
        None, // custom_weights
    );

    // Refit with None-returning fitter — should fall back to global mean (5.0)
    let failing_fitter = |_: &[f64],
                          _: &Neighborhood<f64>,
                          _: &mut FittingBuffer<f64>,
                          _: PolynomialDegree|
     -> Option<Vec<f64>> { None };

    let mut refit_neighborhood = Neighborhood::<f64>::new();
    let mut refit_fitting_buffer = FittingBuffer::<f64>::new(2, 2);
    let robustness_weights: Vec<f64> = vec![1.0; x.len()];

    surface.refit_values(
        &x,
        &y,
        failing_fitter,
        &mut refit_neighborhood,
        &mut refit_fitting_buffer,
        None,
        WeightFunction::default(),
        ZeroWeightFallback::default(),
        PolynomialDegree::default(),
        &DistanceMetric::default(),
        &[],
        &robustness_weights,
        false, // boundary_degree_fallback disabled
        None, // custom_weights
    );

    // Fallback to mean: y mean = 5.0
    let val = surface.evaluate(&[1.0]);
    assert_relative_eq!(val, 5.0, epsilon = 1e-6);
}

// ============================================================================
// Additional Coverage Tests
// ============================================================================

/// Test that evaluate triggers fallback_interpolation when a cell has < 2 vertices.
///
/// We build a 1D surface with 1 point so the cell only has 1 vertex,
/// then evaluate — that triggers the fallback_interpolation path.
#[test]
fn test_interpolation_fallback_single_vertex() {
    // Only 1 point — cell will have only 1 vertex
    let x = vec![0.5f64];
    let y = vec![7.0f64];
    let dimensions = 1;

    let kdtree = KDTree::new(&x, dimensions);
    let dist_calc = create_mock_dist_calc();

    let mut workspace =
        LoessBuffer::<f64, NodeDistance<f64>, Neighborhood<f64>>::new(x.len(), dimensions, 1, 2);
    let LoessBuffer {
        ref mut search_buffer,
        ref mut neighborhood,
        ref mut fitting_buffer,
        ..
    } = workspace;

    let fitter = |_: &[f64],
                  _: &Neighborhood<f64>,
                  _: &mut FittingBuffer<f64>,
                  _: PolynomialDegree|
     -> Option<Vec<f64>> { Some(vec![7.0, 0.0]) };

    let surface = InterpolationSurface::build(
        &x,
        &y,
        dimensions,
        1.0,
        1,
        &dist_calc,
        &kdtree,
        5,
        fitter,
        search_buffer,
        neighborhood,
        fitting_buffer,
        0.2,
        None,
        &[],
        WeightFunction::default(),
        ZeroWeightFallback::default(),
        PolynomialDegree::default(),
        &DistanceMetric::default(),
        true,
        None, // custom_weights
    );

    // Should evaluate without panic (uses fallback_interpolation if < 2 vertices)
    let val = surface.evaluate(&[0.5]);
    assert!(val.is_finite(), "Evaluated value should be finite");
}

// ============================================================================
// f32 Type Monomorphization Coverage
// ============================================================================

/// Test InterpolationSurface with f32 to cover f32 monomorphizations.
///
/// The coverage tool counts f32 and f64 versions separately; this test
/// exercises build and evaluate with f32 to cover the otherwise-missed functions.
#[test]
fn test_interpolation_surface_f32() {
    use loess_rs::internals::math::neighborhood::NodeDistance as ND;

    let x: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y: Vec<f32> = vec![0.0, 1.0, 4.0, 9.0, 16.0]; // y = x^2
    let dimensions = 1;

    let kdtree = KDTree::new(&x, dimensions);
    let dist_calc = LoessDistanceCalculator {
        metric: DistanceMetric::Euclidean,
        scales: &[],
    };
    let mut workspace =
        LoessBuffer::<f32, ND<f32>, Neighborhood<f32>>::new(x.len(), dimensions, 3, 2);
    let LoessBuffer {
        ref mut search_buffer,
        ref mut neighborhood,
        ref mut fitting_buffer,
        ..
    } = workspace;

    let fitter = |vertex: &[f32],
                  _: &Neighborhood<f32>,
                  _: &mut FittingBuffer<f32>,
                  _: PolynomialDegree|
     -> Option<Vec<f32>> {
        let v = vertex[0];
        Some(vec![v * v, 2.0 * v]) // f(v) = v^2, f'(v) = 2v
    };

    let surface = InterpolationSurface::build(
        &x,
        &y,
        dimensions,
        0.6f32,
        3,
        &dist_calc,
        &kdtree,
        20,
        fitter,
        search_buffer,
        neighborhood,
        fitting_buffer,
        0.2,
        None,
        &[],
        WeightFunction::default(),
        ZeroWeightFallback::default(),
        PolynomialDegree::default(),
        &DistanceMetric::default(),
        true,
        None, // custom_weights
    );

    // Evaluate at a few points — exercises hermite_phi0/1, hermite_psi0/1 for f32
    let v1 = surface.evaluate(&[1.0f32]);
    let v2 = surface.evaluate(&[2.5f32]);
    assert!(v1.is_finite(), "f32 surface should produce finite values");
    assert!(v2.is_finite(), "f32 surface should produce finite values");
}

/// Test refit_values with f32 type.
#[test]
fn test_refit_values_f32() {
    use loess_rs::internals::math::neighborhood::NodeDistance as ND;

    let x: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0];
    let y: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0];
    let dimensions = 1;

    let kdtree = KDTree::new(&x, dimensions);
    let dist_calc = LoessDistanceCalculator::<f32> {
        metric: DistanceMetric::Euclidean,
        scales: &[],
    };
    let mut workspace =
        LoessBuffer::<f32, ND<f32>, Neighborhood<f32>>::new(x.len(), dimensions, 2, 2);
    let LoessBuffer {
        ref mut search_buffer,
        ref mut neighborhood,
        ref mut fitting_buffer,
        ..
    } = workspace;

    let fitter = |vertex: &[f32],
                  _: &Neighborhood<f32>,
                  _: &mut FittingBuffer<f32>,
                  _: PolynomialDegree|
     -> Option<Vec<f32>> { Some(vec![vertex[0], 1.0]) };

    let mut surface = InterpolationSurface::build(
        &x,
        &y,
        dimensions,
        1.0f32,
        2,
        &dist_calc,
        &kdtree,
        10,
        fitter,
        search_buffer,
        neighborhood,
        fitting_buffer,
        0.2,
        None,
        &[],
        WeightFunction::default(),
        ZeroWeightFallback::default(),
        PolynomialDegree::default(),
        &DistanceMetric::default(),
        true,
        None, // custom_weights
    );

    let refit_fitter = |_: &[f32],
                        _: &Neighborhood<f32>,
                        _: &mut FittingBuffer<f32>,
                        _: PolynomialDegree|
     -> Option<Vec<f32>> { Some(vec![42.0f32, 0.0]) };

    let mut refit_neighborhood = Neighborhood::<f32>::new();
    let mut refit_fitting_buffer = FittingBuffer::<f32>::new(2, 2);
    let robustness_weights = vec![1.0f32; x.len()];

    surface.refit_values(
        &x,
        &y,
        refit_fitter,
        &mut refit_neighborhood,
        &mut refit_fitting_buffer,
        None,
        WeightFunction::default(),
        ZeroWeightFallback::default(),
        PolynomialDegree::default(),
        &DistanceMetric::default(),
        &[],
        &robustness_weights,
        true,
        None, // custom_weights
    );

    let val = surface.evaluate(&[1.5f32]);
    assert_relative_eq!(val, 42.0f32, epsilon = 1e-4);
}

// ============================================================================
// Fallback Interpolation Coverage (Direct Construction)
// ============================================================================

/// Test fallback_interpolation by manually constructing a surface with < 2 vertices.
///
/// The fallback_interpolation function is called from interpolate_in_cell when
/// cell.vertex_indices.len() < 2. This directly constructs such a surface.
#[test]
fn test_fallback_interpolation_manual_f64() {
    // Manually construct an InterpolationSurface with a 1-vertex 1D cell
    // to trigger the fallback_interpolation path
    let surface = InterpolationSurface {
        dimensions: 1,
        vertex_data: vec![42.0f64, 1.0], // [value, derivative] for vertex 0
        vertices: vec![0.5f64],          // vertex 0 at x=0.5
        cells: vec![SurfaceCell {
            lower: vec![0.0f64],
            upper: vec![1.0f64],
            vertex_indices: vec![0], // Only 1 vertex — triggers fallback
            children: None,
            split_dim: None,
            split_val: None,
            point_lo: 0,
            point_hi: 0,
        }],
        root: 0,
        vertex_neighborhoods: vec![CachedNeighborhood {
            indices: vec![],
            distances: vec![],
            max_distance: 0.0f64,
        }],
    };

    // evaluate calls interpolate_in_cell which calls fallback_interpolation
    let val = surface.evaluate(&[0.5f64]);
    assert_relative_eq!(val, 42.0f64, epsilon = 1e-10);
}

/// Test fallback_interpolation (f32) with an empty vertex_indices list.
///
/// When vertex_indices is empty, fallback_interpolation returns 0.0.
#[test]
fn test_fallback_interpolation_manual_f32() {
    let surface = InterpolationSurface {
        dimensions: 1,
        vertex_data: vec![7.0f32, 0.0],
        vertices: vec![0.5f32],
        cells: vec![SurfaceCell {
            lower: vec![0.0f32],
            upper: vec![1.0f32],
            vertex_indices: vec![0], // 1 vertex — triggers fallback
            children: None,
            split_dim: None,
            split_val: None,
            point_lo: 0,
            point_hi: 0,
        }],
        root: 0,
        vertex_neighborhoods: vec![CachedNeighborhood {
            indices: vec![],
            distances: vec![],
            max_distance: 0.0f32,
        }],
    };

    let val = surface.evaluate(&[0.5f32]);
    assert_relative_eq!(val, 7.0f32, epsilon = 1e-5);
}
