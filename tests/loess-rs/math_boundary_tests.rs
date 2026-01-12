#![cfg(feature = "dev")]

use loess_rs::internals::api::{Adapter, BoundaryPolicy, LoessBuilder as Loess, WeightFunction};
use loess_rs::internals::math::boundary::BoundaryPolicy as BoundaryPolicyInternal;

#[test]
fn test_boundary_policy_comparison() {
    // Dataset where edge points are sensitive to padding
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![10.0, 20.0, 10.0, 20.0, 10.0, 20.0];

    // Use fraction(0.8) to get q=4, so that h is large enough to include padded points
    let base_builder = Loess::new()
        .fraction(0.8)
        .iterations(0)
        .weight_function(WeightFunction::Uniform);

    // Fit with standard Extend (default)
    let res_extend = base_builder
        .clone()
        .boundary_policy(BoundaryPolicy::Extend)
        .adapter(Adapter::Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Fit with Reflect
    let res_reflect = base_builder
        .clone()
        .boundary_policy(BoundaryPolicy::Reflect)
        .adapter(Adapter::Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Fit with Zero
    let res_zero = base_builder
        .clone()
        .boundary_policy(BoundaryPolicy::Zero)
        .adapter(Adapter::Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert_ne!(
        res_extend.y[0], res_reflect.y[0],
        "Extend vs Reflect at edge"
    );
    assert_ne!(res_extend.y[0], res_zero.y[0], "Extend vs Zero at edge");
    assert_ne!(res_reflect.y[0], res_zero.y[0], "Reflect vs Zero at edge");
}

#[test]
fn test_boundary_policy_zero_effect() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![100.0, 100.0, 100.0, 100.0, 100.0, 100.0];

    let res_zero = Loess::new()
        .fraction(0.8)
        .iterations(0)
        .weight_function(WeightFunction::Uniform)
        .boundary_policy(BoundaryPolicy::Zero)
        .adapter(Adapter::Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert!(
        res_zero.y[2] > res_zero.y[0],
        "Center should be higher than Zero-padded edge"
    );
}

#[test]
fn test_boundary_minimal_dataset() {
    let x = vec![0.0, 1.0];
    let y = vec![10.0, 20.0];

    let (px, py, _) = BoundaryPolicyInternal::Extend.apply(&x, &y, 1, 2);

    assert_eq!(px.len(), 4);
    assert_eq!(py.len(), 4);
    assert_eq!(px[0], -1.0);
    assert_eq!(px[3], 2.0);
    assert_eq!(py[0], 10.0);
    assert_eq!(py[3], 20.0);
}

#[test]
fn test_boundary_large_window() {
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![10.0, 20.0, 30.0];

    let (px, _, _) = BoundaryPolicyInternal::Extend.apply(&x, &y, 1, 10);

    assert_eq!(px.len(), 7);
}

#[test]
fn test_boundary_zero_dx() {
    let x = vec![1.0, 1.0, 2.0, 2.0];
    let y = vec![10.0, 20.0, 30.0, 40.0];

    let (px, _, _) = BoundaryPolicyInternal::Extend.apply(&x, &y, 1, 4);

    assert_eq!(px[0], 1.0);
    assert_eq!(px[1], 1.0);
}

#[test]
fn test_boundary_small_window() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];

    let (px, _, _) = BoundaryPolicyInternal::Reflect.apply(&x, &y, 1, 2);

    assert_eq!(px.len(), 7);
    assert_eq!(px[0], -1.0);
    assert_eq!(px[6], 5.0);
}

#[test]
fn test_boundary_no_boundary_policy() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];

    let (px, py, mapping) = BoundaryPolicyInternal::NoBoundary.apply(&x, &y, 1, 3);

    assert_eq!(px.len(), x.len());
    assert_eq!(py.len(), y.len());
    assert_eq!(px, x);
    assert_eq!(py, y);
    assert_eq!(mapping.len(), x.len());
}

#[test]
fn test_boundary_extend_2d() {
    let x = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
    let y = vec![10.0, 20.0, 30.0];

    let (px, py, _) = BoundaryPolicyInternal::Extend.apply(&x, &y, 2, 4);

    assert!(px.len() >= x.len());
    assert!(py.len() >= y.len());
}

#[test]
fn test_boundary_reflect_2d() {
    let x = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
    let y = vec![10.0, 20.0, 30.0, 40.0];

    let (px, py, _) = BoundaryPolicyInternal::Reflect.apply(&x, &y, 2, 4);

    assert!(px.len() >= x.len());
    assert!(py.len() >= y.len());
}

#[test]
fn test_boundary_extend_3d() {
    let x = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
    let y = vec![10.0, 20.0, 30.0];

    let (px, py, mapping) = BoundaryPolicyInternal::Extend.apply(&x, &y, 3, 4);

    assert!(px.len() >= x.len());
    assert!(py.len() >= y.len());
    assert_eq!(mapping.len(), py.len());
}

#[test]
fn test_boundary_mapping_correctness() {
    let x = vec![0.0, 1.0, 2.0, 3.0];
    let y = vec![10.0, 20.0, 30.0, 40.0];

    let (_, _, mapping) = BoundaryPolicyInternal::Extend.apply(&x, &y, 1, 3);

    assert_eq!(mapping[0], 0);
    assert_eq!(mapping[mapping.len() - 1], y.len() - 1);
}

#[test]
fn test_boundary_extend_uniform_spacing() {
    let x = vec![0.0, 2.0, 4.0, 6.0, 8.0];
    let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];

    let (px, _, _) = BoundaryPolicyInternal::Extend.apply(&x, &y, 1, 3);

    assert_eq!(px[0], -2.0);
    assert_eq!(px[px.len() - 1], 10.0);
}

#[test]
fn test_boundary_extend_non_uniform_spacing() {
    let x = vec![0.0, 1.0, 3.0, 6.0, 10.0];
    let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];

    let (px, _, _) = BoundaryPolicyInternal::Extend.apply(&x, &y, 1, 3);

    assert_eq!(px[0], -1.0);
    assert_eq!(px[px.len() - 1], 14.0);
}

#[test]
fn test_boundary_single_point() {
    let x = vec![5.0];
    let y = vec![10.0];

    let (px, py, _) = BoundaryPolicyInternal::Extend.apply(&x, &y, 1, 2);

    assert_eq!(px.len(), 1);
    assert_eq!(py.len(), 1);
}

#[test]
fn test_boundary_window_size_one() {
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![10.0, 20.0, 30.0];

    let (px, py, _) = BoundaryPolicyInternal::Extend.apply(&x, &y, 1, 1);

    assert_eq!(px.len(), x.len());
    assert_eq!(py.len(), y.len());
}

#[test]
fn test_boundary_large_dataset() {
    let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let y: Vec<f64> = (0..100).map(|i| (i * 2) as f64).collect();

    let (px, py, mapping) = BoundaryPolicyInternal::Extend.apply(&x, &y, 1, 20);

    assert!(px.len() > x.len());
    assert!(py.len() > y.len());
    assert_eq!(mapping.len(), py.len());
}

#[test]
fn test_boundary_f32_precision() {
    let x = vec![0.0f32, 1.0, 2.0, 3.0];
    let y = vec![10.0f32, 20.0, 30.0, 40.0];

    let (px, py, _) = BoundaryPolicyInternal::Extend.apply(&x, &y, 1, 3);

    assert!(px.len() > x.len());
    assert!(py.len() > y.len());
    assert!(px[0].is_finite());
    assert!(py[0].is_finite());
}

#[test]
fn test_boundary_negative_values() {
    let x = vec![-5.0, -3.0, -1.0, 1.0, 3.0];
    let y = vec![-10.0, -20.0, -30.0, -40.0, -50.0];

    let (px, py, _) = BoundaryPolicyInternal::Reflect.apply(&x, &y, 1, 3);

    assert!(px.len() > x.len());
    assert!(py.len() > y.len());
    assert!(px[0] < x[0]);
}

#[test]
fn test_boundary_policy_enum_equality() {
    assert_eq!(
        BoundaryPolicyInternal::Extend,
        BoundaryPolicyInternal::Extend
    );
    assert_ne!(
        BoundaryPolicyInternal::Extend,
        BoundaryPolicyInternal::Reflect
    );
    assert_ne!(
        BoundaryPolicyInternal::Zero,
        BoundaryPolicyInternal::NoBoundary
    );
}

#[test]
fn test_boundary_policy_default() {
    let default_policy = BoundaryPolicyInternal::default();
    assert_eq!(default_policy, BoundaryPolicyInternal::Extend);
}
