#![cfg(feature = "dev")]

use loess_rs::internals::algorithms::regression::{PolynomialDegree, ZeroWeightFallback};

// ============================================================================
// PolynomialDegree Tests
// ============================================================================

#[test]
fn test_polynomial_degree_values() {
    assert_eq!(PolynomialDegree::Constant.value(), 0);
    assert_eq!(PolynomialDegree::Linear.value(), 1);
    assert_eq!(PolynomialDegree::Quadratic.value(), 2);
    assert_eq!(PolynomialDegree::Cubic.value(), 3);
    assert_eq!(PolynomialDegree::Quartic.value(), 4);
}

#[test]
fn test_polynomial_degree_coefficients_1d() {
    assert_eq!(PolynomialDegree::Constant.num_coefficients_1d(), 1);
    assert_eq!(PolynomialDegree::Linear.num_coefficients_1d(), 2);
    assert_eq!(PolynomialDegree::Quadratic.num_coefficients_1d(), 3);
    assert_eq!(PolynomialDegree::Cubic.num_coefficients_1d(), 4);
    assert_eq!(PolynomialDegree::Quartic.num_coefficients_1d(), 5);
}

#[test]
fn test_polynomial_degree_coefficients_nd() {
    // 2D
    assert_eq!(PolynomialDegree::Constant.num_coefficients_nd(2), 1);
    assert_eq!(PolynomialDegree::Linear.num_coefficients_nd(2), 3); // 1 + 2
    assert_eq!(PolynomialDegree::Quadratic.num_coefficients_nd(2), 6); // 1 + 2 + 3

    // Cubic 2D: (n+3)(n+2)(n+1)/6 = 5*4*3/6 = 10
    assert_eq!(PolynomialDegree::Cubic.num_coefficients_nd(2), 10);

    // Quartic 2D: (n+4)(n+3)(n+2)(n+1)/24 = 6*5*4*3/24 = 15
    assert_eq!(PolynomialDegree::Quartic.num_coefficients_nd(2), 15);

    // 3D
    assert_eq!(PolynomialDegree::Constant.num_coefficients_nd(3), 1);
    assert_eq!(PolynomialDegree::Linear.num_coefficients_nd(3), 4); // 1 + 3
    assert_eq!(PolynomialDegree::Quadratic.num_coefficients_nd(3), 10); // 1 + 3 + 6
}

#[test]
fn test_build_terms_1d() {
    let degree = PolynomialDegree::Quadratic;
    let point = [2.0];
    let center = [1.0];
    let mut terms = vec![0.0; 3];

    let count = degree.build_terms(&point, &center, &mut terms);

    assert_eq!(count, 3);
    assert_eq!(terms[0], 1.0); // intercept
    assert_eq!(terms[1], 1.0); // x
    assert_eq!(terms[2], 1.0); // x^2
}

#[test]
fn test_build_terms_1d_higher_order() {
    let degree = PolynomialDegree::Quartic;
    let point = [3.0];
    let center = [1.0];
    let mut terms = vec![0.0; 5]; // 1, x, x^2, x^3, x^4

    let count = degree.build_terms(&point, &center, &mut terms);

    // x = 3 - 1 = 2
    assert_eq!(count, 5);
    assert_eq!(terms[0], 1.0);
    assert_eq!(terms[1], 2.0);
    assert_eq!(terms[2], 4.0);
    assert_eq!(terms[3], 8.0);
    assert_eq!(terms[4], 16.0);
}

#[test]
fn test_build_terms_2d_linear() {
    let degree = PolynomialDegree::Linear;
    let point = [2.0, 3.0];
    let center = [1.0, 1.0];
    let mut terms = vec![0.0; 3]; // 1, x, y

    let count = degree.build_terms(&point, &center, &mut terms);

    // x = 1, y = 2
    assert_eq!(count, 3);
    assert_eq!(terms[0], 1.0);
    assert_eq!(terms[1], 1.0);
    assert_eq!(terms[2], 2.0);
}

#[test]
fn test_build_terms_2d_quadratic() {
    let degree = PolynomialDegree::Quadratic;
    let point = [2.0, 3.0];
    let center = [1.0, 1.0];
    let mut terms = vec![0.0; 6]; // 1, x, y, x^2, xy, y^2

    let count = degree.build_terms(&point, &center, &mut terms);

    // x = 1, y = 2
    assert_eq!(count, 6);
    assert_eq!(terms[0], 1.0);
    assert_eq!(terms[1], 1.0);
    assert_eq!(terms[2], 2.0);
    assert_eq!(terms[3], 1.0); // x^2
    assert_eq!(terms[4], 2.0); // xy
    assert_eq!(terms[5], 4.0); // y^2
}

#[test]
fn test_build_terms_nd_linear() {
    let degree = PolynomialDegree::Linear;
    let point = [2.0, 3.0, 4.0];
    let center = [1.0, 1.0, 1.0];
    let mut terms = vec![0.0; 4]; // 1, x, y, z

    let count = degree.build_terms(&point, &center, &mut terms);

    assert_eq!(count, 4);
    assert_eq!(terms[0], 1.0);
    assert_eq!(terms[1], 1.0);
    assert_eq!(terms[2], 2.0);
    assert_eq!(terms[3], 3.0);
}

#[test]
fn test_build_terms_nd_quadratic() {
    // 3D Quadratic
    let degree = PolynomialDegree::Quadratic;
    let point = [2.0, 3.0, 4.0];
    let center = [1.0, 1.0, 1.0];
    // 1 + 3 + 6 = 10 terms
    let mut terms = vec![0.0; 10];

    let count = degree.build_terms(&point, &center, &mut terms);

    // x=1, y=2, z=3
    // Linear: 1, 2, 3 (indices 1,2,3)
    // Quadratic:
    // i=0: x^2(1), xy(2), xz(3)
    // i=1: y^2(4), yz(6)
    // i=2: z^2(9)
    assert_eq!(count, 10);
    assert_eq!(terms[0], 1.0);
    assert_eq!(terms[1], 1.0);
    assert_eq!(terms[2], 2.0);
    assert_eq!(terms[3], 3.0);

    // Check a few quadratic terms
    // terms[4] = x*x = 1
    assert_eq!(terms[4], 1.0);
    // terms[9] = z*z = 9
    assert_eq!(terms[9], 9.0);
}

#[test]
fn test_build_terms_nd_cubic() {
    // 3D Cubic implies generic path since only 1D/2D have special handling
    let degree = PolynomialDegree::Cubic;
    let point = [2.0, 2.0, 2.0];
    let center = [1.0, 1.0, 1.0];
    // 3D Cubic terms: (3+3)(3+2)(3+1)/6 = 6*5*4/6 = 20
    let mut terms = vec![0.0; 20];

    let count = degree.build_terms(&point, &center, &mut terms);

    assert_eq!(count, 20);
    // last term should be z^3 = 1
    assert_eq!(terms[19], 1.0);
}

#[test]
fn test_build_terms_nd_quartic() {
    // 3D Quartic generic path
    let degree = PolynomialDegree::Quartic;
    let point = [2.0, 2.0, 2.0];
    let center = [1.0, 1.0, 1.0];
    // 3D Quartic: (3+4)(3+3)(3+2)(3+1)/24 = 7*6*5*4/24 = 35
    let mut terms = vec![0.0; 35];

    let count = degree.build_terms(&point, &center, &mut terms);

    assert_eq!(count, 35);
    // z^4 = 1
    assert_eq!(terms[34], 1.0);
}

// ============================================================================
// ZeroWeightFallback Tests
// ============================================================================

#[test]
fn test_zero_weight_fallback_u8_conversion() {
    assert_eq!(
        ZeroWeightFallback::UseLocalMean,
        ZeroWeightFallback::from_u8(0)
    );
    assert_eq!(
        ZeroWeightFallback::ReturnOriginal,
        ZeroWeightFallback::from_u8(1)
    );
    assert_eq!(
        ZeroWeightFallback::ReturnNone,
        ZeroWeightFallback::from_u8(2)
    );

    assert_eq!(ZeroWeightFallback::UseLocalMean.to_u8(), 0);
    assert_eq!(ZeroWeightFallback::ReturnOriginal.to_u8(), 1);
    assert_eq!(ZeroWeightFallback::ReturnNone.to_u8(), 2);

    assert_eq!(
        ZeroWeightFallback::from_u8(99),
        ZeroWeightFallback::UseLocalMean
    ); // Default
}

#[test]
fn test_zero_weight_fallback_default() {
    assert_eq!(
        ZeroWeightFallback::default(),
        ZeroWeightFallback::UseLocalMean
    );
}

#[test]
fn test_zero_weight_fallback_properties() {
    let a = ZeroWeightFallback::UseLocalMean;
    let b = a;
    assert_eq!(a, b);
    assert_ne!(a, ZeroWeightFallback::ReturnOriginal);
}
