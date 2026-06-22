//! Generic Solvers and Term Generation
//!
//! ## Design notes
//!
//! * Uses a fixed stack buffer (size 64) for terms during generic accumulation
//!   to avoid heap allocations in the inner loop. This imposes a practical
//!   limit on the complexity of generic regression (e.g., high-dimensional
//!   quadratic fits).
//!
//! ## Features
//!
//! - Trait-based term generation for different polynomial degrees.
//! - Generic accumulation logic for arbitrary dimensions.

// External dependencies
use core::marker::PhantomData;
use num_traits::Float;

// Module dependencies
use super::types::PolynomialDegree;

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
// Term Generators (Strategy Pattern)
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

/// Trait for generating polynomial terms for a given point.
/// This abstracts the dimensionality and degree logic from the solver.
pub trait TermGenerator<T: Float> {
    /// Returns the number of coefficients (terms) generated.
    fn n_coeffs(&self) -> usize;

    /// Generates terms for a single point relative to the query point.
    /// Writes terms into `out`, which must have length >= `n_coeffs()`.
    fn generate(&self, point: &[T], query: &[T], out: &mut [T]);
}

/// Generator for N-dimensional generic polynomial terms.
pub struct GenericTermGenerator<'a, T: Float> {
    degree: PolynomialDegree,
    _d: usize,
    n_coeffs: usize,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T: Float> GenericTermGenerator<'a, T> {
    /// Create a new generic term generator.
    pub fn new(degree: PolynomialDegree, d: usize, n_coeffs: usize) -> Self {
        Self {
            degree,
            _d: d,
            n_coeffs,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: Float> TermGenerator<T> for GenericTermGenerator<'a, T> {
    #[inline(always)]
    fn n_coeffs(&self) -> usize {
        self.n_coeffs
    }

    #[inline(always)]
    fn generate(&self, point: &[T], query: &[T], out: &mut [T]) {
        self.degree.build_terms(point, query, out);
    }
}

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
// Generic Accumulation
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

/// Helper to accumulate Normal Equations using a Generator.
#[allow(clippy::too_many_arguments)]
pub fn accumulate_normal_equations<T: Float, G: TermGenerator<T>>(
    x: &[T],
    y: &[T],
    indices: &[usize],
    weights: &[T],
    generator: G,
    query: &[T],
    xtwx: &mut [T],
    xtwy: &mut [T],
) {
    let n = indices.len();
    let n_coeffs = generator.n_coeffs();
    let dimensions = query.len();

    // Use a small stack buffer for terms to avoid allocation
    let mut terms = [T::zero(); 64];
    assert!(n_coeffs <= 64, "Too many coefficients for stack buffer");

    for i in 0..n {
        let w = weights[i];
        if w <= T::epsilon() {
            continue;
        }

        let idx = indices[i];
        let point = &x[idx * dimensions..(idx + 1) * dimensions];
        let y_val = y[idx];

        generator.generate(point, query, &mut terms[..n_coeffs]);

        // Accumulate X'WX
        for j in 0..n_coeffs {
            let w_tj = w * terms[j];
            for k in j..n_coeffs {
                xtwx[j * n_coeffs + k] = xtwx[j * n_coeffs + k] + w_tj * terms[k];
            }
            // Accumulate X'Wy
            xtwy[j] = xtwy[j] + w_tj * y_val;
        }
    }

    // Fill symmetric part of xtwx
    for j in 0..n_coeffs {
        for k in 0..j {
            xtwx[j * n_coeffs + k] = xtwx[k * n_coeffs + j];
        }
    }
}
