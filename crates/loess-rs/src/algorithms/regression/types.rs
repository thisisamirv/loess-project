//! Regression Types
//!
//! ## Purpose
//!
//! This module defines the core data types and configuration enums used in
//! regression fitting, such as `PolynomialDegree` and `ZeroWeightFallback`.

// External dependencies
use num_traits::Float;

// ============================================================================
// Polynomial Degree
// ============================================================================

/// Polynomial degree for local regression fitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PolynomialDegree {
    /// Degree 0: Local constant (weighted mean)
    Constant,

    /// Degree 1: Local linear regression (default)
    #[default]
    Linear,

    /// Degree 2: Local quadratic regression
    Quadratic,

    /// Degree 3: Local cubic regression
    Cubic,

    /// Degree 4: Local quartic regression
    Quartic,
}

impl PolynomialDegree {
    /// Get the numeric degree value.
    #[inline]
    pub const fn value(&self) -> usize {
        match self {
            PolynomialDegree::Constant => 0,
            PolynomialDegree::Linear => 1,
            PolynomialDegree::Quadratic => 2,
            PolynomialDegree::Cubic => 3,
            PolynomialDegree::Quartic => 4,
        }
    }

    /// Number of coefficients for 1D polynomial of this degree.
    #[inline]
    pub const fn num_coefficients_1d(&self) -> usize {
        self.value() + 1
    }

    /// Number of coefficients for nD polynomial of this degree.
    #[inline]
    pub const fn num_coefficients_nd(&self, dimensions: usize) -> usize {
        match self {
            PolynomialDegree::Constant => 1,
            PolynomialDegree::Linear => 1 + dimensions,
            PolynomialDegree::Quadratic => 1 + dimensions + (dimensions * (dimensions + 1)) / 2,
            PolynomialDegree::Cubic => {
                let n = dimensions;
                (n + 3) * (n + 2) * (n + 1) / 6
            }
            PolynomialDegree::Quartic => {
                let n = dimensions;
                (n + 4) * (n + 3) * (n + 2) * (n + 1) / 24
            }
        }
    }

    /// Build polynomial terms for a point relative to center.
    pub fn build_terms<T: Float>(&self, point: &[T], center: &[T], terms: &mut [T]) -> usize {
        let d = point.len();
        let degree = self.value();

        // Intercept
        terms[0] = T::one();
        if degree == 0 {
            return 1;
        }

        // Special case 1D for speed
        if d == 1 {
            let x = point[0] - center[0];
            terms[1] = x;
            if degree == 1 {
                return 2;
            }
            terms[2] = x * x;
            if degree == 2 {
                return 3;
            }
            terms[3] = x * x * x;
            if degree == 3 {
                return 4;
            }
            terms[4] = x * x * x * x;
            return 5;
        }

        // Special case 2D for speed
        if d == 2 {
            let x = point[0] - center[0];
            let y = point[1] - center[1];
            terms[1] = x;
            terms[2] = y;
            if degree == 1 {
                return 3;
            }

            // Quadratic: x^2, xy, y^2
            terms[3] = x * x;
            terms[4] = x * y;
            terms[5] = y * y;
            if degree == 2 {
                return 6;
            }

            // Cubic: x^3, x^2y, xy^2, y^3
            terms[6] = x * x * x;
            terms[7] = x * x * y;
            terms[8] = x * y * y;
            terms[9] = y * y * y;
            if degree == 3 {
                return 10;
            }

            // Quartic: x^4, x^3y, x^2y^2, xy^3, y^4
            terms[10] = x * x * x * x;
            terms[11] = x * x * x * y;
            terms[12] = x * x * y * y;
            terms[13] = x * y * y * y;
            terms[14] = y * y * y * y;
            return 15;
        }

        // General nD case
        let mut count = 1;

        // Linear terms (and store centered values for higher degrees)
        for i in 0..d {
            let val = point[i] - center[i];
            terms[count] = val;
            count += 1;
        }

        if degree == 1 {
            return count;
        }

        // Quadratic
        for i in 0..d {
            for j in i..d {
                terms[count] = terms[1 + i] * terms[1 + j];
                count += 1;
            }
        }
        if degree == 2 {
            return count;
        }

        // Cubic
        for i in 0..d {
            for j in i..d {
                for k in j..d {
                    terms[count] = terms[1 + i] * terms[1 + j] * terms[1 + k];
                    count += 1;
                }
            }
        }
        if degree == 3 {
            return count;
        }

        // Quartic
        for i in 0..d {
            for j in i..d {
                for k in j..d {
                    for l in k..d {
                        terms[count] = terms[1 + i] * terms[1 + j] * terms[1 + k] * terms[1 + l];
                        count += 1;
                    }
                }
            }
        }
        count
    }
}

// ============================================================================
// Zero-Weight Fallback Policy
// ============================================================================

/// Policy for handling cases where all weights are zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZeroWeightFallback {
    /// Use local mean (default).
    #[default]
    UseLocalMean,

    /// Return the original y-value.
    ReturnOriginal,

    /// Return None (propagate failure).
    ReturnNone,
}

impl ZeroWeightFallback {
    /// Create from u8 flag.
    #[inline]
    pub fn from_u8(flag: u8) -> Self {
        match flag {
            0 => ZeroWeightFallback::UseLocalMean,
            1 => ZeroWeightFallback::ReturnOriginal,
            2 => ZeroWeightFallback::ReturnNone,
            _ => ZeroWeightFallback::UseLocalMean,
        }
    }

    /// Convert to u8 flag.
    #[inline]
    pub fn to_u8(self) -> u8 {
        match self {
            ZeroWeightFallback::UseLocalMean => 0,
            ZeroWeightFallback::ReturnOriginal => 1,
            ZeroWeightFallback::ReturnNone => 2,
        }
    }
}
