//! # LOESS — Locally Estimated Scatterplot Smoothing for Rust
//!
//! The fastest, most robust, and most feature-complete language-agnostic
//! LOESS (Locally Estimated Scatterplot Smoothing) implementation for **Rust**.
//!
//! ## What is LOESS?
//!
//! LOESS (Locally Estimated Scatterplot Smoothing) is a nonparametric regression
//! method that fits smooth curves through scatter plots. At each point, it fits
//! a weighted polynomial (typically linear or quadratic) using nearby data points,
//! with weights decreasing smoothly with distance. This creates flexible,
//! data-adaptive curves without assuming a global functional form.
//!
//! ## Documentation
//!
//! > 📚 **Full Documentation**: [loess.readthedocs.io](https://loess.readthedocs.io/)
//! >
//! > Comprehensive guides, API references, and tutorials.
//!
//! ## Quick Start
//!
//! ### Typical Use
//!
//! ```rust
//! use loess_rs::prelude::*;
//!
//! let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! // Build the model
//! let model = Loess::new()
//!     .fraction(0.5)      // Use 50% of data for each local fit
//!     .iterations(3)      // 3 robustness iterations
//!     .build()?;
//!
//! // Fit the model to the data
//! let result = model.fit(&x, &y)?;
//!
//! println!("{}", result);
//! # Result::<(), LoessError>::Ok(())
//! ```
//!
//! ```text
//! Summary:
//!   Data points: 5
//!   Fraction: 0.5
//!
//! Smoothed Data:
//!        X     Y_smooth
//!   --------------------
//!     1.00     2.00000
//!     2.00     4.10000
//!     3.00     5.90000
//!     4.00     8.20000
//!     5.00     9.80000
//! ```
//!
//! ### Full Features
//!
//! ```rust
//! use loess_rs::prelude::*;
//!
//! let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//! let y = vec![2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7];
//!
//! // Build model with all features enabled
//! let model = Loess::new()
//!     .fraction(0.5)                                   // Use 50% of data for each local fit
//!     .iterations(3)                                   // 3 robustness iterations
//!     .degree("linear")                               // Polynomial degree (case-insensitive)
//!     .dimensions(1)                                   // Number of dimensions
//!     .distance_metric("euclidean")                   // Distance metric
//!     .weight_function("tricube")                     // Kernel function
//!     .robustness_method("bisquare")                  // Outlier handling
//!     .surface_mode("interpolation")                  // Surface evaluation mode
//!     .boundary_policy("extend")                       // Boundary handling
//!     .boundary_degree_fallback(true)                  // Boundary degree fallback
//!     .scaling_method("mad")                          // Scaling method
//!     .cell(0.2)                                       // Interpolation cell size
//!     .interpolation_vertices(1000)                    // Maximum vertices for interpolation
//!     .zero_weight_fallback("use_local_mean")         // Fallback policy
//!     .auto_converge(1e-6)                             // Auto-convergence threshold
//!     .confidence_intervals(0.95)                      // 95% confidence intervals
//!     .prediction_intervals(0.95)                      // 95% prediction intervals
//!     .return_diagnostics()                            // Fit quality metrics
//!     .return_residuals()                              // Include residuals
//!     .return_robustness_weights()                     // Include robustness weights
//!     .return_se()                                     // Enable standard error computation
//!     .cv_method("kfold")                              // Case-insensitive: "kfold" or "loocv"
//!     .cv_k(5)                                          // Number of folds for k-fold CV
//!     .cv_fractions(vec![0.3, 0.7])                     // Candidate fractions to evaluate
//!     .cv_seed(123)                                     // Reproducible fold split
//!     .build()?;
//!
//! let result = model.fit(&x, &y)?;
//! println!("{}", result);
//! # Result::<(), LoessError>::Ok(())
//! ```
//!
//! ```text
//! Summary:
//!   Data points: 8
//!   Fraction: 0.5
//!   Robustness: Applied
//!
//! LOESS Diagnostics:
//!   RMSE:         0.191925
//!   MAE:          0.181676
//!   R^2:           0.998205
//!   Residual SD:  0.297750
//!   Effective DF: 8.00
//!   AIC:          -10.41
//!   AICc:         inf
//!
//! Smoothed Data:
//!        X     Y_smooth      Std_Err   Conf_Lower   Conf_Upper   Pred_Lower   Pred_Upper     Residual Rob_Weight
//!   ----------------------------------------------------------------------------------------------------------------
//!     1.00     2.01963     0.389365     1.256476     2.782788     1.058911     2.980353     0.080368     1.0000
//!     2.00     4.00251     0.345447     3.325438     4.679589     3.108641     4.896386    -0.202513     1.0000
//!     3.00     5.99959     0.423339     5.169846     6.829335     4.985168     7.014013     0.200410     1.0000
//!     4.00     8.09859     0.489473     7.139224     9.057960     6.975666     9.221518    -0.198592     1.0000
//!     5.00    10.03881     0.551687     8.957506    11.120118     8.810073    11.267551     0.261188     1.0000
//!     6.00    12.02872     0.539259    10.971775    13.085672    10.821364    13.236083    -0.228723     1.0000
//!     7.00    13.89828     0.371149    13.170829    14.625733    12.965670    14.830892     0.201719     1.0000
//!     8.00    15.77990     0.408300    14.979631    16.580167    14.789441    16.770356    -0.079899     1.0000
//! ```
//!
//! ### Result and Error Handling
//!
//! The `fit` method returns a `Result<LoessResult<T>, LoessError>`.
//!
//! - **`Ok(LoessResult<T>)`**: Contains the smoothed data and diagnostics.
//! - **`Err(LoessError)`**: Indicates a failure (e.g., mismatched input lengths, insufficient data).
//!
//! The `?` operator is idiomatic:
//!
//! ```rust
//! use loess_rs::prelude::*;
//! # let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! # let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! let model = Loess::new().build()?;
//!
//! let result = model.fit(&x, &y)?;
//! // or to be more explicit:
//! // let result: LoessResult<f64> = model.fit(&x, &y)?;
//! # Result::<(), LoessError>::Ok(())
//! ```
//!
//! But you can also handle results explicitly:
//!
//! ```rust
//! use loess_rs::prelude::*;
//! # let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! # let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! let model = Loess::new().build()?;
//!
//! match model.fit(&x, &y) {
//!     Ok(result) => {
//!         // result is LoessResult<f64>
//!         println!("Smoothed: {:?}", result.y);
//!     }
//!     Err(e) => {
//!         // e is LoessError
//!         eprintln!("Fitting failed: {}", e);
//!     }
//! }
//! # Result::<(), LoessError>::Ok(())
//! ```
//!
//! ## Builder Arguments
//!
//! All builder methods return `Self` and can be chained. Finalize the builder with
//! `build()`.
//!
//! ### Core Smoothing
//!
//! - **`fraction(f: T)`** — Smoothing bandwidth: fraction of the data used for each local
//!   fit (range `(0, 1]`). Smaller → more local and jagged; larger → smoother.
//!   Default: `0.67`.
//!
//! - **`iterations(n: usize)`** — Number of robustness (IRLS) iterations for outlier
//!   resistance. `0` disables robustness weighting. Default: `3`.
//!
//! - **`degree(d: PolynomialDegree)`** — Degree of the local polynomial fitted at each point.
//!   - `Constant` / `"constant"` (0): weighted mean — fastest, least flexible
//!   - `Linear` / `"linear"` (1, **default**): standard LOESS — good balance of speed and accuracy
//!   - `Quadratic` / `"quadratic"` (2): better for curved regions
//!   - `Cubic` / `"cubic"` (3) / `Quartic` / `"quartic"` (4): higher flexibility, more expensive
//!
//! - **`weight_function(wf: WeightFunction)`** — Kernel function for distance-based local
//!   weighting. Options: `Tricube` / `"tricube"` (**default**), `Epanechnikov` / `"epanechnikov"`,
//!   `Biweight` / `"biweight"`, `Gaussian` / `"gaussian"`, `Triangle` / `"triangle"`,
//!   `Cosine` / `"cosine"`, `Uniform` / `"uniform"`.
//!
//! - **`robustness_method(rm: RobustnessMethod)`** — Downweighting method applied to
//!   outliers during robustness iterations. Options: `Bisquare` / `"bisquare"` (**default**),
//!   `Huber` / `"huber"`, `Talwar` / `"talwar"`.
//!
//! - **`scaling_method(sm: ScalingMethod)`** — Residual scale estimator used in robustness
//!   weighting. Options: `MAD` / `"mad"` (**default**), `MAR` / `"mar"`, `Mean` / `"mean"`.
//!
//! - **`custom_weights(w: Vec<T>)`** — Per-observation case weights applied as
//!   `w_ij = custom_weights[j] × K(d_ij / h)`. Higher values increase the influence of an
//!   observation on nearby local fits (analogous to `weights` in R's `stats::loess`).
//!   Must have the same length as `y`. Only applied in Batch mode.
//!
//! ### Surface Evaluation
//!
//! - **`surface_mode(m: SurfaceMode)`** — How the fitted surface is evaluated.
//!   - `Interpolation` / `"interpolation"` (**default**): fits at a sparse grid of vertices then interpolates —
//!     fast for large datasets.
//!   - `Direct` / `"direct"`: fits exactly at every data point — exact but O(n²).
//!
//! - **`cell(c: T)`** — Cell size for the interpolation vertex grid (default: `0.2`).
//!   Smaller → more vertices, higher accuracy, slower.
//!
//! - **`interpolation_vertices(n: usize)`** — Hard cap on the number of interpolation
//!   vertices regardless of `cell`.
//!
//! - **`boundary_degree_fallback(enabled: bool)`** — When `true` (**default**), vertices
//!   outside the tight data range use a `Linear` fit to avoid unstable extrapolation.
//!   Set to `false` to match R's `stats::loess` behavior exactly.
//!
//! ### Neighborhood & Distance
//!
//! - **`dimensions(n: usize)`** — Number of predictor dimensions (default: `1`).
//!
//! - **`distance_metric(m: DistanceMetric<T>)`** — Distance metric for neighbor selection.
//!   - `Normalized` / `"normalized"` (**default**): each dimension scaled to `[0, 1]`
//!   - `Euclidean` / `"euclidean"`: standard L² distance
//!   - `Manhattan` / `"manhattan"`: L¹ distance
//!   - `Chebyshev` / `"chebyshev"`: L∞ (max) distance
//!   - `Minkowski(p)` / `"minkowski:p"`: Lᵖ distance for arbitrary `p`
//!   - `Weighted(w)`: dimension-weighted Euclidean (no string form — requires a weights vector)
//!
//! ### Boundary Handling
//!
//! - **`boundary_policy(p: BoundaryPolicy)`** — How query points outside the observed data
//!   range are handled. Options: `Extend` / `"extend"` (**default**), `Reflect` / `"reflect"`,
//!   `Zero` / `"zero"`, `NoBoundary` / `"noboundary"`.
//!
//! - **`zero_weight_fallback(p: ZeroWeightFallback)`** — Fallback when all neighbors of a
//!   point have zero weight (degenerate neighborhood).
//!   - `UseLocalMean` / `"use_local_mean"` (**default**): return the weighted mean of nearby values
//!   - `ReturnOriginal` / `"return_original"`: return the raw `y` value
//!   - `ReturnNone` / `"return_none"`: return `NaN`
//!
//! ### Convergence
//!
//! - **`auto_converge(tol: T)`** — Stop robustness iterations early when the relative change
//!   in fitted values falls below `tol`. Disabled by default.
//!
//! ### Output Options
//!
//! - **`return_diagnostics()`** — Include fit-quality diagnostics in the result (RMSE, MAE,
//!   R², AIC, effective degrees of freedom, residual SD, etc.).
//!
//! - **`return_residuals()`** — Include raw residuals `r_i = y_i − ŷ_i` in the result.
//!
//! - **`return_robustness_weights()`** — Include the final robustness weights `w_i`.
//!
//! - **`return_se()`** — Compute standard errors, hat-matrix trace, and effective number of
//!   parameters. Required for confidence/prediction intervals.
//!
//! - **`confidence_intervals(level: T)`** — Enable confidence intervals at the given coverage
//!   level (e.g., `0.95`). Requires `return_se()` to also be set.
//!
//! - **`prediction_intervals(level: T)`** — Enable prediction intervals at the given coverage
//!   level. Requires `return_se()` to also be set.
//!
//! ### Cross-Validation
//!
//! - **`cv_method(m: &str)`** — Cross-validation method, string-based and case-insensitive:
//!   - `"kfold"`
//!   - `"loocv"`
//!
//! - **`cv_k(k: usize)`** — Number of folds for `"kfold"` mode (default: `5`).
//!
//! - **`cv_fractions(f: Vec<T>)`** — Candidate fractions to evaluate during CV.
//!
//! - **`cv_seed(s: u64)`** — Optional seed for reproducible fold splitting.
//!
//! ### Adapter-Specific Options
//!
//! **StreamingLoess** (`StreamingLoess::new()`):
//!
//! - **`chunk_size(n: usize)`** — Number of points processed per streaming chunk.
//! - **`overlap(n: usize)`** — Point overlap between consecutive chunks for smooth boundaries.
//! - **`merge_strategy(s: MergeStrategy)`** — How overlapping region fits are combined.
//!   Options: `Average` / `"average"`, `WeightedAverage` / `"weighted_average"`,
//!   `TakeFirst` / `"take_first"`, `TakeLast` / `"take_last"`.
//!
//! **OnlineLoess** (`OnlineLoess::new()`):
//!
//! - **`window_capacity(n: usize)`** — Maximum points kept in the sliding window.
//! - **`min_points(n: usize)`** — Minimum points required before returning a fit.
//! - **`update_mode(m: UpdateMode)`** — Window update strategy.
//!   - `Full` / `"full"` (**default**): full refit on every update
//!   - `Incremental` / `"incremental"`: lightweight incremental update
//!
//! The crate supports `no_std` environments for embedded devices and resource-constrained systems.
//! Disable default features to remove the standard library dependency:
//!
//! ```toml
//! [dependencies]
//! loess_rs = { version = "0.1", default-features = false }
//! ```
//!
//! **Minimal example for embedded systems:**
//!
//! ```rust
//! # #[cfg(feature = "std")] {
//! use loess_rs::prelude::*;
//!
//! // In an embedded context (e.g., sensor data processing)
//! fn smooth_sensor_data() -> Result<(), LoessError> {
//!     // Small dataset from sensor readings
//!     let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
//!     let y = vec![2.1, 3.9, 6.2, 7.8, 10.1];
//!
//!     // Build minimal model (no intervals, no diagnostics)
//!     let model = Loess::new()
//!         .fraction(0.5)
//!         .iterations(2)      // Fewer iterations for speed
//!         .build()?;
//!
//!     // Fit the model
//!     let result = model.fit(&x, &y)?;
//!
//!     // Use smoothed values (result.y)
//!     // ...
//!
//!     Ok(())
//! }
//! # smooth_sensor_data().unwrap();
//! # }
//! ```
//!
//! **Tips for embedded/no_std usage:**
//! - Use `f32` instead of `f64` to reduce memory footprint
//! - Keep datasets small (< 1000 points)
//! - Disable optional features (intervals, diagnostics) to reduce code size
//! - Use fewer iterations (1-2) to reduce computation time
//! - Allocate buffers statically when possible to avoid heap fragmentation
//!
//! ## References
//!
//! - Cleveland, W. S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots"
//! - Cleveland, W. S. & Devlin, S. J. (1988). "Locally Weighted Regression: An Approach to Regression Analysis by Local Fitting"
//!
//! ## srrstats Compliance for rOpenSci Statistical Software Review
//!
//! @srrstats {G1.0} Statistical literature references documented above (Cleveland 1979, 1988).
//! @srrstats {G1.1} This package provides LOESS smoothing, a nonparametric regression method
//!   for fitting smooth curves to scatterplot data using locally weighted polynomial regression.
//! @srrstats {G1.4} All exported functions and types are documented with rustdoc comments.
//! @srrstats {G1.6} Performance characteristics documented: SIMD-optimized solvers, O(n*k)
//!   complexity where k is the window size, supports streaming and online modes.
//!
//! ## License
//!
//! See the repository for license information and contribution guidelines.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
#[macro_use]
extern crate alloc;

// Layer 1: Primitives - data structures and basic utilities.
mod primitives;

// Layer 2: Math - pure mathematical functions.
mod math;

// Layer 3: Algorithms - core LOESS algorithms.
mod algorithms;

// Layer 4: Evaluation - post-processing and diagnostics.
mod evaluation;

// Layer 5: Engine - orchestration and execution control.
mod engine;

// Layer 6: Adapters - execution mode adapters.
mod adapters;

// High-level fluent API for LOESS smoothing.
mod api;

// Standard LOESS prelude.
pub mod prelude {
    pub use crate::api::{Loess, LoessError, LoessResult, OnlineLoess, StreamingLoess};
}

// Internal modules for development and testing.
//
// This module re-exports internal modules for development and testing purposes.
// It is only available with the `dev` feature enabled.
#[cfg(feature = "dev")]
pub mod internals {
    pub mod primitives {
        pub use crate::primitives::*;
    }
    pub mod math {
        pub use crate::math::*;
    }
    pub mod algorithms {
        pub use crate::algorithms::*;
    }
    pub mod engine {
        pub use crate::engine::*;
    }
    pub mod evaluation {
        pub use crate::evaluation::*;
    }
    pub mod adapters {
        pub use crate::adapters::*;
    }
    pub mod api {
        pub use crate::api::*;
    }
    pub mod alias {
        pub use crate::api::helpers::*;
    }
    pub mod defaults {
        pub use crate::adapters::defaults::*;
        pub use crate::algorithms::defaults::*;
        pub use crate::engine::defaults::*;
        pub use crate::math::defaults::*;
    }
}
