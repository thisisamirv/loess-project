//! # Fast LOESS (Locally Estimated Scatterplot Smoothing)
//!
//! The fastest, most robust, and most feature-complete language-agnostic
//! LOESS (Locally Estimated Scatterplot Smoothing) implementation for **Rust**,
//! **Python**, and **R**.
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
//! The [`doc`] module contains the full user guide, browsable on [docs.rs](https://docs.rs/fastLoess):
//!
//! - **Getting Started**
//!   - [Concepts](doc::introduction::concepts)
//!   - [Installation](doc::introduction::installation)
//!   - [Quick Start](doc::introduction::quickstart)
//! - **API Reference**
//!   - [Batch API](doc::api)
//!   - [Streaming API](doc::api::streaming)
//!   - [Online API](doc::api::online)
//! - **Adapters**
//!   - [Choosing an Adapter](doc::guide::adapter_choice)
//! - **Analysis**
//!   - [Intervals](doc::guide::intervals)
//!   - [Cross-Validation](doc::guide::cross_validation)
//!   - [Prediction](doc::guide::predict)
//! - **Customization**
//!   - [Kernels](doc::weighting::kernels)
//!   - [Robustness](doc::weighting::robustness)
//!   - [Scaling](doc::weighting::scaling)
//!   - [Custom Weights](doc::weighting::custom_weights)
//!   - [Boundary](doc::advanced::boundary)
//!   - [Merge Strategies](doc::advanced::merge)
//!   - [Degree](doc::advanced::degree)
//!   - [Dimensions](doc::advanced::dimensions)
//! - **Use Cases**
//!   - [Genomics](doc::use_case::genomics)
//!   - [Time Series](doc::use_case::time_series)
//!   - [Real-Time](doc::use_case::real_time)
//! - **News**
//!   - [Release Notes](doc::news)
//!
//! ## Quick Start (Batch)
//!
//! ### Typical Use
//!
//! ```rust
//! use fastLoess::prelude::*;
//! use ndarray::Array1;
//!
//! let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
//! let y = Array1::from_vec(vec![2.0, 4.1, 5.9, 8.2, 9.8]);
//!
//! // Build the model with parallel execution (default)
//! let model = Loess::new()
//!     .fraction(0.5)      // Use 50% of data for each local fit
//!     .iterations(3)      // 3 robustness iterations
//!     .build()?;
//!
//! // Fit the model to the data
//! let result = model.fit(&x, &y)?;
//!
//! println!("LOESS result:\n{}", result);
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
//! use fastLoess::prelude::*;
//! use ndarray::Array1;
//!
//! let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
//! let y = Array1::from_vec(vec![2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7]);
//!
//! // Build model with all features enabled
//! let model = Loess::new()
//!     .fraction(0.5)                                  // Use 50% of data for each local fit
//!     .iterations(3)                                  // 3 robustness iterations
//!     .weight_function("tricube")                     // Kernel function
//!     .robustness_method("bisquare")                  // Outlier handling
//!     .degree("linear")                               // Polynomial degree (case-insensitive)
//!     .dimensions(1)                                  // Number of dimensions
//!     .distance_metric("euclidean")                   // Distance metric
//!     .surface_mode("direct")                         // Required for per-point gradients
//!     .cell(0.2)                                      // Interpolation cell size
//!     .interpolation_vertices(1000)                   // Maximum vertices for interpolation
//!     .zero_weight_fallback("use_local_mean")         // Fallback policy
//!     .boundary_policy("extend")                      // Boundary handling
//!     .boundary_degree_fallback(true)                 // Boundary degree fallback
//!     .scaling_method("mad")                          // Scaling method
//!     .auto_converge(1e-6)                            // Auto-convergence threshold
//!     .missing("error")                               // Reject non-finite (NaN/Inf) input
//!     .parallel(true)                                 // Enable parallel execution
//!     .outputs([
//!         "se",                                       // Standard errors
//!         "diagnostics",                              // Fit quality metrics
//!         "residuals",                                // Include residuals
//!         "weights",                                  // Include robustness weights
//!         "gradient",                                 // Include per-point local fit gradient
//!         "sorted"                                    // Sort output ascending by x
//!     ])
//!     .intervals(
//!         IntervalsBuilder::new()
//!         .confidence(0.95)                           // 95% confidence intervals
//!         .prediction(0.95)                           // 95% prediction intervals
//!     )
//!     .cv(
//!         CVBuilder::new()
//!         .method("kfold")                            // Use k-fold CV (or "loocv")
//!         .k(5)                                       // Split observations into five folds
//!         .fraction(vec![0.3, 0.7])                   // Candidate smoothing fractions
//!     )
//!     .seed(123)
//!     .retain_model(true)                             // Retain state for out-of-sample predict()
//!     .custom_weights(vec![1.0; 8])                   // Per-observation case weights
//!     .build()?;
//!
//! let result = model.fit(&x, &y)?;
//! println!("LOESS result:\n{}", result);
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
//!   R2:           0.998205
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
//! use fastLoess::prelude::*;
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
//! use fastLoess::prelude::*;
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
//! ### ndarray Integration
//!
//! `fastLoess` supports [ndarray](https://docs.rs/ndarray) natively, allowing for zero-copy
//! data passing and efficient numerical operations.
//!
//! ```rust
//! use fastLoess::prelude::*;
//! use ndarray::Array1;
//!
//! // Data as ndarray types
//! let x = Array1::from_vec((0..100).map(|i| i as f64 * 0.1).collect());
//! let y = Array1::from_elem(100, 1.0); // Replace with real data
//!
//! let model = Loess::new().build()?;
//!
//! // fit() accepts &Array1<f64>, &[f64], or Vec<f64>
//! let result = model.fit(&x, &y)?;
//!
//! // result.y is an Array1<f64>
//! let smoothed_values = result.y;
//! # Result::<(), LoessError>::Ok(())
//! ```
//!
//! **Benefits:**
//! - **Zero-copy**: Pass data directly from your numerical pipeline.
//! - **Consistency**: If your project already uses `ndarray`, `fastLoess` fits right in.
//! - **Performance**: Optimized internal operations using `ndarray` primitives.
//!
//!
//! ### Predict
//!
//! Retain the fitted Batch model to evaluate new coordinates with optional uncertainty
//! estimates, gradients, and bounded extrapolation:
//!
//! ```rust
//! use fastLoess::prelude::*;
//!
//! let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//! let y = vec![2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7];
//! let fitted = Loess::new()
//!     .fraction(0.5)
//!     .surface_mode("direct")
//!     .retain_model(true)
//!     .build()?
//!     .fit(&x, &y)?;
//! let prediction = Predict::new()
//!     .outputs(["se", "derivative"])
//!     .intervals(IntervalsBuilder::new()
//!         .confidence(0.95)
//!         .prediction(0.95)
//!         .bootstrap(20)
//!     )
//!     .seed(42)
//!     .extrapolation("linear")
//!     .max_extrapolation_distance(2.0)
//!     .max_neighbor_distance(10.0)
//!     .build()?
//!     .call(&fitted, &[2.5, 8.5])?;
//! println!("{prediction:#?}");
//! # assert_eq!(prediction.y.len(), 2);
//! # assert!(prediction.standard_errors.is_some());
//! # assert!(prediction.confidence_lower.is_some());
//! # assert!(prediction.prediction_upper.is_some());
//! # assert!(prediction.derivative.is_some());
//! # Result::<(), LoessError>::Ok(())
//! ```
//!
//! ## Quick Start (Streaming)
//!
//! ### Typical Use
//!
//! Process a dataset in chunks, then flush the points retained for overlap:
//!
//! ```rust
//! use fastLoess::prelude::*;
//!
//! let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
//! let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
//! let mut model = StreamingLoess::new()
//!     .fraction(0.5)
//!     .chunk_size(10)
//!     .overlap(2)
//!     .build()?;
//!
//! let mut emitted = 0;
//! for (xs, ys) in x.chunks(10).zip(y.chunks(10)) {
//!     emitted += model.process_chunk(xs, ys)?.y.len();
//! }
//! emitted += model.finalize()?.y.len();
//! println!("Smoothed {emitted} points across two chunks");
//! # assert_eq!(emitted, x.len());
//! # Result::<(), LoessError>::Ok(())
//! ```
//!
//! ```text
//! Smoothed 20 points across two chunks
//! ```
//!
//! ### Full Features
//!
//! Configure uncertainty estimates and optional outputs per chunk:
//! The example lists common LOESS settings explicitly. Cell size, vertex limits, and
//! boundary-degree fallback apply only with `surface_mode("interpolation")`; gradients
//! require the direct mode shown here. CV, case weights, retained prediction, and sorted
//! output are Batch-only options and are not part of this Streaming example.
//!
//! ```rust
//! use fastLoess::prelude::*;
//!
//! let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.2).collect();
//! let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1 * (xi * 5.0).sin()).collect();
//! let mut model = StreamingLoess::new()
//!     .fraction(0.6)                          // Local smoothing span
//!     .iterations(1)                          // Robustness iterations
//!     .weight_function("tricube")             // Kernel function
//!     .robustness_method("bisquare")          // Outlier downweighting
//!     .degree("linear")                       // Local polynomial degree
//!     .dimensions(1)                          // One predictor per observation
//!     .distance_metric("euclidean")           // Neighbor distance metric
//!     .surface_mode("direct")                 // Required for per-point gradients
//!     .cell(0.2)                              // Applies only in interpolation mode
//!     .interpolation_vertices(1000)           // Applies only in interpolation mode
//!     .zero_weight_fallback("use_local_mean") // Zero-weight neighborhood fallback
//!     .boundary_policy("extend")              // Boundary padding
//!     .boundary_degree_fallback(true)         // Applies only in interpolation mode
//!     .scaling_method("mad")                  // Robust residual scale
//!     .auto_converge(1e-6)                    // Robustness convergence tolerance
//!     .missing("error")                       // Reject non-finite observations
//!     .chunk_size(10)                         // Points per input chunk
//!     .overlap(2)                             // Points retained between chunks
//!     .merge_strategy("weighted_average")     // Blend estimates in the overlap
//!     .parallel(true)                         // Enable parallel per-point fitting
//!     .outputs([
//!         "se",                               // Standard errors
//!         "diagnostics",                      // Cumulative fit diagnostics
//!         "residuals",                        // Observed minus fitted values
//!         "weights",                          // Final robustness weights
//!         "gradient"                          // Local slope at each point
//!     ])
//!     .intervals(IntervalsBuilder::new()
//!         .confidence(0.95)                   // 95% confidence intervals
//!         .prediction(0.95)                   // 95% prediction intervals
//!         .bootstrap(20)                      // Residual-bootstrap refits per chunk
//!     )
//!     .seed(7)                                // Reproducible bootstrap draws
//!     .build()?;
//!
//! let first = model.process_chunk(&x[..10], &y[..10])?;
//! # assert!(first.standard_errors.is_some());
//! # assert!(first.confidence_lower.is_some());
//! # assert!(first.prediction_upper.is_some());
//! # assert!(first.gradient.is_some());
//! # assert!(first.diagnostics.is_some());
//! # assert!(first.residuals.is_some());
//! # assert!(first.robustness_weights.is_some());
//! let second = model.process_chunk(&x[10..], &y[10..])?;
//! let final_chunk = model.finalize()?;
//! let emitted = first.y.len() + second.y.len() + final_chunk.y.len();
//! println!("Streaming intervals: {emitted} points");
//! # assert_eq!(emitted, x.len());
//! # Result::<(), LoessError>::Ok(())
//! ```
//!
//! ```text
//! Streaming intervals: 20 points
//! ```
//!
//! ### Result and Error Handling
//!
//! `process_chunk()` and `finalize()` return `Result<LoessResult<T>, LoessError>`.
//! A mismatched chunk is rejected without silently dropping points:
//!
//! ```rust
//! use fastLoess::prelude::*;
//!
//! let mut model = StreamingLoess::new().chunk_size(10).overlap(2).build()?;
//! let invalid = model.process_chunk(&[1.0, 2.0][..], &[3.0][..]);
//! assert!(invalid.is_err());
//! let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
//! let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi).collect();
//! let chunk = model.process_chunk(&x, &y)?;
//! let tail = model.finalize()?;
//! println!("Recovered {} points", chunk.y.len() + tail.y.len());
//! # Result::<(), LoessError>::Ok(())
//! ```
//!
//! ```text
//! Recovered 10 points
//! ```
//!
//! ### ndarray Integration
//!
//! Contiguous ndarray arrays expose slices for chunked processing with `fastLoess`.
//! The slices are passed without copying the source arrays:
//!
//! ```rust
//! use fastLoess::prelude::*;
//! use ndarray::Array1;
//!
//! let x = Array1::from_vec((0..20).map(|i| i as f64).collect());
//! let y = x.mapv(|xi| 2.0 * xi + 1.0);
//! let mut model = StreamingLoess::new().chunk_size(10).overlap(2).build()?;
//! let mut emitted = 0;
//! for (xs, ys) in x.as_slice().unwrap().chunks(10).zip(y.as_slice().unwrap().chunks(10)) {
//!     emitted += model.process_chunk(xs, ys)?.y.len();
//! }
//! emitted += model.finalize()?.y.len();
//! println!("Streaming ndarray points: {emitted}");
//! # assert_eq!(emitted, x.len());
//! # Result::<(), LoessError>::Ok(())
//! ```
//!
//! ```text
//! Streaming ndarray points: 20
//! ```
//!
//! ## Quick Start (Online)
//!
//! ### Typical Use
//!
//! Add points to a sliding window; updates begin once `min_points` is reached:
//!
//! ```rust
//! use fastLoess::prelude::*;
//!
//! let x = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
//! let y = [2.0, 4.0, 6.0, 8.0, 10.0];
//! let mut model = OnlineLoess::new()
//!     .fraction(0.5)
//!     .window_capacity(5)
//!     .min_points(3)
//!     .build()?;
//!
//! let mut updates = 0;
//! let mut latest = None;
//! for (&xi, &yi) in x.iter().zip(&y) {
//!     if let Some(output) = model.add_point(&[xi], yi)? {
//!         updates += 1;
//!         latest = Some(output.y);
//!     }
//! }
//! let latest = latest.expect("the window has enough points");
//! println!("Online updates: {updates}; latest estimate: {latest:.1}");
//! # assert_eq!(updates, 3);
//! # assert!((latest - 10.0).abs() < 1e-8);
//! # Result::<(), LoessError>::Ok(())
//! ```
//!
//! ```text
//! Online updates: 3; latest estimate: 10.0
//! ```
//!
//! ### Full Features
//!
//! Full updates support robust fitting and uncertainty estimates for each latest point:
//! Kernel, robustness, scaling, boundary, and convergence controls are shared with Batch.
//! Interpolation controls are listed but inactive in direct mode. Full updates are required
//! for robustness iterations and intervals; CV, case weights, and retained prediction remain
//! Batch-only. Online output describes the latest point rather than cumulative diagnostics.
//!
//! ```rust
//! use fastLoess::prelude::*;
//!
//! let mut model = OnlineLoess::new()
//!     .fraction(0.7)                          // Local smoothing span
//!     .iterations(1)                          // Robustness iterations
//!     .weight_function("tricube")             // Kernel function
//!     .robustness_method("bisquare")          // Outlier downweighting
//!     .degree("linear")                       // Local polynomial degree
//!     .dimensions(1)                          // One predictor per observation
//!     .distance_metric("euclidean")           // Neighbor distance metric
//!     .surface_mode("direct")                 // Required for per-point gradients
//!     .cell(0.2)                              // Applies only in interpolation mode
//!     .interpolation_vertices(1000)           // Applies only in interpolation mode
//!     .zero_weight_fallback("use_local_mean") // Zero-weight neighborhood fallback
//!     .boundary_policy("extend")              // Boundary padding
//!     .boundary_degree_fallback(true)         // Applies only in interpolation mode
//!     .scaling_method("mad")                  // Robust residual scale
//!     .auto_converge(1e-6)                    // Robustness convergence tolerance
//!     .missing("error")                       // Reject non-finite observations
//!     .window_capacity(20)                    // Maximum sliding-window size
//!     .min_points(5)                          // Wait for five points before smoothing
//!     .update_mode("full")                    // Refit the whole window for intervals
//!     .outputs([
//!         "se",                               // Latest-point standard error
//!         "weights",                          // Latest-point robustness weight
//!         "gradient"                          // Latest-point local slope
//!     ])
//!     .intervals(IntervalsBuilder::new()
//!         .confidence(0.95)                   // 95% confidence interval
//!         .prediction(0.95)                   // 95% prediction interval
//!         .bootstrap(20)                      // Refit the current window 20 times
//!     )
//!     .seed(7)                                // Reproducible bootstrap draws
//!     .build()?;
//!
//! let mut updates = 0;
//! for i in 0..12 {
//!     let x = i as f64 * 0.2;
//!     if let Some(output) = model.add_point(&[x], x.sin() + 0.1 * (5.0 * x).sin())? {
//!         updates += 1;
//!         assert!(output.standard_error.is_some());
//!         assert!(output.confidence_lower.is_some());
//!         assert!(output.prediction_upper.is_some());
//!         assert!(output.gradient.is_some());
//!         assert!(output.robustness_weight.is_some());
//!     }
//! }
//! println!("Online full updates: {updates}");
//! # assert_eq!(updates, 8);
//! # Result::<(), LoessError>::Ok(())
//! ```
//!
//! ```text
//! Online full updates: 8
//! ```
//!
//! ### Result and Error Handling
//!
//! `build()` and `add_point(&[coordinate], response)` return `Result`; `add_point(&[coordinate], response)` returns `None` until
//! `min_points` is reached. Invalid configurations and non-finite points return errors:
//!
//! ```rust
//! use fastLoess::prelude::*;
//!
//! let invalid = OnlineLoess::new()
//!     .intervals(IntervalsBuilder::new().confidence(0.95))
//!     .build();
//! assert!(matches!(invalid, Err(LoessError::StandardErrorRequiresFullUpdateMode)));
//!
//! let mut model = OnlineLoess::new().window_capacity(5).min_points(3).build()?;
//! assert!(model.add_point(&[f64::NAN], 2.0).is_err());
//! let mut ready = 0;
//! for i in 1..=3 {
//!     if model.add_point(&[i as f64], 2.0 * i as f64)?.is_some() {
//!         ready += 1;
//!     }
//! }
//! println!("Ready online outputs: {ready}");
//! # Result::<(), LoessError>::Ok(())
//! ```
//!
//! ```text
//! Ready online outputs: 1
//! ```
//!
//! ### ndarray Integration
//!
//! With `ndarray` in your dependencies, iterate array values into `add_point(&[coordinate], response)`:
//!
//! ```rust
//! use fastLoess::prelude::*;
//! use ndarray::Array1;
//!
//! let x = Array1::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
//! let y = x.mapv(|xi| 2.0 * xi);
//! let mut model = OnlineLoess::new().window_capacity(5).min_points(3).build()?;
//! let mut ready = 0;
//! for (&xi, &yi) in x.iter().zip(y.iter()) {
//!     if model.add_point(&[xi], yi)?.is_some() {
//!         ready += 1;
//!     }
//! }
//! println!("Online ndarray updates: {ready}");
//! # assert_eq!(ready, 3);
//! # Result::<(), LoessError>::Ok(())
//! ```
//!
//! ```text
//! Online ndarray updates: 3
//! ```
//!
//! ## References
//!
//! - Cleveland, W. S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots"
//! - Cleveland, W. S. & Devlin, S. J. (1988). "Locally Weighted Regression: An Approach to Regression Analysis by Local Fitting"
// ## srrstats Compliance for rOpenSci Statistical Software Review
//
// @srrstats {G1.0} Statistical literature references documented above (Cleveland 1979, 1988).
// @srrstats {G1.5} Parallel execution via Rayon for multi-threaded performance.
// @srrstats {G3.0} ndarray integration for zero-copy data passing and numerical operations.

//! ## License
//!
//! See the repository for license information and contribution guidelines.

#![allow(non_snake_case)]

#[cfg(doc)]
pub mod doc;

// Layer 2: Math - pure mathematical functions.
mod math;

// Layer 4: Evaluation - post-processing and diagnostics.
mod evaluation;

// Layer 5: Engine - orchestration and execution control.
mod engine;

// Layer 6: Adapters - execution mode adapters.
mod adapters;

// High-level fluent API for LOESS smoothing.
mod api;

pub use loess_rs::{CVOptions, IntervalsBuilder};

// Input data handling.
mod input;

// Shared option parsing helpers for language bindings.
#[cfg(feature = "dev")]
mod binding_support;

// Standard fastLoess prelude.
pub mod prelude {
    pub use crate::api::{
        CVBuilder, IntervalsBuilder, Loess, LoessError, LoessResult, OnlineLoess, Predict,
        StreamingLoess,
    };
}

// Internal modules for development and testing.
//
// This module re-exports internal modules for development and testing purposes.
// It is only available with the `dev` feature enabled.
#[cfg(feature = "dev")]
pub mod internals {
    pub mod math {
        pub use crate::math::*;
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
    pub mod binding_support {
        pub use crate::binding_support::*;
    }
}
