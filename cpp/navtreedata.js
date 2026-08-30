/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "fastLoess", "index.html", [
    [ "LOESS Project", "index.html#autotoc_md0", [
      [ "Installation & Documentation", "index.html#autotoc_md2", [
        [ "📚 <a href=\"https://thisisamirv.github.io/loess-project/cpp/\" >View the full documentation</a>", "index.html#autotoc_md3", null ]
      ] ],
      [ "</blockquote>", "index.html#autotoc_md4", null ],
      [ "LOESS vs. LOWESS", "index.html#autotoc_md5", null ],
      [ "</blockquote>", "index.html#autotoc_md6", null ],
      [ "Why this package?", "index.html#autotoc_md7", [
        [ "Speed", "index.html#autotoc_md8", null ],
        [ "Robustness", "index.html#autotoc_md9", null ],
        [ "Features", "index.html#autotoc_md10", null ]
      ] ],
      [ "Validation", "index.html#autotoc_md11", null ],
      [ "API Reference", "index.html#autotoc_md12", null ],
      [ "Contributing", "index.html#autotoc_md14", null ],
      [ "Changelog", "index.html#autotoc_md15", null ],
      [ "License", "index.html#autotoc_md16", null ],
      [ "Citation", "index.html#autotoc_md17", null ]
    ] ],
    [ "Execution Modes", "md_docs_2adapter-choice.html", [
      [ "Overview", "md_docs_2adapter-choice.html#autotoc_md19", null ]
    ] ],
    [ "OnlineLoess API", "md_docs_2api-online.html", [
      [ "When to Use", "md_docs_2api-online.html#autotoc_md22", null ],
      [ "Class", "md_docs_2api-online.html#autotoc_md23", [
        [ "<tt>fastloess::OnlineLoess</tt>", "md_docs_2api-online.html#autotoc_md24", null ]
      ] ],
      [ "Options Structure", "md_docs_2api-online.html#autotoc_md25", [
        [ "<tt>OnlineOptions</tt> (inherits <tt>LoessOptions</tt>)", "md_docs_2api-online.html#autotoc_md26", null ]
      ] ],
      [ "Result Structure", "md_docs_2api-online.html#autotoc_md27", [
        [ "<tt>fastloess::OnlineOutput</tt>", "md_docs_2api-online.html#autotoc_md28", null ]
      ] ],
      [ "Options", "md_docs_2api-online.html#autotoc_md29", [
        [ "update_mode", "md_docs_2api-online.html#autotoc_md30", null ]
      ] ]
    ] ],
    [ "StreamingLoess API", "md_docs_2api-streaming.html", [
      [ "When to Use", "md_docs_2api-streaming.html#autotoc_md32", null ],
      [ "Class", "md_docs_2api-streaming.html#autotoc_md33", [
        [ "<tt>fastloess::StreamingLoess</tt>", "md_docs_2api-streaming.html#autotoc_md34", null ]
      ] ],
      [ "Result Structure", "md_docs_2api-streaming.html#autotoc_md35", [
        [ "<tt>fastloess::LoessResult</tt>", "md_docs_2api-streaming.html#autotoc_md36", null ]
      ] ],
      [ "Options Structure", "md_docs_2api-streaming.html#autotoc_md37", [
        [ "<tt>StreamingOptions</tt> (inherits <tt>LoessOptions</tt>)", "md_docs_2api-streaming.html#autotoc_md38", null ]
      ] ],
      [ "Options", "md_docs_2api-streaming.html#autotoc_md39", [
        [ "merge_strategy", "md_docs_2api-streaming.html#autotoc_md40", null ]
      ] ]
    ] ],
    [ "API", "md_docs_2api.html", [
      [ "When to Use", "md_docs_2api.html#autotoc_md43", null ],
      [ "Classes", "md_docs_2api.html#autotoc_md44", [
        [ "<tt>fastloess::Loess</tt>", "md_docs_2api.html#autotoc_md45", null ]
      ] ],
      [ "Options Structures", "md_docs_2api.html#autotoc_md46", [
        [ "<tt>LoessOptions</tt>", "md_docs_2api.html#autotoc_md47", null ]
      ] ],
      [ "Result Structure", "md_docs_2api.html#autotoc_md48", [
        [ "<tt>fastloess::LoessResult</tt>", "md_docs_2api.html#autotoc_md49", null ],
        [ "<tt>fastloess::Diagnostics</tt>", "md_docs_2api.html#autotoc_md50", null ]
      ] ],
      [ "Options", "md_docs_2api.html#autotoc_md51", [
        [ "weight_function", "md_docs_2api.html#autotoc_md52", null ],
        [ "robustness_method", "md_docs_2api.html#autotoc_md53", null ],
        [ "boundary_policy", "md_docs_2api.html#autotoc_md54", null ],
        [ "scaling_method", "md_docs_2api.html#autotoc_md55", null ],
        [ "zero_weight_fallback", "md_docs_2api.html#autotoc_md56", null ],
        [ "degree", "md_docs_2api.html#autotoc_md57", null ],
        [ "distance_metric", "md_docs_2api.html#autotoc_md58", null ],
        [ "surface_mode", "md_docs_2api.html#autotoc_md59", null ],
        [ "merge_strategy", "md_docs_2api.html#autotoc_md60", null ],
        [ "update_mode", "md_docs_2api.html#autotoc_md61", null ]
      ] ],
      [ "Example", "md_docs_2api.html#autotoc_md62", null ]
    ] ],
    [ "Benchmarks", "md_docs_2benchmarks.html", [
      [ "CPU Benchmarks", "md_docs_2benchmarks.html#autotoc_md64", null ],
      [ "Reproducing Benchmarks", "md_docs_2benchmarks.html#autotoc_md66", null ]
    ] ],
    [ "Boundary Handling", "md_docs_2boundary.html", [
      [ "Overview", "md_docs_2boundary.html#autotoc_md68", null ],
      [ "Extend (Default)", "md_docs_2boundary.html#autotoc_md70", null ],
      [ "Reflect", "md_docs_2boundary.html#autotoc_md72", null ],
      [ "Zero", "md_docs_2boundary.html#autotoc_md74", null ],
      [ "No Boundary", "md_docs_2boundary.html#autotoc_md76", null ],
      [ "Choosing a Policy", "md_docs_2boundary.html#autotoc_md78", null ]
    ] ],
    [ "Concepts", "md_docs_2concepts.html", [
      [ "What is LOESS?", "md_docs_2concepts.html#autotoc_md80", null ],
      [ "How It Works", "md_docs_2concepts.html#autotoc_md82", null ],
      [ "The Fraction Parameter", "md_docs_2concepts.html#autotoc_md84", null ],
      [ "Robustness Iterations", "md_docs_2concepts.html#autotoc_md86", null ],
      [ "Confidence vs Prediction Intervals", "md_docs_2concepts.html#autotoc_md88", null ],
      [ "Execution Modes", "md_docs_2concepts.html#autotoc_md90", null ],
      [ "Quick Decision Guide", "md_docs_2concepts.html#autotoc_md92", null ],
      [ "Key Advantages", "md_docs_2concepts.html#autotoc_md94", null ],
      [ "Next Steps", "md_docs_2concepts.html#autotoc_md96", null ]
    ] ],
    [ "Cross-Validation", "md_docs_2cross-validation.html", [
      [ "Overview", "md_docs_2cross-validation.html#autotoc_md98", null ],
      [ "K-Fold Cross-Validation", "md_docs_2cross-validation.html#autotoc_md100", null ],
      [ "Leave-One-Out (LOOCV)", "md_docs_2cross-validation.html#autotoc_md102", null ],
      [ "Seeded Randomization", "md_docs_2cross-validation.html#autotoc_md104", null ],
      [ "Comparison", "md_docs_2cross-validation.html#autotoc_md106", null ],
      [ "CV Metrics", "md_docs_2cross-validation.html#autotoc_md108", null ],
      [ "Interpreting Results", "md_docs_2cross-validation.html#autotoc_md110", null ],
      [ "Availability", "md_docs_2cross-validation.html#autotoc_md112", null ],
      [ "Best Practices", "md_docs_2cross-validation.html#autotoc_md114", null ]
    ] ],
    [ "Custom Weights", "md_docs_2custom-weights.html", [
      [ "How Custom Weights Work", "md_docs_2custom-weights.html#autotoc_md116", null ],
      [ "When to Use Custom Weights", "md_docs_2custom-weights.html#autotoc_md118", [
        [ "Custom Weights vs. Robustness Iterations", "md_docs_2custom-weights.html#autotoc_md119", null ]
      ] ],
      [ "Basic Usage", "md_docs_2custom-weights.html#autotoc_md121", [
        [ "Suppress a Known Outlier", "md_docs_2custom-weights.html#autotoc_md122", null ]
      ] ],
      [ "Validation Rules", "md_docs_2custom-weights.html#autotoc_md124", null ],
      [ "See Also", "md_docs_2custom-weights.html#autotoc_md126", null ]
    ] ],
    [ "Polynomial Degree", "md_docs_2degree.html", [
      [ "Overview", "md_docs_2degree.html#autotoc_md128", null ],
      [ "Degree 0 — Local Constant", "md_docs_2degree.html#autotoc_md130", null ],
      [ "Degree 1 — Local Linear (Default)", "md_docs_2degree.html#autotoc_md132", null ],
      [ "Degree 2 — Local Quadratic", "md_docs_2degree.html#autotoc_md134", null ],
      [ "Degree 3 — Local Cubic", "md_docs_2degree.html#autotoc_md136", null ],
      [ "Degree 4 — Local Quartic", "md_docs_2degree.html#autotoc_md138", null ],
      [ "Choosing the Right Degree", "md_docs_2degree.html#autotoc_md140", null ],
      [ "Higher Degree Effects", "md_docs_2degree.html#autotoc_md142", null ],
      [ "Surface Mode", "md_docs_2degree.html#autotoc_md144", null ]
    ] ],
    [ "Multivariate LOESS", "md_docs_2dimensions.html", [
      [ "Overview", "md_docs_2dimensions.html#autotoc_md146", null ],
      [ "1D — Standard (Default)", "md_docs_2dimensions.html#autotoc_md148", null ],
      [ "2D — Spatial Surface", "md_docs_2dimensions.html#autotoc_md150", null ],
      [ "3D and Higher", "md_docs_2dimensions.html#autotoc_md152", null ],
      [ "Distance Metrics for Multivariate Data", "md_docs_2dimensions.html#autotoc_md154", null ]
    ] ],
    [ "Installation", "md_docs_2installation.html", [
      [ "Pre-built Binaries (Linux (x64))", "md_docs_2installation.html#autotoc_md156", null ],
      [ "Pre-built Binaries (macOS (x64))", "md_docs_2installation.html#autotoc_md157", null ],
      [ "Pre-built Binaries (Windows (x64))", "md_docs_2installation.html#autotoc_md158", null ],
      [ "From Source", "md_docs_2installation.html#autotoc_md159", null ],
      [ "From conda-forge", "md_docs_2installation.html#autotoc_md160", null ],
      [ "Verify Installation", "md_docs_2installation.html#autotoc_md162", null ]
    ] ],
    [ "Intervals", "md_docs_2intervals.html", [
      [ "Overview", "md_docs_2intervals.html#autotoc_md164", null ],
      [ "Confidence Intervals", "md_docs_2intervals.html#autotoc_md166", null ],
      [ "Prediction Intervals", "md_docs_2intervals.html#autotoc_md168", null ],
      [ "Both Intervals", "md_docs_2intervals.html#autotoc_md170", null ],
      [ "Confidence Levels", "md_docs_2intervals.html#autotoc_md172", null ],
      [ "Standard Errors", "md_docs_2intervals.html#autotoc_md174", null ],
      [ "Availability", "md_docs_2intervals.html#autotoc_md176", null ]
    ] ],
    [ "Weight Functions", "md_docs_2kernels.html", [
      [ "Overview", "md_docs_2kernels.html#autotoc_md178", null ],
      [ "Available Kernels", "md_docs_2kernels.html#autotoc_md180", null ],
      [ "Tricube (Default)", "md_docs_2kernels.html#autotoc_md182", null ],
      [ "Epanechnikov", "md_docs_2kernels.html#autotoc_md184", null ],
      [ "Gaussian", "md_docs_2kernels.html#autotoc_md186", null ],
      [ "Biweight", "md_docs_2kernels.html#autotoc_md188", null ],
      [ "Cosine", "md_docs_2kernels.html#autotoc_md190", null ],
      [ "Triangle", "md_docs_2kernels.html#autotoc_md192", null ],
      [ "Uniform", "md_docs_2kernels.html#autotoc_md194", null ],
      [ "Choosing a Kernel", "md_docs_2kernels.html#autotoc_md196", null ]
    ] ],
    [ "Merge Strategies", "md_docs_2merge.html", [
      [ "Overview", "md_docs_2merge.html#autotoc_md198", null ],
      [ "Average", "md_docs_2merge.html#autotoc_md200", null ],
      [ "Take First", "md_docs_2merge.html#autotoc_md202", null ],
      [ "Take Last", "md_docs_2merge.html#autotoc_md204", null ],
      [ "Weighted Average", "md_docs_2merge.html#autotoc_md206", null ],
      [ "Choosing a Strategy", "md_docs_2merge.html#autotoc_md208", null ]
    ] ],
    [ "Parameters", "md_docs_2parameters.html", [
      [ "Quick Reference", "md_docs_2parameters.html#autotoc_md210", null ],
      [ "Parameter Options Summary", "md_docs_2parameters.html#autotoc_md212", null ],
      [ "Core Parameters", "md_docs_2parameters.html#autotoc_md214", [
        [ "fraction", "md_docs_2parameters.html#autotoc_md215", null ],
        [ "iterations", "md_docs_2parameters.html#autotoc_md217", null ],
        [ "degree", "md_docs_2parameters.html#autotoc_md219", null ],
        [ "surface_mode", "md_docs_2parameters.html#autotoc_md221", null ],
        [ "cell", "md_docs_2parameters.html#autotoc_md223", null ],
        [ "interpolation_vertices", "md_docs_2parameters.html#autotoc_md225", null ],
        [ "dimensions", "md_docs_2parameters.html#autotoc_md227", null ],
        [ "distance_metric / weighted_metric_weights", "md_docs_2parameters.html#autotoc_md229", null ],
        [ "weight_function", "md_docs_2parameters.html#autotoc_md231", null ],
        [ "robustness_method", "md_docs_2parameters.html#autotoc_md233", null ],
        [ "boundary_policy", "md_docs_2parameters.html#autotoc_md235", null ],
        [ "boundary_degree_fallback", "md_docs_2parameters.html#autotoc_md237", null ],
        [ "scaling_method", "md_docs_2parameters.html#autotoc_md239", null ],
        [ "zero_weight_fallback", "md_docs_2parameters.html#autotoc_md241", null ],
        [ "auto_converge", "md_docs_2parameters.html#autotoc_md243", null ],
        [ "parallel", "md_docs_2parameters.html#autotoc_md245", null ],
        [ "custom_weights", "md_docs_2parameters.html#autotoc_md247", null ]
      ] ],
      [ "Output Options", "md_docs_2parameters.html#autotoc_md249", [
        [ "return_residuals", "md_docs_2parameters.html#autotoc_md250", null ],
        [ "return_diagnostics", "md_docs_2parameters.html#autotoc_md252", null ],
        [ "return_robustness_weights", "md_docs_2parameters.html#autotoc_md254", null ],
        [ "confidence_intervals / prediction_intervals", "md_docs_2parameters.html#autotoc_md256", null ]
      ] ],
      [ "CV Methods", "md_docs_2parameters.html#autotoc_md258", [
        [ "cv_method", "md_docs_2parameters.html#autotoc_md259", null ]
      ] ],
      [ "Adapter Parameters", "md_docs_2parameters.html#autotoc_md261", [
        [ "chunk_size", "md_docs_2parameters.html#autotoc_md262", null ],
        [ "overlap", "md_docs_2parameters.html#autotoc_md264", null ],
        [ "merge_strategy", "md_docs_2parameters.html#autotoc_md266", null ],
        [ "window_capacity", "md_docs_2parameters.html#autotoc_md268", null ],
        [ "min_points", "md_docs_2parameters.html#autotoc_md270", null ],
        [ "update_mode", "md_docs_2parameters.html#autotoc_md272", null ]
      ] ]
    ] ],
    [ "Quick Start", "md_docs_2quickstart.html", [
      [ "Basic Smoothing", "md_docs_2quickstart.html#autotoc_md274", null ],
      [ "With Confidence Intervals", "md_docs_2quickstart.html#autotoc_md276", null ],
      [ "Handling Outliers", "md_docs_2quickstart.html#autotoc_md278", null ],
      [ "Streaming Mode", "md_docs_2quickstart.html#autotoc_md280", null ],
      [ "Next Steps", "md_docs_2quickstart.html#autotoc_md282", null ]
    ] ],
    [ "Robustness", "md_docs_2robustness.html", [
      [ "How Robustness Works", "md_docs_2robustness.html#autotoc_md284", null ],
      [ "Robustness Methods", "md_docs_2robustness.html#autotoc_md286", [
        [ "Bisquare (Default)", "md_docs_2robustness.html#autotoc_md287", null ],
        [ "Huber", "md_docs_2robustness.html#autotoc_md289", null ],
        [ "Talwar", "md_docs_2robustness.html#autotoc_md291", null ]
      ] ],
      [ "Comparison", "md_docs_2robustness.html#autotoc_md293", null ],
      [ "Detecting Outliers", "md_docs_2robustness.html#autotoc_md295", null ],
      [ "Scale Estimation", "md_docs_2robustness.html#autotoc_md297", null ],
      [ "Auto-Convergence", "md_docs_2robustness.html#autotoc_md299", null ]
    ] ],
    [ "Scaling Methods", "md_docs_2scaling.html", [
      [ "Overview", "md_docs_2scaling.html#autotoc_md301", null ],
      [ "MAD — Median Absolute Deviation (Default)", "md_docs_2scaling.html#autotoc_md303", null ],
      [ "MAR — Median Absolute Residual", "md_docs_2scaling.html#autotoc_md305", null ],
      [ "Mean — Mean Absolute Residual", "md_docs_2scaling.html#autotoc_md307", null ],
      [ "Choosing a Scaling Method", "md_docs_2scaling.html#autotoc_md309", null ]
    ] ],
    [ "Genomic Data Smoothing", "md_docs_2use-case-genomics.html", [
      [ "Overview", "md_docs_2use-case-genomics.html#autotoc_md311", null ],
      [ "Methylation Profile Smoothing", "md_docs_2use-case-genomics.html#autotoc_md313", [
        [ "The Challenge", "md_docs_2use-case-genomics.html#autotoc_md314", null ],
        [ "Solution", "md_docs_2use-case-genomics.html#autotoc_md315", null ]
      ] ],
      [ "ChIP-seq Signal Smoothing", "md_docs_2use-case-genomics.html#autotoc_md317", [
        [ "Application", "md_docs_2use-case-genomics.html#autotoc_md318", null ]
      ] ],
      [ "Large Genome Coverage (Streaming)", "md_docs_2use-case-genomics.html#autotoc_md320", null ],
      [ "Best Practices for Genomic Data", "md_docs_2use-case-genomics.html#autotoc_md322", null ],
      [ "See Also", "md_docs_2use-case-genomics.html#autotoc_md324", null ]
    ] ],
    [ "Real-Time Processing", "md_docs_2use-case-real-time.html", [
      [ "Overview", "md_docs_2use-case-real-time.html#autotoc_md326", null ],
      [ "Online Mode: Point-by-Point", "md_docs_2use-case-real-time.html#autotoc_md328", [
        [ "Sensor Data Example", "md_docs_2use-case-real-time.html#autotoc_md329", null ]
      ] ],
      [ "Streaming Mode: Chunk Processing", "md_docs_2use-case-real-time.html#autotoc_md331", [
        [ "Log File Processing", "md_docs_2use-case-real-time.html#autotoc_md332", null ]
      ] ],
      [ "Real-Time Dashboard Example", "md_docs_2use-case-real-time.html#autotoc_md334", null ],
      [ "Choosing Parameters", "md_docs_2use-case-real-time.html#autotoc_md336", [
        [ "Online Mode", "md_docs_2use-case-real-time.html#autotoc_md337", null ],
        [ "Streaming Mode", "md_docs_2use-case-real-time.html#autotoc_md338", null ]
      ] ],
      [ "Performance Considerations", "md_docs_2use-case-real-time.html#autotoc_md340", null ],
      [ "See Also", "md_docs_2use-case-real-time.html#autotoc_md342", null ]
    ] ],
    [ "Time Series Analysis", "md_docs_2use-case-time-series.html", [
      [ "Overview", "md_docs_2use-case-time-series.html#autotoc_md344", null ],
      [ "Basic Trend Extraction", "md_docs_2use-case-time-series.html#autotoc_md346", null ],
      [ "Detrending", "md_docs_2use-case-time-series.html#autotoc_md348", null ],
      [ "Forecasting with Prediction Intervals", "md_docs_2use-case-time-series.html#autotoc_md350", null ],
      [ "Handling Missing Data", "md_docs_2use-case-time-series.html#autotoc_md352", null ],
      [ "Multi-Scale Analysis", "md_docs_2use-case-time-series.html#autotoc_md354", null ],
      [ "Gene Expression Time Course", "md_docs_2use-case-time-series.html#autotoc_md356", null ],
      [ "Choosing Fraction for Time Series", "md_docs_2use-case-time-series.html#autotoc_md358", null ],
      [ "See Also", "md_docs_2use-case-time-series.html#autotoc_md360", null ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"index.html"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';