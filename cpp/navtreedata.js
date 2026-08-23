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
    [ "Concepts", "index.html#autotoc_md57", [
      [ "What is LOESS?", "index.html#autotoc_md58", null ],
      [ "How It Works", "index.html#autotoc_md60", null ],
      [ "The Fraction Parameter", "index.html#autotoc_md62", null ],
      [ "Robustness Iterations", "index.html#autotoc_md64", null ],
      [ "Confidence vs Prediction Intervals", "index.html#autotoc_md66", null ],
      [ "Execution Modes", "index.html#autotoc_md68", null ],
      [ "Key Advantages", "index.html#autotoc_md70", null ],
      [ "Next Steps", "index.html#autotoc_md72", null ]
    ] ],
    [ "adapter-choice", "md_docs_2adapter-choice.html", [
      [ "Execution Modes", "md_docs_2adapter-choice.html#autotoc_md0", [
        [ "Overview", "md_docs_2adapter-choice.html#autotoc_md1", null ]
      ] ]
    ] ],
    [ "OnlineLoess — C++ API Reference", "md_docs_2api-online.html", [
      [ "Class", "md_docs_2api-online.html#autotoc_md4", [
        [ "<tt>fastloess::OnlineLoess</tt>", "md_docs_2api-online.html#autotoc_md5", null ]
      ] ],
      [ "Options Structure", "md_docs_2api-online.html#autotoc_md6", [
        [ "<tt>OnlineOptions</tt> (inherits <tt>LoessOptions</tt>)", "md_docs_2api-online.html#autotoc_md7", null ]
      ] ],
      [ "Result Structure", "md_docs_2api-online.html#autotoc_md8", [
        [ "<tt>fastloess::OnlineOutput</tt>", "md_docs_2api-online.html#autotoc_md9", null ]
      ] ],
      [ "Options", "md_docs_2api-online.html#autotoc_md10", [
        [ "update_mode", "md_docs_2api-online.html#autotoc_md11", null ]
      ] ]
    ] ],
    [ "StreamingLoess — C++ API Reference", "md_docs_2api-streaming.html", [
      [ "Class", "md_docs_2api-streaming.html#autotoc_md13", [
        [ "<tt>fastloess::StreamingLoess</tt>", "md_docs_2api-streaming.html#autotoc_md14", null ]
      ] ],
      [ "Result Structure", "md_docs_2api-streaming.html#autotoc_md15", [
        [ "<tt>fastloess::LoessResult</tt>", "md_docs_2api-streaming.html#autotoc_md16", null ]
      ] ],
      [ "Options Structure", "md_docs_2api-streaming.html#autotoc_md17", [
        [ "<tt>StreamingOptions</tt> (inherits <tt>LoessOptions</tt>)", "md_docs_2api-streaming.html#autotoc_md18", null ]
      ] ],
      [ "Options", "md_docs_2api-streaming.html#autotoc_md19", [
        [ "merge_strategy", "md_docs_2api-streaming.html#autotoc_md20", null ]
      ] ]
    ] ],
    [ "fastLoess C++ API Reference", "md_docs_2api.html", [
      [ "Classes", "md_docs_2api.html#autotoc_md22", [
        [ "<tt>fastloess::Loess</tt>", "md_docs_2api.html#autotoc_md23", null ]
      ] ],
      [ "Options Structures", "md_docs_2api.html#autotoc_md24", [
        [ "<tt>LoessOptions</tt>", "md_docs_2api.html#autotoc_md25", null ]
      ] ],
      [ "Result Structure", "md_docs_2api.html#autotoc_md26", [
        [ "<tt>fastloess::LoessResult</tt>", "md_docs_2api.html#autotoc_md27", null ],
        [ "<tt>fastloess::Diagnostics</tt>", "md_docs_2api.html#autotoc_md28", null ]
      ] ],
      [ "Options", "md_docs_2api.html#autotoc_md29", [
        [ "weight_function", "md_docs_2api.html#autotoc_md30", null ],
        [ "robustness_method", "md_docs_2api.html#autotoc_md31", null ],
        [ "boundary_policy", "md_docs_2api.html#autotoc_md32", null ],
        [ "scaling_method", "md_docs_2api.html#autotoc_md33", null ],
        [ "zero_weight_fallback", "md_docs_2api.html#autotoc_md34", null ],
        [ "degree", "md_docs_2api.html#autotoc_md35", null ],
        [ "distance_metric", "md_docs_2api.html#autotoc_md36", null ],
        [ "surface_mode", "md_docs_2api.html#autotoc_md37", null ],
        [ "merge_strategy", "md_docs_2api.html#autotoc_md38", null ],
        [ "update_mode", "md_docs_2api.html#autotoc_md39", null ]
      ] ],
      [ "Example", "md_docs_2api.html#autotoc_md40", null ]
    ] ],
    [ "Batch Adapter", "md_docs_2batch.html", [
      [ "When to Use", "md_docs_2batch.html#autotoc_md42", null ],
      [ "Example", "md_docs_2batch.html#autotoc_md43", null ]
    ] ],
    [ "boundary", "md_docs_2boundary.html", [
      [ "Boundary Handling", "md_docs_2boundary.html#autotoc_md45", [
        [ "Overview", "md_docs_2boundary.html#autotoc_md46", null ],
        [ "Extend (Default)", "md_docs_2boundary.html#autotoc_md48", null ],
        [ "Reflect", "md_docs_2boundary.html#autotoc_md50", null ],
        [ "Zero", "md_docs_2boundary.html#autotoc_md52", null ],
        [ "No Boundary", "md_docs_2boundary.html#autotoc_md54", null ],
        [ "Choosing a Policy", "md_docs_2boundary.html#autotoc_md56", null ]
      ] ]
    ] ],
    [ "cross-validation", "md_docs_2cross-validation.html", [
      [ "Cross-Validation", "md_docs_2cross-validation.html#autotoc_md73", [
        [ "Overview", "md_docs_2cross-validation.html#autotoc_md74", null ],
        [ "K-Fold Cross-Validation", "md_docs_2cross-validation.html#autotoc_md76", null ],
        [ "Leave-One-Out (LOOCV)", "md_docs_2cross-validation.html#autotoc_md78", null ],
        [ "Seeded Randomization", "md_docs_2cross-validation.html#autotoc_md80", null ],
        [ "Comparison", "md_docs_2cross-validation.html#autotoc_md82", null ],
        [ "CV Metrics", "md_docs_2cross-validation.html#autotoc_md84", null ],
        [ "Interpreting Results", "md_docs_2cross-validation.html#autotoc_md86", null ],
        [ "Availability", "md_docs_2cross-validation.html#autotoc_md88", null ],
        [ "Best Practices", "md_docs_2cross-validation.html#autotoc_md90", null ]
      ] ]
    ] ],
    [ "degree", "md_docs_2degree.html", [
      [ "Polynomial Degree", "md_docs_2degree.html#autotoc_md91", [
        [ "Overview", "md_docs_2degree.html#autotoc_md92", null ],
        [ "Degree 0 — Local Constant", "md_docs_2degree.html#autotoc_md94", null ],
        [ "Degree 1 — Local Linear (Default)", "md_docs_2degree.html#autotoc_md96", null ],
        [ "Degree 2 — Local Quadratic", "md_docs_2degree.html#autotoc_md98", null ],
        [ "Degree 3 — Local Cubic", "md_docs_2degree.html#autotoc_md100", null ],
        [ "Degree 4 — Local Quartic", "md_docs_2degree.html#autotoc_md102", null ],
        [ "Choosing the Right Degree", "md_docs_2degree.html#autotoc_md104", null ],
        [ "Higher Degree Effects", "md_docs_2degree.html#autotoc_md106", null ],
        [ "Surface Mode", "md_docs_2degree.html#autotoc_md108", null ]
      ] ]
    ] ],
    [ "dimensions", "md_docs_2dimensions.html", [
      [ "Multivariate LOESS", "md_docs_2dimensions.html#autotoc_md109", [
        [ "Overview", "md_docs_2dimensions.html#autotoc_md110", null ],
        [ "1D — Standard (Default)", "md_docs_2dimensions.html#autotoc_md112", null ],
        [ "2D — Spatial Surface", "md_docs_2dimensions.html#autotoc_md114", null ],
        [ "3D and Higher", "md_docs_2dimensions.html#autotoc_md116", null ],
        [ "Distance Metrics for Multivariate Data", "md_docs_2dimensions.html#autotoc_md118", null ]
      ] ]
    ] ],
    [ "installation", "md_docs_2installation.html", [
      [ "Installation", "md_docs_2installation.html#autotoc_md119", [
        [ "Pre-built Binaries (Linux (x64))", "md_docs_2installation.html#autotoc_md120", null ],
        [ "Pre-built Binaries (macOS (x64))", "md_docs_2installation.html#autotoc_md121", null ],
        [ "Pre-built Binaries (Windows (x64))", "md_docs_2installation.html#autotoc_md122", null ],
        [ "From Source", "md_docs_2installation.html#autotoc_md123", null ],
        [ "From conda-forge", "md_docs_2installation.html#autotoc_md124", null ],
        [ "Verify Installation", "md_docs_2installation.html#autotoc_md126", null ]
      ] ]
    ] ],
    [ "intervals", "md_docs_2intervals.html", [
      [ "Intervals", "md_docs_2intervals.html#autotoc_md127", [
        [ "Overview", "md_docs_2intervals.html#autotoc_md128", null ],
        [ "Confidence Intervals", "md_docs_2intervals.html#autotoc_md130", null ],
        [ "Prediction Intervals", "md_docs_2intervals.html#autotoc_md132", null ],
        [ "Both Intervals", "md_docs_2intervals.html#autotoc_md134", null ],
        [ "Confidence Levels", "md_docs_2intervals.html#autotoc_md136", null ],
        [ "Standard Errors", "md_docs_2intervals.html#autotoc_md138", null ],
        [ "Availability", "md_docs_2intervals.html#autotoc_md140", null ]
      ] ]
    ] ],
    [ "kernels", "md_docs_2kernels.html", [
      [ "Weight Functions", "md_docs_2kernels.html#autotoc_md141", [
        [ "Overview", "md_docs_2kernels.html#autotoc_md142", null ],
        [ "Available Kernels", "md_docs_2kernels.html#autotoc_md144", null ],
        [ "Tricube (Default)", "md_docs_2kernels.html#autotoc_md146", null ],
        [ "Epanechnikov", "md_docs_2kernels.html#autotoc_md148", null ],
        [ "Gaussian", "md_docs_2kernels.html#autotoc_md150", null ],
        [ "Biweight", "md_docs_2kernels.html#autotoc_md152", null ],
        [ "Cosine", "md_docs_2kernels.html#autotoc_md154", null ],
        [ "Triangle", "md_docs_2kernels.html#autotoc_md156", null ],
        [ "Uniform", "md_docs_2kernels.html#autotoc_md158", null ],
        [ "Choosing a Kernel", "md_docs_2kernels.html#autotoc_md160", null ]
      ] ]
    ] ],
    [ "merge", "md_docs_2merge.html", [
      [ "Merge Strategies", "md_docs_2merge.html#autotoc_md161", [
        [ "Overview", "md_docs_2merge.html#autotoc_md162", null ],
        [ "Average", "md_docs_2merge.html#autotoc_md164", null ],
        [ "Take First", "md_docs_2merge.html#autotoc_md166", null ],
        [ "Take Last", "md_docs_2merge.html#autotoc_md168", null ],
        [ "Weighted Average", "md_docs_2merge.html#autotoc_md170", null ],
        [ "Choosing a Strategy", "md_docs_2merge.html#autotoc_md172", null ]
      ] ]
    ] ],
    [ "Online Adapter", "md_docs_2online.html", [
      [ "When to Use", "md_docs_2online.html#autotoc_md174", null ],
      [ "Parameters", "md_docs_2online.html#autotoc_md175", null ],
      [ "Update Modes", "md_docs_2online.html#autotoc_md176", null ],
      [ "Example", "md_docs_2online.html#autotoc_md177", null ]
    ] ],
    [ "parameters", "md_docs_2parameters.html", [
      [ "Parameters", "md_docs_2parameters.html#autotoc_md179", [
        [ "Quick Reference", "md_docs_2parameters.html#autotoc_md180", null ],
        [ "Parameter Options Summary", "md_docs_2parameters.html#autotoc_md182", null ],
        [ "Core Parameters", "md_docs_2parameters.html#autotoc_md184", [
          [ "fraction", "md_docs_2parameters.html#autotoc_md185", null ],
          [ "iterations", "md_docs_2parameters.html#autotoc_md187", null ],
          [ "degree", "md_docs_2parameters.html#autotoc_md189", null ],
          [ "surface_mode", "md_docs_2parameters.html#autotoc_md191", null ],
          [ "cell", "md_docs_2parameters.html#autotoc_md193", null ],
          [ "interpolation_vertices", "md_docs_2parameters.html#autotoc_md195", null ],
          [ "dimensions", "md_docs_2parameters.html#autotoc_md197", null ],
          [ "distance_metric / weighted_metric_weights", "md_docs_2parameters.html#autotoc_md199", null ],
          [ "weight_function", "md_docs_2parameters.html#autotoc_md201", null ],
          [ "robustness_method", "md_docs_2parameters.html#autotoc_md203", null ],
          [ "boundary_policy", "md_docs_2parameters.html#autotoc_md205", null ],
          [ "boundary_degree_fallback", "md_docs_2parameters.html#autotoc_md207", null ],
          [ "scaling_method", "md_docs_2parameters.html#autotoc_md209", null ],
          [ "zero_weight_fallback", "md_docs_2parameters.html#autotoc_md211", null ],
          [ "auto_converge", "md_docs_2parameters.html#autotoc_md213", null ],
          [ "parallel", "md_docs_2parameters.html#autotoc_md215", null ],
          [ "custom_weights", "md_docs_2parameters.html#autotoc_md217", null ]
        ] ],
        [ "Output Options", "md_docs_2parameters.html#autotoc_md219", [
          [ "return_residuals", "md_docs_2parameters.html#autotoc_md220", null ],
          [ "return_diagnostics", "md_docs_2parameters.html#autotoc_md222", null ],
          [ "return_robustness_weights", "md_docs_2parameters.html#autotoc_md224", null ],
          [ "confidence_intervals / prediction_intervals", "md_docs_2parameters.html#autotoc_md226", null ]
        ] ],
        [ "CV Methods", "md_docs_2parameters.html#autotoc_md228", [
          [ "cv_method", "md_docs_2parameters.html#autotoc_md229", null ]
        ] ],
        [ "Adapter Parameters", "md_docs_2parameters.html#autotoc_md231", [
          [ "chunk_size", "md_docs_2parameters.html#autotoc_md232", null ],
          [ "overlap", "md_docs_2parameters.html#autotoc_md234", null ],
          [ "merge_strategy", "md_docs_2parameters.html#autotoc_md236", null ],
          [ "window_capacity", "md_docs_2parameters.html#autotoc_md238", null ],
          [ "min_points", "md_docs_2parameters.html#autotoc_md240", null ],
          [ "update_mode", "md_docs_2parameters.html#autotoc_md242", null ]
        ] ]
      ] ]
    ] ],
    [ "quickstart", "md_docs_2quickstart.html", [
      [ "Quick Start", "md_docs_2quickstart.html#autotoc_md243", [
        [ "Basic Smoothing", "md_docs_2quickstart.html#autotoc_md244", null ],
        [ "With Confidence Intervals", "md_docs_2quickstart.html#autotoc_md246", null ],
        [ "Handling Outliers", "md_docs_2quickstart.html#autotoc_md248", null ],
        [ "Streaming Mode", "md_docs_2quickstart.html#autotoc_md250", null ],
        [ "Next Steps", "md_docs_2quickstart.html#autotoc_md252", null ]
      ] ]
    ] ],
    [ "robustness", "md_docs_2robustness.html", [
      [ "Robustness", "md_docs_2robustness.html#autotoc_md253", [
        [ "How Robustness Works", "md_docs_2robustness.html#autotoc_md254", null ],
        [ "Robustness Methods", "md_docs_2robustness.html#autotoc_md256", [
          [ "Bisquare (Default)", "md_docs_2robustness.html#autotoc_md257", null ],
          [ "Huber", "md_docs_2robustness.html#autotoc_md259", null ],
          [ "Talwar", "md_docs_2robustness.html#autotoc_md261", null ]
        ] ],
        [ "Comparison", "md_docs_2robustness.html#autotoc_md263", null ],
        [ "Detecting Outliers", "md_docs_2robustness.html#autotoc_md265", null ],
        [ "Scale Estimation", "md_docs_2robustness.html#autotoc_md267", null ],
        [ "Auto-Convergence", "md_docs_2robustness.html#autotoc_md269", null ]
      ] ]
    ] ],
    [ "scaling", "md_docs_2scaling.html", [
      [ "Scaling Methods", "md_docs_2scaling.html#autotoc_md270", [
        [ "Overview", "md_docs_2scaling.html#autotoc_md271", null ],
        [ "MAD — Median Absolute Deviation (Default)", "md_docs_2scaling.html#autotoc_md273", null ],
        [ "MAR — Median Absolute Residual", "md_docs_2scaling.html#autotoc_md275", null ],
        [ "Mean — Mean Absolute Residual", "md_docs_2scaling.html#autotoc_md277", null ],
        [ "Choosing a Scaling Method", "md_docs_2scaling.html#autotoc_md279", null ]
      ] ]
    ] ],
    [ "Streaming Adapter", "md_docs_2streaming.html", [
      [ "When to Use", "md_docs_2streaming.html#autotoc_md281", null ],
      [ "Parameters", "md_docs_2streaming.html#autotoc_md282", null ],
      [ "Merge Strategies", "md_docs_2streaming.html#autotoc_md283", null ],
      [ "Example", "md_docs_2streaming.html#autotoc_md284", null ]
    ] ],
    [ "use-case-genomics", "md_docs_2use-case-genomics.html", [
      [ "Genomic Data Smoothing", "md_docs_2use-case-genomics.html#autotoc_md286", [
        [ "Overview", "md_docs_2use-case-genomics.html#autotoc_md287", null ],
        [ "Methylation Profile Smoothing", "md_docs_2use-case-genomics.html#autotoc_md289", [
          [ "The Challenge", "md_docs_2use-case-genomics.html#autotoc_md290", null ],
          [ "Solution", "md_docs_2use-case-genomics.html#autotoc_md291", null ]
        ] ],
        [ "ChIP-seq Signal Smoothing", "md_docs_2use-case-genomics.html#autotoc_md293", [
          [ "Application", "md_docs_2use-case-genomics.html#autotoc_md294", null ]
        ] ],
        [ "Large Genome Coverage (Streaming)", "md_docs_2use-case-genomics.html#autotoc_md296", null ],
        [ "Best Practices for Genomic Data", "md_docs_2use-case-genomics.html#autotoc_md298", null ],
        [ "See Also", "md_docs_2use-case-genomics.html#autotoc_md300", null ]
      ] ]
    ] ],
    [ "use-case-real-time", "md_docs_2use-case-real-time.html", [
      [ "Real-Time Processing", "md_docs_2use-case-real-time.html#autotoc_md301", [
        [ "Overview", "md_docs_2use-case-real-time.html#autotoc_md302", null ],
        [ "Online Mode: Point-by-Point", "md_docs_2use-case-real-time.html#autotoc_md304", [
          [ "Sensor Data Example", "md_docs_2use-case-real-time.html#autotoc_md305", null ]
        ] ],
        [ "Streaming Mode: Chunk Processing", "md_docs_2use-case-real-time.html#autotoc_md307", [
          [ "Log File Processing", "md_docs_2use-case-real-time.html#autotoc_md308", null ]
        ] ],
        [ "Real-Time Dashboard Example", "md_docs_2use-case-real-time.html#autotoc_md310", null ],
        [ "Choosing Parameters", "md_docs_2use-case-real-time.html#autotoc_md312", [
          [ "Online Mode", "md_docs_2use-case-real-time.html#autotoc_md313", null ],
          [ "Streaming Mode", "md_docs_2use-case-real-time.html#autotoc_md314", null ]
        ] ],
        [ "Performance Considerations", "md_docs_2use-case-real-time.html#autotoc_md316", null ],
        [ "See Also", "md_docs_2use-case-real-time.html#autotoc_md318", null ]
      ] ]
    ] ],
    [ "use-case-time-series", "md_docs_2use-case-time-series.html", [
      [ "Time Series Analysis", "md_docs_2use-case-time-series.html#autotoc_md319", [
        [ "Overview", "md_docs_2use-case-time-series.html#autotoc_md320", null ],
        [ "Basic Trend Extraction", "md_docs_2use-case-time-series.html#autotoc_md322", null ],
        [ "Detrending", "md_docs_2use-case-time-series.html#autotoc_md324", null ],
        [ "Forecasting with Prediction Intervals", "md_docs_2use-case-time-series.html#autotoc_md326", null ],
        [ "Handling Missing Data", "md_docs_2use-case-time-series.html#autotoc_md328", null ],
        [ "Multi-Scale Analysis", "md_docs_2use-case-time-series.html#autotoc_md330", null ],
        [ "Gene Expression Time Course", "md_docs_2use-case-time-series.html#autotoc_md332", null ],
        [ "Choosing Fraction for Time Series", "md_docs_2use-case-time-series.html#autotoc_md334", null ],
        [ "See Also", "md_docs_2use-case-time-series.html#autotoc_md336", null ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"index.html"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';