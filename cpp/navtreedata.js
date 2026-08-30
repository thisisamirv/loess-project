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
      [ "Installation & Documentation", "index.html#autotoc_md2", null ],
      [ "LOESS vs. LOWESS", "index.html#autotoc_md3", null ],
      [ "Why this package?", "index.html#autotoc_md4", [
        [ "Speed", "index.html#autotoc_md5", null ],
        [ "Robustness", "index.html#autotoc_md6", null ],
        [ "Features", "index.html#autotoc_md7", null ]
      ] ],
      [ "Validation", "index.html#autotoc_md8", null ],
      [ "Contributing", "index.html#autotoc_md10", null ],
      [ "License", "index.html#autotoc_md11", null ],
      [ "Citation", "index.html#autotoc_md12", null ]
    ] ],
    [ "Execution Modes", "md_docs_2adapter-choice.html", [
      [ "Overview", "md_docs_2adapter-choice.html#autotoc_md14", null ]
    ] ],
    [ "OnlineLoess API", "md_docs_2api-online.html", [
      [ "When to Use", "md_docs_2api-online.html#autotoc_md17", null ],
      [ "Class", "md_docs_2api-online.html#autotoc_md18", [
        [ "fastloess::OnlineLoess", "md_docs_2api-online.html#autotoc_md19", null ]
      ] ],
      [ "Options Structure", "md_docs_2api-online.html#autotoc_md20", [
        [ "OnlineOptions (inherits LoessOptions)", "md_docs_2api-online.html#autotoc_md21", null ]
      ] ],
      [ "Result Structure", "md_docs_2api-online.html#autotoc_md22", [
        [ "fastloess::OnlineOutput", "md_docs_2api-online.html#autotoc_md23", null ]
      ] ],
      [ "Options", "md_docs_2api-online.html#autotoc_md24", [
        [ "update_mode", "md_docs_2api-online.html#autotoc_md25", null ]
      ] ]
    ] ],
    [ "StreamingLoess API", "md_docs_2api-streaming.html", [
      [ "When to Use", "md_docs_2api-streaming.html#autotoc_md27", null ],
      [ "Class", "md_docs_2api-streaming.html#autotoc_md28", [
        [ "fastloess::StreamingLoess", "md_docs_2api-streaming.html#autotoc_md29", null ]
      ] ],
      [ "Result Structure", "md_docs_2api-streaming.html#autotoc_md30", [
        [ "fastloess::LoessResult", "md_docs_2api-streaming.html#autotoc_md31", null ]
      ] ],
      [ "Options Structure", "md_docs_2api-streaming.html#autotoc_md32", [
        [ "StreamingOptions (inherits LoessOptions)", "md_docs_2api-streaming.html#autotoc_md33", null ]
      ] ],
      [ "Options", "md_docs_2api-streaming.html#autotoc_md34", [
        [ "merge_strategy", "md_docs_2api-streaming.html#autotoc_md35", null ]
      ] ]
    ] ],
    [ "API", "md_docs_2api.html", [
      [ "When to Use Batch Adapter", "md_docs_2api.html#autotoc_md38", null ],
      [ "Classes", "md_docs_2api.html#autotoc_md39", [
        [ "fastloess::Loess", "md_docs_2api.html#autotoc_md40", null ]
      ] ],
      [ "Options Structures", "md_docs_2api.html#autotoc_md41", [
        [ "LoessOptions", "md_docs_2api.html#autotoc_md42", null ]
      ] ],
      [ "Result Structure", "md_docs_2api.html#autotoc_md43", [
        [ "fastloess::LoessResult", "md_docs_2api.html#autotoc_md44", null ],
        [ "fastloess::Diagnostics", "md_docs_2api.html#autotoc_md45", null ]
      ] ],
      [ "Options", "md_docs_2api.html#autotoc_md46", [
        [ "weight_function", "md_docs_2api.html#autotoc_md47", null ],
        [ "robustness_method", "md_docs_2api.html#autotoc_md48", null ],
        [ "boundary_policy", "md_docs_2api.html#autotoc_md49", null ],
        [ "scaling_method", "md_docs_2api.html#autotoc_md50", null ],
        [ "zero_weight_fallback", "md_docs_2api.html#autotoc_md51", null ],
        [ "degree", "md_docs_2api.html#autotoc_md52", null ],
        [ "distance_metric", "md_docs_2api.html#autotoc_md53", null ],
        [ "surface_mode", "md_docs_2api.html#autotoc_md54", null ]
      ] ],
      [ "Example", "md_docs_2api.html#autotoc_md55", null ]
    ] ],
    [ "Benchmarks", "md_docs_2benchmarks.html", [
      [ "CPU Benchmarks", "md_docs_2benchmarks.html#autotoc_md57", null ],
      [ "Reproducing Benchmarks", "md_docs_2benchmarks.html#autotoc_md59", null ]
    ] ],
    [ "Boundary Handling", "md_docs_2boundary.html", [
      [ "Overview", "md_docs_2boundary.html#autotoc_md61", null ],
      [ "Extend (Default)", "md_docs_2boundary.html#autotoc_md63", null ],
      [ "Reflect", "md_docs_2boundary.html#autotoc_md65", null ],
      [ "Zero", "md_docs_2boundary.html#autotoc_md67", null ],
      [ "No Boundary", "md_docs_2boundary.html#autotoc_md69", null ],
      [ "Choosing a Policy", "md_docs_2boundary.html#autotoc_md71", null ]
    ] ],
    [ "Concepts", "md_docs_2concepts.html", [
      [ "What is LOESS?", "md_docs_2concepts.html#autotoc_md73", null ],
      [ "How It Works", "md_docs_2concepts.html#autotoc_md75", null ],
      [ "The Fraction Parameter", "md_docs_2concepts.html#autotoc_md77", null ],
      [ "Robustness Iterations", "md_docs_2concepts.html#autotoc_md78", null ],
      [ "Confidence vs Prediction Intervals", "md_docs_2concepts.html#autotoc_md80", null ],
      [ "Execution Modes", "md_docs_2concepts.html#autotoc_md82", null ],
      [ "Quick Decision Guide", "md_docs_2concepts.html#autotoc_md84", null ],
      [ "Key Advantages", "md_docs_2concepts.html#autotoc_md86", null ],
      [ "Next Steps", "md_docs_2concepts.html#autotoc_md88", null ]
    ] ],
    [ "Cross-Validation", "md_docs_2cross-validation.html", [
      [ "Overview", "md_docs_2cross-validation.html#autotoc_md90", null ],
      [ "K-Fold Cross-Validation", "md_docs_2cross-validation.html#autotoc_md92", null ],
      [ "Leave-One-Out (LOOCV)", "md_docs_2cross-validation.html#autotoc_md94", null ],
      [ "Seeded Randomization", "md_docs_2cross-validation.html#autotoc_md96", null ],
      [ "Comparison", "md_docs_2cross-validation.html#autotoc_md98", null ],
      [ "CV Metrics", "md_docs_2cross-validation.html#autotoc_md99", null ],
      [ "Interpreting Results", "md_docs_2cross-validation.html#autotoc_md101", null ],
      [ "Availability", "md_docs_2cross-validation.html#autotoc_md103", null ],
      [ "Best Practices", "md_docs_2cross-validation.html#autotoc_md105", null ]
    ] ],
    [ "Custom Weights", "md_docs_2custom-weights.html", [
      [ "How Custom Weights Work", "md_docs_2custom-weights.html#autotoc_md107", null ],
      [ "When to Use Custom Weights", "md_docs_2custom-weights.html#autotoc_md108", [
        [ "Custom Weights vs. Robustness Iterations", "md_docs_2custom-weights.html#autotoc_md109", null ]
      ] ],
      [ "Basic Usage", "md_docs_2custom-weights.html#autotoc_md111", [
        [ "Suppress a Known Outlier", "md_docs_2custom-weights.html#autotoc_md112", null ]
      ] ],
      [ "Validation Rules", "md_docs_2custom-weights.html#autotoc_md114", null ],
      [ "See Also", "md_docs_2custom-weights.html#autotoc_md115", null ]
    ] ],
    [ "Polynomial Degree", "md_docs_2degree.html", [
      [ "Overview", "md_docs_2degree.html#autotoc_md117", null ],
      [ "Degree 0 — Local Constant", "md_docs_2degree.html#autotoc_md119", null ],
      [ "Degree 1 — Local Linear (Default)", "md_docs_2degree.html#autotoc_md121", null ],
      [ "Degree 2 — Local Quadratic", "md_docs_2degree.html#autotoc_md123", null ],
      [ "Degree 3 — Local Cubic", "md_docs_2degree.html#autotoc_md125", null ],
      [ "Degree 4 — Local Quartic", "md_docs_2degree.html#autotoc_md127", null ],
      [ "Choosing the Right Degree", "md_docs_2degree.html#autotoc_md129", null ],
      [ "Higher Degree Effects", "md_docs_2degree.html#autotoc_md131", null ],
      [ "Surface Mode", "md_docs_2degree.html#autotoc_md133", null ]
    ] ],
    [ "Multivariate LOESS", "md_docs_2dimensions.html", [
      [ "Overview", "md_docs_2dimensions.html#autotoc_md135", null ],
      [ "1D — Standard (Default)", "md_docs_2dimensions.html#autotoc_md136", null ],
      [ "2D — Spatial Surface", "md_docs_2dimensions.html#autotoc_md138", null ],
      [ "3D and Higher", "md_docs_2dimensions.html#autotoc_md140", null ],
      [ "Distance Metrics for Multivariate Data", "md_docs_2dimensions.html#autotoc_md142", null ]
    ] ],
    [ "Installation", "md_docs_2installation.html", [
      [ "Pre-built Binaries (Linux (x64))", "md_docs_2installation.html#autotoc_md144", null ],
      [ "Pre-built Binaries (macOS (x64))", "md_docs_2installation.html#autotoc_md145", null ],
      [ "Pre-built Binaries (Windows (x64))", "md_docs_2installation.html#autotoc_md146", null ],
      [ "From Source", "md_docs_2installation.html#autotoc_md147", null ],
      [ "From conda-forge", "md_docs_2installation.html#autotoc_md148", null ],
      [ "Verify Installation", "md_docs_2installation.html#autotoc_md150", null ]
    ] ],
    [ "Intervals", "md_docs_2intervals.html", [
      [ "Overview", "md_docs_2intervals.html#autotoc_md152", null ],
      [ "Confidence Intervals", "md_docs_2intervals.html#autotoc_md154", null ],
      [ "Prediction Intervals", "md_docs_2intervals.html#autotoc_md156", null ],
      [ "Both Intervals", "md_docs_2intervals.html#autotoc_md158", null ],
      [ "Confidence Levels", "md_docs_2intervals.html#autotoc_md160", null ],
      [ "Standard Errors", "md_docs_2intervals.html#autotoc_md162", null ],
      [ "Availability", "md_docs_2intervals.html#autotoc_md164", null ]
    ] ],
    [ "Weight Functions", "md_docs_2kernels.html", [
      [ "Overview", "md_docs_2kernels.html#autotoc_md166", null ],
      [ "Available Kernels", "md_docs_2kernels.html#autotoc_md168", null ],
      [ "Tricube (Default)", "md_docs_2kernels.html#autotoc_md170", null ],
      [ "Epanechnikov", "md_docs_2kernels.html#autotoc_md172", null ],
      [ "Gaussian", "md_docs_2kernels.html#autotoc_md174", null ],
      [ "Biweight", "md_docs_2kernels.html#autotoc_md176", null ],
      [ "Cosine", "md_docs_2kernels.html#autotoc_md178", null ],
      [ "Triangle", "md_docs_2kernels.html#autotoc_md180", null ],
      [ "Uniform", "md_docs_2kernels.html#autotoc_md182", null ],
      [ "Choosing a Kernel", "md_docs_2kernels.html#autotoc_md184", null ]
    ] ],
    [ "Merge Strategies", "md_docs_2merge.html", [
      [ "Overview", "md_docs_2merge.html#autotoc_md186", null ],
      [ "Average", "md_docs_2merge.html#autotoc_md188", null ],
      [ "Take First", "md_docs_2merge.html#autotoc_md190", null ],
      [ "Take Last", "md_docs_2merge.html#autotoc_md192", null ],
      [ "Weighted Average", "md_docs_2merge.html#autotoc_md194", null ],
      [ "Choosing a Strategy", "md_docs_2merge.html#autotoc_md196", null ]
    ] ],
    [ "NEWS", "md_docs_2NEWS.html", [
      [ "fastloess (C++) (development version)", "md_docs_2NEWS.html#autotoc_md197", [
        [ "Changed", "md_docs_2NEWS.html#autotoc_md198", null ],
        [ "Fixed", "md_docs_2NEWS.html#autotoc_md199", null ]
      ] ],
      [ "fastloess (C++) 1.1.0", "md_docs_2NEWS.html#autotoc_md200", [
        [ "Added", "md_docs_2NEWS.html#autotoc_md201", null ],
        [ "Changed", "md_docs_2NEWS.html#autotoc_md202", null ],
        [ "Fixed", "md_docs_2NEWS.html#autotoc_md203", null ]
      ] ],
      [ "fastloess (C++) 1.0.0", "md_docs_2NEWS.html#autotoc_md204", [
        [ "Changed", "md_docs_2NEWS.html#autotoc_md205", null ]
      ] ],
      [ "fastloess (C++) 0.9.0", "md_docs_2NEWS.html#autotoc_md206", [
        [ "Added", "md_docs_2NEWS.html#autotoc_md207", null ],
        [ "Changed", "md_docs_2NEWS.html#autotoc_md208", null ]
      ] ]
    ] ],
    [ "Quick Start", "md_docs_2quickstart.html", [
      [ "Basic Smoothing", "md_docs_2quickstart.html#autotoc_md210", null ],
      [ "With Confidence Intervals", "md_docs_2quickstart.html#autotoc_md212", null ],
      [ "Handling Outliers", "md_docs_2quickstart.html#autotoc_md214", null ],
      [ "Streaming Mode", "md_docs_2quickstart.html#autotoc_md216", null ],
      [ "Next Steps", "md_docs_2quickstart.html#autotoc_md218", null ]
    ] ],
    [ "Robustness", "md_docs_2robustness.html", [
      [ "How Robustness Works", "md_docs_2robustness.html#autotoc_md220", null ],
      [ "Robustness Methods", "md_docs_2robustness.html#autotoc_md222", [
        [ "Bisquare (Default)", "md_docs_2robustness.html#autotoc_md223", null ],
        [ "Huber", "md_docs_2robustness.html#autotoc_md225", null ],
        [ "Talwar", "md_docs_2robustness.html#autotoc_md227", null ]
      ] ],
      [ "Comparison", "md_docs_2robustness.html#autotoc_md229", null ],
      [ "Detecting Outliers", "md_docs_2robustness.html#autotoc_md231", null ],
      [ "Scale Estimation", "md_docs_2robustness.html#autotoc_md233", null ],
      [ "Auto-Convergence", "md_docs_2robustness.html#autotoc_md235", null ]
    ] ],
    [ "Scaling Methods", "md_docs_2scaling.html", [
      [ "Overview", "md_docs_2scaling.html#autotoc_md237", null ],
      [ "MAD — Median Absolute Deviation (Default)", "md_docs_2scaling.html#autotoc_md239", null ],
      [ "MAR — Median Absolute Residual", "md_docs_2scaling.html#autotoc_md241", null ],
      [ "Mean — Mean Absolute Residual", "md_docs_2scaling.html#autotoc_md243", null ],
      [ "Choosing a Scaling Method", "md_docs_2scaling.html#autotoc_md245", null ]
    ] ],
    [ "Genomic Data Smoothing", "md_docs_2use-case-genomics.html", [
      [ "Overview", "md_docs_2use-case-genomics.html#autotoc_md247", null ],
      [ "Methylation Profile Smoothing", "md_docs_2use-case-genomics.html#autotoc_md249", [
        [ "The Challenge", "md_docs_2use-case-genomics.html#autotoc_md250", null ],
        [ "Solution", "md_docs_2use-case-genomics.html#autotoc_md251", null ]
      ] ],
      [ "ChIP-seq Signal Smoothing", "md_docs_2use-case-genomics.html#autotoc_md253", [
        [ "Application", "md_docs_2use-case-genomics.html#autotoc_md254", null ]
      ] ],
      [ "Large Genome Coverage (Streaming)", "md_docs_2use-case-genomics.html#autotoc_md256", null ],
      [ "Best Practices for Genomic Data", "md_docs_2use-case-genomics.html#autotoc_md258", null ],
      [ "See Also", "md_docs_2use-case-genomics.html#autotoc_md260", null ]
    ] ],
    [ "Real-Time Processing", "md_docs_2use-case-real-time.html", [
      [ "Overview", "md_docs_2use-case-real-time.html#autotoc_md262", null ],
      [ "Online Mode: Point-by-Point", "md_docs_2use-case-real-time.html#autotoc_md264", [
        [ "Sensor Data Example", "md_docs_2use-case-real-time.html#autotoc_md265", null ]
      ] ],
      [ "Streaming Mode: Chunk Processing", "md_docs_2use-case-real-time.html#autotoc_md267", [
        [ "Log File Processing", "md_docs_2use-case-real-time.html#autotoc_md268", null ]
      ] ],
      [ "Real-Time Dashboard Example", "md_docs_2use-case-real-time.html#autotoc_md270", null ],
      [ "Choosing Parameters", "md_docs_2use-case-real-time.html#autotoc_md272", [
        [ "Online Mode", "md_docs_2use-case-real-time.html#autotoc_md273", null ],
        [ "Streaming Mode", "md_docs_2use-case-real-time.html#autotoc_md274", null ]
      ] ],
      [ "Performance Considerations", "md_docs_2use-case-real-time.html#autotoc_md276", null ],
      [ "See Also", "md_docs_2use-case-real-time.html#autotoc_md278", null ]
    ] ],
    [ "Time Series Analysis", "md_docs_2use-case-time-series.html", [
      [ "Overview", "md_docs_2use-case-time-series.html#autotoc_md280", null ],
      [ "Basic Trend Extraction", "md_docs_2use-case-time-series.html#autotoc_md282", null ],
      [ "Detrending", "md_docs_2use-case-time-series.html#autotoc_md284", null ],
      [ "Forecasting with Prediction Intervals", "md_docs_2use-case-time-series.html#autotoc_md286", null ],
      [ "Handling Missing Data", "md_docs_2use-case-time-series.html#autotoc_md288", null ],
      [ "Multi-Scale Analysis", "md_docs_2use-case-time-series.html#autotoc_md290", null ],
      [ "Gene Expression Time Course", "md_docs_2use-case-time-series.html#autotoc_md292", null ],
      [ "Choosing Fraction for Time Series", "md_docs_2use-case-time-series.html#autotoc_md294", null ],
      [ "See Also", "md_docs_2use-case-time-series.html#autotoc_md296", null ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"index.html"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';