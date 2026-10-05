"""
	FastLOESS

High-performance LOESS (Locally Estimated Scatterplot Smoothing) for Julia.

Provides bindings to the fastloess Rust library for fast, robust LOESS smoothing.

# Main API
- `Loess(; kwargs...)`: Configure batch LOESS
- `fit(model, x, y)`: Fit and return results
- `StreamingLoess(; kwargs...)` / `process_chunk` / `finalize`: Streaming mode
- `OnlineLoess(; kwargs...)` / `add_point`: Online sliding-window mode

# Example
```julia
using FastLOESS

x = collect(1.0:0.1:10.0)
y = sin.(x) .+ 0.1 .* randn(length(x))

result = fit(Loess(fraction=0.3), x, y)
println("Smoothed values: ", result.y)
```
"""
module FastLOESS

using TOML

export Loess, StreamingLoess, OnlineLoess
export fit, process_chunk, finalize, add_point, predict, window_diagnostics, predict_window
export LoessResult, OnlineOutput, Diagnostics, PredictModel, PredictResult
export version

import Base: finalize

function _package_version()
    project_file = joinpath(dirname(@__DIR__), "Project.toml")
    return get(TOML.parsefile(project_file), "version", "unknown")
end

"""
	version() -> String

Return the installed FastLOESS Julia package version from `Project.toml`.
This is the Julia binding version, not the native Rust library version.
"""
version() = _package_version()

function _output_flags(
    outputs;
    allowed = (
        "diagnostics",
        "residuals",
        "weights",
        "gradient",
        "derivative",
        "se",
        "sorted",
    ),
)
    selected = String.(outputs)
    unknown = setdiff(selected, String.(allowed))
    isempty(unknown) || throw(ArgumentError("Unknown outputs: $(join(unknown, ", "))"))
    return (
        diagnostics = "diagnostics" in selected,
        residuals = "residuals" in selected,
        weights = "weights" in selected,
        gradient = "gradient" in selected || "derivative" in selected,
        se = "se" in selected,
        sorted = "sorted" in selected,
    )
end

function _check_group_keys(group, allowed, name)
    unknown = setdiff(Symbol.(collect(keys(group))), collect(allowed))
    isempty(unknown) ||
        throw(ArgumentError("Unknown $name options: $(join(unknown, ", "))"))
end

function _interval_options(intervals)
    intervals === nothing && return (confidence = NaN, prediction = NaN)
    _check_group_keys(intervals, (:confidence, :prediction), "intervals")
    confidence = get(intervals, :confidence, nothing)
    prediction = get(intervals, :prediction, nothing)
    return (
        confidence = confidence === nothing ? NaN : Float64(confidence),
        prediction = prediction === nothing ? NaN : Float64(prediction),
    )
end

function _cv_options(cv)
    cv === nothing && return (fractions = Float64[], method = "kfold", k = 5)
    _check_group_keys(cv, (:fractions, :method, :k), "cv")
    haskey(cv, :fractions) || throw(ArgumentError("cv requires fractions"))
    return (
        fractions = Float64.(get(cv, :fractions, Float64[])),
        method = String(get(cv, :method, "kfold")),
        k = Int(get(cv, :k, 5)),
    )
end

function _constructor_error(fallback::String)
    error_ptr = ccall((:jl_last_error_message, libfastloess), Ptr{Cchar}, ())
    return error_ptr == C_NULL ? fallback : unsafe_string(error_ptr)
end

# Try to import JLL package first
try
    using fastloess_jll
catch e
    # JLL not available, will use fallback
end

# Library name varies by platform
const LIBNAME =
    Sys.iswindows() ? "fastloess_jl.dll" :
    Sys.isapple() ? "libfastloess_jl.dylib" : "libfastloess_jl.so"

# Try to load from JLL package first, fall back to local build
function find_library()
    # Option 1: Check environment variable (PRIORITY)
    if haskey(ENV, "FASTLOESS_LIB")
        lib = ENV["FASTLOESS_LIB"]
        @info "Using library from FASTLOESS_LIB: $lib"
        return lib
    end

    # Option 2: Use JLL package if available (for registered package)
    if @isdefined(fastloess_jll)
        try
            if hasproperty(fastloess_jll, :libfastloess_jl)
                lib = fastloess_jll.libfastloess_jl
                @info "Using fastloess_jll library: $lib"
                return lib
            end
        catch e
            @warn "Failed to load from fastloess_jll" exception = e
        end
    end

    # Option 3: Check relative paths (development mode)
    # Path: julia/src/fastloess.jl -> julia/ -> bindings/julia/ -> bindings/ -> loess-project/
    src_dir = @__DIR__                        # julia/src/
    julia_dir = dirname(src_dir)              # julia/
    bindings_julia_dir = dirname(julia_dir)   # bindings/julia/
    bindings_dir = dirname(bindings_julia_dir)# bindings/
    workspace_root = dirname(bindings_dir)    # loess-project/

    candidates = [
        # Workspace root target (most common for workspace members)
        joinpath(workspace_root, "target", "release", LIBNAME),
        joinpath(workspace_root, "target", "debug", LIBNAME),
        # Local target (if built standalone)
        joinpath(bindings_julia_dir, "target", "release", LIBNAME),
        joinpath(bindings_julia_dir, "target", "debug", LIBNAME),
        # Same directory as module
        joinpath(julia_dir, LIBNAME),
    ]

    for path ∈ candidates
        if isfile(path)
            @info "Using local library: $path"
            return path
        end
    end

    # Fall back to system path
    @warn "Library not found in JLL or local paths, falling back to system path"
    return LIBNAME
end

libfastloess = ""

function current_library()
    if isempty(libfastloess)
        global libfastloess = find_library()
    end
    return libfastloess
end

function __init__()
    global libfastloess = find_library()
end

"""
	Diagnostics

Diagnostic statistics for LOESS fit quality.

# Fields
- `rmse::Float64`: Root Mean Squared Error
- `mae::Float64`: Mean Absolute Error
- `r_squared::Float64`: R-squared (coefficient of determination)
- `aic::Union{Float64, Nothing}`: Akaike Information Criterion
- `aicc::Union{Float64, Nothing}`: Corrected AIC
- `effective_df::Union{Float64, Nothing}`: Effective degrees of freedom
- `residual_sd::Float64`: Batch uses robust residual scale estimate (`1.4826 * MAD`); Streaming uses cumulative sample SD of emitted residuals
"""
struct Diagnostics
    rmse::Float64
    mae::Float64
    r_squared::Float64
    aic::Union{Float64,Nothing}
    aicc::Union{Float64,Nothing}
    effective_df::Union{Float64,Nothing}
    residual_sd::Float64
end

# C FFI result struct for predict() (must match Rust definition).
struct CJlPredictResult
    y::Ptr{Cdouble}
    n::Culong
    standard_errors::Ptr{Cdouble}
    confidence_lower::Ptr{Cdouble}
    confidence_upper::Ptr{Cdouble}
    prediction_lower::Ptr{Cdouble}
    prediction_upper::Ptr{Cdouble}
    derivative::Ptr{Cdouble}
    dimensions::Cint
    error::Ptr{Cchar}
end

"""
	PredictResult

Result from `predict(model, new_x)`.

# Fields
- `y::Vector{Float64}`: Predicted y values, one per query point
- `standard_errors::Union{Vector{Float64}, Nothing}`: Standard errors (if requested)
- `confidence_lower::Union{Vector{Float64}, Nothing}`: Lower confidence bounds (if requested)
- `confidence_upper::Union{Vector{Float64}, Nothing}`: Upper confidence bounds (if requested)
- `prediction_lower::Union{Vector{Float64}, Nothing}`: Lower prediction bounds (if requested)
- `prediction_upper::Union{Vector{Float64}, Nothing}`: Upper prediction bounds (if requested)
- `derivative::Union{Vector{Float64}, Nothing}`: Local fit's gradient at each query point (if requested)
"""
struct PredictResult
    y::Vector{Float64}
    standard_errors::Union{Vector{Float64},Nothing}
    confidence_lower::Union{Vector{Float64},Nothing}
    confidence_upper::Union{Vector{Float64},Nothing}
    prediction_lower::Union{Vector{Float64},Nothing}
    prediction_upper::Union{Vector{Float64},Nothing}
    derivative::Union{Vector{Float64},Nothing}
end

"""
	PredictModel

Retained fitted-model state enabling out-of-sample `predict()`, obtained via
`LoessResult.predict_model` when `retain_model=true` was passed to `Loess`.
"""
mutable struct PredictModel
    handle::Ptr{Cvoid}

    function PredictModel(handle::Ptr{Cvoid})
        obj = new(handle)
        finalizer(
            x -> begin
                if x.handle != C_NULL
                    ccall(
                        (:jl_predict_handle_free, libfastloess),
                        Cvoid,
                        (Ptr{Cvoid},),
                        x.handle,
                    )
                end
            end,
            obj,
        )
        return obj
    end
end

"""
	predict(model::PredictModel, new_x::Vector{Float64}; kwargs...) -> PredictResult

Evaluate the fitted model at out-of-sample query points not in the training set
(flattened, `dimensions` values per point).

# Keyword Arguments
- `outputs::Vector{String} = String[]`: optional output components, including
	`"se"`, `"gradient"`, and/or `"derivative"`.
- `intervals = nothing`: grouped confidence and prediction coverage levels.
- `extrapolation::String = "clamp"`: one of "clamp", "linear", "error".
- `max_extrapolation_distance::Union{Float64, Nothing} = nothing`
- `max_neighbor_distance::Union{Float64, Nothing} = nothing`
"""
function predict(
    model::PredictModel,
    new_x::Vector{Float64};
    outputs::Vector{String} = String[],
    intervals = nothing,
    extrapolation::String = "clamp",
    max_extrapolation_distance::Union{Float64,Nothing} = nothing,
    max_neighbor_distance::Union{Float64,Nothing} = nothing,
)
    flags = _output_flags(outputs; allowed = ("se", "gradient", "derivative"))
    interval_options = _interval_options(intervals)

    if model.handle == C_NULL
        error(
            "fastloess error: predict() called on an invalid PredictModel (was retain_model set?)",
        )
    end

    c_result = GC.@preserve model new_x ccall(
        (:jl_predict, libfastloess),
        CJlPredictResult,
        (
            Ptr{Cvoid},
            Ptr{Cdouble},
            Culong,
            Cint,
            Cdouble,
            Cdouble,
            Cint,
            Cstring,
            Cdouble,
            Cdouble,
        ),
        model.handle,
        pointer(new_x),
        Culong(length(new_x)),
        Cint(flags.se),
        interval_options.confidence,
        interval_options.prediction,
        Cint(flags.gradient),
        extrapolation,
        (max_extrapolation_distance === nothing ? NaN : max_extrapolation_distance),
        (max_neighbor_distance === nothing ? NaN : max_neighbor_distance),
    )

    if c_result.error != C_NULL
        error_msg = unsafe_string(Ptr{UInt8}(c_result.error))
        ccall(
            (:jl_predict_free_result, libfastloess),
            Cvoid,
            (Ptr{CJlPredictResult},),
            Ref(c_result),
        )
        error("fastloess error: $error_msg")
    end

    n = Int(c_result.n)
    dims = max(Int(c_result.dimensions), 1)

    result = PredictResult(
        ptr_to_vector(c_result.y, n),
        ptr_to_vector(c_result.standard_errors, n),
        ptr_to_vector(c_result.confidence_lower, n),
        ptr_to_vector(c_result.confidence_upper, n),
        ptr_to_vector(c_result.prediction_lower, n),
        ptr_to_vector(c_result.prediction_upper, n),
        ptr_to_vector(c_result.derivative, n * dims),
    )

    ccall(
        (:jl_predict_free_result, libfastloess),
        Cvoid,
        (Ptr{CJlPredictResult},),
        Ref(c_result),
    )

    return result
end

"""
	LoessResult

Result from LOESS smoothing.

# Fields
- `x::Vector{Float64}`: x values (same order as input)
- `y::Vector{Float64}`: Smoothed y values
- `standard_errors::Union{Vector{Float64}, Nothing}`: Standard errors (if computed)
- `confidence_lower::Union{Vector{Float64}, Nothing}`: Lower confidence bounds
- `confidence_upper::Union{Vector{Float64}, Nothing}`: Upper confidence bounds
- `prediction_lower::Union{Vector{Float64}, Nothing}`: Lower prediction bounds
- `prediction_upper::Union{Vector{Float64}, Nothing}`: Upper prediction bounds
- `residuals::Union{Vector{Float64}, Nothing}`: Residuals
- `robustness_weights::Union{Vector{Float64}, Nothing}`: Robustness weights
- `gradient::Union{Vector{Float64}, Nothing}`: Local fit's gradient at each point
  (flattened, `dimensions` values per point); only populated when
  `surface_mode="direct"` was passed to `Loess`/`StreamingLoess`
- `fraction_used::Float64`: Fraction used for smoothing
- `iterations_used::Union{Int, Nothing}`: Number of iterations performed (`nothing` if not applicable)
- `diagnostics::Union{Diagnostics, Nothing}`: Diagnostic metrics
- `enp::Union{Float64, Nothing}`: Equivalent number of parameters
- `trace_hat::Union{Float64, Nothing}`: Trace of hat matrix
- `delta1::Union{Float64, Nothing}`: First delta statistic
- `delta2::Union{Float64, Nothing}`: Second delta statistic
- `residual_scale::Union{Float64, Nothing}`: Residual scale estimate
- `leverage::Union{Vector{Float64}, Nothing}`: Per-point leverage (hat matrix diagonal)
- `dimensions::Int`: Number of predictor dimensions
"""
struct LoessResult
    x::Vector{Float64}
    y::Vector{Float64}
    standard_errors::Union{Vector{Float64},Nothing}
    confidence_lower::Union{Vector{Float64},Nothing}
    confidence_upper::Union{Vector{Float64},Nothing}
    prediction_lower::Union{Vector{Float64},Nothing}
    prediction_upper::Union{Vector{Float64},Nothing}
    residuals::Union{Vector{Float64},Nothing}
    robustness_weights::Union{Vector{Float64},Nothing}
    gradient::Union{Vector{Float64},Nothing}
    fraction_used::Float64
    iterations_used::Union{Int,Nothing}
    diagnostics::Union{Diagnostics,Nothing}
    enp::Union{Float64,Nothing}
    trace_hat::Union{Float64,Nothing}
    delta1::Union{Float64,Nothing}
    delta2::Union{Float64,Nothing}
    residual_scale::Union{Float64,Nothing}
    leverage::Union{Vector{Float64},Nothing}
    dimensions::Int
    cv_scores::Union{Vector{Float64},Nothing}
    predict_model::Union{PredictModel,Nothing}
end

"""
	OnlineOutput

Result from a single `add_point` call.

# Fields
- `y::Float64`: Smoothed value for the latest point
- `standard_error::Union{Float64, Nothing}`: Standard error (if computed)
- `residual::Union{Float64, Nothing}`: Residual (raw input y minus this output's y) (if computed)
- `robustness_weight::Union{Float64, Nothing}`: Robustness weight (if computed)
- `iterations_used::Union{Int, Nothing}`: Number of robustness iterations
- `confidence_lower::Union{Float64, Nothing}`: Confidence interval lower bound
  (`update_mode="full"` only, if requested)
- `confidence_upper::Union{Float64, Nothing}`: Confidence interval upper bound
  (`update_mode="full"` only, if requested)
- `prediction_lower::Union{Float64, Nothing}`: Prediction interval lower bound
  (`update_mode="full"` only, if requested)
- `prediction_upper::Union{Float64, Nothing}`: Prediction interval upper bound
  (`update_mode="full"` only, if requested)
- `gradient::Union{Vector{Float64}, Nothing}`: Latest point's gradient (length =
  `dimensions`); only populated when `surface_mode="direct"` was passed to `OnlineLoess`
"""
struct OnlineOutput
    y::Float64
    standard_error::Union{Float64,Nothing}
    residual::Union{Float64,Nothing}
    robustness_weight::Union{Float64,Nothing}
    iterations_used::Union{Int,Nothing}
    confidence_lower::Union{Float64,Nothing}
    confidence_upper::Union{Float64,Nothing}
    prediction_lower::Union{Float64,Nothing}
    prediction_upper::Union{Float64,Nothing}
    gradient::Union{Vector{Float64},Nothing}
end

# C FFI struct for per-point online output (must match Rust definition).
struct CJlOnlineOutput
    has_value::Cint
    y::Cdouble
    standard_error::Cdouble
    residual::Cdouble
    robustness_weight::Cdouble
    iterations_used::Cint
    confidence_lower::Cdouble
    confidence_upper::Cdouble
    prediction_lower::Cdouble
    prediction_upper::Cdouble
    gradient::Ptr{Cdouble}
    dimensions::Cint
    error::Ptr{Cchar}
end

struct CJlOnlineDiagnostics
    has_value::Cint
    rmse::Cdouble
    mae::Cdouble
    r_squared::Cdouble
    aic::Cdouble
    aicc::Cdouble
    effective_df::Cdouble
    residual_sd::Cdouble
    error::Ptr{Cchar}
end

# C FFI result struct (must match Rust definition)
struct CJlLoessResult
    x::Ptr{Cdouble}
    y::Ptr{Cdouble}
    n::Culong
    standard_errors::Ptr{Cdouble}
    confidence_lower::Ptr{Cdouble}
    confidence_upper::Ptr{Cdouble}
    prediction_lower::Ptr{Cdouble}
    prediction_upper::Ptr{Cdouble}
    residuals::Ptr{Cdouble}
    robustness_weights::Ptr{Cdouble}
    gradient::Ptr{Cdouble}
    fraction_used::Cdouble
    iterations_used::Cint
    rmse::Cdouble
    mae::Cdouble
    r_squared::Cdouble
    aic::Cdouble
    aicc::Cdouble
    effective_df::Cdouble
    residual_sd::Cdouble
    enp::Cdouble
    trace_hat::Cdouble
    delta1::Cdouble
    delta2::Cdouble
    residual_scale::Cdouble
    leverage::Ptr{Cdouble}
    dimensions::Cint
    cv_scores::Ptr{Cdouble}
    cv_scores_len::Culong
    predict_handle::Ptr{Cvoid}
    error::Ptr{Cchar}
end

function ptr_to_vector(ptr::Ptr{Cdouble}, n::Int)
    if ptr == C_NULL
        return nothing
    end
    return unsafe_wrap(Array, ptr, n, own = false) |> copy
end

function convert_result(c_result::CJlLoessResult)
    # Check for error
    if c_result.error != C_NULL
        error_msg = unsafe_string(Ptr{UInt8}(c_result.error))
        # Free the result before throwing
        ccall(
            (:jl_loess_free_result, libfastloess),
            Cvoid,
            (Ptr{CJlLoessResult},),
            Ref(c_result),
        )
        error("fastloess error: $error_msg")
    end

    n = Int(c_result.n)

    # Extract arrays
    x = ptr_to_vector(c_result.x, n)
    y = ptr_to_vector(c_result.y, n)

    if x === nothing || y === nothing
        ccall(
            (:jl_loess_free_result, libfastloess),
            Cvoid,
            (Ptr{CJlLoessResult},),
            Ref(c_result),
        )
        error("fastloess error: result arrays are null")
    end

    x = x::Vector{Float64}
    y = y::Vector{Float64}

    standard_errors = ptr_to_vector(c_result.standard_errors, n)
    confidence_lower = ptr_to_vector(c_result.confidence_lower, n)
    confidence_upper = ptr_to_vector(c_result.confidence_upper, n)
    prediction_lower = ptr_to_vector(c_result.prediction_lower, n)
    prediction_upper = ptr_to_vector(c_result.prediction_upper, n)
    residuals = ptr_to_vector(c_result.residuals, n)
    robustness_weights = ptr_to_vector(c_result.robustness_weights, n)

    # Extract hat-matrix statistics
    enp = isnan(c_result.enp) ? nothing : c_result.enp
    trace_hat = isnan(c_result.trace_hat) ? nothing : c_result.trace_hat
    delta1 = isnan(c_result.delta1) ? nothing : c_result.delta1
    delta2 = isnan(c_result.delta2) ? nothing : c_result.delta2
    residual_scale = isnan(c_result.residual_scale) ? nothing : c_result.residual_scale
    leverage = ptr_to_vector(c_result.leverage, n)
    gradient = ptr_to_vector(c_result.gradient, n * max(Int(c_result.dimensions), 1))

    # Extract diagnostics
    diagnostics = if !isnan(c_result.rmse)
        Diagnostics(
            c_result.rmse,
            c_result.mae,
            c_result.r_squared,
            isnan(c_result.aic) ? nothing : c_result.aic,
            isnan(c_result.aicc) ? nothing : c_result.aicc,
            isnan(c_result.effective_df) ? nothing : c_result.effective_df,
            c_result.residual_sd,
        )
    else
        nothing
    end

    cv_scores = if c_result.cv_scores != C_NULL && c_result.cv_scores_len > 0
        unsafe_wrap(Array, c_result.cv_scores, Int(c_result.cv_scores_len), own = false) |> copy
    else
        nothing
    end

    predict_model = if c_result.predict_handle != C_NULL
        PredictModel(c_result.predict_handle)
    else
        nothing
    end

    result = LoessResult(
        x,
        y,
        standard_errors,
        confidence_lower,
        confidence_upper,
        prediction_lower,
        prediction_upper,
        residuals,
        robustness_weights,
        gradient,
        c_result.fraction_used,
        c_result.iterations_used == -1 ? nothing : Int(c_result.iterations_used),
        diagnostics,
        enp,
        trace_hat,
        delta1,
        delta2,
        residual_scale,
        leverage,
        Int(c_result.dimensions),
        cv_scores,
        predict_model,
    )

    # Free the C result
    ccall(
        (:jl_loess_free_result, libfastloess),
        Cvoid,
        (Ptr{CJlLoessResult},),
        Ref(c_result),
    )

    return result
end

"""
	Base.append!(a::LoessResult, b::LoessResult) -> LoessResult

Append the results from `b` to `a`. This modifies `a` in place.
"""
const _APPENDABLE_RESULT_FIELDS = (
    :standard_errors,
    :confidence_lower,
    :confidence_upper,
    :prediction_lower,
    :prediction_upper,
    :residuals,
    :robustness_weights,
    :gradient,
)

function Base.append!(a::LoessResult, b::LoessResult)
    a.dimensions == b.dimensions ||
        throw(ArgumentError("cannot append results with different dimensions"))
    for field ∈ _APPENDABLE_RESULT_FIELDS
        left = getfield(a, field)
        right = getfield(b, field)
        (left === nothing) == (right === nothing) || throw(
            ArgumentError("cannot append results with mismatched optional field: $field"),
        )
    end

    append!(a.x, b.x)
    append!(a.y, b.y)

    for field ∈ _APPENDABLE_RESULT_FIELDS
        values = getfield(a, field)
        values === nothing || append!(values, getfield(b, field))
    end

    # Update fraction_used and iterations_used if they differ?
    # Streaming usually keeps them constant. We'll keep a's values.

    return a
end



"""
	Loess(; kwargs...)

Stateful batch LOESS smoother.

# Keyword Arguments
- `fraction::Float64 = 0.67`: Smoothing fraction (proportion of data used for each fit). See Notes for guidance on choosing a value.
- `iterations::Int = 3`: Number of robustness iterations, between 0 and 1000. See Notes for guidance on choosing a value.
- `weight_function::String = "tricube"`: Kernel function
- `robustness_method::String = "bisquare"`: Robustness method
- `scaling_method::String = "mad"`: Scaling method for robustness
- `boundary_policy::String = "extend"`: Handling of edge effects
- `intervals = nothing`: Grouped `confidence` and `prediction` coverage levels.
- `outputs::Vector{String} = String[]`: Grouped optional outputs: `"diagnostics"`,
  `"residuals"`, `"weights"`, `"gradient"`/`"derivative"`, `"se"`, and `"sorted"`.
- `outputs::Vector{String} = String[]`: Optional result components such as
	`"diagnostics"`, `"residuals"`, `"weights"`, `"gradient"`, `"se"`, and
	`"sorted"`.
- `zero_weight_fallback::String = "use_local_mean"`: Fallback when all weights are zero. See Notes for a description of each option.
- `auto_converge::Float64 = NaN`: Tolerance for auto-convergence, NaN to disable
- `cv = nothing`: Grouped cross-validation settings (`fractions`, `method`, `k`).
- `parallel::Bool = true`: Enable parallel execution
- `degree::String = "linear"`: Polynomial degree ("constant", "linear", "quadratic", etc.)
- `dimensions::Int = 1`: Number of predictor dimensions
- `distance_metric::String = "normalized"`: Distance metric ("normalized", "euclidean",
  "manhattan", "chebyshev", "minkowski"). Use "minkowski:p" to set a custom p value,
  e.g. `distance_metric="minkowski:3"`.
- `weighted_metric_weights::Union{Vector{Float64}, Nothing} = nothing`: Per-dimension
  weights, one per dimension declared in `dimensions`. Only used when
  `distance_metric="weighted"`; setting `distance_metric="weighted"` without providing
  this raises an error. `nothing` (default) has no effect unless `distance_metric="weighted"`
  is set.
- `surface_mode::String = "interpolation"`: Surface mode ("interpolation" or "direct")
- `outputs` may include `"se"` and/or `"sorted"`. Results are sorted ascending by `x` instead
  of in the original input order. To get both orderings without re-fitting, sort
  the default (unsorted) result client-side (e.g. `sortperm(result.x)`) rather
  than calling `fit` twice.
- `cell::Union{Float64, Nothing} = nothing`: Cell size tuning parameter for the
  interpolation grid, in `(0, 1]`. `nothing` (default) uses the library default (`0.2`).
  Only applies when `surface_mode = "interpolation"`.
- `interpolation_vertices::Union{Int, Nothing} = nothing`: Caps the number of
  interpolation vertices. `nothing` (default) uses the library default (no explicit cap).
  Only applies when `surface_mode = "interpolation"`.
- `boundary_degree_fallback::Union{Bool, Nothing} = nothing`: Whether to reduce the
  polynomial degree at boundary vertices when the requested `degree` can't be fit there.
  `nothing` (default) uses the library default (enabled). Only applies when
  `surface_mode = "interpolation"`.
- `seed::Union{Int, Nothing} = nothing`: Random seed for reproducible k-fold
  cross-validation shuffling. `nothing` (default) uses a random seed.
- `missing::String = "error"`: Policy for non-finite (NaN/Inf) values in input
  data. See Notes for a description of each option.

# Notes
`fraction` is the most important parameter: it controls the size of the local neighbourhood used at each point.

| Range | Effect | Use case |
| --- | --- | --- |
| 0.1-0.3 | Fine detail | Rapidly changing signals |
| 0.3-0.5 | Balanced | General purpose |
| 0.5-0.7 | Heavy smoothing | Noisy data |
| 0.7-1.0 | Very smooth | Trend extraction |

`iterations` controls robustness to outliers, at the cost of speed.

| Value | Effect | Performance |
| --- | --- | --- |
| 0 | No robustness | Fastest |
| 1-3 | Moderate | Recommended |
| 4-6 | Strong | Contaminated data |
| 7+ | Very strong | Heavy outliers |

`zero_weight_fallback` controls the behavior when all neighborhood weights are zero:

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` (default) | Use the mean of the neighborhood |
| `"return_original"` | Return the original y value |
| `"return_none"` | Return `NaN` |

`missing` controls the behavior when `x`/`y` (or `custom_weights`) contain non-finite (NaN/Inf) values:

| Option | Behavior |
| --- | --- |
| `"error"` (default) | Raise an error if any value is non-finite |
| `"drop"` | Silently remove observations (rows) where any x dimension or y is non-finite before fitting |

A length mismatch between `x` and `y` always errors, even under `"drop"`.

# Example
```julia
l = Loess(fraction=0.3)
result = fit(l, x, y)
```
"""
mutable struct Loess
    handle::Ptr{Cvoid}
    dimensions::Int

    function Loess(;
        fraction::Float64 = 0.67,
        iterations::Int = 3,
        weight_function::String = "tricube",
        robustness_method::String = "bisquare",
        scaling_method::String = "mad",
        boundary_policy::String = "extend",
        outputs::Vector{String} = String[],
        intervals = nothing,
        cv = nothing,
        seed::Union{Int,Nothing} = nothing,
        zero_weight_fallback::String = "use_local_mean",
        auto_converge::Float64 = NaN,
        parallel::Bool = true,
        degree::String = "linear",
        dimensions::Int = 1,
        distance_metric::String = "normalized",
        weighted_metric_weights::Union{Vector{Float64},Nothing} = nothing,
        surface_mode::String = "interpolation",
        cell::Union{Float64,Nothing} = nothing,
        interpolation_vertices::Union{Int,Nothing} = nothing,
        boundary_degree_fallback::Union{Bool,Nothing} = nothing,
        missing::String = "error",
        retain_model::Bool = false,
    )
        flags = _output_flags(outputs)
        interval_options = _interval_options(intervals)
        confidence_intervals = interval_options.confidence
        prediction_intervals = interval_options.prediction
        cv_options = _cv_options(cv)
        cv_fractions = cv_options.fractions
        cv_method = cv_options.method
        cv_k = cv_options.k
        seed === nothing || seed >= 0 || throw(ArgumentError("seed must be non-negative"))
        cv_seed = seed === nothing ? nothing : UInt64(seed)
        interpolation_vertices_value = if interpolation_vertices === nothing
            nothing
        else
            interpolation_vertices > 0 ||
                throw(ArgumentError("interpolation_vertices must be positive"))
            Csize_t(interpolation_vertices)
        end
        configured_dimensions = max(dimensions, 1)
        cv_ptr = isempty(cv_fractions) ? Ptr{Cdouble}(C_NULL) : pointer(cv_fractions)
        cv_len = length(cv_fractions)

        handle = GC.@preserve cv_fractions weighted_metric_weights ccall(
            (:jl_loess_new, libfastloess),
            Ptr{Cvoid},
            (
                Cdouble,
                Cint,
                Cstring,
                Cstring,
                Cstring,
                Cstring,
                Cdouble,
                Cdouble,
                Cint,
                Cint,
                Cint,
                Cstring,
                Cdouble,
                Ptr{Cdouble},
                Culong,
                Cstring,
                Cint,
                Cint,
                Cstring,
                Cint,
                Cstring,
                Cstring,
                Cint,
                Cint,
                Ptr{Cdouble},
                Culong,
                Cstring,
                Cint,
                Cint,
            ),
            fraction,
            Cint(iterations),
            weight_function,
            robustness_method,
            scaling_method,
            boundary_policy,
            confidence_intervals,
            prediction_intervals,
            Cint(flags.diagnostics),
            Cint(flags.residuals),
            Cint(flags.weights),
            zero_weight_fallback,
            auto_converge,
            cv_ptr,
            Culong(cv_len),
            cv_method,
            Cint(cv_k),
            Cint(parallel),
            degree,
            Cint(configured_dimensions),
            distance_metric,
            surface_mode,
            Cint(flags.se),
            Cint(flags.sorted),
            (
                weighted_metric_weights !== nothing ? pointer(weighted_metric_weights) :
                Ptr{Cdouble}(C_NULL)
            ),
            Culong(
                weighted_metric_weights !== nothing ? length(weighted_metric_weights) : 0,
            ),
            missing,
            Cint(retain_model),
            Cint(flags.gradient),
        )

        if handle == C_NULL
            error_message = _constructor_error("Failed to create Loess configuration")
            error("fastloess error: $error_message")
        end

        # Apply optional overrides via setters
        if cell !== nothing
            ccall(
                (:jl_loess_set_cell, libfastloess),
                Cvoid,
                (Ptr{Cvoid}, Cdouble),
                handle,
                cell,
            )
        end
        if interpolation_vertices_value !== nothing
            ccall(
                (:jl_loess_set_interpolation_vertices, libfastloess),
                Cvoid,
                (Ptr{Cvoid}, Csize_t),
                handle,
                interpolation_vertices_value,
            )
        end
        if boundary_degree_fallback !== nothing
            ccall(
                (:jl_loess_set_boundary_degree_fallback, libfastloess),
                Cvoid,
                (Ptr{Cvoid}, Cint),
                handle,
                Cint(boundary_degree_fallback),
            )
        end
        if cv_seed !== nothing
            ccall(
                (:jl_loess_set_cv_seed, libfastloess),
                Cvoid,
                (Ptr{Cvoid}, UInt64),
                handle,
                UInt64(cv_seed),
            )
        end

        obj = new(handle, configured_dimensions)
        finalizer(
            x -> ccall((:jl_loess_free, libfastloess), Cvoid, (Ptr{Cvoid},), x.handle),
            obj,
        )
        return obj
    end
end

"""
	fit(l::Loess, x, y; custom_weights=nothing) -> LoessResult

Fit the LOESS model to data.

# Arguments
- `l::Loess`: LOESS model.
- `x::Vector{Float64}`: Predictor values.
- `y::Vector{Float64}`: Response values.

# Keyword Arguments
- `custom_weights::Union{Vector{Float64}, Nothing} = nothing`: Per-observation weights
  (same length as `y`). Each weight multiplies the local kernel weight:
  `w_ij = custom_weights[j] * K(d_ij/h) * rob_j`. Analogous to the `weights` argument
  in R's `stats::loess`. `nothing` disables custom weighting.
"""
function fit(
    l::Loess,
    x::Vector{Float64},
    y::Vector{Float64};
    custom_weights::Union{Vector{Float64},Nothing} = nothing,
)
    if l.dimensions != 1
        throw(
            ArgumentError(
                "vector x input requires dimensions=1; use an n-by-d Matrix for multivariate fits",
            ),
        )
    end
    n = length(x)
    if n != length(y)
        throw(ArgumentError("x and y must have the same length"))
    end

    if custom_weights !== nothing
        if length(custom_weights) != n
            throw(ArgumentError("custom_weights must have the same length as y"))
        end
    end

    c_result = GC.@preserve l x y custom_weights ccall(
        (:jl_loess_fit, libfastloess),
        CJlLoessResult,
        (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Culong, Ptr{Cdouble}, Culong),
        l.handle,
        x,
        y,
        Culong(n),
        custom_weights !== nothing ? pointer(custom_weights) : Ptr{Cdouble}(C_NULL),
        Culong(custom_weights !== nothing ? length(custom_weights) : 0),
    )

    return convert_result(c_result)
end

"""
	fit(l::Loess, x::Matrix{Float64}, y::Vector{Float64}; custom_weights=nothing) -> LoessResult

Fit the LOESS model to multivariate data.

The matrix `x` has shape `(n, d)` where `n` is the number of observations and `d` is
the number of predictor dimensions (must match `l.dimensions`). Julia matrices are
column-major; this method converts them to the row-major layout expected by the C
library before calling the underlying routine.

# Arguments
- `l::Loess`: LOESS model (configured with `dimensions=d`).
- `x::Matrix{Float64}`: Predictor matrix of shape `(n, d)`.
- `y::Vector{Float64}`: Response values of length `n`.

# Keyword Arguments
- `custom_weights::Union{Vector{Float64}, Nothing} = nothing`: Per-observation weights.
"""
function fit(
    l::Loess,
    x::Matrix{Float64},
    y::Vector{Float64};
    custom_weights::Union{Vector{Float64},Nothing} = nothing,
)
    n = size(x, 1)
    if size(x, 2) != l.dimensions
        throw(
            ArgumentError(
                "x has $(size(x, 2)) columns but model has dimensions=$(l.dimensions); " *
                "pass dimensions=$(size(x, 2)) to Loess()",
            ),
        )
    end
    if n != length(y)
        throw(ArgumentError("x and y must have the same length"))
    end

    # Convert (n, d) column-major Julia matrix to row-major flat vector for the C FFI
    x_flat = vec(permutedims(x))  # shape (d, n) then flatten → [p1_d1, p1_d2, p2_d1, …]

    if custom_weights !== nothing
        if length(custom_weights) != n
            throw(ArgumentError("custom_weights must have the same length as y"))
        end
    end

    c_result = GC.@preserve l x_flat y custom_weights ccall(
        (:jl_loess_fit, libfastloess),
        CJlLoessResult,
        (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Culong, Ptr{Cdouble}, Culong),
        l.handle,
        x_flat,
        y,
        Culong(n),
        custom_weights !== nothing ? pointer(custom_weights) : Ptr{Cdouble}(C_NULL),
        Culong(custom_weights !== nothing ? length(custom_weights) : 0),
    )

    return convert_result(c_result)
end

"""
	StreamingLoess(; kwargs...)

Stateful streaming LOESS smoother.

# Keyword Arguments
- `fraction::Float64 = 0.67`: Smoothing fraction
- `chunk_size::Int = 5000`: Size of each processing chunk
- `overlap::Int = -1`: Overlap between chunks. Negative (the default) means "use the library default" (`chunk_size / 10`, clamped to `[1, chunk_size - 10]`).
- `iterations::Int = 3`: Number of robustness iterations
- `weight_function::String = "tricube"`: Kernel function
- `robustness_method::String = "bisquare"`: Robustness method
- `scaling_method::String = "mad"`: Scaling method
- `boundary_policy::String = "extend"`: Boundary handling
- `auto_converge::Float64 = NaN`: Auto-convergence tolerance
- `outputs::Vector{String} = String[]`: Grouped optional outputs: `"diagnostics"`,
  `"residuals"`, `"weights"`, `"gradient"`/`"derivative"`, and `"se"`.
- `outputs::Vector{String} = String[]`: Optional result components such as
	`"diagnostics"`, `"residuals"`, `"weights"`, `"gradient"`, and `"se"`.
- `zero_weight_fallback::String = "use_local_mean"`: Zero weight handling
- `parallel::Bool = true`: Enable parallel execution
- `degree::String = "linear"`: Polynomial degree
- `dimensions::Int = 1`: Number of predictor dimensions
- `distance_metric::String = "normalized"`: Distance metric ("normalized", "euclidean",
  "manhattan", "chebyshev", "minkowski"). Use "minkowski:p" for a custom p value.
- `surface_mode::String = "interpolation"`: Surface mode
- `merge_strategy::String = "weighted_average"`: Strategy for merging overlapping chunk regions:
  "average", "weighted_average", "take_first", "take_last"
- `weighted_metric_weights::Union{Vector{Float64}, Nothing} = nothing`: Per-dimension
  weights, one per dimension declared in `dimensions`. Only used when
  `distance_metric="weighted"`; setting `distance_metric="weighted"` without providing
  this raises an error.
- `cell::Union{Float64, Nothing} = nothing`: Cell size tuning parameter for the
  interpolation grid.
- `interpolation_vertices::Union{Int, Nothing} = nothing`: Number of interpolation vertices.
- `boundary_degree_fallback::Union{Bool, Nothing} = nothing`: Fall back to lower polynomial
  degree at boundaries when higher degrees fail.
- `missing::String = "error"`: Policy for non-finite (NaN/Inf) values in each chunk.
- `intervals = nothing`: Grouped `confidence` and `prediction` coverage levels,
	computed per chunk and merged across overlap boundaries via `merge_strategy`.
  See `Loess` for a description of each option.
"""
mutable struct StreamingLoess
    handle::Ptr{Cvoid}
    dimensions::Int
    lock::ReentrantLock

    function StreamingLoess(;
        fraction::Float64 = 0.67,
        chunk_size::Int = 5000,
        overlap::Int = -1,
        iterations::Int = 3,
        weight_function::String = "tricube",
        robustness_method::String = "bisquare",
        scaling_method::String = "mad",
        boundary_policy::String = "extend",
        auto_converge::Float64 = NaN,
        outputs::Vector{String} = String[],
        zero_weight_fallback::String = "use_local_mean",
        parallel::Bool = true,
        degree::String = "linear",
        dimensions::Int = 1,
        distance_metric::String = "normalized",
        surface_mode::String = "interpolation",
        merge_strategy::String = "weighted_average",
        weighted_metric_weights::Union{Vector{Float64},Nothing} = nothing,
        cell::Union{Float64,Nothing} = nothing,
        interpolation_vertices::Union{Int,Nothing} = nothing,
        boundary_degree_fallback::Union{Bool,Nothing} = nothing,
        missing::String = "error",
        intervals = nothing,
    )
        configured_dimensions = max(dimensions, 1)
        interval_options = _interval_options(intervals)
        confidence_intervals = interval_options.confidence
        prediction_intervals = interval_options.prediction
        flags = _output_flags(
            outputs;
            allowed = (
                "diagnostics",
                "residuals",
                "weights",
                "gradient",
                "derivative",
                "se",
            ),
        )
        # Resolve weighted metric arguments
        wm_ptr, wm_len = if !isnothing(weighted_metric_weights)
            weighted_metric_weights, Culong(length(weighted_metric_weights))
        else
            C_NULL, Culong(0)
        end
        cell_val = isnothing(cell) ? NaN : Float64(cell)
        iv_val = isnothing(interpolation_vertices) ? Cint(-1) : Cint(interpolation_vertices)
        bdf_val =
            isnothing(boundary_degree_fallback) ? Cint(-1) :
            (boundary_degree_fallback ? Cint(1) : Cint(0))

        handle = GC.@preserve weighted_metric_weights ccall(
            (:jl_streaming_loess_new, libfastloess),
            Ptr{Cvoid},
            (
                Cdouble,
                Cint,
                Cint,
                Cint,
                Cstring,
                Cstring,
                Cstring,
                Cstring,
                Cdouble,
                Cint,
                Cint,
                Cint,
                Cstring,
                Cstring,
                Cint,
                Cstring,
                Cint,
                Cstring,
                Cstring,
                Cdouble,
                Cint,
                Cint,
                Ptr{Cdouble},
                Culong,
                Cstring,
                Cint,
                Cdouble,
                Cdouble,
                Cint,
            ),
            fraction,
            Cint(chunk_size),
            Cint(overlap),
            Cint(iterations),
            weight_function,
            robustness_method,
            scaling_method,
            boundary_policy,
            auto_converge,
            Cint(flags.diagnostics),
            Cint(flags.residuals),
            Cint(flags.weights),
            zero_weight_fallback,
            merge_strategy,
            Cint(parallel),
            degree,
            Cint(configured_dimensions),
            distance_metric,
            surface_mode,
            cell_val,
            iv_val,
            bdf_val,
            wm_ptr,
            wm_len,
            missing,
            Cint(flags.gradient),
            isnothing(confidence_intervals) ? NaN : Float64(confidence_intervals),
            isnothing(prediction_intervals) ? NaN : Float64(prediction_intervals),
            Cint(flags.se),
        )

        if handle == C_NULL
            error_message = _constructor_error("Failed to create StreamingLoess")
            error("fastloess error: $error_message")
        end

        obj = new(handle, configured_dimensions, ReentrantLock())
        finalizer(
            x -> ccall(
                (:jl_streaming_loess_free, libfastloess),
                Cvoid,
                (Ptr{Cvoid},),
                x.handle,
            ),
            obj,
        )
        return obj
    end
end

"""
	process_chunk(s::StreamingLoess, x, y) -> LoessResult

Process a chunk of data.
"""
function process_chunk(
    s::StreamingLoess,
    x::Vector{Float64},
    y::Vector{Float64};
    custom_weights::Union{Vector{Float64},Nothing} = nothing,
)
    if s.dimensions != 1
        throw(
            ArgumentError(
                "vector x input requires dimensions=1; pass an n-by-d Matrix for multivariate chunks",
            ),
        )
    end
    n = length(x)
    if n != length(y)
        throw(ArgumentError("x and y must have the same length"))
    end
    if !isnothing(custom_weights) && length(custom_weights) != n
        throw(ArgumentError("custom_weights must have the same length as y"))
    end

    c_result = lock(s.lock) do
        if isnothing(custom_weights)
            GC.@preserve s x y ccall(
                (:jl_streaming_loess_process_chunk, libfastloess),
                CJlLoessResult,
                (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Culong),
                s.handle,
                x,
                y,
                Culong(n),
            )
        else
            GC.@preserve s x y custom_weights ccall(
                (:jl_streaming_loess_process_chunk_weighted, libfastloess),
                CJlLoessResult,
                (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Culong),
                s.handle,
                x,
                y,
                custom_weights,
                Culong(n),
            )
        end
    end

    return convert_result(c_result)
end

"""
	process_chunk(s::StreamingLoess, x::Matrix{Float64}, y::Vector{Float64}) -> LoessResult

Process a multivariate chunk. Rows are observations and columns are predictor
dimensions.
"""
function process_chunk(
    s::StreamingLoess,
    x::Matrix{Float64},
    y::Vector{Float64};
    custom_weights::Union{Vector{Float64},Nothing} = nothing,
)
    n = size(x, 1)
    if size(x, 2) != s.dimensions
        throw(
            ArgumentError(
                "x has $(size(x, 2)) columns but model has dimensions=$(s.dimensions)",
            ),
        )
    end
    if n != length(y)
        throw(ArgumentError("x and y must have the same number of observations"))
    end
    if !isnothing(custom_weights) && length(custom_weights) != n
        throw(
            ArgumentError("custom_weights must have the same number of observations as y"),
        )
    end
    x_flat = vec(permutedims(x))
    c_result = lock(s.lock) do
        if isnothing(custom_weights)
            GC.@preserve s x_flat y ccall(
                (:jl_streaming_loess_process_chunk, libfastloess),
                CJlLoessResult,
                (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Culong),
                s.handle,
                x_flat,
                y,
                Culong(n),
            )
        else
            GC.@preserve s x_flat y custom_weights ccall(
                (:jl_streaming_loess_process_chunk_weighted, libfastloess),
                CJlLoessResult,
                (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Culong),
                s.handle,
                x_flat,
                y,
                custom_weights,
                Culong(n),
            )
        end
    end

    return convert_result(c_result)
end

"""
	finalize(s::StreamingLoess) -> LoessResult

Finalize streaming and return remaining buffered data.
"""
function finalize(s::StreamingLoess)
    c_result = lock(s.lock) do
        GC.@preserve s ccall(
            (:jl_streaming_loess_finalize, libfastloess),
            CJlLoessResult,
            (Ptr{Cvoid},),
            s.handle,
        )
    end

    return convert_result(c_result)
end

"""
	OnlineLoess(; kwargs...)

Stateful online LOESS smoother.

# Keyword Arguments
- `fraction::Float64 = 0.67`: Smoothing fraction
- `window_capacity::Int = 1000`: Maximum points to retain in window
- `min_points::Int = 2`: Minimum points before smoothing starts
- `iterations::Int = 3`: Number of robustness iterations
- `weight_function::String = "tricube"`: Kernel function
- `robustness_method::String = "bisquare"`: Robustness method
- `scaling_method::String = "mad"`: Scaling method
- `boundary_policy::String = "extend"`: Boundary handling
- `update_mode::String = "incremental"`: Update strategy ("full" or "incremental")
- `auto_converge::Float64 = NaN`: Auto-convergence tolerance
- `outputs::Vector{String} = String[]`: Grouped optional outputs: `"weights"`,
  `"gradient"`/`"derivative"`, and `"se"`.
- `outputs::Vector{String} = String[]`: Optional outputs including `"weights"`,
  `"gradient"`, and `"se"`.
- `zero_weight_fallback::String = "use_local_mean"`: Zero weight handling
- `degree::String = "linear"`: Polynomial degree
- `dimensions::Int = 1`: Number of predictor dimensions; multivariate `add_point` calls take coordinate vectors
- `distance_metric::String = "normalized"`: Distance metric ("normalized", "euclidean",
  "manhattan", "chebyshev", "minkowski"). Use "minkowski:p" for a custom p value.
- `surface_mode::String = "interpolation"`: Surface mode
- `weighted_metric_weights::Union{Vector{Float64}, Nothing} = nothing`: Per-dimension
  weights, one per dimension declared in `dimensions`. Only used when
  `distance_metric="weighted"`; setting `distance_metric="weighted"` without providing
  this raises an error.
- `cell::Union{Float64, Nothing} = nothing`: Cell size tuning parameter for the
  interpolation grid.
- `interpolation_vertices::Union{Int, Nothing} = nothing`: Number of interpolation vertices.
- `boundary_degree_fallback::Union{Bool, Nothing} = nothing`: Fall back to lower polynomial
  degree at boundaries when higher degrees fail.
- `missing::String = "error"`: Policy for non-finite (NaN/Inf) `x`/`y` values passed
  to `add_point`. `"error"` (default) raises, `"drop"` silently ignores the point
  (returns `nothing` instead of adding it to the window).
- `intervals = nothing`: Grouped `confidence` and `prediction` coverage levels.
	Interval levels and the `"se"` output require `update_mode="full"`.
"""
mutable struct OnlineLoess
    handle::Ptr{Cvoid}
    dimensions::Int
    lock::ReentrantLock

    function OnlineLoess(;
        fraction::Float64 = 0.67,
        window_capacity::Int = 1000,
        min_points::Int = 2,
        iterations::Int = 0,
        weight_function::String = "tricube",
        robustness_method::String = "bisquare",
        scaling_method::String = "mad",
        boundary_policy::String = "extend",
        update_mode::String = "incremental",
        auto_converge::Float64 = NaN,
        outputs::Vector{String} = String[],
        zero_weight_fallback::String = "use_local_mean",
        degree::String = "linear",
        dimensions::Int = 1,
        distance_metric::String = "normalized",
        surface_mode::String = "interpolation",
        weighted_metric_weights::Union{Vector{Float64},Nothing} = nothing,
        cell::Union{Float64,Nothing} = nothing,
        interpolation_vertices::Union{Int,Nothing} = nothing,
        boundary_degree_fallback::Union{Bool,Nothing} = nothing,
        missing::String = "error",
        intervals = nothing,
    )
        configured_dimensions = max(dimensions, 1)
        interval_options = _interval_options(intervals)
        confidence_intervals = interval_options.confidence
        prediction_intervals = interval_options.prediction
        flags =
            _output_flags(outputs; allowed = ("weights", "gradient", "derivative", "se"))
        # Resolve weighted metric arguments
        wm_ptr, wm_len = if !isnothing(weighted_metric_weights)
            weighted_metric_weights, Culong(length(weighted_metric_weights))
        else
            C_NULL, Culong(0)
        end
        cell_val = isnothing(cell) ? NaN : Float64(cell)
        iv_val = isnothing(interpolation_vertices) ? Cint(-1) : Cint(interpolation_vertices)
        bdf_val =
            isnothing(boundary_degree_fallback) ? Cint(-1) :
            (boundary_degree_fallback ? Cint(1) : Cint(0))

        handle = GC.@preserve weighted_metric_weights ccall(
            (:jl_online_loess_new, libfastloess),
            Ptr{Cvoid},
            (
                Cdouble,
                Cint,
                Cint,
                Cint,
                Cstring,
                Cstring,
                Cstring,
                Cstring,
                Cstring,
                Cdouble,
                Cint,
                Cstring,
                Cstring,
                Cint,
                Cstring,
                Cstring,
                Cdouble,
                Cint,
                Cint,
                Ptr{Cdouble},
                Culong,
                Cstring,
                Cint,
                Cdouble,
                Cdouble,
                Cint,
            ),
            fraction,
            Cint(window_capacity),
            Cint(min_points),
            Cint(iterations),
            weight_function,
            robustness_method,
            scaling_method,
            boundary_policy,
            update_mode,
            auto_converge,
            Cint(flags.weights),
            zero_weight_fallback,
            degree,
            Cint(configured_dimensions),
            distance_metric,
            surface_mode,
            cell_val,
            iv_val,
            bdf_val,
            wm_ptr,
            wm_len,
            missing,
            Cint(flags.gradient),
            isnothing(confidence_intervals) ? NaN : Float64(confidence_intervals),
            isnothing(prediction_intervals) ? NaN : Float64(prediction_intervals),
            Cint(flags.se),
        )

        if handle == C_NULL
            error_message = _constructor_error("Failed to create OnlineLoess")
            error("fastloess error: $error_message")
        end

        obj = new(handle, configured_dimensions, ReentrantLock())
        finalizer(
            x -> ccall(
                (:jl_online_loess_free, libfastloess),
                Cvoid,
                (Ptr{Cvoid},),
                x.handle,
            ),
            obj,
        )
        return obj
    end
end

"""
	add_point(o::OnlineLoess, x, y) -> Union{OnlineOutput, Nothing}

Add one point to the online processor and return its smoothed value. For a
one-dimensional model, pass a scalar coordinate; for multivariate models, pass
a vector with one coordinate per configured dimension. Returns `nothing` while
the window is still filling (fewer than `min_points` have been seen).
"""
function add_point(o::OnlineLoess, x::Real, y::Real; weight::Real = 1.0)
    return add_point(o, Float64[x], Float64(y); weight)
end

function add_point(o::OnlineLoess, x::AbstractVector{<:Real}, y::Real; weight::Real = 1.0)
    x_values = Float64.(x)
    if length(x_values) != o.dimensions
        throw(
            ArgumentError(
                "x must have exactly $(o.dimensions) values for dimensions=$(o.dimensions)",
            ),
        )
    end
    response = Float64(y)
    c_result = lock(o.lock) do
        GC.@preserve o x_values ccall(
            (:jl_online_loess_add_point, libfastloess),
            CJlOnlineOutput,
            (Ptr{Cvoid}, Ptr{Cdouble}, Culong, Cdouble, Cdouble),
            o.handle,
            x_values,
            Culong(length(x_values)),
            response,
            Float64(weight),
        )
    end

    if c_result.error != C_NULL
        error_msg = unsafe_string(Ptr{UInt8}(c_result.error))
        ccall(
            (:jl_online_free_output, libfastloess),
            Cvoid,
            (Ptr{CJlOnlineOutput},),
            Ref(c_result),
        )
        error("fastloess error: $error_msg")
    end

    if c_result.has_value == 0
        return nothing
    end

    gradient = ptr_to_vector(c_result.gradient, max(Int(c_result.dimensions), 1))

    output = OnlineOutput(
        c_result.y,
        isnan(c_result.standard_error) ? nothing : c_result.standard_error,
        isnan(c_result.residual) ? nothing : c_result.residual,
        isnan(c_result.robustness_weight) ? nothing : c_result.robustness_weight,
        c_result.iterations_used == -1 ? nothing : Int(c_result.iterations_used),
        isnan(c_result.confidence_lower) ? nothing : c_result.confidence_lower,
        isnan(c_result.confidence_upper) ? nothing : c_result.confidence_upper,
        isnan(c_result.prediction_lower) ? nothing : c_result.prediction_lower,
        isnan(c_result.prediction_upper) ? nothing : c_result.prediction_upper,
        gradient,
    )

    # Free the C-allocated gradient buffer now that it has been copied above.
    ccall(
        (:jl_online_free_output, libfastloess),
        Cvoid,
        (Ptr{CJlOnlineOutput},),
        Ref(c_result),
    )

    return output
end

function window_diagnostics(o::OnlineLoess)
    c_result = lock(o.lock) do
        GC.@preserve o ccall(
            (:jl_online_loess_window_diagnostics, libfastloess),
            CJlOnlineDiagnostics,
            (Ptr{Cvoid},),
            o.handle,
        )
    end
    if c_result.error != C_NULL
        error_message = unsafe_string(Ptr{UInt8}(c_result.error))
        ccall(
            (:jl_online_free_diagnostics, libfastloess),
            Cvoid,
            (Ptr{CJlOnlineDiagnostics},),
            Ref(c_result),
        )
        error("fastloess error: $error_message")
    end
    if c_result.has_value == 0
        return nothing
    end
    result = Diagnostics(
        c_result.rmse,
        c_result.mae,
        c_result.r_squared,
        isnan(c_result.aic) ? nothing : c_result.aic,
        isnan(c_result.aicc) ? nothing : c_result.aicc,
        isnan(c_result.effective_df) ? nothing : c_result.effective_df,
        c_result.residual_sd,
    )
    ccall(
        (:jl_online_free_diagnostics, libfastloess),
        Cvoid,
        (Ptr{CJlOnlineDiagnostics},),
        Ref(c_result),
    )
    return result
end

function predict_window(
    o::OnlineLoess,
    new_x::Vector{Float64};
    outputs::Vector{String} = String[],
    intervals = nothing,
    extrapolation::String = "clamp",
    max_extrapolation_distance::Union{Float64,Nothing} = nothing,
    max_neighbor_distance::Union{Float64,Nothing} = nothing,
)
    flags = _output_flags(outputs; allowed = ("se", "gradient", "derivative"))
    interval_options = _interval_options(intervals)
    c_result = lock(o.lock) do
        GC.@preserve o new_x ccall(
            (:jl_online_loess_predict_window, libfastloess),
            CJlPredictResult,
            (
                Ptr{Cvoid},
                Ptr{Cdouble},
                Culong,
                Cint,
                Cdouble,
                Cdouble,
                Cint,
                Cstring,
                Cdouble,
                Cdouble,
            ),
            o.handle,
            new_x,
            Culong(length(new_x)),
            Cint(flags.se),
            interval_options.confidence,
            interval_options.prediction,
            Cint(flags.gradient),
            extrapolation,
            isnothing(max_extrapolation_distance) ? NaN : max_extrapolation_distance,
            isnothing(max_neighbor_distance) ? NaN : max_neighbor_distance,
        )
    end
    if c_result.error != C_NULL
        error_message = unsafe_string(Ptr{UInt8}(c_result.error))
        ccall(
            (:jl_predict_free_result, libfastloess),
            Cvoid,
            (Ptr{CJlPredictResult},),
            Ref(c_result),
        )
        error("fastloess error: $error_message")
    end
    n = Int(c_result.n)
    dimensions = max(Int(c_result.dimensions), 1)
    result = PredictResult(
        ptr_to_vector(c_result.y, n),
        ptr_to_vector(c_result.standard_errors, n),
        ptr_to_vector(c_result.confidence_lower, n),
        ptr_to_vector(c_result.confidence_upper, n),
        ptr_to_vector(c_result.prediction_lower, n),
        ptr_to_vector(c_result.prediction_upper, n),
        ptr_to_vector(c_result.derivative, n * dimensions),
    )
    ccall(
        (:jl_predict_free_result, libfastloess),
        Cvoid,
        (Ptr{CJlPredictResult},),
        Ref(c_result),
    )
    return result
end

end # module FastLOESS
