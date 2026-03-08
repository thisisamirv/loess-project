"""
    FastLOESS

High-performance LOESS (Locally Weighted Scatterplot Smoothing) for Julia.

Provides bindings to the fastloess Rust library for fast, robust LOESS smoothing.

# Main Functions
- `smooth(x, y; kwargs...)`: Batch LOESS smoothing
- `smooth_streaming(x, y; kwargs...)`: Streaming mode for large datasets
- `smooth_online(x, y; kwargs...)`: Online mode with sliding window

# Example
```julia
using FastLOESS

x = collect(1.0:0.1:10.0)
y = sin.(x) .+ 0.1 .* randn(length(x))

result = smooth(x, y, fraction=0.3)
println("Smoothed values: ", result.y)
```
"""
module FastLOESS

export Loess, StreamingLoess, OnlineLoess
export fit, process_chunk, finalize, add_points
export LoessResult, Diagnostics

import Base: finalize

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

    for path in candidates
        if isfile(path)
            @info "Using local library: $path"
            return path
        end
    end

    # Fall back to system path
    @warn "Library not found in JLL or local paths, falling back to system path"
    return LIBNAME
end

const libfastloess = find_library()

"""
    Diagnostics

Diagnostic statistics for LOESS fit quality.

# Fields
- `rmse::Float64`: Root Mean Squared Error
- `mae::Float64`: Mean Absolute Error
- `r_squared::Float64`: R-squared (coefficient of determination)
- `aic::Float64`: Akaike Information Criterion (NaN if not computed)
- `aicc::Float64`: Corrected AIC (NaN if not computed)
- `effective_df::Float64`: Effective degrees of freedom (NaN if not computed)
- `residual_sd::Float64`: Residual standard deviation
"""
struct Diagnostics
    rmse::Float64
    mae::Float64
    r_squared::Float64
    aic::Float64
    aicc::Float64
    effective_df::Float64
    residual_sd::Float64
end

"""
    LoessResult

Result from LOESS smoothing.

# Fields
- `x::Vector{Float64}`: Sorted x values
- `y::Vector{Float64}`: Smoothed y values
- `standard_errors::Union{Vector{Float64}, Nothing}`: Standard errors (if computed)
- `confidence_lower::Union{Vector{Float64}, Nothing}`: Lower confidence bounds
- `confidence_upper::Union{Vector{Float64}, Nothing}`: Upper confidence bounds
- `prediction_lower::Union{Vector{Float64}, Nothing}`: Lower prediction bounds
- `prediction_upper::Union{Vector{Float64}, Nothing}`: Upper prediction bounds
- `residuals::Union{Vector{Float64}, Nothing}`: Residuals
- `robustness_weights::Union{Vector{Float64}, Nothing}`: Robustness weights
- `fraction_used::Float64`: Fraction used for smoothing
- `iterations_used::Int`: Number of iterations performed (-1 if not available)
- `diagnostics::Union{Diagnostics, Nothing}`: Diagnostic metrics
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
    fraction_used::Float64
    iterations_used::Int
    diagnostics::Union{Diagnostics,Nothing}
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
    fraction_used::Cdouble
    iterations_used::Cint
    rmse::Cdouble
    mae::Cdouble
    r_squared::Cdouble
    aic::Cdouble
    aicc::Cdouble
    effective_df::Cdouble
    residual_sd::Cdouble
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
        error_msg = unsafe_string(c_result.error)
        # Free the result before throwing
        @ccall libfastloess.jl_loess_free_result(
            Ref(c_result)::Ptr{CJlLoessResult},
        )::Cvoid
        error("fastloess error: $error_msg")
    end

    n = Int(c_result.n)

    # Extract arrays
    x = ptr_to_vector(c_result.x, n)
    y = ptr_to_vector(c_result.y, n)

    if x === nothing || y === nothing
        @ccall libfastloess.jl_loess_free_result(
            Ref(c_result)::Ptr{CJlLoessResult},
        )::Cvoid
        error("fastloess error: result arrays are null")
    end

    standard_errors = ptr_to_vector(c_result.standard_errors, n)
    confidence_lower = ptr_to_vector(c_result.confidence_lower, n)
    confidence_upper = ptr_to_vector(c_result.confidence_upper, n)
    prediction_lower = ptr_to_vector(c_result.prediction_lower, n)
    prediction_upper = ptr_to_vector(c_result.prediction_upper, n)
    residuals = ptr_to_vector(c_result.residuals, n)
    robustness_weights = ptr_to_vector(c_result.robustness_weights, n)

    # Extract diagnostics
    diagnostics = if !isnan(c_result.rmse)
        Diagnostics(
            c_result.rmse,
            c_result.mae,
            c_result.r_squared,
            c_result.aic,
            c_result.aicc,
            c_result.effective_df,
            c_result.residual_sd,
        )
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
        c_result.fraction_used,
        Int(c_result.iterations_used),
        diagnostics,
    )

    # Free the C result
    @ccall libfastloess.jl_loess_free_result(Ref(c_result)::Ptr{CJlLoessResult})::Cvoid

    return result
end

"""
    Base.append!(a::LoessResult, b::LoessResult) -> LoessResult

Append the results from `b` to `a`. This modifies `a` in place.
"""
function Base.append!(a::LoessResult, b::LoessResult)
    append!(a.x, b.x)
    append!(a.y, b.y)

    if a.standard_errors !== nothing && b.standard_errors !== nothing
        append!(a.standard_errors, b.standard_errors)
    end
    if a.confidence_lower !== nothing && b.confidence_lower !== nothing
        append!(a.confidence_lower, b.confidence_lower)
    end
    if a.confidence_upper !== nothing && b.confidence_upper !== nothing
        append!(a.confidence_upper, b.confidence_upper)
    end
    if a.prediction_lower !== nothing && b.prediction_lower !== nothing
        append!(a.prediction_lower, b.prediction_lower)
    end
    if a.prediction_upper !== nothing && b.prediction_upper !== nothing
        append!(a.prediction_upper, b.prediction_upper)
    end
    if a.residuals !== nothing && b.residuals !== nothing
        append!(a.residuals, b.residuals)
    end
    if a.robustness_weights !== nothing && b.robustness_weights !== nothing
        append!(a.robustness_weights, b.robustness_weights)
    end

    # Update fraction_used and iterations_used if they differ?
    # Streaming usually keeps them constant. We'll keep a's values.

    return a
end



"""
    Loess(; kwargs...)

Stateful batch LOESS smoother.

# Keyword Arguments
- `fraction::Float64 = 0.67`: Smoothing fraction (proportion of data used for each fit)
- `iterations::Int = 3`: Number of robustness iterations
- `delta::Float64 = NaN`: Interpolation optimization threshold (NaN for auto)
- `weight_function::String = "tricube"`: Kernel function
- `robustness_method::String = "bisquare"`: Robustness method
- `scaling_method::String = "mad"`: Scaling method for robustness
- `boundary_policy::String = "extend"`: Handling of edge effects
- `confidence_intervals::Float64 = NaN`: Confidence level (e.g., 0.95), NaN to disable
- `prediction_intervals::Float64 = NaN`: Prediction interval level, NaN to disable
- `return_diagnostics::Bool = false`: Whether to compute RMSE, MAE, R², etc.
- `return_residuals::Bool = false`: Whether to include residuals
- `return_robustness_weights::Bool = false`: Whether to include robustness weights
- `zero_weight_fallback::String = "use_local_mean"`: Fallback when all weights are zero
- `auto_converge::Float64 = NaN`: Tolerance for auto-convergence, NaN to disable
- `cv_fractions::Vector{Float64} = Float64[]`: Fractions for cross-validation
- `cv_method::String = "kfold"`: CV method ("loocv" or "kfold")
- `cv_k::Int = 5`: Number of folds for k-fold CV
- `parallel::Bool = true`: Enable parallel execution

# Example
```julia
l = Loess(fraction=0.3)
result = fit(l, x, y)
```
"""
mutable struct Loess
    handle::Ptr{Cvoid}

    function Loess(;
        fraction::Float64 = 0.67,
        iterations::Int = 3,
        delta::Float64 = NaN,
        weight_function::String = "tricube",
        robustness_method::String = "bisquare",
        scaling_method::String = "mad",
        boundary_policy::String = "extend",
        confidence_intervals::Float64 = NaN,
        prediction_intervals::Float64 = NaN,
        return_diagnostics::Bool = false,
        return_residuals::Bool = false,
        return_robustness_weights::Bool = false,
        zero_weight_fallback::String = "use_local_mean",
        auto_converge::Float64 = NaN,
        cv_fractions::Vector{Float64} = Float64[],
        cv_method::String = "kfold",
        cv_k::Int = 5,
        parallel::Bool = true,
    )
        cv_ptr = isempty(cv_fractions) ? C_NULL : pointer(cv_fractions)
        cv_len = length(cv_fractions)

        handle = @ccall libfastloess.jl_loess_new(
            fraction::Cdouble,
            Cint(iterations)::Cint,
            delta::Cdouble,
            weight_function::Cstring,
            robustness_method::Cstring,
            scaling_method::Cstring,
            boundary_policy::Cstring,
            confidence_intervals::Cdouble,
            prediction_intervals::Cdouble,
            Cint(return_diagnostics)::Cint,
            Cint(return_residuals)::Cint,
            Cint(return_robustness_weights)::Cint,
            zero_weight_fallback::Cstring,
            auto_converge::Cdouble,
            cv_ptr::Ptr{Cdouble},
            Culong(cv_len)::Culong,
            cv_method::Cstring,
            Cint(cv_k)::Cint,
            Cint(parallel)::Cint,
        )::Ptr{Cvoid}

        if handle == C_NULL
            error("Failed to create Loess configuration")
        end

        obj = new(handle)
        finalizer(
            x -> @ccall(libfastloess.jl_loess_free(x.handle::Ptr{Cvoid})::Cvoid),
            obj,
        )
        return obj
    end
end

"""
    fit(l::Loess, x, y) -> LoessResult

Fit the LOESS model to data.
"""
function fit(l::Loess, x::Vector{Float64}, y::Vector{Float64})
    n = length(x)
    if n != length(y)
        throw(ArgumentError("x and y must have the same length"))
    end

    c_result = @ccall libfastloess.jl_loess_fit(
        l.handle::Ptr{Cvoid},
        x::Ptr{Cdouble},
        y::Ptr{Cdouble},
        Culong(n)::Culong,
    )::CJlLoessResult

    return convert_result(c_result)
end

"""
    StreamingLoess(; kwargs...)

Stateful streaming LOESS smoother.

# Keyword Arguments
- `fraction::Float64 = 0.3`: Smoothing fraction
- `chunk_size::Int = 5000`: Size of each processing chunk
- `overlap::Int = -1`: Overlap between chunks (-1 for auto = 10% of chunk_size)
- `iterations::Int = 3`: Number of robustness iterations
- `delta::Float64 = NaN`: Interpolation threshold
- `weight_function::String = "tricube"`: Kernel function
- `robustness_method::String = "bisquare"`: Robustness method
- `scaling_method::String = "mad"`: Scaling method
- `boundary_policy::String = "extend"`: Boundary handling
- `auto_converge::Float64 = NaN`: Auto-convergence tolerance
- `return_diagnostics::Bool = false`: Compute diagnostics
- `return_residuals::Bool = false`: Include residuals
- `return_robustness_weights::Bool = false`: Include weights
- `zero_weight_fallback::String = "use_local_mean"`: Zero weight handling
- `parallel::Bool = true`: Enable parallel execution
"""
mutable struct StreamingLoess
    handle::Ptr{Cvoid}

    function StreamingLoess(;
        fraction::Float64 = 0.3,
        chunk_size::Int = 5000,
        overlap::Int = -1,
        iterations::Int = 3,
        delta::Float64 = NaN,
        weight_function::String = "tricube",
        robustness_method::String = "bisquare",
        scaling_method::String = "mad",
        boundary_policy::String = "extend",
        auto_converge::Float64 = NaN,
        return_diagnostics::Bool = false,
        return_residuals::Bool = false,
        return_robustness_weights::Bool = false,
        zero_weight_fallback::String = "use_local_mean",
        parallel::Bool = true,
    )
        handle = @ccall libfastloess.jl_streaming_loess_new(
            fraction::Cdouble,
            Cint(chunk_size)::Cint,
            Cint(overlap)::Cint,
            Cint(iterations)::Cint,
            delta::Cdouble,
            weight_function::Cstring,
            robustness_method::Cstring,
            scaling_method::Cstring,
            boundary_policy::Cstring,
            auto_converge::Cdouble,
            Cint(return_diagnostics)::Cint,
            Cint(return_residuals)::Cint,
            Cint(return_robustness_weights)::Cint,
            zero_weight_fallback::Cstring,
            Cint(parallel)::Cint,
        )::Ptr{Cvoid}

        if handle == C_NULL
            error("Failed to create StreamingLoess")
        end

        obj = new(handle)
        finalizer(
            x ->
                @ccall(libfastloess.jl_streaming_loess_free(x.handle::Ptr{Cvoid})::Cvoid),
            obj,
        )
        return obj
    end
end

"""
    process_chunk(s::StreamingLoess, x, y) -> LoessResult

Process a chunk of data.
"""
function process_chunk(s::StreamingLoess, x::Vector{Float64}, y::Vector{Float64})
    n = length(x)
    if n != length(y)
        throw(ArgumentError("x and y must have the same length"))
    end

    c_result = @ccall libfastloess.jl_streaming_loess_process_chunk(
        s.handle::Ptr{Cvoid},
        x::Ptr{Cdouble},
        y::Ptr{Cdouble},
        Culong(n)::Culong,
    )::CJlLoessResult

    return convert_result(c_result)
end

"""
    finalize(s::StreamingLoess) -> LoessResult

Finalize streaming and return remaining buffered data.
"""
function finalize(s::StreamingLoess)
    c_result = @ccall libfastloess.jl_streaming_loess_finalize(
        s.handle::Ptr{Cvoid},
    )::CJlLoessResult

    return convert_result(c_result)
end

"""
    OnlineLoess(; kwargs...)

Stateful online LOESS smoother.

# Keyword Arguments
- `fraction::Float64 = 0.2`: Smoothing fraction
- `window_capacity::Int = 100`: Maximum points to retain in window
- `min_points::Int = 2`: Minimum points before smoothing starts
- `iterations::Int = 3`: Number of robustness iterations
- `delta::Float64 = NaN`: Interpolation threshold
- `weight_function::String = "tricube"`: Kernel function
- `robustness_method::String = "bisquare"`: Robustness method
- `scaling_method::String = "mad"`: Scaling method
- `boundary_policy::String = "extend"`: Boundary handling
- `update_mode::String = "full"`: Update strategy ("full" or "incremental")
- `auto_converge::Float64 = NaN`: Auto-convergence tolerance
- `return_robustness_weights::Bool = false`: Include weights
- `zero_weight_fallback::String = "use_local_mean"`: Zero weight handling
- `parallel::Bool = false`: Enable parallel execution
"""
mutable struct OnlineLoess
    handle::Ptr{Cvoid}

    function OnlineLoess(;
        fraction::Float64 = 0.2,
        window_capacity::Int = 100,
        min_points::Int = 2,
        iterations::Int = 3,
        delta::Float64 = NaN,
        weight_function::String = "tricube",
        robustness_method::String = "bisquare",
        scaling_method::String = "mad",
        boundary_policy::String = "extend",
        update_mode::String = "full",
        auto_converge::Float64 = NaN,
        return_robustness_weights::Bool = false,
        zero_weight_fallback::String = "use_local_mean",
        parallel::Bool = false,
    )
        handle = @ccall libfastloess.jl_online_loess_new(
            fraction::Cdouble,
            Cint(window_capacity)::Cint,
            Cint(min_points)::Cint,
            Cint(iterations)::Cint,
            delta::Cdouble,
            weight_function::Cstring,
            robustness_method::Cstring,
            scaling_method::Cstring,
            boundary_policy::Cstring,
            update_mode::Cstring,
            auto_converge::Cdouble,
            Cint(return_robustness_weights)::Cint,
            zero_weight_fallback::Cstring,
            Cint(parallel)::Cint,
        )::Ptr{Cvoid}

        if handle == C_NULL
            error("Failed to create OnlineLoess")
        end

        obj = new(handle)
        finalizer(
            x -> @ccall(libfastloess.jl_online_loess_free(x.handle::Ptr{Cvoid})::Cvoid),
            obj,
        )
        return obj
    end
end

"""
    add_points(o::OnlineLoess, x, y) -> LoessResult

Add points to the online processor.
"""
function add_points(o::OnlineLoess, x::Vector{Float64}, y::Vector{Float64})
    n = length(x)
    if n != length(y)
        throw(ArgumentError("x and y must have the same length"))
    end

    c_result = @ccall libfastloess.jl_online_loess_add_points(
        o.handle::Ptr{Cvoid},
        x::Ptr{Cdouble},
        y::Ptr{Cdouble},
        Culong(n)::Culong,
    )::CJlLoessResult

    return convert_result(c_result)
end

end # module FastLOESS
