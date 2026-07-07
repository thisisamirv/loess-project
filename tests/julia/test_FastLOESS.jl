#!/usr/bin/env julia
"""
Tests for fastloess Julia bindings.

Comprehensive test suite covering:
- Stateful Loess class (batch smoothing)
- Reusability of Loess instances
- StreamingLoess class
- OnlineLoess class
- Error handling
- Edge cases

Run with: julia --project=bindings/julia/julia tests/julia/test_fastloess.jl
"""

using Test
using Random
using Statistics

# Handle package loading - check if we're already in the fastloess project
using Pkg
project_name = Pkg.project().name
if project_name != "FastLOESS"
    # Not in the fastloess project, need to develop it
    script_dir = @__DIR__
    project_root = dirname(dirname(script_dir))
    julia_pkg_dir = joinpath(project_root, "bindings", "julia", "julia")
    if !haskey(Pkg.project().dependencies, "FastLOESS")
        Pkg.develop(path = julia_pkg_dir)
    end
end

using FastLOESS

@testset "fastloess Julia Bindings" begin

    @testset "Loess (Batch)" begin
        @testset "default parameters" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            y = [2.0, 4.1, 5.9, 8.2, 9.8]

            model = Loess(fraction = 0.5)
            result = fit(model, x, y)

            @test result isa LoessResult
            @test length(result.y) == length(x)
            @test length(result.x) == length(x)
            @test result.fraction_used ≈ 0.5
        end

        @testset "reuse Loess instance" begin
            x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
            y1 = [2.0, 4.1, 5.9, 8.2, 9.8]
            x2 = [1.0, 2.0, 3.0]
            y2 = [1.0, 2.0, 3.0]

            model = Loess(fraction = 0.5)

            # First fit
            result1 = fit(model, x1, y1)
            @test length(result1.y) == length(x1)

            # Second fit with different data
            result2 = fit(model, x2, y2)
            @test length(result2.y) == length(x2)
        end

        @testset "serial execution" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            y = [2.0, 4.1, 5.9, 8.2, 9.8]

            model = Loess(fraction = 0.5, parallel = false)
            result = fit(model, x, y)

            @test result isa LoessResult
            @test length(result.y) == length(x)
        end

        @testset "with diagnostics" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            y = [2.0, 4.1, 5.9, 8.2, 9.8]

            model = Loess(fraction = 0.5, return_diagnostics = true)
            result = fit(model, x, y)

            @test result.diagnostics !== nothing
            @test result.diagnostics isa Diagnostics
            @test result.diagnostics.rmse >= 0
            @test result.diagnostics.mae >= 0
            @test 0 <= result.diagnostics.r_squared <= 1
            @test result.diagnostics.residual_sd >= 0
        end

        @testset "with residuals" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            y = [2.0, 4.1, 5.9, 8.2, 9.8]

            model = Loess(fraction = 0.5, return_residuals = true)
            result = fit(model, x, y)

            @test result.residuals !== nothing
            @test length(result.residuals) == length(x)
        end

        @testset "with robustness weights" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            y = [2.0, 4.1, 100.0, 8.2, 9.8]  # Outlier

            model = Loess(fraction = 0.7, iterations = 3, return_robustness_weights = true)
            result = fit(model, x, y)

            @test result.robustness_weights !== nothing
            @test length(result.robustness_weights) == length(x)
            @test all(result.robustness_weights .>= 0)
            @test all(result.robustness_weights .<= 1)
        end

        @testset "with confidence intervals" begin
            Random.seed!(42)
            x = collect(range(0, 10, length = 20))
            y = 2 .* x .+ randn(20)

            model = Loess(fraction = 0.5, confidence_intervals = 0.95)
            result = fit(model, x, y)

            @test result.confidence_lower !== nothing
            @test result.confidence_upper !== nothing
            @test length(result.confidence_lower) == length(x)
            @test length(result.confidence_upper) == length(x)
            @test all(result.confidence_lower .<= result.confidence_upper)
        end

        @testset "with prediction intervals" begin
            Random.seed!(42)
            x = collect(range(0, 10, length = 20))
            y = 2 .* x .+ randn(20)

            model = Loess(fraction = 0.5, prediction_intervals = 0.95)
            result = fit(model, x, y)

            @test result.prediction_lower !== nothing
            @test result.prediction_upper !== nothing
            @test length(result.prediction_lower) == length(x)
            @test length(result.prediction_upper) == length(x)
        end
    end

    @testset "Weight Functions" begin
        x = collect(range(0, 10, length = 20))
        y = sin.(x)

        kernels = ["tricube", "epanechnikov", "gaussian", "uniform", "biweight", "triangle"]

        for kernel ∈ kernels
            @testset "$kernel" begin
                model = Loess(fraction = 0.5, weight_function = kernel)
                result = fit(model, x, y)
                @test length(result.y) == length(x)
            end
        end
    end

    @testset "Robustness Methods" begin
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 100.0, 8.0, 10.0]  # Outlier

        methods = ["bisquare", "huber", "talwar"]

        for method ∈ methods
            @testset "$method" begin
                model = Loess(fraction = 0.7, iterations = 3, robustness_method = method)
                result = fit(model, x, y)
                @test length(result.y) == length(x)
            end
        end
    end

    @testset "Iterations" begin
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 100.0, 8.0, 10.0]

        for iterations ∈ [0, 1, 3, 5]
            @testset "iterations=$iterations" begin
                model = Loess(fraction = 0.7, iterations = iterations)
                result = fit(model, x, y)
                @test length(result.y) == length(x)
            end
        end
    end

    @testset "StreamingLoess" begin
        @testset "basic streaming" begin
            x = collect(range(0, 1000, length = 2000))
            y = sin.(x ./ 100)

            stream = StreamingLoess(fraction = 0.1, chunk_size = 1000)

            # First chunk
            r1 = process_chunk(stream, x[1:1000], y[1:1000])
            @test r1 isa LoessResult # Partial results might be empty or not, but generally empty until finalized or enough data

            # Second chunk
            r2 = process_chunk(stream, x[1001:end], y[1001:end])

            r_final = finalize(stream)

            # Rust streaming usually returns results per chunk if possible, or buffered. 
            # The test just checks it runs and returns a LoessResult structure.
            @test r_final isa LoessResult
        end

        @testset "larger dataset streaming results" begin
            Random.seed!(42)
            x = collect(range(0, 1000, length = 5000))
            y = sin.(x ./ 100) .+ randn(5000) .* 0.1

            stream = StreamingLoess(fraction = 0.05, chunk_size = 1500)

            # We just verify it runs without error
            process_chunk(stream, x[1:2500], y[1:2500])
            process_chunk(stream, x[2501:end], y[2501:end])
            finalize(stream)
        end

        @testset "streaming accuracy" begin
            x = collect(range(0, 100, length = 200))
            y = 2 .* x .+ 1  # Perfect linear

            stream = StreamingLoess(fraction = 0.5, chunk_size = 1000)
            r1 = process_chunk(stream, x, y)
            r2 = finalize(stream)

            model_batch = Loess(fraction = 0.5)
            result_batch = fit(model_batch, x, y)

            # Combine streaming results
            stream_y = vcat(r1.y, r2.y)

            @test stream_y ≈ result_batch.y rtol = 1e-10
        end
    end

    @testset "OnlineLoess" begin
        @testset "basic online" begin
            x = collect(Float64, 1:10)
            y = collect(Float64, 2:2:20)

            online = OnlineLoess(fraction = 0.5, window_capacity = 10, min_points = 3)
            results = [add_point(online, x[i], y[i]) for i ∈ eachindex(x)]

            @test any(r !== nothing for r ∈ results)
            @test all(r === nothing || r isa OnlineOutput for r ∈ results)
        end

        @testset "with noise" begin
            Random.seed!(42)
            x = collect(range(0, 20, length = 50))
            y = 2 .* x .+ randn(50)

            online = OnlineLoess(fraction = 0.3, window_capacity = 20, min_points = 5)
            results = [add_point(online, x[i], y[i]) for i ∈ eachindex(x)]

            @test any(r !== nothing for r ∈ results)
        end

        @testset "update modes" begin
            x = collect(Float64, 0:99)
            y = 20.0 .+ 5.0 .* sin.(x .* 0.1)

            o1 = OnlineLoess(fraction = 0.3, window_capacity = 50, update_mode = "full")
            results_full = [add_point(o1, x[i], y[i]) for i ∈ eachindex(x)]

            o2 = OnlineLoess(
                fraction = 0.3,
                window_capacity = 50,
                update_mode = "incremental",
            )
            results_inc = [add_point(o2, x[i], y[i]) for i ∈ eachindex(x)]

            @test any(r !== nothing for r ∈ results_full)
            @test any(r !== nothing for r ∈ results_inc)
        end
    end

    @testset "Result Fields" begin
        @testset "optional fields none" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            y = [2.0, 4.0, 6.0, 8.0, 10.0]

            model = Loess(fraction = 0.5)
            result = fit(model, x, y)

            @test result.diagnostics === nothing
            @test result.residuals === nothing
            @test result.robustness_weights === nothing
            @test result.confidence_lower === nothing
            @test result.confidence_upper === nothing
            @test result.prediction_lower === nothing
            @test result.prediction_upper === nothing
        end
    end

    @testset "Diagnostics Values" begin
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]  # Perfect linear

        model = Loess(fraction = 0.5, return_diagnostics = true)
        result = fit(model, x, y)

        diag = result.diagnostics
        @test diag !== nothing
        @test diag.rmse < 0.1
        @test diag.mae < 0.1
        @test diag.r_squared > 0.99
    end

    @testset "Edge Cases" begin
        @testset "two points" begin
            x = [1.0, 2.0]
            y = [2.0, 4.0]

            model = Loess(fraction = 1.0)
            result = fit(model, x, y)
            @test length(result.y) == 2
        end

        @testset "large dataset" begin
            Random.seed!(42)
            n = 1000
            x = collect(range(0, 100, length = n))
            y = sin.(x ./ 10) .+ randn(n) .* 0.1

            model = Loess(fraction = 0.1)
            result = fit(model, x, y)
            @test length(result.y) == n
        end

        @testset "unsorted input" begin
            x = [3.0, 1.0, 5.0, 2.0, 4.0]
            y = [6.0, 2.0, 10.0, 4.0, 8.0]

            model = Loess(fraction = 0.7)
            result = fit(model, x, y)
            @test length(result.y) == 5
        end

        @testset "duplicate x values" begin
            x = [1.0, 1.0, 2.0, 2.0, 3.0]
            y = [2.0, 2.1, 4.0, 3.9, 6.0]

            model = Loess(fraction = 0.7)
            result = fit(model, x, y)
            @test length(result.y) == 5
        end

        @testset "constant y values" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            y = [5.0, 5.0, 5.0, 5.0, 5.0]

            model = Loess(fraction = 0.5)
            result = fit(model, x, y)
            @test result.y ≈ y rtol = 1e-10
        end
    end

    @testset "Cross-Validation" begin
        @testset "basic CV" begin
            x = collect(range(0, 10, length = 50))
            y = 2 .* x .+ sin.(x)

            model = Loess(cv_fractions = [0.2, 0.3, 0.5, 0.7])
            result = fit(model, x, y)

            @test result.fraction_used in [0.2, 0.3, 0.5, 0.7]
            @test length(result.y) == length(x)
        end

        @testset "k-fold CV" begin
            x = collect(range(0, 10, length = 30))
            y = x .^ 2

            model = Loess(cv_fractions = [0.3, 0.5], cv_method = "kfold", cv_k = 5)
            result = fit(model, x, y)

            @test result.fraction_used in [0.3, 0.5]
        end

        @testset "LOOCV" begin
            x = collect(range(0, 10, length = 20))
            y = sin.(x)

            model = Loess(cv_fractions = [0.4, 0.6], cv_method = "loocv")
            result = fit(model, x, y)

            @test result.fraction_used in [0.4, 0.6]
        end
    end

    @testset "Error Handling" begin
        @testset "mismatched lengths" begin
            x = [1.0, 2.0, 3.0]
            y = [2.0, 4.0]

            model = Loess(fraction = 0.5)
            @test_throws ArgumentError fit(model, x, y)
        end

        @testset "invalid weight function" begin
            # Error happens at construction time now
            @test_throws ErrorException Loess(fraction = 0.5, weight_function = "invalid")
        end

        @testset "invalid robustness method" begin
            @test_throws ErrorException Loess(fraction = 0.5, robustness_method = "invalid")
        end
    end

    @testset "Parameter Coverage" begin
        x20 = collect(range(0, 10, length = 20))
        y20 = sin.(x20)
        x5 = [1.0, 2.0, 3.0, 4.0, 5.0]
        y5 = [2.0, 4.0, 6.0, 8.0, 10.0]

        @testset "Loess: scaling_method" begin
            for sm ∈ ["mad", "mar", "mean"]
                r = fit(Loess(fraction = 0.5, scaling_method = sm), x5, y5)
                @test length(r.y) == 5
            end
        end

        @testset "Loess: boundary_policy" begin
            for bp ∈ ["extend", "reflect", "zero", "noboundary"]
                r = fit(Loess(fraction = 0.5, boundary_policy = bp), x5, y5)
                @test length(r.y) == 5
            end
        end

        @testset "Loess: zero_weight_fallback" begin
            for zwf ∈ ["use_local_mean", "return_original", "return_none"]
                r = fit(Loess(fraction = 0.5, zero_weight_fallback = zwf), x5, y5)
                @test length(r.y) == 5
            end
        end

        @testset "Loess: auto_converge" begin
            r = fit(Loess(fraction = 0.5, auto_converge = 1e-4), x5, y5)
            @test length(r.y) == 5
        end

        @testset "Loess: degree" begin
            for deg ∈ ["constant", "linear", "quadratic"]
                r = fit(Loess(fraction = 0.9, degree = deg), x5, y5)
                @test length(r.y) == 5
            end
        end

        @testset "Loess: distance_metric variants" begin
            for dm ∈ ["normalized", "euclidean", "manhattan", "chebyshev"]
                r = fit(Loess(fraction = 0.5, distance_metric = dm), x20, y20)
                @test length(r.y) == 20
            end
        end

        @testset "Loess: minkowski via distance_metric string" begin
            r = fit(Loess(fraction = 0.5, distance_metric = "minkowski:3"), x20, y20)
            @test length(r.y) == 20
        end

        @testset "Loess: surface_mode=direct" begin
            r = fit(Loess(fraction = 0.5, surface_mode = "direct"), x5, y5)
            @test length(r.y) == 5
        end

        @testset "Loess: return_se" begin
            r = fit(
                Loess(fraction = 0.5, return_se = true, surface_mode = "direct"),
                x20,
                y20,
            )
            @test r.trace_hat !== nothing
            @test r.leverage !== nothing
        end

        @testset "StreamingLoess: merge_strategy" begin
            xlong = collect(range(0, 100, length = 200))
            ylong = sin.(xlong ./ 10)
            for ms ∈ ["average", "weighted_average", "take_first", "take_last"]
                s = StreamingLoess(fraction = 0.3, chunk_size = 100, merge_strategy = ms)
                r1 = process_chunk(s, xlong, ylong)
                r2 = finalize(s)
                @test length(r1.y) + length(r2.y) == 200
            end
        end

        @testset "StreamingLoess: minkowski via distance_metric string" begin
            xlong = collect(range(0, 50, length = 100))
            ylong = sin.(xlong)
            s = StreamingLoess(
                fraction = 0.3,
                chunk_size = 60,
                distance_metric = "minkowski:2.5",
            )
            r1 = process_chunk(s, xlong, ylong)
            r2 = finalize(s)
            @test length(r1.y) + length(r2.y) == 100
        end

        @testset "StreamingLoess: misc params" begin
            xlong = collect(range(0, 100, length = 200))
            ylong = sin.(xlong ./ 10)
            s = StreamingLoess(
                fraction = 0.3,
                chunk_size = 100,
                overlap = 10,
                scaling_method = "mar",
                boundary_policy = "reflect",
                auto_converge = 1e-3,
                return_diagnostics = true,
                return_residuals = true,
                return_robustness_weights = true,
                zero_weight_fallback = "return_original",
                degree = "quadratic",
                surface_mode = "direct",
                return_se = true,
            )
            r1 = process_chunk(s, xlong, ylong)
            r2 = finalize(s)
            @test length(r1.y) + length(r2.y) == 200
        end

        @testset "OnlineLoess: update_mode" begin
            xo = collect(Float64, 1:20)
            yo = xo .* 2.0
            for um ∈ ["full", "incremental"]
                o = OnlineLoess(fraction = 0.5, window_capacity = 10, update_mode = um)
                results = [add_point(o, xo[i], yo[i]) for i ∈ eachindex(xo)]
                @test any(r !== nothing for r ∈ results)
            end
        end

        @testset "OnlineLoess: minkowski via distance_metric string" begin
            xo = collect(Float64, 1:20)
            yo = xo .* 2.0
            o = OnlineLoess(
                fraction = 0.5,
                window_capacity = 10,
                distance_metric = "minkowski:3",
            )
            results = [add_point(o, xo[i], yo[i]) for i ∈ eachindex(xo)]
            @test any(r !== nothing for r ∈ results)
        end

        @testset "OnlineLoess: misc params" begin
            xo = collect(range(0, 10, length = 30))
            yo = sin.(xo)
            o = OnlineLoess(
                fraction = 0.5,
                window_capacity = 20,
                degree = "quadratic",
                return_se = true,
                auto_converge = 1e-3,
                scaling_method = "mean",
                boundary_policy = "zero",
                return_robustness_weights = true,
                zero_weight_fallback = "return_none",
                surface_mode = "direct",
            )
            results = [add_point(o, xo[i], yo[i]) for i ∈ eachindex(xo)]
            @test any(r !== nothing for r ∈ results)
        end
    end

    @testset "Custom Weights" begin
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        y_outlier = [1.0, 2.0, 3.0, 100.0, 5.0, 6.0, 7.0]

        @testset "zero weight on outlier reduces error" begin
            w_zero = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
            model = Loess(fraction = 0.6)
            r_no_w = fit(model, x, y_outlier)
            r_w = fit(model, x, y_outlier; custom_weights = w_zero)

            non_outlier = [1, 2, 3, 5, 6, 7]
            err_no_w = mean(abs.(r_no_w.y[non_outlier] .- y_true[non_outlier]))
            err_w = mean(abs.(r_w.y[non_outlier] .- y_true[non_outlier]))
            @test err_w < err_no_w
        end

        @testset "uniform weights equal no weights" begin
            y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
            w_uniform = ones(length(y))
            model = Loess(fraction = 0.6)
            r_no_w = fit(model, x, y)
            r_w = fit(model, x, y; custom_weights = w_uniform)
            @test r_w.y ≈ r_no_w.y atol = 1e-6
        end

        @testset "wrong length raises error" begin
            y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
            w_bad = [1.0, 1.0, 1.0]
            model = Loess(fraction = 0.6)
            @test_throws Exception fit(model, x, y; custom_weights = w_bad)
        end

        @testset "negative weight raises error" begin
            y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
            w_neg = [1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            model = Loess(fraction = 0.6)
            @test_throws Exception fit(model, x, y; custom_weights = w_neg)
        end
    end

end  # main testset

println("\n✓ All tests passed!")
