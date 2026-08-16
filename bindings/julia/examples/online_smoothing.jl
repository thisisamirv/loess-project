#!/usr/bin/env julia
"""
FastLOESS Online Smoothing - Comprehensive Examples

 1. Basic incremental processing
 2. Real-time sensor data simulation
 3. Outlier handling in online mode
 4. Window size comparison
 5. Memory-bounded processing (embedded systems)
 6. Sliding window behavior
 7. Benchmark (sequential online)
 8. Update modes (Full vs Incremental) and min_points
 9. Advanced online options
"""

using Printf

# Handle package loading
using Pkg
project_name = Pkg.project().name
if project_name != "FastLOESS"
    script_dir = @__DIR__
    julia_pkg_dir = joinpath(dirname(script_dir), "julia")
    if !haskey(Pkg.project().dependencies, "FastLOESS")
        Pkg.develop(path = julia_pkg_dir)
    end
end

using FastLOESS

# Helper: feed all (x, y) pairs through add_point; return vector of smoothed values (NaN when None).
function add_all_points(model, x, y)
    [
        let r = add_point(model, x[i], y[i]);
            r !== nothing ? r.smoothed : NaN
        end for i ∈ eachindex(x)
    ]
end

# ── Example 1: Basic Incremental Processing ──────────────────────────────────
function example_1_basic_streaming()
    println("Example 1: Basic Incremental Processing")
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    y = [3.1, 5.0, 7.2, 8.9, 11.1, 13.0, 15.2, 16.8, 19.1, 21.0]
    model = OnlineLoess(fraction = 0.5, iterations = 2, window_capacity = 5)
    smoothed = add_all_points(model, x, y)
    @printf("  %8s %12s %12s\n", "X", "Y_obs", "Y_smooth")
    for i ∈ eachindex(x)
        sv = isnan(smoothed[i]) ? "  (buffering)" : @sprintf("%12.2f", smoothed[i])
        @printf("  %8.2f %12.2f %s\n", x[i], y[i], sv)
    end
    println()
end

# ── Example 2: Real-Time Sensor Data Simulation ───────────────────────────────
function example_2_sensor_data_simulation()
    println("Example 2: Real-Time Sensor Data Simulation")
    println("  Simulating temperature sensor with noise...")
    hours = collect(Float64, 0:23)
    temp = 20 .+ 5 .* sin.(hours .* π ./ 12) .+ (mod.(hours .* 7, 11)) .* 0.3 .- 1.5
    model = OnlineLoess(
        fraction = 0.4,
        iterations = 3,
        robustness_method = "bisquare",
        window_capacity = 12,
    )
    smoothed = add_all_points(model, hours, temp)
    @printf("  %6s %12s %12s\n", "Hour", "Raw", "Smoothed")
    for i ∈ eachindex(hours)
        sv = isnan(smoothed[i]) ? "  (warming up)" : @sprintf("%10.2f°C", smoothed[i])
        @printf("  %6.0f %10.2f°C %s\n", hours[i], temp[i], sv)
    end
    println()
end

# ── Example 3: Outlier Handling in Online Mode ────────────────────────────────
function example_3_outlier_handling()
    println("Example 3: Outlier Handling in Online Mode")
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    y = [2.0, 4.1, 5.9, 25.0, 10.1, 12.0, 14.1, 50.0, 18.0, 20.1]
    for method ∈ ["bisquare", "talwar"]
        model = OnlineLoess(
            fraction = 0.5,
            iterations = 5,
            robustness_method = method,
            window_capacity = 6,
        )
        smoothed = filter(!isnan, add_all_points(model, x, y))
        println("  $method: [$(join(round.(smoothed, digits=1), ", "))]")
    end
    println()
end

# ── Example 4: Window Size Comparison ────────────────────────────────────────
function example_4_window_size_comparison()
    println("Example 4: Window Size Comparison")
    x = collect(Float64, 1:20)
    y = 2 .* x .+ sin.(x .* 0.5) .* 3
    for w ∈ [5, 10, 15]
        model = OnlineLoess(fraction = 0.5, iterations = 2, window_capacity = w)
        smoothed = filter(!isnan, add_all_points(model, x, y))
        last5 = smoothed[(end-4):end]
        println("  window_capacity=$w: last 5 = [$(join(round.(last5, digits=2), ", "))]")
    end
    println()
end

# ── Example 5: Memory-Bounded Processing ──────────────────────────────────────
function example_5_memory_bounded_processing()
    println("Example 5: Memory-Bounded Processing (Embedded Systems)")
    total = 1000
    x = collect(Float64, 0:(total-1))
    y = 2 .* x .+ sin.(x .* 0.1) .* 5 .+ (mod.(0:(total-1), 7) .- 3) .* 0.5
    model = OnlineLoess(fraction = 0.3, iterations = 1, window_capacity = 20)
    smoothed = filter(!isnan, add_all_points(model, x, y))
    n_out = length(smoothed)
    for milestone ∈ [200, 400, 600, 800, 1000]
        milestone <= n_out && @printf(
            "  Processed: %4d pts | smoothed=%.2f\n",
            milestone,
            smoothed[milestone]
        )
    end
    @printf("  Total: %d, final smoothed: %.2f\n", n_out, smoothed[end])
    println("  Memory: constant (window=20)")
    println()
end

# ── Example 6: Sliding Window Behavior ───────────────────────────────────────
function example_6_sliding_window_behavior()
    println("Example 6: Sliding Window Behavior")
    x_all = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    y_all = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    model = OnlineLoess(fraction = 0.6, iterations = 0, window_capacity = 4)
    @printf("  %4s %4s %6s %10s %-22s\n", "Pt", "X", "Y", "Smoothed", "Status")
    for i ∈ eachindex(x_all)
        r = add_point(model, x_all[i], y_all[i])
        if r !== nothing
            @printf(
                "  %4d %4.0f %6.0f %10.2f %-22s\n",
                i,
                x_all[i],
                y_all[i],
                r.smoothed,
                "Window full (sliding)"
            )
        else
            @printf(
                "  %4d %4.0f %6.0f %10s %-22s\n",
                i,
                x_all[i],
                y_all[i],
                "-",
                "Filling ($i/4)"
            )
        end
    end
    println("  Output starts after window fills (4 pts), then slides.")
    println()
end

# ── Example 7: Benchmark (Sequential Online) ──────────────────────────────────
function example_7_benchmark()
    println("Example 7: Benchmark (Sequential Online)")
    n = 1000
    x = collect(Float64, 0:(n-1))
    y = sin.(x .* 0.1) .+ cos.(x .* 0.01)
    model = OnlineLoess(fraction = 0.5, iterations = 3, window_capacity = 10)
    t0 = time()
    smoothed = filter(!isnan, add_all_points(model, x, y))
    ms = (time() - t0) * 1000
    @printf("  %d pts processed in %.2fms (window_capacity=10)\n", length(smoothed), ms)
    println()
end

# ── Example 8: Update Modes (Full vs Incremental) and min_points ───────────────
function example_8_update_modes()
    println("Example 8: Update Modes (Full vs Incremental) and min_points")
    x = collect(Float64, 0:29)
    y = 2 .* x .+ 1
    for mode ∈ ["full", "incremental"]
        model = OnlineLoess(
            fraction = 0.5,
            iterations = 2,
            update_mode = mode,
            min_points = 5,
            window_capacity = 15,
        )
        smoothed = filter(!isnan, add_all_points(model, x, y))
        println("  $mode: $(length(smoothed)) pts emitted (out of $(length(x)))")
    end
    model =
        OnlineLoess(fraction = 0.5, iterations = 2, window_capacity = 10, min_points = 3)
    smoothed = filter(!isnan, add_all_points(model, x, y))
    @printf("  last smoothed: %.3f\n", smoothed[end])
    println()
end

# ── Example 9: Advanced Online Options ────────────────────────────────────────
function example_9_advanced_online_options()
    println("Example 9: Advanced Online Options")
    x = collect(Float64, 0:29)
    y = 2 .* x .+ 1
    model = OnlineLoess(
        fraction = 0.5,
        iterations = 2,
        degree = "quadratic",
        scaling_method = "mar",
        boundary_policy = "reflect",
        zero_weight_fallback = "return_original",
        distance_metric = "chebyshev",
        auto_converge = 1e-3,
        return_robustness_weights = true,
        min_points = 5,
        window_capacity = 15,
    )
    smoothed = filter(!isnan, add_all_points(model, x, y))
    println("  emitted: $(length(smoothed))")
    length(smoothed) > 0 && @printf("  last smoothed: %.3f\n", smoothed[end])
    println()
end

# ── Main ──────────────────────────────────────────────────────────────────────
function main()
    println("=" ^ 60)
    println("FastLOESS Online Smoothing - Comprehensive Examples")
    println("=" ^ 60)
    println()

    example_1_basic_streaming()
    example_2_sensor_data_simulation()
    example_3_outlier_handling()
    example_4_window_size_comparison()
    example_5_memory_bounded_processing()
    example_6_sliding_window_behavior()
    example_7_benchmark()
    example_8_update_modes()
    example_9_advanced_online_options()

    println("=== Online Smoothing Examples Complete ===")
end

main()
