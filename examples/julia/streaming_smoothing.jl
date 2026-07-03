#!/usr/bin/env julia
"""
FastLOESS Streaming Smoothing - Comprehensive Examples

 1. Basic chunked processing
 2. Chunk size comparison
 3. Overlap strategies
 4. Large dataset processing
 5. Outlier handling in streaming mode
 6. File-based streaming simulation
 7. Benchmark (sequential streaming)
 8. Merge strategies
 9. Advanced streaming options
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

make_linear(n) = (collect(Float64, 0:(n-1)), collect(Float64, 0:(n-1)) .* 2 .+ 1)

"""Feed only full-size chunks; finalize() handles remaining data."""
function collect_chunks(model, x, y, chunk_size, overlap)
	step = chunk_size - overlap
	n = length(x)
	result = nothing
	start = 1
	while start + chunk_size - 1 <= n
		res = process_chunk(model, x[start:(start+chunk_size-1)], y[start:(start+chunk_size-1)])
		if result === nothing
			result = res
		else
			append!(result, res)
		end
		start += step
	end
	fin = finalize(model)
	if result === nothing
		result = fin
	else
		append!(result, fin)
	end
	return result
end

# ── Example 1: Basic Chunked Processing ─────────────────────────────────────
function example_1_basic_chunked_processing()
	println("Example 1: Basic Chunked Processing")
	n = 50
	x, y = make_linear(n)
	chunk_size, overlap = 15, 5
	model = StreamingLoess(fraction = 0.5, iterations = 2, chunk_size = chunk_size,
		overlap = overlap, return_residuals = true)
	println("  Dataset: $n pts, chunk=$chunk_size, overlap=$overlap")
	total = 0
	ci = 0
	result = nothing
	start = 1
	while start + chunk_size - 1 <= n
		res = process_chunk(model, x[start:(start+chunk_size-1)], y[start:(start+chunk_size-1)])
		if length(res.x) > 0
			total += length(res.x)
			@printf("  Chunk %d: %d pts (x: %.0f..%.0f)\n", ci, length(res.x), res.x[1], res.x[end])
			result = result === nothing ? res : (append!(result, res); result)
		end
		start += chunk_size - overlap
		ci += 1
	end
	fin = finalize(model)
	if length(fin.x) > 0
		total += length(fin.x)
		@printf("  Finalize: %d remaining pts\n", length(fin.x))
	end
	println("  Total: $total/$n")
	println()
end

# ── Example 2: Chunk Size Comparison ─────────────────────────────────────────
function example_2_chunk_size_comparison()
	println("Example 2: Chunk Size Comparison")
	n = 100
	x, y = make_linear(n)
	for (cs, ov, label) ∈ [(20, 5, "Small"), (50, 10, "Medium"), (80, 15, "Large")]
		model = StreamingLoess(fraction = 0.5, iterations = 1, chunk_size = cs, overlap = ov)
		chunks = 0;
		total = 0;
		start = 1
		while start + cs - 1 <= n
			res = process_chunk(model, x[start:(start+cs-1)], y[start:(start+cs-1)])
			if length(res.x) > 0
				;
				chunks += 1;
				total += length(res.x);
			end
			start += cs - ov
		end
		fin = finalize(model)
		if length(fin.x) > 0
			;
			chunks += 1;
			total += length(fin.x);
		end
		println("  $label (size=$cs, overlap=$ov): chunks=$chunks, total=$total")
	end
	println()
end

# ── Example 3: Overlap Strategies ────────────────────────────────────────────
function example_3_overlap_strategies()
	println("Example 3: Overlap Strategies")
	n = 100
	x, y = make_linear(n)
	cs = 40
	for (overlap, label) ∈ [(0, "No overlap"), (10, "10-pt overlap"), (20, "20-pt overlap")]
		model = StreamingLoess(fraction = 0.5, chunk_size = cs, overlap = overlap)
		total = 0;
		step = cs - overlap;
		start = 1
		while start + cs - 1 <= n
			total += length(process_chunk(model, x[start:(start+cs-1)], y[start:(start+cs-1)]).x)
			start += step
		end
		total += length(finalize(model).x)
		println("  $label: total output=$total")
	end
	println()
end

# ── Example 4: Large Dataset Processing ──────────────────────────────────────
function example_4_large_dataset_processing()
	println("Example 4: Large Dataset Processing")
	n = 10_000
	x = collect(Float64, 0:(n-1))
	y = sin.(x .* 0.01) .+ x .* 0.001
	cs, ov = 500, 50
	model = StreamingLoess(fraction = 0.05, iterations = 2, chunk_size = cs, overlap = ov)
	total = 0;
	step = cs - ov;
	start = 1
	while start + cs - 1 <= n
		total += length(process_chunk(model, x[start:(start+cs-1)], y[start:(start+cs-1)]).x)
		if total > 0 && total % 2000 < step
			println("  Progress: ~$total pts smoothed")
		end
		start += step
	end
	total += length(finalize(model).x)
	println("  Total: $total/$n, memory: constant (chunk=$cs)")
	println()
end

# ── Example 5: Outlier Handling in Streaming Mode ─────────────────────────────
function example_5_outlier_handling()
	println("Example 5: Outlier Handling in Streaming Mode")
	n = 100
	x = collect(Float64, 0:(n-1))
	y = 2 .* x .+ 1 .+ sin.(x .* 0.2) .* 2
	y[[26, 51, 76]] .+= 50  # Outliers (1-indexed)
	for method ∈ ["bisquare", "huber", "talwar"]
		model = StreamingLoess(fraction = 0.5, iterations = 5, robustness_method = method,
			chunk_size = 30, overlap = 10, return_residuals = true)
		large = 0;
		start = 1
		while start + 29 <= n
			res = process_chunk(model, x[start:(start+29)], y[start:(start+29)])
			if res.residuals !== nothing
				large += count(r -> abs(r) > 10, res.residuals)
			end
			start += 20
		end
		fin = finalize(model)
		if fin.residuals !== nothing
			large += count(r -> abs(r) > 10, fin.residuals)
		end
		println("  $method: pts with |residual|>10: $large")
	end
	println()
end

# ── Example 6: File-Based Streaming Simulation ───────────────────────────────
function example_6_file_simulation()
	println("Example 6: File-Based Streaming Simulation")
	println("  Simulating: input.csv -> Smooth -> output.csv")
	total_lines, cs, ov = 200, 50, 10
	model = StreamingLoess(fraction = 0.5, iterations = 2, chunk_size = cs,
		overlap = ov, return_residuals = true)
	out_count = 0;
	ci = 0;
	start_line = 0
	while start_line < total_lines
		end_line = min(start_line + cs, total_lines)
		xc = collect(Float64, start_line:(end_line-1))
		yc = 2 .* xc .+ 1 .+ sin.(xc .* 0.1) .* 3
		println("  Reading chunk $ci (lines $start_line..$(end_line-1))")
		res = process_chunk(model, xc, yc)
		if length(res.x) > 0
			out_count += length(res.x)
			println("    -> Writing $(length(res.x)) smoothed pts (total: $out_count)")
		end
		start_line += cs - ov
		ci += 1
	end
	fin = finalize(model)
	if length(fin.x) > 0
		out_count += length(fin.x)
		println("  Finalizing: $(length(fin.x)) remaining pts")
	end
	println("  Input: $total_lines, Output: $out_count")
	println()
end

# ── Example 7: Benchmark (Sequential Streaming) ───────────────────────────────
function example_7_benchmark()
	println("Example 7: Benchmark (Sequential Streaming)")
	n, cs, ov = 1000, 100, 10
	model = StreamingLoess(fraction = 0.5, iterations = 3, chunk_size = cs, overlap = ov)
	t0 = time()
	total = 0;
	start = 1
	while start + cs - 1 <= n
		xc = collect(Float64, (start-1):(start+cs-2))
		yc = sin.(xc .* 0.1) .+ cos.(xc .* 0.01)
		total += length(process_chunk(model, xc, yc).x)
		start += cs - ov
	end
	total += length(finalize(model).x)
	ms = (time() - t0) * 1000
	@printf("  %d pts in %.2fms (chunk=%d, overlap=%d)\n", total, ms, cs, ov)
	println()
end

# ── Example 8: Merge Strategies ──────────────────────────────────────────────
function example_8_merge_strategies()
	println("Example 8: Merge Strategies")
	n = 50
	x, y = make_linear(n)
	for strategy ∈ ["average", "weighted_average", "take_first", "take_last"]
		model = StreamingLoess(fraction = 0.5, iterations = 2, chunk_size = 20,
			overlap = 5, merge_strategy = strategy)
		total = 0;
		start = 1
		while start + 19 <= n
			total += length(process_chunk(model, x[start:(start+19)], y[start:(start+19)]).x)
			start += 15
		end
		total += length(finalize(model).x)
		println("  $strategy: total=$total")
	end
	println()
end

# ── Example 9: Advanced Streaming Options ─────────────────────────────────────
function example_9_advanced_options()
	println("Example 9: Advanced Streaming Options")
	n = 50
	x, y = make_linear(n)
	model = StreamingLoess(
		fraction = 0.5, iterations = 2,
		degree = "quadratic",
		scaling_method = "mar",
		boundary_policy = "reflect",
		zero_weight_fallback = "return_original",
		distance_metric = "manhattan",
		surface_mode = "direct",
		return_se = true,
		return_diagnostics = true,
		return_robustness_weights = true,
		auto_converge = 1e-3,
		chunk_size = 20, overlap = 5,
	)
	total = 0;
	start = 1
	while start + 19 <= n
		total += length(process_chunk(model, x[start:(start+19)], y[start:(start+19)]).x)
		start += 15
	end
	fin = finalize(model)
	total += length(fin.x)
	println("  total pts: $total")
	if fin.standard_errors !== nothing && !isempty(fin.standard_errors)
		@printf("  standard_errors[1]: %.4f\n", fin.standard_errors[1])
	end
	if fin.diagnostics !== nothing
		@printf("  diagnostics.rmse: %.3f\n", fin.diagnostics.rmse)
		@printf("  diagnostics.r_squared: %.3f\n", fin.diagnostics.r_squared)
		!isnan(fin.diagnostics.aic) && @printf("  diagnostics.aic: %.3f\n", fin.diagnostics.aic)
	end
	if fin.robustness_weights !== nothing && !isempty(fin.robustness_weights)
		@printf("  robustness_weights[1]: %.4f\n", fin.robustness_weights[1])
	end
	println()
end

# ── Main ──────────────────────────────────────────────────────────────────────
function main()
	println("=" ^ 60)
	println("FastLOESS Streaming Smoothing - Comprehensive Examples")
	println("=" ^ 60)
	println()

	example_1_basic_chunked_processing()
	example_2_chunk_size_comparison()
	example_3_overlap_strategies()
	example_4_large_dataset_processing()
	example_5_outlier_handling()
	example_6_file_simulation()
	example_7_benchmark()
	example_8_merge_strategies()
	example_9_advanced_options()

	println("=== Streaming Smoothing Examples Complete ===")
end

main()

