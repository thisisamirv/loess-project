#!/usr/bin/env python3
"""
fastloess Streaming Smoothing - Comprehensive Examples

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

import time

import numpy as np
from fastloess import StreamingLoess


def make_linear(n: int):
    x = np.arange(n, dtype=float)
    y = 2.0 * x + 1.0
    return x, y


def process_all(
    model: StreamingLoess, x: np.ndarray, y: np.ndarray, chunk_size: int, overlap: int
):
    """Feed only full-size chunks; finalize() handles remaining data."""
    step = chunk_size - overlap
    result_x: list = []
    result_y: list = []
    n = len(x)
    start = 0
    while start + chunk_size <= n:
        res = model.process_chunk(
            x[start : start + chunk_size], y[start : start + chunk_size]
        )
        result_x.extend(res.x)
        result_y.extend(res.y)
        start += step
    fin = model.finalize()
    result_x.extend(fin.x)
    result_y.extend(fin.y)
    return np.array(result_x), np.array(result_y)


# ── Example 1: Basic Chunked Processing ─────────────────────────────────────
def example_1_basic_chunked_processing():
    print("Example 1: Basic Chunked Processing")

    n = 50
    x, y = make_linear(n)
    chunk_size, overlap = 15, 5

    model = StreamingLoess(
        fraction=0.5,
        iterations=2,
        chunk_size=chunk_size,
        overlap=overlap,
        return_residuals=True,
    )

    print(f"  Dataset: {n} pts, chunk={chunk_size}, overlap={overlap}")
    total_x: list = []
    total_y: list = []
    ci = 0
    start = 0
    while start + chunk_size <= n:
        res = model.process_chunk(
            x[start : start + chunk_size], y[start : start + chunk_size]
        )
        if len(res.x) > 0:
            total_x.extend(res.x)
            total_y.extend(res.y)
            print(
                f"  Chunk {ci}: {len(res.x)} pts (x: {res.x[0]:.0f}..{res.x[-1]:.0f})"
            )
        start += chunk_size - overlap
        ci += 1
    fin = model.finalize()
    if len(fin.x) > 0:
        total_x.extend(fin.x)
        total_y.extend(fin.y)
        print(f"  Finalize: {len(fin.x)} remaining pts")
    print(f"  Total: {len(total_y)}/{n}")
    print()


# ── Example 2: Chunk Size Comparison ─────────────────────────────────────────
def example_2_chunk_size_comparison():
    print("Example 2: Chunk Size Comparison")

    n = 100
    x, y = make_linear(n)

    for cs, ov, label in [(20, 5, "Small"), (50, 10, "Medium"), (80, 15, "Large")]:
        model = StreamingLoess(fraction=0.5, iterations=1, chunk_size=cs, overlap=ov)
        chunks = 0
        total = 0
        start = 0
        while start + cs <= n:
            res = model.process_chunk(x[start : start + cs], y[start : start + cs])
            if len(res.x) > 0:
                chunks += 1
                total += len(res.x)
            start += cs - ov
        fin = model.finalize()
        if len(fin.x) > 0:
            chunks += 1
            total += len(fin.x)
        print(f"  {label} (size={cs}, overlap={ov}): chunks={chunks}, total={total}")
    print()


# ── Example 3: Overlap Strategies ────────────────────────────────────────────
def example_3_overlap_strategies():
    print("Example 3: Overlap Strategies")

    n = 100
    x, y = make_linear(n)
    cs = 40

    for overlap, label in [
        (0, "No overlap"),
        (10, "10-pt overlap"),
        (20, "20-pt overlap"),
    ]:
        model = StreamingLoess(fraction=0.5, chunk_size=cs, overlap=overlap)
        total = 0
        step = cs - overlap
        start = 0
        while start + cs <= n:
            total += len(
                model.process_chunk(x[start : start + cs], y[start : start + cs]).x
            )
            start += step
        total += len(model.finalize().x)
        print(f"  {label}: total output={total}")
    print()


# ── Example 4: Large Dataset Processing ──────────────────────────────────────
def example_4_large_dataset_processing():
    print("Example 4: Large Dataset Processing")

    n = 10_000
    x = np.arange(n, dtype=float)
    y = np.sin(x * 0.01) + x * 0.001

    cs, ov = 500, 50
    model = StreamingLoess(fraction=0.05, iterations=2, chunk_size=cs, overlap=ov)

    total = 0
    step = cs - ov
    start = 0
    while start + cs <= n:
        total += len(
            model.process_chunk(x[start : start + cs], y[start : start + cs]).x
        )
        if total > 0 and total % 2000 < step:
            print(f"  Progress: ~{total} pts smoothed")
        start += step
    total += len(model.finalize().x)
    print(f"  Total: {total}/{n}, memory: constant (chunk={cs})")
    print()


# ── Example 5: Outlier Handling in Streaming Mode ─────────────────────────────
def example_5_outlier_handling():
    print("Example 5: Outlier Handling in Streaming Mode")

    n = 100
    x = np.arange(n, dtype=float)
    y = 2 * x + 1 + np.sin(x * 0.2) * 2
    y[[25, 50, 75]] += 50  # Outliers

    for method in ["bisquare", "huber", "talwar"]:
        model = StreamingLoess(
            fraction=0.5,
            iterations=5,
            robustness_method=method,
            chunk_size=30,
            overlap=10,
            return_residuals=True,
        )
        large = 0
        start = 0
        while start + 30 <= n:
            res = model.process_chunk(x[start : start + 30], y[start : start + 30])
            if res.residuals is not None:
                large += int(np.sum(np.abs(res.residuals) > 10))
            start += 20
        fin = model.finalize()
        if fin.residuals is not None:
            large += int(np.sum(np.abs(fin.residuals) > 10))
        print(f"  {method}: pts with |residual|>10: {large}")
    print()


# ── Example 6: File-Based Streaming Simulation ───────────────────────────────
def example_6_file_simulation():
    print("Example 6: File-Based Streaming Simulation")
    print("  Simulating: input.csv -> Smooth -> output.csv")

    total_lines, cs, ov = 200, 50, 10
    model = StreamingLoess(
        fraction=0.5, iterations=2, chunk_size=cs, overlap=ov, return_residuals=True
    )

    out_count = 0
    ci = 0
    start_line = 0
    while start_line < total_lines:
        end_line = min(start_line + cs, total_lines)
        xc = np.arange(start_line, end_line, dtype=float)
        yc = 2 * xc + 1 + np.sin(xc * 0.1) * 3
        print(f"  Reading chunk {ci} (lines {start_line}..{end_line - 1})")
        res = model.process_chunk(xc, yc)
        if len(res.x) > 0:
            out_count += len(res.x)
            print(f"    -> Writing {len(res.x)} smoothed pts (total: {out_count})")
        start_line += cs - ov
        ci += 1
    fin = model.finalize()
    if len(fin.x) > 0:
        out_count += len(fin.x)
        print(f"  Finalizing: {len(fin.x)} remaining pts")
    print(f"  Input: {total_lines}, Output: {out_count}")
    print()


# ── Example 7: Benchmark (Sequential Streaming) ───────────────────────────────
def example_7_benchmark():
    print("Example 7: Benchmark (Sequential Streaming)")

    n, cs, ov = 1000, 100, 10
    model = StreamingLoess(fraction=0.5, iterations=3, chunk_size=cs, overlap=ov)

    t0 = time.perf_counter()
    total = 0
    start = 0
    while start + cs <= n:
        xc = np.arange(start, start + cs, dtype=float)
        yc = np.sin(xc * 0.1) + np.cos(xc * 0.01)
        total += len(model.process_chunk(xc, yc).x)
        start += cs - ov
    total += len(model.finalize().x)
    ms = (time.perf_counter() - t0) * 1000

    print(f"  {total} pts in {ms:.2f}ms (chunk={cs}, overlap={ov})")
    print()


# ── Example 8: Merge Strategies ──────────────────────────────────────────────
def example_8_merge_strategies():
    print("Example 8: Merge Strategies")

    n = 50
    x, y = make_linear(n)

    for strategy in ["average", "weighted_average", "take_first", "take_last"]:
        model = StreamingLoess(
            fraction=0.5,
            iterations=2,
            chunk_size=20,
            overlap=5,
            merge_strategy=strategy,
        )
        total = 0
        start = 0
        while start + 20 <= n:
            total += len(
                model.process_chunk(x[start : start + 20], y[start : start + 20]).x
            )
            start += 15
        total += len(model.finalize().x)
        print(f"  {strategy}: total={total}")
    print()


# ── Example 9: Advanced Streaming Options ─────────────────────────────────────
def example_9_advanced_options():
    print("Example 9: Advanced Streaming Options")

    n = 50
    x, y = make_linear(n)

    model = StreamingLoess(
        fraction=0.5,
        iterations=2,
        degree="quadratic",
        scaling_method="mar",
        boundary_policy="reflect",
        zero_weight_fallback="return_original",
        distance_metric="manhattan",
        surface_mode="direct",
        return_se=True,
        return_diagnostics=True,
        return_robustness_weights=True,
        auto_converge=1e-3,
        chunk_size=20,
        overlap=5,
    )

    total = 0
    start = 0
    while start + 20 <= n:
        total += len(
            model.process_chunk(x[start : start + 20], y[start : start + 20]).x
        )
        start += 15
    fin = model.finalize()
    total += len(fin.x)

    print(f"  total pts: {total}")
    if fin.standard_errors is not None and len(fin.standard_errors) > 0:
        print(f"  standard_errors[0]: {fin.standard_errors[0]:.4f}")
    if fin.diagnostics is not None:
        print(f"  diagnostics.rmse: {fin.diagnostics.rmse:.3f}")
        print(f"  diagnostics.r_squared: {fin.diagnostics.r_squared:.3f}")
        if fin.diagnostics.aic is not None:
            print(f"  diagnostics.aic: {fin.diagnostics.aic:.3f}")
    if fin.robustness_weights is not None and len(fin.robustness_weights) > 0:
        print(f"  robustness_weights[0]: {fin.robustness_weights[0]:.4f}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("fastloess Streaming Smoothing - Comprehensive Examples")
    print("=" * 60)
    print()

    example_1_basic_chunked_processing()
    example_2_chunk_size_comparison()
    example_3_overlap_strategies()
    example_4_large_dataset_processing()
    example_5_outlier_handling()
    example_6_file_simulation()
    example_7_benchmark()
    example_8_merge_strategies()
    example_9_advanced_options()

    print("=== Streaming Smoothing Examples Complete ===")


if __name__ == "__main__":
    main()
