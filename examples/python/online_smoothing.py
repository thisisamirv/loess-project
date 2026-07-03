#!/usr/bin/env python3
"""
fastloess Online Smoothing - Comprehensive Examples

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

import time

import numpy as np
from fastloess import OnlineLoess


# ── Example 1: Basic Incremental Processing ──────────────────────────────────
def example_1_basic_streaming():
    print("Example 1: Basic Incremental Processing")

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    y = np.array([3.1, 5.0, 7.2, 8.9, 11.1, 13.0, 15.2, 16.8, 19.1, 21.0])

    model = OnlineLoess(fraction=0.5, iterations=2, window_capacity=5)
    result = model.add_points(x, y)

    print(f"  {'X':>8} {'Y_obs':>12} {'Y_smooth':>12}")
    for i in range(len(result.y)):
        print(f"  {result.x[i]:8.2f} {y[i]:12.2f} {result.y[i]:12.2f}")
    print()


# ── Example 2: Real-Time Sensor Data Simulation ───────────────────────────────
def example_2_sensor_data_simulation():
    print("Example 2: Real-Time Sensor Data Simulation")
    print("  Simulating temperature sensor with noise...")

    hours = np.arange(24, dtype=float)
    temp = 20 + 5 * np.sin(hours * np.pi / 12) + (hours * 7 % 11) * 0.3 - 1.5

    model = OnlineLoess(
        fraction=0.4, iterations=3, robustness_method="bisquare", window_capacity=12
    )
    result = model.add_points(hours, temp)

    print(f"  {'Hour':>6} {'Raw':>12} {'Smoothed':>12}")
    for i in range(len(result.y)):
        print(f"  {result.x[i]:6.0f} {temp[i]:10.2f}°C {result.y[i]:10.2f}°C")
    print()


# ── Example 3: Outlier Handling in Online Mode ────────────────────────────────
def example_3_outlier_handling():
    print("Example 3: Outlier Handling in Online Mode")

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    y = np.array([2.0, 4.1, 5.9, 25.0, 10.1, 12.0, 14.1, 50.0, 18.0, 20.1])

    for method in ["bisquare", "talwar"]:
        model = OnlineLoess(
            fraction=0.5, iterations=5, robustness_method=method, window_capacity=6
        )
        result = model.add_points(x, y)
        print(f"  {method}: [{', '.join(f'{v:.1f}' for v in result.y)}]")
    print()


# ── Example 4: Window Size Comparison ────────────────────────────────────────
def example_4_window_size_comparison():
    print("Example 4: Window Size Comparison")

    x = np.arange(1, 21, dtype=float)
    y = 2 * x + np.sin(x * 0.5) * 3

    for w in [5, 10, 15]:
        model = OnlineLoess(fraction=0.5, iterations=2, window_capacity=w)
        result = model.add_points(x, y)
        last5 = result.y[-5:]
        print(
            f"  window_capacity={w}: last 5 = [{', '.join(f'{v:.2f}' for v in last5)}]"
        )
    print()


# ── Example 5: Memory-Bounded Processing ──────────────────────────────────────
def example_5_memory_bounded_processing():
    print("Example 5: Memory-Bounded Processing (Embedded Systems)")

    total = 1000
    x = np.arange(total, dtype=float)
    y = 2 * x + np.sin(x * 0.1) * 5 + (np.arange(total) % 7 - 3) * 0.5

    model = OnlineLoess(fraction=0.3, iterations=1, window_capacity=20)
    result = model.add_points(x, y)

    n_out = len(result.y)
    for milestone in [200, 400, 600, 800, 1000]:
        if milestone <= n_out:
            print(
                f"  Processed: {milestone:4d} pts | smoothed={result.y[milestone - 1]:.2f}"
            )
    print(f"  Total: {n_out}, final smoothed: {result.y[-1]:.2f}")
    print("  Memory: constant (window=20)")
    print()


# ── Example 6: Sliding Window Behavior ───────────────────────────────────────
def example_6_sliding_window_behavior():
    print("Example 6: Sliding Window Behavior")

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    y = np.array([2, 4, 6, 8, 10, 12, 14, 16], dtype=float)

    # Use .update() to process point-by-point to show buffering behaviour
    model = OnlineLoess(fraction=0.6, iterations=0, window_capacity=4)
    print(f"  {'Pt':>4} {'X':>4} {'Y':>6} {'Smoothed':>10} Status")
    for i, (xi, yi) in enumerate(zip(x, y), 1):
        smoothed = model.update(xi, yi)
        if smoothed is not None:
            print(
                f"  {i:4d} {xi:4.0f} {yi:6.0f} {smoothed:10.2f} Window full (sliding)"
            )
        else:
            print(f"  {i:4d} {xi:4.0f} {yi:6.0f} {'-':>10} Filling ({i}/4)")
    print("  Output starts after window fills (4 pts), then slides.")
    print()


# ── Example 7: Benchmark (Sequential Online) ──────────────────────────────────
def example_7_benchmark():
    print("Example 7: Benchmark (Sequential Online)")

    n = 1000
    x = np.arange(n, dtype=float)
    y = np.sin(x * 0.1) + np.cos(x * 0.01)

    model = OnlineLoess(fraction=0.5, iterations=3, window_capacity=10)

    t0 = time.perf_counter()
    result = model.add_points(x, y)
    ms = (time.perf_counter() - t0) * 1000

    print(f"  {len(result.y)} pts processed in {ms:.2f}ms (window_capacity=10)")
    print()


# ── Example 8: Update Modes (Full vs Incremental) and min_points ───────────────
def example_8_update_modes():
    print("Example 8: Update Modes (Full vs Incremental) and min_points")

    x = np.arange(30, dtype=float)
    y = 2 * x + 1.0

    for mode in ["full", "incremental"]:
        model = OnlineLoess(
            fraction=0.5,
            iterations=2,
            update_mode=mode,
            min_points=5,
            window_capacity=15,
        )
        result = model.add_points(x, y)
        print(f"  {mode}: {len(result.y)} pts emitted (out of {len(x)})")

    # Show fraction_used and iterations_used from LoessResult
    model = OnlineLoess(fraction=0.5, iterations=2, window_capacity=10, min_points=3)
    result = model.add_points(x, y)
    print(f"  last smoothed: {result.y[-1]:.3f}")
    print(f"  fraction_used: {result.fraction_used}")
    if result.iterations_used is not None:
        print(f"  iterations_used: {result.iterations_used}")
    print()


# ── Example 9: Advanced Online Options ────────────────────────────────────────
def example_9_advanced_online_options():
    print("Example 9: Advanced Online Options")

    x = np.arange(30, dtype=float)
    y = 2 * x + 1.0

    model = OnlineLoess(
        fraction=0.5,
        iterations=2,
        degree="quadratic",
        scaling_method="mar",
        boundary_policy="reflect",
        zero_weight_fallback="return_original",
        distance_metric="chebyshev",
        auto_converge=1e-3,
        return_robustness_weights=True,
        min_points=5,
        window_capacity=15,
    )
    result = model.add_points(x, y)
    print(f"  emitted: {len(result.y)}")
    print(f"  last smoothed: {result.y[-1]:.3f}")
    print(f"  fraction_used: {result.fraction_used}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("fastloess Online Smoothing - Comprehensive Examples")
    print("=" * 60)
    print()

    example_1_basic_streaming()
    example_2_sensor_data_simulation()
    example_3_outlier_handling()
    example_4_window_size_comparison()
    example_5_memory_bounded_processing()
    example_6_sliding_window_behavior()
    example_7_benchmark()
    example_8_update_modes()
    example_9_advanced_online_options()

    print("=== Online Smoothing Examples Complete ===")


if __name__ == "__main__":
    main()
