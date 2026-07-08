#!/usr/bin/env python3
"""
Convert Criterion benchmark results to JSON format for comparison.
Criterion stores results in:
- target/criterion/<group>/<bench_id>/<param>/new/estimates.json
"""

import json
from pathlib import Path
from typing import Dict, List


def parse_estimates(estimates_file: Path) -> dict | None:
    """Parse a criterion estimates.json file and return timing in ms."""
    try:
        with open(estimates_file) as f:
            estimates = json.load(f)

        # Criterion stores in nanoseconds
        mean_ns = estimates.get("mean", {}).get("point_estimate", 0)
        std_ns = estimates.get("std_dev", {}).get("point_estimate", 0)
        median_ns = estimates.get("median", {}).get("point_estimate", 0)

        # Try to get min/max from sample data
        sample_file = estimates_file.parent / "sample.json"
        min_ns = mean_ns
        max_ns = mean_ns
        if sample_file.exists():
            try:
                with open(sample_file) as f:
                    sample = json.load(f)
                times = sample.get("times", [])
                if times:
                    min_ns = min(times)
                    max_ns = max(times)
            except Exception:
                pass

        return {
            "mean_time_ms": mean_ns / 1_000_000,
            "std_time_ms": std_ns / 1_000_000,
            "median_time_ms": median_ns / 1_000_000,
            "min_time_ms": min_ns / 1_000_000,
            "max_time_ms": max_ns / 1_000_000,
        }
    except Exception as e:
        print(f"Error parsing {estimates_file}: {e}")
        return None


def find_criterion_results(criterion_dir: Path, baseline: str) -> Dict[str, List[dict]]:
    """Find criterion results for a named baseline (e.g. 'parallel' or 'serial').

    Run benches with:
      cargo bench -p benchmark -- --save-baseline parallel
      $env:LOESS_PARALLEL="false"; cargo bench -p benchmark -- --save-baseline serial
    """
    results: Dict[str, List[dict]] = {}

    if not criterion_dir.exists():
        print(f"Criterion directory not found: {criterion_dir}")
        return results

    for group_dir in criterion_dir.iterdir():
        if not group_dir.is_dir() or group_dir.name == "report":
            continue

        category = group_dir.name
        if category not in results:
            results[category] = []

        for bench_dir in group_dir.iterdir():
            if not bench_dir.is_dir() or bench_dir.name == "report":
                continue

            bench_id = bench_dir.name

            # Non-parameterized: bench_dir/<baseline>/estimates.json
            baseline_dir = bench_dir / baseline
            if baseline_dir.exists() and (baseline_dir / "estimates.json").exists():
                timing = parse_estimates(baseline_dir / "estimates.json")
                if timing:
                    result = {
                        "name": bench_id,
                        "size": 5000,
                        "iterations": 10,
                        **timing,
                    }
                    results[category].append(result)
            else:
                # Parameterized: bench_dir/<param>/<baseline>/estimates.json
                for param_dir in bench_dir.iterdir():
                    if not param_dir.is_dir() or param_dir.name in (
                        "report",
                        "new",
                        "base",
                        "change",
                        baseline,
                    ):
                        continue

                    param = param_dir.name
                    estimates_file = param_dir / baseline / "estimates.json"

                    if estimates_file.exists():
                        timing = parse_estimates(estimates_file)
                        if timing:
                            try:
                                size = int(param)
                            except ValueError:
                                size = 0

                            if category == "scalability":
                                name = f"scale_{param}"
                            elif category in (
                                "financial",
                                "scientific",
                                "genomic",
                                "fraction",
                                "iterations",
                            ):
                                name = f"{category}_{param}"
                            else:
                                name = f"{bench_id}_{param}"

                            result = {
                                "name": name,
                                "size": size,
                                "iterations": 10,
                                **timing,
                            }
                            results[category].append(result)

    for key in results:
        results[key].sort(key=lambda x: x["name"])

    return results


def main():
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent
    criterion_dir = workspace_root / "target" / "criterion"
    output_dir = script_dir / "output"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading criterion results from: {criterion_dir}")

    def save_results(data, filename):
        if not data:
            return
        output_path = output_dir / filename
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to: {output_path}")

    parallel_results = find_criterion_results(criterion_dir, "parallel")
    serial_results = find_criterion_results(criterion_dir, "serial")

    if not parallel_results and not serial_results:
        print(
            "No criterion results found. Run 'cargo bench -- --save-baseline parallel' first."
        )
        return 1

    save_results(parallel_results, "rust_benchmark_cpu.json")
    save_results(serial_results, "rust_benchmark_cpu_serial.json")

    return 0


if __name__ == "__main__":
    exit(main())
