#!/usr/bin/env python3
"""
Compare fastLoess validation results against R's loess implementation.
R is the reference implementation (original Cleveland algorithm).
"""

import json
import numpy as np
from pathlib import Path

def compare_implementations():
    r_dir = Path("output/r")
    fastLoess_dir = Path("output/fastLoess")
    
    if not r_dir.exists():
        print("Error: R output directory not found. Run R/validate.R first.")
        return
    
    if not fastLoess_dir.exists():
        print("Error: fastLoess output directory not found. Run fastLoess/validate first.")
        return
    
    print("=" * 85)
    print("VALIDATION: fastLoess vs R (Reference Implementation)")
    print("=" * 85)
    print()
    print(f"{'SCENARIO':<30} | {'STATUS':<15} | {'MAX DIFF':<15} | {'RMSE':<15}")
    print("-" * 85)
    
    scenarios = sorted([f.stem for f in r_dir.glob("*.json")])
    
    matches = []
    mismatches = []
    
    for scenario in scenarios:
        r_file = r_dir / f"{scenario}.json"
        fastLoess_file = fastLoess_dir / f"{scenario}.json"
        
        if not fastLoess_file.exists():
            print(f"{scenario:<30} | {'MISSING':<15} | {'-':<15} | {'-':<15}")
            continue
        
        # Load data
        with open(r_file) as f:
            r_data = json.load(f)
        with open(fastLoess_file) as f:
            fastLoess_data = json.load(f)
        
        # Compare fitted values
        r_fitted = np.array(r_data['result']['fitted'])
        fastLoess_fitted = np.array(fastLoess_data['result']['fitted'])
        
        diff = np.abs(r_fitted - fastLoess_fitted)
        max_diff = np.max(diff)
        rmse = np.sqrt(np.mean(diff**2))
        
        # Determine status
        if max_diff < 1e-10:
            status = "EXACT MATCH"
            matches.append(scenario)
        elif max_diff < 1e-6:
            status = "MATCH"
            matches.append(scenario)
        elif max_diff < 1e-3:
            status = "CLOSE"
            matches.append(scenario)
        else:
            status = "MISMATCH"
            mismatches.append(scenario)
        
        print(f"{scenario:<30} | {status:<15} | {max_diff:<15.2e} | {rmse:<15.2e}")
    
    print("-" * 85)
    print()
    print(f"Summary: {len(matches)} matches, {len(mismatches)} mismatches")
    
    if mismatches:
        print(f"\nFAILURES ({len(mismatches)}):")
        for scenario in mismatches:
            print(f"  - {scenario}")
    else:
        print("\n✓ All scenarios PASS!")
    
    return len(mismatches) == 0

if __name__ == "__main__":
    success = compare_implementations()
    exit(0 if success else 1)
