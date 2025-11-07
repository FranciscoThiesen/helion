#!/usr/bin/env python3
"""
Extract and display best configurations from benchmark results.

Author: Francisco Geiman Thiesen
Date: 2025-11-06
"""

import json
import re
from pathlib import Path


def extract_config_params(config_str):
    """Extract key parameters from config string."""
    params = {}

    # Extract block_sizes
    match = re.search(r'block_sizes=\[([^\]]+)\]', config_str)
    if match:
        params['block_sizes'] = f"[{match.group(1)}]"

    # Extract num_warps
    match = re.search(r'num_warps=(\d+)', config_str)
    if match:
        params['num_warps'] = match.group(1)

    # Extract num_stages
    match = re.search(r'num_stages=(\d+)', config_str)
    if match:
        params['num_stages'] = match.group(1)

    # Extract indexing
    match = re.search(r"indexing=\[([^\]]+)\]", config_str)
    if match:
        params['indexing'] = f"[{match.group(1)}]"

    # Extract pid_type
    match = re.search(r"pid_type='([^']+)'", config_str)
    if match:
        params['pid_type'] = match.group(1)

    # Extract load_eviction_policies
    match = re.search(r"load_eviction_policies=\[([^\]]+)\]", config_str)
    if match:
        params['load_eviction'] = f"[{match.group(1)}]"

    return params


def main():
    # Try to load results from final benchmark
    results_files = [
        'final_three_kernel_results.json',
        'de_convergence_results.json',
        'three_kernel_results.json'
    ]

    results_data = None
    used_file = None

    for fname in results_files:
        if Path(fname).exists():
            with open(fname, 'r') as f:
                results_data = json.load(f)
            used_file = fname
            break

    if not results_data:
        print("Error: No results file found")
        return

    print("="*90)
    print(f"BEST CONFIGURATION ANALYSIS")
    print(f"Source: {used_file}")
    print("="*90)
    print()

    # Extract best configs for each kernel/algorithm pair
    for kernel_result in results_data:
        kernel_name = kernel_result['kernel']

        print("="*90)
        print(f"KERNEL: {kernel_name}")
        print("="*90)
        print()

        # Sort results by performance
        sorted_results = sorted(
            kernel_result['results'],
            key=lambda x: x['performance']
        )

        for result in sorted_results:
            algo = result['algorithm']
            perf = result['performance']
            time_taken = result['time']
            evals = result['evaluations']

            print(f"{algo}:")
            print(f"  Performance: {perf:.4f} ms")
            print(f"  Time: {time_taken:.1f}s")
            print(f"  Evaluations: {evals}")

            # Try to load full config if available
            if 'best_config' in result and result['best_config']:
                config_str = result['best_config']
                params = extract_config_params(str(config_str))

                if params:
                    print(f"  Key Parameters:")
                    if 'block_sizes' in params:
                        print(f"    block_sizes: {params['block_sizes']}")
                    if 'num_warps' in params:
                        print(f"    num_warps: {params['num_warps']}")
                    if 'num_stages' in params:
                        print(f"    num_stages: {params['num_stages']}")
                    if 'indexing' in params:
                        print(f"    indexing: {params['indexing']}")
                    if 'pid_type' in params:
                        print(f"    pid_type: {params['pid_type']}")

            print()

    print("="*90)


if __name__ == '__main__':
    main()
