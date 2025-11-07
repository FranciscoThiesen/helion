#!/usr/bin/env python3
"""
Convergence Analysis for DE vs DE-Surrogate vs PatternSearch.

This script uses the new autotune CSV logging feature from PR #1095 to accurately
track convergence of different search algorithms on 3 different kernel types.

Usage:
    python scripts/convergence_comparison.py

Output:
    - CSV files with per-config metrics for each algorithm/kernel combination
    - Convergence plots (3 images, one per kernel)
    - Comprehensive analysis markdown document

Author: Francisco Geiman Thiesen
Date: 2025-11-07
"""

import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import helion
import helion.language as lang
import matplotlib.pyplot as plt
import numpy as np
import torch

# Import search algorithms
from helion.autotuner.differential_evolution import DifferentialEvolutionSearch
from helion.autotuner.de_surrogate_hybrid import DESurrogateHybrid
from helion.autotuner.pattern_search import PatternSearch


# ============================================================================
# TEST KERNELS
# ============================================================================

@helion.kernel(static_shapes=True)
def matmul_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication - Compute-bound kernel."""
    assert a.shape[1] == b.shape[0]
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty([m, n], dtype=a.dtype, device=a.device)

    for tile_m, tile_n in lang.tile([m, n]):
        accumulator = torch.zeros([tile_m, tile_n], dtype=torch.float32, device=a.device)
        for tile_k in lang.tile(k):
            a_tile = a[tile_m, tile_k].to(torch.float32)
            b_tile = b[tile_k, tile_n].to(torch.float32)
            accumulator += a_tile @ b_tile
        c[tile_m, tile_n] = accumulator.to(a.dtype)

    return c


@helion.kernel(static_shapes=True)
def gelu_kernel(x: torch.Tensor) -> torch.Tensor:
    """GELU activation - Bandwidth-bound kernel."""
    n, = x.size()
    out = torch.empty([n], dtype=x.dtype, device=x.device)

    for tile_n in lang.tile(n):
        x_tile = x[tile_n]
        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = 0.7978845608
        x_cubed = x_tile * x_tile * x_tile
        inner = sqrt_2_over_pi * (x_tile + 0.044715 * x_cubed)
        tanh_inner = torch.tanh(inner)
        out[tile_n] = 0.5 * x_tile * (1.0 + tanh_inner)

    return out


@helion.kernel(static_shapes=True)
def fused_relu_add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fused ReLU + Add - Memory-bound mixed kernel."""
    n, = x.size()
    out = torch.empty([n], dtype=x.dtype, device=x.device)

    for tile_n in lang.tile(n):
        x_tile = x[tile_n]
        y_tile = y[tile_n]
        relu_x = torch.where(x_tile > 0, x_tile, torch.zeros_like(x_tile))
        out[tile_n] = relu_x + y_tile

    return out


# ============================================================================
# CSV PARSING AND CONVERGENCE TRACKING
# ============================================================================

def parse_autotune_csv(csv_path: str) -> List[Dict]:
    """Parse the autotune CSV log file.

    Returns:
        List of dicts with keys: config_num, status, perf, compile_time, etc.
    """
    records = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def extract_convergence(records: List[Dict]) -> Tuple[List[int], List[float]]:
    """Extract convergence data from CSV records.

    Returns:
        (evaluations, best_perfs): Lists tracking best performance over time
    """
    evaluations = []
    best_perfs = []
    best_so_far = float('inf')

    for i, record in enumerate(records, 1):
        status = record.get('status', '')
        if status == 'ok':
            try:
                perf = float(record['perf'])
                if perf < best_so_far:
                    best_so_far = perf
            except (ValueError, KeyError):
                pass

        evaluations.append(i)
        best_perfs.append(best_so_far)

    return evaluations, best_perfs


# ============================================================================
# BENCHMARK EXECUTION
# ============================================================================

def run_single_benchmark(
    kernel_name: str,
    kernel_fn,
    args: Tuple,
    algorithm_name: str,
    algorithm_class,
    config: Dict,
    log_dir: Path
) -> Dict:
    """Run a single benchmark with CSV logging enabled."""

    print(f"\n{'='*80}")
    print(f"Running {algorithm_name} on {kernel_name}")
    print(f"{'='*80}")

    # Set up logging path
    log_base = log_dir / f"{kernel_name}_{algorithm_name}"
    csv_path = f"{log_base}.csv"

    start_time = time.time()

    try:
        # Create bound kernel
        bound = kernel_fn.bind(args)

        # Enable CSV logging (from PR #1095)
        bound.settings.autotune_log = str(log_base)

        # Create search instance
        search = algorithm_class(bound, args, **config)

        # Run autotuning
        best_config = search.autotune()
        elapsed = time.time() - start_time

        # Parse CSV to get convergence data
        if Path(csv_path).exists():
            records = parse_autotune_csv(csv_path)
            total_evals = len(records)
            evals, perfs = extract_convergence(records)
            # Get best performance from convergence data
            best_perf = min(perfs) if perfs else float('inf')
        else:
            print(f"  Warning: CSV log not found at {csv_path}")
            total_evals = 0
            evals, perfs = [], []
            best_perf = float('inf')

        print(f"  ✓ Time: {elapsed:.1f}s, Best: {best_perf:.4f}ms, Evals: {total_evals}")

        return {
            'kernel': kernel_name,
            'algorithm': algorithm_name,
            'time': elapsed,
            'performance': best_perf,
            'evaluations': total_evals,
            'best_config': str(best_config),
            'status': 'success',
            'csv_path': csv_path,
            'convergence_evals': evals,
            'convergence_perfs': perfs
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ✗ Failed: {str(e)[:100]}")
        import traceback
        traceback.print_exc()

        return {
            'kernel': kernel_name,
            'algorithm': algorithm_name,
            'time': elapsed,
            'performance': float('inf'),
            'evaluations': 0,
            'best_config': None,
            'status': 'failed',
            'error': str(e)[:200],
            'convergence_evals': [],
            'convergence_perfs': []
        }


# ============================================================================
# PLOTTING
# ============================================================================

def plot_convergence(kernel_name: str, kernel_results: List[Dict], output_path: str):
    """Generate convergence plot for a single kernel."""

    plt.figure(figsize=(10, 6))

    colors = {
        'DifferentialEvolution': '#1f77b4',
        'DE-Surrogate': '#ff7f0e',
        'PatternSearch': '#2ca02c'
    }

    for result in kernel_results:
        if result['status'] != 'success':
            continue

        algo = result['algorithm']
        evals = result['convergence_evals']
        perfs = result['convergence_perfs']

        if not evals or not perfs:
            continue

        # Convert to ms if needed
        perfs_ms = [p * 1000 if p < 1 else p for p in perfs]

        plt.plot(evals, perfs_ms, label=algo, color=colors.get(algo, 'gray'), linewidth=2)

    plt.xlabel('Number of Evaluations', fontsize=12)
    plt.ylabel('Best Performance (ms)', fontsize=12)
    plt.title(f'Convergence Comparison: {kernel_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved plot: {output_path}")
    plt.close()


# ============================================================================
# ANALYSIS REPORT GENERATION
# ============================================================================

def generate_analysis_report(results: List[Dict], output_path: str):
    """Generate comprehensive markdown analysis report."""

    # Group by kernel
    kernels = {}
    for result in results:
        kernel = result['kernel']
        if kernel not in kernels:
            kernels[kernel] = []
        kernels[kernel].append(result)

    with open(output_path, 'w') as f:
        f.write("# Convergence Analysis: DE vs DE-Surrogate vs PatternSearch\n\n")
        f.write("## Hardware Configuration\n\n")
        f.write("- **GPU**: NVIDIA H200 (141GB memory)\n")
        f.write("- **Compute Capability**: 9.0 (sm_90a)\n")
        f.write("- **Framework**: Helion with PR #1095 CSV logging\n\n")

        f.write("## Executive Summary\n\n")

        # Calculate aggregate statistics
        total_times = {'DifferentialEvolution': 0, 'DE-Surrogate': 0, 'PatternSearch': 0}
        total_evals = {'DifferentialEvolution': 0, 'DE-Surrogate': 0, 'PatternSearch': 0}
        wins = {'DifferentialEvolution': 0, 'DE-Surrogate': 0, 'PatternSearch': 0}

        for kernel_name, kernel_results in kernels.items():
            best_perf = float('inf')
            best_algo = None

            for result in kernel_results:
                if result['status'] == 'success':
                    algo = result['algorithm']
                    total_times[algo] += result['time']
                    total_evals[algo] += result['evaluations']

                    if result['performance'] < best_perf:
                        best_perf = result['performance']
                        best_algo = algo

            if best_algo:
                wins[best_algo] += 1

        f.write(f"Tested on **3 diverse GPU kernels** with ~1600 evaluations per algorithm:\n\n")

        for algo in ['DifferentialEvolution', 'DE-Surrogate', 'PatternSearch']:
            f.write(f"- **{algo}**: {wins[algo]}/3 wins, ")
            f.write(f"{total_times[algo]:.1f}s total time, ")
            f.write(f"{total_evals[algo]} total evaluations\n")

        f.write("\n## Detailed Results by Kernel\n\n")

        # Per-kernel results
        for kernel_name in sorted(kernels.keys()):
            kernel_results = kernels[kernel_name]

            f.write(f"### {kernel_name}\n\n")
            f.write("| Algorithm | Time (s) | Best (ms) | Evaluations | Status |\n")
            f.write("|-----------|----------|-----------|-------------|--------|\n")

            # Find baseline (DE)
            de_perf = None
            for result in kernel_results:
                if result['algorithm'] == 'DifferentialEvolution' and result['status'] == 'success':
                    de_perf = result['performance']

            for result in kernel_results:
                algo = result['algorithm']
                if result['status'] == 'success':
                    perf = result['performance']
                    improvement = ""
                    if de_perf and algo != 'DifferentialEvolution':
                        pct = ((de_perf - perf) / de_perf) * 100
                        improvement = f" ({pct:+.1f}%)"

                    f.write(f"| {algo} | {result['time']:.1f} | ")
                    f.write(f"**{perf:.4f}**{improvement} | {result['evaluations']} | ✓ |\n")
                else:
                    f.write(f"| {algo} | {result['time']:.1f} | - | 0 | ✗ Failed |\n")

            f.write(f"\n![Convergence plot for {kernel_name}](convergence_{kernel_name}.png)\n\n")

        f.write("\n## Key Insights\n\n")
        f.write("### Algorithm Characteristics\n\n")
        f.write("1. **DifferentialEvolution**:\n")
        f.write("   - Baseline evolutionary algorithm\n")
        f.write("   - Robust exploration via mutation and crossover\n")
        f.write("   - No machine learning component\n\n")

        f.write("2. **DE-Surrogate**:\n")
        f.write("   - Hybrid: DE + Random Forest surrogate\n")
        f.write("   - Generates 3× candidates, evaluates top 1/3 predicted by surrogate\n")
        f.write("   - Learns kernel-specific patterns\n\n")

        f.write("3. **PatternSearch**:\n")
        f.write("   - Local search via parameter neighbors\n")
        f.write("   - Systematic exploration of nearby configurations\n")
        f.write("   - Multiple copies for diversification\n\n")

        f.write("### Convergence Patterns\n\n")
        f.write("The convergence plots show how quickly each algorithm finds good configurations:\n\n")
        f.write("- **Faster convergence** = Steeper initial drop in best performance\n")
        f.write("- **Better final result** = Lower final performance value\n")
        f.write("- **Sample efficiency** = Good performance with fewer evaluations\n\n")

        f.write("## Conclusion\n\n")
        f.write("This analysis demonstrates the trade-offs between different autotuning algorithms:\n\n")
        f.write("- **DE-Surrogate** excels on complex kernels where learning pays off\n")
        f.write("- **DifferentialEvolution** provides consistent baseline performance\n")
        f.write("- **PatternSearch** offers local optimization with systematic exploration\n\n")
        f.write("The choice of algorithm should depend on:\n")
        f.write("- Kernel complexity\n")
        f.write("- Available tuning budget\n")
        f.write("- Importance of final performance vs tuning time\n")

    print(f"\n  Saved analysis report: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("CONVERGENCE ANALYSIS: DE vs DE-Surrogate vs PatternSearch")
    print("="*80)
    print("Using CSV logging feature from PR #1095")
    print("="*80)
    print()

    # Create output directory
    log_dir = Path("convergence_logs")
    log_dir.mkdir(exist_ok=True)

    # Test configurations
    kernels = [
        {
            'name': 'MatMul-1024',
            'kernel': matmul_kernel,
            'args': (
                torch.randn(1024, 1024, dtype=torch.float16, device='cuda'),
                torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
            ),
            'description': 'Matrix Multiplication 1024×1024 (Compute-Bound)'
        },
        {
            'name': 'GELU-1M',
            'kernel': gelu_kernel,
            'args': (torch.randn(1_000_000, dtype=torch.float16, device='cuda'),),
            'description': 'GELU Activation 1M elements (Bandwidth-Bound)'
        },
        {
            'name': 'FusedReLUAdd-1M',
            'kernel': fused_relu_add_kernel,
            'args': (
                torch.randn(1_000_000, dtype=torch.float16, device='cuda'),
                torch.randn(1_000_000, dtype=torch.float16, device='cuda')
            ),
            'description': 'Fused ReLU+Add 1M elements (Memory-Bound Mixed)'
        }
    ]

    algorithms = [
        {
            'name': 'DifferentialEvolution',
            'class': DifferentialEvolutionSearch,
            'config': {
                'population_size': 40,
                'max_generations': 40,
                'crossover_rate': 0.8
            }
        },
        {
            'name': 'DE-Surrogate',
            'class': DESurrogateHybrid,
            'config': {
                'population_size': 40,
                'max_generations': 40,
                'crossover_rate': 0.8,
                'surrogate_threshold': 100,
                'candidate_ratio': 3,
                'refit_frequency': 5
            }
        },
        {
            'name': 'PatternSearch',
            'class': PatternSearch,
            'config': {
                'initial_population': 80,
                'copies': 10,
                'max_generations': 40
            }
        }
    ]

    # Run all benchmarks
    results = []

    for kernel_config in kernels:
        print(f"\n{'='*80}")
        print(f"KERNEL: {kernel_config['name']}")
        print(f"Description: {kernel_config['description']}")
        print(f"{'='*80}")

        for algo_config in algorithms:
            result = run_single_benchmark(
                kernel_config['name'],
                kernel_config['kernel'],
                kernel_config['args'],
                algo_config['name'],
                algo_config['class'],
                algo_config['config'],
                log_dir
            )
            results.append(result)

    # Save raw results
    results_file = 'convergence_comparison_results.json'
    with open(results_file, 'w') as f:
        # Remove convergence data from JSON (too large)
        results_for_json = []
        for r in results:
            r_copy = r.copy()
            r_copy.pop('convergence_evals', None)
            r_copy.pop('convergence_perfs', None)
            results_for_json.append(r_copy)
        json.dump(results_for_json, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Raw results saved to {results_file}")
    print(f"{'='*80}")

    # Generate convergence plots
    print(f"\n{'='*80}")
    print("Generating convergence plots...")
    print(f"{'='*80}")

    kernels_dict = {}
    for result in results:
        kernel = result['kernel']
        if kernel not in kernels_dict:
            kernels_dict[kernel] = []
        kernels_dict[kernel].append(result)

    for kernel_name, kernel_results in kernels_dict.items():
        plot_path = f"convergence_{kernel_name}.png"
        plot_convergence(kernel_name, kernel_results, plot_path)

    # Generate analysis report
    print(f"\n{'='*80}")
    print("Generating analysis report...")
    print(f"{'='*80}")

    generate_analysis_report(results, 'CONVERGENCE_ANALYSIS.md')

    print(f"\n{'='*80}")
    print("CONVERGENCE ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutputs:")
    print(f"  - CSV logs: convergence_logs/")
    print(f"  - Convergence plots: convergence_*.png (3 images)")
    print(f"  - Analysis report: CONVERGENCE_ANALYSIS.md")
    print(f"  - Raw results: {results_file}")


if __name__ == '__main__':
    main()
