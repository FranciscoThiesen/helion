#!/usr/bin/env python3
"""
Convergence Analysis for DE vs DE-Surrogate.

Parses benchmark logs to extract generation-by-generation convergence data
and generates convergence curves showing how algorithms improve over time.

Author: Francisco Geiman Thiesen
Date: 2025-11-06
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")


def parse_convergence_from_log(log_file: str) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    """
    Parse convergence data from benchmark log file.

    Returns:
        Dict mapping kernel_name -> algorithm_name -> [(eval_count, best_perf), ...]
    """
    convergence_data = {}
    current_kernel = None
    current_algorithm = None
    eval_count = 0
    population_size = 40

    with open(log_file, 'r') as f:
        for line in f:
            # Detect kernel
            kernel_match = re.search(r'^KERNEL: (.+)$', line)
            if kernel_match:
                current_kernel = kernel_match.group(1).strip()
                if current_kernel not in convergence_data:
                    convergence_data[current_kernel] = {}
                continue

            # Detect algorithm
            algo_match = re.search(r'Running ([\w-]+)\.\.\.', line)
            if algo_match:
                current_algorithm = algo_match.group(1)
                if current_kernel and current_algorithm:
                    convergence_data[current_kernel][current_algorithm] = []
                    eval_count = 0
                continue

            # Parse generation data for DifferentialEvolution
            if current_algorithm == 'DifferentialEvolution':
                # Initial population: "Initial population: error=2 ok=78 min=0.0182"
                init_match = re.search(r'Initial population:.*min=([\d.]+)', line)
                if init_match:
                    perf = float(init_match.group(1))
                    eval_count = population_size * 2  # Initial pop is 2x
                    convergence_data[current_kernel][current_algorithm].append((eval_count, perf))
                    continue

                # Generation complete: "Generation 5 complete: replaced=20 ok=40 min=0.0152"
                gen_match = re.search(r'Generation (\d+) complete:.*min=([\d.]+)', line)
                if gen_match:
                    gen_num = int(gen_match.group(1))
                    perf = float(gen_match.group(2))
                    eval_count = population_size * 2 + (gen_num - 1) * population_size
                    convergence_data[current_kernel][current_algorithm].append((eval_count, perf))
                    continue

            # Parse generation data for DE-Surrogate
            if current_algorithm == 'DE-Surrogate':
                # "Gen 5: SURROGATE | best=0.0114 ms | replaced=13/40 | total_evals=280"
                # or "Gen 2: STANDARD | best=0.0178 ms | replaced=10/40 | total_evals=116"
                gen_match = re.search(r'Gen (\d+):\s*(?:SURROGATE|STANDARD)\s*\|.*best=([\d.]+)\s*ms.*total_evals=(\d+)', line)
                if gen_match:
                    perf = float(gen_match.group(2))
                    evals = int(gen_match.group(3))
                    convergence_data[current_kernel][current_algorithm].append((evals, perf))
                    continue

    return convergence_data


def smooth_convergence(data: List[Tuple[int, float]], window: int = 3) -> List[Tuple[int, float]]:
    """Apply moving average smoothing to convergence data."""
    if len(data) <= window:
        return data

    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2 + 1)
        window_data = [perf for _, perf in data[start:end]]
        avg_perf = sum(window_data) / len(window_data)
        smoothed.append((data[i][0], avg_perf))

    return smoothed


def plot_convergence_per_kernel(convergence_data: Dict[str, Dict[str, List[Tuple[int, float]]]],
                                 output_dir: Path):
    """Generate convergence plots for each kernel."""
    if not HAS_MATPLOTLIB:
        return

    colors = {
        'DifferentialEvolution': '#1f77b4',  # Blue
        'DE-Surrogate': '#ff7f0e',           # Orange
        'PatternSearch': '#2ca02c',          # Green
    }

    for kernel_name, algorithms in convergence_data.items():
        plt.figure(figsize=(10, 6))

        for algo_name, data in algorithms.items():
            if not data:
                continue

            evals, perfs = zip(*data)

            # Plot raw data with transparency
            plt.plot(evals, perfs,
                    color=colors.get(algo_name, 'gray'),
                    alpha=0.3,
                    linewidth=1)

            # Plot smoothed data
            smoothed = smooth_convergence(data, window=5)
            evals_smooth, perfs_smooth = zip(*smoothed)
            final_perf = perfs_smooth[-1]

            plt.plot(evals_smooth, perfs_smooth,
                    label=f'{algo_name} (final: {final_perf:.4f}ms)',
                    color=colors.get(algo_name, 'gray'),
                    linewidth=2.5,
                    marker='o',
                    markersize=4,
                    markevery=max(1, len(evals_smooth) // 10))

        plt.xlabel('Number of Evaluations', fontsize=12, fontweight='bold')
        plt.ylabel('Best Performance (ms, lower is better)', fontsize=12, fontweight='bold')
        plt.title(f'Convergence Analysis: {kernel_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()

        # Save plot
        safe_name = kernel_name.replace(' ', '_').replace('-', '_')
        output_file = output_dir / f'convergence_{safe_name}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved plot: {output_file}")
        plt.close()


def plot_combined_convergence(convergence_data: Dict[str, Dict[str, List[Tuple[int, float]]]],
                               output_dir: Path):
    """Generate combined convergence plot with all kernels."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {
        'DifferentialEvolution': '#1f77b4',
        'DE-Surrogate': '#ff7f0e',
        'PatternSearch': '#2ca02c',
    }

    kernel_names = list(convergence_data.keys())

    for idx, (kernel_name, algorithms) in enumerate(convergence_data.items()):
        ax = axes[idx]

        for algo_name, data in algorithms.items():
            if not data:
                continue

            smoothed = smooth_convergence(data, window=5)
            evals, perfs = zip(*smoothed)
            final_perf = perfs[-1]

            ax.plot(evals, perfs,
                   label=f'{algo_name}\n(final: {final_perf:.4f}ms)',
                   color=colors.get(algo_name, 'gray'),
                   linewidth=2.5,
                   marker='o',
                   markersize=3,
                   markevery=max(1, len(evals) // 8))

        ax.set_xlabel('Evaluations', fontsize=11, fontweight='bold')
        ax.set_ylabel('Best Perf (ms)', fontsize=11, fontweight='bold')
        ax.set_title(kernel_name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_file = output_dir / 'convergence_combined.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved combined plot: {output_file}")
    plt.close()


def calculate_convergence_metrics(convergence_data: Dict[str, Dict[str, List[Tuple[int, float]]]]):
    """Calculate convergence metrics for analysis."""
    metrics = {}

    for kernel_name, algorithms in convergence_data.items():
        metrics[kernel_name] = {}

        for algo_name, data in algorithms.items():
            if not data:
                continue

            evals, perfs = zip(*data)

            # Find when algorithm reached 90%, 95%, 99% of final performance
            final_perf = perfs[-1]
            thresholds = {'90%': 0.9, '95%': 0.95, '99%': 0.99}
            convergence_points = {}

            for name, ratio in thresholds.items():
                target = final_perf / ratio
                for i, perf in enumerate(perfs):
                    if perf <= target:
                        convergence_points[name] = (evals[i], perf)
                        break

            # Calculate improvement rate (initial to final)
            initial_perf = perfs[0]
            improvement = (initial_perf - final_perf) / initial_perf * 100

            metrics[kernel_name][algo_name] = {
                'initial_perf': initial_perf,
                'final_perf': final_perf,
                'improvement_pct': improvement,
                'total_evals': evals[-1],
                'convergence_points': convergence_points
            }

    return metrics


def print_convergence_report(convergence_data: Dict[str, Dict[str, List[Tuple[int, float]]]],
                              metrics: Dict):
    """Print detailed convergence analysis report."""
    print("\n" + "="*90)
    print("CONVERGENCE ANALYSIS REPORT")
    print("="*90)

    for kernel_name, algorithms in convergence_data.items():
        print(f"\n{'='*90}")
        print(f"KERNEL: {kernel_name}")
        print(f"{'='*90}")

        for algo_name, data in algorithms.items():
            if not data or algo_name not in metrics[kernel_name]:
                continue

            m = metrics[kernel_name][algo_name]

            print(f"\n{algo_name}:")
            print(f"  Initial Performance:  {m['initial_perf']:.4f} ms")
            print(f"  Final Performance:    {m['final_perf']:.4f} ms")
            print(f"  Improvement:          {m['improvement_pct']:.1f}%")
            print(f"  Total Evaluations:    {m['total_evals']}")

            if m['convergence_points']:
                print(f"  Convergence Points:")
                for name, (evals, perf) in sorted(m['convergence_points'].items()):
                    print(f"    {name} of final: {evals} evals ({perf:.4f} ms)")

        # Compare algorithms
        if 'DifferentialEvolution' in algorithms and 'DE-Surrogate' in algorithms:
            de_final = metrics[kernel_name]['DifferentialEvolution']['final_perf']
            des_final = metrics[kernel_name]['DE-Surrogate']['final_perf']
            improvement = (de_final - des_final) / de_final * 100

            print(f"\n  Comparison:")
            print(f"    DE-Surrogate vs DE: {improvement:+.2f}%")

            # Check if DE-Surrogate converged faster
            de_90 = metrics[kernel_name]['DifferentialEvolution']['convergence_points'].get('90%')
            des_90 = metrics[kernel_name]['DE-Surrogate']['convergence_points'].get('90%')

            if de_90 and des_90:
                de_evals = de_90[0]
                des_evals = des_90[0]
                if des_evals < de_evals:
                    speedup = de_evals / des_evals
                    print(f"    DE-Surrogate reached 90% {speedup:.2f}× faster ({des_evals} vs {de_evals} evals)")


def main():
    print("="*90)
    print("CONVERGENCE ANALYSIS: DE vs DE-Surrogate")
    print("="*90)
    print()

    # Parse log file
    log_file = 'final_three_kernel_results.txt'

    if not Path(log_file).exists():
        print(f"Error: Log file '{log_file}' not found")
        print("Please run the benchmark first to generate logs.")
        return

    print(f"Parsing convergence data from {log_file}...")
    convergence_data = parse_convergence_from_log(log_file)

    # Check if we got data
    total_series = sum(len(algos) for algos in convergence_data.values())
    print(f"  Found {len(convergence_data)} kernels with {total_series} convergence series")

    if total_series == 0:
        print("  Warning: No convergence data found in log file")
        return

    # Calculate metrics
    print("\nCalculating convergence metrics...")
    metrics = calculate_convergence_metrics(convergence_data)

    # Generate plots
    output_dir = Path('.')
    if HAS_MATPLOTLIB:
        print("\nGenerating convergence plots...")
        plot_convergence_per_kernel(convergence_data, output_dir)
        plot_combined_convergence(convergence_data, output_dir)

    # Print report
    print_convergence_report(convergence_data, metrics)

    # Save metrics to JSON
    output_file = 'convergence_metrics.json'
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Convergence metrics saved to {output_file}")

    print("\n" + "="*90)
    print("CONVERGENCE ANALYSIS COMPLETE")
    print("="*90)


if __name__ == '__main__':
    main()
