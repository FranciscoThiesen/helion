#!/usr/bin/env python3
"""Regenerate convergence plots from CSV files."""

import csv
from pathlib import Path
import matplotlib.pyplot as plt

def parse_csv_convergence(csv_path):
    """Parse CSV and extract convergence data."""
    evals = []
    perfs = []
    best_so_far = float('inf')

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, record in enumerate(reader, 1):
            if record.get('status') == 'ok':
                perf = float(record['perf_ms'])
                if perf < best_so_far:
                    best_so_far = perf
            evals.append(i)
            perfs.append(best_so_far)

    return evals, perfs


def find_convergence_point(perfs, patience=100, threshold=0.001):
    """Find where performance stopped improving significantly.

    Args:
        perfs: List of best performance values over time
        patience: Number of evaluations without improvement to consider converged
        threshold: Relative improvement threshold (0.001 = 0.1%)

    Returns:
        Index where convergence occurred
    """
    if len(perfs) < patience:
        return len(perfs)

    for i in range(len(perfs) - patience):
        current_best = perfs[i]
        future_best = min(perfs[i:i+patience])

        # Check if improvement is less than threshold
        if current_best == 0:
            improvement = 0
        else:
            improvement = (current_best - future_best) / current_best

        if improvement < threshold:
            return i + int(patience * 0.5)  # Return midpoint for better visualization

    return len(perfs)

def plot_convergence(kernel_name):
    """Generate convergence plot for a kernel."""
    plt.figure(figsize=(10, 6))

    colors = {
        'DifferentialEvolution': '#1f77b4',
        'DE-Surrogate': '#ff7f0e',
        'PatternSearch': '#2ca02c'
    }

    log_dir = Path('convergence_logs')
    convergence_points = []

    for algo in ['DifferentialEvolution', 'DE-Surrogate', 'PatternSearch']:
        csv_path = log_dir / f"{kernel_name}_{algo}.csv"

        if not csv_path.exists():
            print(f"  Warning: {csv_path} not found")
            continue

        evals, perfs = parse_csv_convergence(csv_path)

        if evals and perfs:
            # Find where this algorithm converged
            conv_idx = find_convergence_point(perfs)
            convergence_points.append(conv_idx)

            plt.plot(evals, perfs, label=algo, color=colors.get(algo, 'gray'), linewidth=2)
            print(f"  Added {algo}: {len(evals)} evals, best={min(perfs):.4f}ms, converged at ~{conv_idx}")

    # Set X-axis to where the last algorithm converged (with 10% margin)
    if convergence_points:
        max_convergence = max(convergence_points)
        plt.xlim(0, max_convergence * 1.1)
        print(f"  X-axis limited to {int(max_convergence * 1.1)} (last convergence at {max_convergence})")

    plt.xlabel('Number of Evaluations', fontsize=12)
    plt.ylabel('Best Performance (ms)', fontsize=12)
    plt.title(f'Convergence Comparison: {kernel_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = f"convergence_{kernel_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}\n")
    plt.close()

if __name__ == '__main__':
    kernels = ['MatMul-1024', 'GELU-1M', 'FusedReLUAdd-1M']

    for kernel in kernels:
        print(f"Generating plot for {kernel}:")
        plot_convergence(kernel)

    print("All plots regenerated!")
