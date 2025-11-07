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

def plot_convergence(kernel_name):
    """Generate convergence plot for a kernel."""
    plt.figure(figsize=(10, 6))

    colors = {
        'DifferentialEvolution': '#1f77b4',
        'DE-Surrogate': '#ff7f0e',
        'PatternSearch': '#2ca02c'
    }

    log_dir = Path('convergence_logs')
    max_eval_limit = 2000  # Limit X-axis for fair comparison

    for algo in ['DifferentialEvolution', 'DE-Surrogate', 'PatternSearch']:
        csv_path = log_dir / f"{kernel_name}_{algo}.csv"

        if not csv_path.exists():
            print(f"  Warning: {csv_path} not found")
            continue

        evals, perfs = parse_csv_convergence(csv_path)

        if evals and perfs:
            # Limit to first max_eval_limit evaluations for fair comparison
            if len(evals) > max_eval_limit:
                evals = evals[:max_eval_limit]
                perfs = perfs[:max_eval_limit]

            plt.plot(evals, perfs, label=algo, color=colors.get(algo, 'gray'), linewidth=2)
            print(f"  Added {algo}: {len(evals)} points (displayed), best={min(perfs):.4f}ms")

    plt.xlabel('Number of Evaluations', fontsize=12)
    plt.ylabel('Best Performance (ms)', fontsize=12)
    plt.title(f'Convergence Comparison: {kernel_name}', fontsize=14, fontweight='bold')
    plt.xlim(0, max_eval_limit)
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
