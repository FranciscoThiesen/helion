#!/usr/bin/env python3
"""Generate a single combined plot with all 3 kernels."""

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
    """Find where performance stopped improving significantly."""
    if len(perfs) < patience:
        return len(perfs)

    for i in range(len(perfs) - patience):
        current_best = perfs[i]
        future_best = min(perfs[i:i+patience])

        if current_best == 0:
            improvement = 0
        else:
            improvement = (current_best - future_best) / current_best

        if improvement < threshold:
            return i + int(patience * 0.5)

    return len(perfs)


def plot_kernel(ax, kernel_name, kernel_title):
    """Plot convergence for a single kernel on given axes."""
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
            continue

        evals, perfs = parse_csv_convergence(csv_path)

        if evals and perfs:
            conv_idx = find_convergence_point(perfs)
            convergence_points.append(conv_idx)
            ax.plot(evals, perfs, label=algo, color=colors.get(algo, 'gray'), linewidth=2)

    # Set X-axis to where the last algorithm converged
    if convergence_points:
        max_convergence = max(convergence_points)
        ax.set_xlim(0, max_convergence * 1.1)

    ax.set_xlabel('Number of Evaluations', fontsize=10)
    ax.set_ylabel('Best Performance (ms)', fontsize=10)
    ax.set_title(kernel_title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)


# Create combined plot
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

plot_kernel(axes[0], 'MatMul-1024', 'MatMul-1024 (Compute-Bound)')
plot_kernel(axes[1], 'GELU-1M', 'GELU-1M (Bandwidth-Bound)')
plot_kernel(axes[2], 'FusedReLUAdd-1M', 'FusedReLUAdd-1M (Memory-Bound)')

plt.suptitle('Convergence Analysis: DE vs DE-Surrogate vs PatternSearch',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

output_path = 'convergence_combined.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved combined plot: {output_path}")
plt.close()
