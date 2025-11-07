#!/usr/bin/env python3
"""Benchmark MatMul-1024 with early stopping for all 3 algorithms."""

import time
import csv
from pathlib import Path

import helion
import helion.language as lang
import torch
from helion.autotuner.differential_evolution import DifferentialEvolutionSearch
from helion.autotuner.de_surrogate_hybrid import DESurrogateHybrid
from helion.autotuner.pattern_search import PatternSearch


@helion.kernel(static_shapes=True)
def matmul_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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


# Kernel configuration
kernel_name = 'MatMul-1024'
kernel_fn = matmul_kernel
args = (
    torch.randn(1024, 1024, dtype=torch.float16, device='cuda'),
    torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
)

# Algorithm configurations with early stopping
algorithms = {
    'DifferentialEvolution': {
        'class': DifferentialEvolutionSearch,
        'config': {
            'population_size': 40,
            'max_generations': 40,
            'min_improvement_delta': 0.001,  # 0.1% improvement threshold
            'patience': 3  # Stop after 3 generations without improvement
        }
    },
    'DE-Surrogate': {
        'class': DESurrogateHybrid,
        'config': {
            'population_size': 40,
            'max_generations': 40,
            'min_improvement_delta': 0.001,
            'patience': 3
        }
    },
    'PatternSearch': {
        'class': PatternSearch,
        'config': {
            'initial_population': 80,
            'copies': 10,
            'max_generations': 40,
            'min_improvement_delta': 0.001  # PatternSearch already has this parameter
        }
    }
}

log_dir = Path('convergence_logs')
log_dir.mkdir(exist_ok=True)

results = {}

print("=" * 80)
print(f"MatMul-1024 Benchmark with Early Stopping (delta=0.001, patience=3)")
print("=" * 80)
print()

for algo_name, algo_info in algorithms.items():
    print(f"\n{'=' * 80}")
    print(f"Running {algo_name} on {kernel_name}")
    print(f"{'=' * 80}")

    log_base = log_dir / f"{kernel_name}_{algo_name}_early_stopping"

    start_time = time.time()

    try:
        # Create bound kernel
        bound = kernel_fn.bind(args)

        # Enable CSV logging (PR #1095)
        bound.settings.autotune_log = str(log_base)

        # Run algorithm with early stopping
        search = algo_info['class'](bound, args, **algo_info['config'])
        best_config = search.autotune()
        elapsed = time.time() - start_time

        # Read best performance from CSV
        csv_path = f"{log_base}.csv"
        if Path(csv_path).exists():
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                records = list(reader)

            best_perf = float('inf')
            for record in records:
                if record.get('status') == 'ok':
                    perf = float(record['perf_ms'])
                    if perf < best_perf:
                        best_perf = perf

            total_evals = len(records)
        else:
            best_perf = float('inf')
            total_evals = 0

        results[algo_name] = {
            'time': elapsed,
            'best_perf': best_perf,
            'total_evals': total_evals
        }

        print(f"  ✓ Time: {elapsed:.1f}s, Best: {best_perf:.4f}ms, Evals: {total_evals}")

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ✗ Failed: {str(e)[:100]}")
        import traceback
        traceback.print_exc()

# Print summary table
print("\n" + "=" * 80)
print("BENCHMARK RESULTS (with early stopping)")
print("=" * 80)
print(f"\n{'Algorithm':<25} {'Time (s)':<12} {'Best (ms)':<12} {'Evaluations':<12}")
print("-" * 80)

for algo_name, result in results.items():
    print(f"{algo_name:<25} {result['time']:<12.1f} {result['best_perf']:<12.4f} {result['total_evals']:<12}")

print("\n" + "=" * 80)
print("Benchmark complete!")
print("=" * 80)
