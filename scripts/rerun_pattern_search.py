#!/usr/bin/env python3
"""Re-run PatternSearch with fair evaluation budget (~1600 configs)."""

import time
import csv
from pathlib import Path

import helion
import helion.language as lang
import torch
from helion.autotuner.pattern_search import PatternSearch


# Test kernels
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


@helion.kernel(static_shapes=True)
def gelu_kernel(x: torch.Tensor) -> torch.Tensor:
    n, = x.size()
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile_n in lang.tile(n):
        x_tile = x[tile_n]
        sqrt_2_over_pi = 0.7978845608
        x_cubed = x_tile * x_tile * x_tile
        inner = sqrt_2_over_pi * (x_tile + 0.044715 * x_cubed)
        tanh_inner = torch.tanh(inner)
        out[tile_n] = 0.5 * x_tile * (1.0 + tanh_inner)
    return out


@helion.kernel(static_shapes=True)
def fused_relu_add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n, = x.size()
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile_n in lang.tile(n):
        x_tile = x[tile_n]
        y_tile = y[tile_n]
        relu_x = torch.where(x_tile > 0, x_tile, torch.zeros_like(x_tile))
        out[tile_n] = relu_x + y_tile
    return out


kernels = {
    'MatMul-1024': {
        'kernel': matmul_kernel,
        'args': (
            torch.randn(1024, 1024, dtype=torch.float16, device='cuda'),
            torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
        )
    },
    'GELU-1M': {
        'kernel': gelu_kernel,
        'args': (torch.randn(1_000_000, dtype=torch.float16, device='cuda'),)
    },
    'FusedReLUAdd-1M': {
        'kernel': fused_relu_add_kernel,
        'args': (
            torch.randn(1_000_000, dtype=torch.float16, device='cuda'),
            torch.randn(1_000_000, dtype=torch.float16, device='cuda')
        )
    }
}

log_dir = Path('convergence_logs')

# PatternSearch config with restricted generations for fair comparison
# initial_population=80 evaluates 80*2=160 configs initially
# Each generation evaluates ~300-400 neighbors
# To get ~1600 total: 160 initial + 4 generations * ~360 = ~1600
pattern_config = {
    'initial_population': 80,
    'copies': 10,
    'max_generations': 5  # Reduced from 40 to limit to ~1600 evals
}

print("="*80)
print("Re-running PatternSearch with fair evaluation budget (~1600 configs)")
print("="*80)
print()

for kernel_name, kernel_info in kernels.items():
    print(f"\n{'='*80}")
    print(f"Running PatternSearch on {kernel_name}")
    print(f"{'='*80}")

    log_base = log_dir / f"{kernel_name}_PatternSearch"

    start_time = time.time()

    try:
        # Create bound kernel
        bound = kernel_info['kernel'].bind(kernel_info['args'])

        # Enable CSV logging
        bound.settings.autotune_log = str(log_base)

        # Run PatternSearch with limited generations
        search = PatternSearch(bound, kernel_info['args'], **pattern_config)
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

        print(f"  ✓ Time: {elapsed:.1f}s, Best: {best_perf:.4f}ms, Evals: {total_evals}")

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ✗ Failed: {str(e)[:100]}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("PatternSearch re-run complete!")
print("="*80)
