#!/usr/bin/env python3
"""
Final Three-Kernel Championship Benchmark

Tests all autotuner algorithms across 3 different kernel types:
1. Matrix Multiplication (compute-bound)
2. Element-wise GELU (bandwidth-bound)
3. Element-wise ReLU + Add (memory-bound mixed)

Each algorithm gets exactly 1600 evaluations per kernel.
Fixed: TPE hashable issue, eval counting, better 3rd kernel.
"""

import json
import time
from typing import Any
import torch
import helion
import helion.language as lang
from helion.autotuner import (
    DifferentialEvolutionSearch,
    GeneticAlgorithmSearch,
    ParticleSwarmSearch,
    TreeStructuredParzenEstimator,
    DESurrogateHybrid,
    MultiFidelityRandomForestSearch,
)


# ============================================================================
# Kernel 1: Matrix Multiplication (Compute-Bound)
# ============================================================================

@helion.kernel(
    static_shapes=True,
    autotune_config_overrides={
        "range_unroll_factors": [0, 0],
        "range_num_stages": [0, 0],
    },
    autotune_compile_timeout=20,
)
def matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication - compute-bound."""
    m, k = x.size()
    k2, n = y.size()
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)

    for tile_m, tile_n in lang.tile([m, n]):
        acc = lang.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in lang.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc

    return out


# ============================================================================
# Kernel 2: Element-wise GELU (Bandwidth-Bound)
# ============================================================================

@helion.kernel(
    static_shapes=True,
    autotune_config_overrides={
        "range_unroll_factors": [0],
        "range_num_stages": [0],
    },
    autotune_compile_timeout=20,
)
def gelu_kernel(x: torch.Tensor) -> torch.Tensor:
    """GELU activation - bandwidth-bound."""
    n, = x.size()
    out = torch.empty([n], dtype=x.dtype, device=x.device)

    for tile_n in lang.tile(n):
        # GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x_tile = x[tile_n]
        x_cubed = x_tile * x_tile * x_tile
        inner = 0.7978845608 * (x_tile + 0.044715 * x_cubed)  # sqrt(2/pi) ≈ 0.797
        out[tile_n] = x_tile * 0.5 * (1.0 + torch.tanh(inner))

    return out


# ============================================================================
# Kernel 3: Fused ReLU + Add (Memory-Bound Mixed)
# ============================================================================

@helion.kernel(
    static_shapes=True,
    autotune_config_overrides={
        "range_unroll_factors": [0],
        "range_num_stages": [0],
    },
    autotune_compile_timeout=20,
)
def fused_relu_add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fused ReLU + Add: out = ReLU(x) + y - memory-bound mixed."""
    n, = x.size()
    out = torch.empty([n], dtype=x.dtype, device=x.device)

    for tile_n in lang.tile(n):
        # Fused: ReLU(x) + y
        x_tile = x[tile_n]
        y_tile = y[tile_n]
        relu_x = torch.where(x_tile > 0, x_tile, torch.zeros_like(x_tile))
        out[tile_n] = relu_x + y_tile

    return out


# ============================================================================
# Algorithm Configurations
# ============================================================================

ALGORITHMS = {
    "DifferentialEvolution": (
        DifferentialEvolutionSearch,
        {"population_size": 40, "max_generations": 40},  # 80 + 38*40 = 1600
    ),
    "GeneticAlgorithm": (
        GeneticAlgorithmSearch,
        {
            "population_size": 40,
            "max_generations": 38,
            "elite_size": 0,
            "local_search_freq": 1000,
        },  # 80 + 38*40 = 1600
    ),
    "PSO": (
        ParticleSwarmSearch,
        {"swarm_size": 40, "max_iterations": 39},  # 40 + 39*40 = 1600
    ),
    "TPE": (
        TreeStructuredParzenEstimator,
        {
            "n_initial": 100,
            "n_iterations": 1500,
            "gamma": 0.25,
            "n_ei_candidates": 24,
        },  # 100 + 1500 = 1600
    ),
    "DE-Surrogate": (
        DESurrogateHybrid,
        {
            "population_size": 40,
            "max_generations": 40,
            "crossover_rate": 0.8,
            "surrogate_threshold": 100,
            "candidate_ratio": 3,
        },  # 80 + 38*40 = 1600
    ),
    "RandomForestMFBO": (
        MultiFidelityRandomForestSearch,
        {
            "n_low_fidelity": 1200,
            "n_medium_fidelity": 300,
            "n_high_fidelity": 80,
            "n_ultra_fidelity": 20,
            "fidelity_low": 10,
            "fidelity_medium": 15,
            "fidelity_high": 30,
            "fidelity_ultra": 50,
        },  # 1200 + 300 + 80 + 20 = 1600
    ),
}


# ============================================================================
# Kernel Test Cases
# ============================================================================

KERNELS = {
    "MatMul-1024": {
        "kernel": matmul_kernel,
        "args_fn": lambda: (
            torch.randn([1024, 1024], device="cuda", dtype=torch.float16),
            torch.randn([1024, 1024], device="cuda", dtype=torch.float16),
        ),
        "description": "Matrix Multiplication 1024×1024 (Compute-Bound)",
    },
    "GELU-1M": {
        "kernel": gelu_kernel,
        "args_fn": lambda: (
            torch.randn([1_000_000], device="cuda", dtype=torch.float32),
        ),
        "description": "GELU Activation 1M elements (Bandwidth-Bound)",
    },
    "FusedReLUAdd-1M": {
        "kernel": fused_relu_add_kernel,
        "args_fn": lambda: (
            torch.randn([1_000_000], device="cuda", dtype=torch.float32),
            torch.randn([1_000_000], device="cuda", dtype=torch.float32),
        ),
        "description": "Fused ReLU+Add 1M elements (Memory-Bound Mixed)",
    },
}


def count_evaluations(search) -> int:
    """
    Count total evaluations from the search object.
    Fixed to properly count all evaluations, not just population size.
    """
    # For TPE: count observations
    if hasattr(search, 'observations') and search.observations:
        return len(search.observations)

    # For DE-Surrogate: count all_observations
    if hasattr(search, 'all_observations') and search.all_observations:
        return len(search.all_observations)

    # For MFBO: sum across all fidelity levels
    if hasattr(search, 'evaluated_ultra'):
        return (
            len(search.evaluated_low) +
            len(search.evaluated_medium) +
            len(search.evaluated_high) +
            len(search.evaluated_ultra)
        )

    # For population-based (DE, GA, PSO): count from counters
    if hasattr(search, 'counters') and 'benchmark' in search.counters:
        return search.counters['benchmark']

    # Fallback: estimate from population and generations
    if hasattr(search, 'population_size') and hasattr(search, 'max_generations'):
        # For DE/GA: pop*2 initial + (gen-2)*pop
        if hasattr(search, 'max_generations'):
            return search.population_size * 2 + (search.max_generations - 2) * search.population_size

    # For PSO: swarm + iter*swarm
    if hasattr(search, 'swarm_size') and hasattr(search, 'max_iterations'):
        return search.swarm_size + search.max_iterations * search.swarm_size

    return 0


def run_benchmark_for_kernel(kernel_name: str, kernel_info: dict) -> dict:
    """Run all algorithms on a single kernel."""
    print("=" * 80)
    print(f"KERNEL: {kernel_name}")
    print(f"Description: {kernel_info['description']}")
    print("=" * 80)
    print()

    kernel = kernel_info["kernel"]
    args = kernel_info["args_fn"]()

    results = []

    for algo_name, (algo_class, kwargs) in ALGORITHMS.items():
        print(f"Running {algo_name}...")

        try:
            bound_kernel = kernel.bind(args)

            start_time = time.time()
            search = algo_class(bound_kernel, args, **kwargs)
            best_config = search.autotune()
            elapsed = time.time() - start_time

            # Get best performance and evaluation count
            eval_count = count_evaluations(search)

            if hasattr(search, 'population') and search.population:
                valid_members = [m for m in search.population if m.perf != float("inf")]
                best_perf = min(m.perf for m in valid_members) if valid_members else float("inf")
            elif hasattr(search, 'observations') and search.observations:
                valid_members = [m for m in search.observations if m.perf != float("inf")]
                best_perf = min(m.perf for m in valid_members) if valid_members else float("inf")
            elif hasattr(search, 'evaluated_ultra') and search.evaluated_ultra:
                valid_members = [m for m in search.evaluated_ultra if m.perf != float("inf")]
                best_perf = min(m.perf for m in valid_members) if valid_members else float("inf")
            else:
                best_perf = float("inf")

            result = {
                "algorithm": algo_name,
                "time": elapsed,
                "performance": best_perf,
                "evaluations": eval_count,
            }

            print(f"  ✓ Time: {elapsed:.1f}s, Best: {best_perf:.4f}ms, Evals: {eval_count}")
            results.append(result)

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            result = {
                "algorithm": algo_name,
                "time": 0,
                "performance": float("inf"),
                "evaluations": 0,
                "error": str(e)[:200],  # Truncate error message
            }
            results.append(result)

        print()

    return {"kernel": kernel_name, "results": results}


def main():
    """Run final three-kernel championship."""
    print("=" * 80)
    print("FINAL THREE-KERNEL CHAMPIONSHIP BENCHMARK")
    print("=" * 80)
    print(f"Kernels: {len(KERNELS)}")
    print(f"Algorithms: {len(ALGORITHMS)}")
    print(f"Budget: ~1600 evaluations per algorithm per kernel")
    print(f"Fixes: TPE hashable, eval counting, better 3rd kernel")
    print("=" * 80)
    print()

    all_results = []

    for kernel_name, kernel_info in KERNELS.items():
        kernel_results = run_benchmark_for_kernel(kernel_name, kernel_info)
        all_results.append(kernel_results)

    # Save results
    with open("final_three_kernel_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("FINAL THREE-KERNEL CHAMPIONSHIP SUMMARY")
    print("=" * 80)
    print()

    # Create summary table
    print(f"{'Kernel':<25} {'Algorithm':<25} {'Time (s)':<12} {'Best (ms)':<12} {'Evals':<10} {'vs DE':<10}")
    print("-" * 105)

    for kernel_result in all_results:
        kernel_name = kernel_result["kernel"]
        results = kernel_result["results"]

        # Find DE baseline
        de_result = next((r for r in results if r["algorithm"] == "DifferentialEvolution"), None)
        de_perf = de_result["performance"] if de_result and de_result["performance"] != float("inf") else None

        # Sort by performance
        valid_results = [r for r in results if r["performance"] != float("inf")]
        sorted_results = sorted(valid_results, key=lambda x: x["performance"])

        for r in sorted_results:
            perf_vs_de = ""
            if de_perf and de_perf > 0 and r["performance"] != float("inf"):
                diff = ((r["performance"] / de_perf - 1) * 100)
                perf_vs_de = f"{diff:+.1f}%"

            print(
                f"{kernel_name:<25} {r['algorithm']:<25} {r['time']:<12.1f} "
                f"{r['performance']:<12.4f} {r['evaluations']:<10} {perf_vs_de:<10}"
            )

        # Show failed algorithms
        failed = [r for r in results if r["performance"] == float("inf")]
        for r in failed:
            error_msg = r.get("error", "Unknown")[:40]
            print(f"{kernel_name:<25} {r['algorithm']:<25} {'FAILED':<12} {'-':<12} {0:<10} {error_msg}")

        print()

    print("Results saved to final_three_kernel_results.json")


if __name__ == "__main__":
    main()
