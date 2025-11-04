#!/usr/bin/env python3
"""
Multi-Kernel Autotuner Benchmark

Comprehensive benchmark testing all autotuner algorithms across diverse GPU kernels:
1. Matrix Multiplication - compute-bound, regular memory access
2. Vector Reduction - memory-bound, irregular access patterns
3. Softmax - mixed pattern with reductions and element-wise operations

This provides a more robust evaluation than single-kernel testing.

Author: Francisco Geiman Thiesen
Date: 2025-11-04
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import helion
import helion.language as lang
from helion.autotuner import (
    CMAESSearch,
    DifferentialEvolutionSearch,
    GeneticAlgorithmSearch,
    MultiFidelityBayesianSearch,
    MultiFidelityRandomForestSearch,
    ParticleSwarmSearch,
    PatternSearch,
    RandomSearch,
)


# ============================================================================
# Kernel Definitions
# ============================================================================


@helion.kernel(
    static_shapes=True,
    autotune_config_overrides={
        "range_unroll_factors": [0, 0],
        "range_num_stages": [0, 0],
    },
)
def matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication: compute-bound with regular memory access.

    Characteristics:
    - High arithmetic intensity
    - Regular, predictable memory patterns
    - Benefits from block tiling and warp configuration
    """
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"

    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )

    for tile_m, tile_n in lang.tile([m, n]):
        acc = lang.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in lang.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc

    return out


@helion.kernel(
    static_shapes=True,
    autotune_config_overrides={
        "range_unroll_factors": [0, 0],
        "range_num_stages": [0, 0],
    },
)
def reduction_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    Vector reduction (sum): memory-bound with irregular access.

    Characteristics:
    - Memory bandwidth limited
    - Requires efficient reduction strategies
    - Benefits from different warp configurations than matmul
    """
    n, = x.size()

    result = torch.zeros([1], dtype=x.dtype, device=x.device)

    for tile_n in lang.tile(n):
        partial = torch.sum(x[tile_n])
        result[0] += partial

    return result


@helion.kernel(
    static_shapes=True,
    autotune_config_overrides={
        "range_unroll_factors": [0, 0],
        "range_num_stages": [0, 0],
    },
)
def softmax_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    Softmax: mixed compute and memory pattern.

    Characteristics:
    - Combines reductions (max, sum) with element-wise operations
    - Requires multiple passes over data
    - Benefits from different tuning than pure compute or memory kernels
    """
    m, n = x.size()

    out = torch.empty_like(x)

    for tile_m in lang.tile(m):
        # Find max for numerical stability
        row_max = torch.max(x[tile_m, :])

        # Compute exp and sum
        exp_vals = torch.exp(x[tile_m, :] - row_max)
        exp_sum = torch.sum(exp_vals)

        # Normalize
        out[tile_m, :] = exp_vals / exp_sum

    return out


# ============================================================================
# Benchmark Configuration
# ============================================================================


@dataclass
class KernelConfig:
    """Configuration for a kernel benchmark."""

    name: str
    kernel: object
    args_fn: callable  # Function to generate args given device
    description: str


def get_kernel_configs():
    """
    Define all kernels to benchmark.

    Returns:
        List of KernelConfig objects
    """
    return [
        KernelConfig(
            name="matmul",
            kernel=matmul_kernel,
            args_fn=lambda device: (
                torch.randn([1024, 1024], device=device, dtype=torch.float16),
                torch.randn([1024, 1024], device=device, dtype=torch.float16),
            ),
            description="Matrix Multiplication (compute-bound)",
        ),
        KernelConfig(
            name="reduction",
            kernel=reduction_kernel,
            args_fn=lambda device: (
                torch.randn([1048576], device=device, dtype=torch.float32),
            ),
            description="Vector Reduction (memory-bound)",
        ),
        KernelConfig(
            name="softmax",
            kernel=softmax_kernel,
            args_fn=lambda device: (
                torch.randn([512, 512], device=device, dtype=torch.float32),
            ),
            description="Softmax (mixed pattern)",
        ),
    ]


# ============================================================================
# Algorithm Configurations
# ============================================================================


def get_algorithm_configs(time_budget: float) -> dict:
    """
    Generate algorithm configurations calibrated to time budget.

    Args:
        time_budget: Target time budget in seconds

    Returns:
        Dictionary mapping algorithm name to (class, kwargs)
    """
    # Empirical time per config (approximate)
    TIME_PER_CONFIG = {
        "PatternSearch": 0.15,
        "RandomSearch": 0.10,
        "DifferentialEvolution": 0.15,
        "GeneticAlgorithm": 0.12,
        "CMA-ES": 0.10,
        "PSO": 0.10,
        "MFBO": 0.08,
        "MFBO-RF": 0.08,
    }

    # Calculate config counts
    pattern_configs = int(time_budget / TIME_PER_CONFIG["PatternSearch"])
    random_configs = int(time_budget / TIME_PER_CONFIG["RandomSearch"])
    de_configs = int(time_budget / TIME_PER_CONFIG["DifferentialEvolution"])
    ga_configs = int(time_budget / TIME_PER_CONFIG["GeneticAlgorithm"])
    cmaes_configs = int(time_budget / TIME_PER_CONFIG["CMA-ES"])
    pso_configs = int(time_budget / TIME_PER_CONFIG["PSO"])
    mfbo_configs = int(time_budget / TIME_PER_CONFIG["MFBO"])

    # PatternSearch config
    pattern_initial_pop = min(100, pattern_configs // 10)
    pattern_copies = 5
    pattern_gens = max(2, pattern_configs // (pattern_initial_pop * pattern_copies))

    # DifferentialEvolution config
    de_pop = min(40, de_configs // 10)
    de_gens = max(2, de_configs // de_pop)

    # GeneticAlgorithm config
    ga_pop = min(50, ga_configs // 8)
    ga_gens = max(2, ga_configs // ga_pop)

    # CMA-ES config
    cmaes_gens = max(20, min(50, cmaes_configs // 10))

    # PSO config
    pso_swarm = min(30, pso_configs // 10)
    pso_iters = max(10, pso_configs // pso_swarm)

    # MFBO config (multi-fidelity budget split)
    mfbo_low = int(mfbo_configs * 0.4)
    mfbo_med = int(mfbo_configs * 0.3)
    mfbo_high = int(mfbo_configs * 0.2)
    mfbo_ultra = int(mfbo_configs * 0.1)

    return {
        "PatternSearch": (
            PatternSearch,
            {
                "initial_population_size": pattern_initial_pop,
                "copies_per_population": pattern_copies,
                "max_generations": pattern_gens,
            },
        ),
        "RandomSearch": (
            RandomSearch,
            {"count": random_configs},
        ),
        "DifferentialEvolution": (
            DifferentialEvolutionSearch,
            {
                "population_size": de_pop,
                "max_generations": de_gens,
            },
        ),
        "GeneticAlgorithm": (
            GeneticAlgorithmSearch,
            {
                "population_size": ga_pop,
                "max_generations": ga_gens,
            },
        ),
        "CMA-ES": (
            CMAESSearch,
            {
                "max_generations": cmaes_gens,
            },
        ),
        "PSO": (
            ParticleSwarmSearch,
            {
                "swarm_size": pso_swarm,
                "max_iterations": pso_iters,
            },
        ),
        "MFBO": (
            MultiFidelityBayesianSearch,
            {
                "n_low_fidelity": mfbo_low,
                "n_medium_fidelity": mfbo_med,
                "n_high_fidelity": mfbo_high,
                "n_ultra_fidelity": mfbo_ultra,
                "fidelity_low": 10,
                "fidelity_ultra": 50,
            },
        ),
        "MFBO-RF": (
            MultiFidelityRandomForestSearch,
            {
                "n_low_fidelity": mfbo_low,
                "n_medium_fidelity": mfbo_med,
                "n_high_fidelity": mfbo_high,
                "n_ultra_fidelity": mfbo_ultra,
                "fidelity_low": 10,
                "fidelity_ultra": 50,
            },
        ),
    }


# ============================================================================
# Benchmark Runner
# ============================================================================


def run_single_benchmark(
    kernel_config: KernelConfig,
    algorithm_name: str,
    algorithm_class: type,
    algorithm_kwargs: dict,
    device: str,
) -> dict:
    """
    Run a single autotuner on a single kernel.

    Returns:
        Dictionary with results: {
            'kernel': kernel_name,
            'algorithm': algorithm_name,
            'time_elapsed': seconds,
            'best_performance': milliseconds,
            'status': 'success' or 'failure',
            'error': error message if failed
        }
    """
    print(f"\n{'=' * 70}")
    print(f"Kernel: {kernel_config.name} | Algorithm: {algorithm_name}")
    print(f"{'=' * 70}")

    result = {
        "kernel": kernel_config.name,
        "algorithm": algorithm_name,
        "time_elapsed": 0.0,
        "best_performance": float("inf"),
        "status": "pending",
        "error": None,
    }

    try:
        # Generate arguments
        args = kernel_config.args_fn(device)

        # Bind kernel
        bound_kernel = kernel_config.kernel.bind(args)

        # Create search instance
        search = algorithm_class(bound_kernel, args, **algorithm_kwargs)

        # Run autotuning
        start_time = time.time()
        best_config = search.autotune()
        end_time = time.time()

        # Get best performance
        # Re-benchmark the best config to get accurate timing
        final_result = bound_kernel.run_with_config(best_config, args)
        best_perf = final_result.benchmark(fidelity=50)

        result["time_elapsed"] = end_time - start_time
        result["best_performance"] = best_perf * 1000  # Convert to ms
        result["status"] = "success"

        print(f"✓ Completed in {result['time_elapsed']:.1f}s")
        print(f"✓ Best performance: {result['best_performance']:.4f}ms")

    except Exception as e:
        result["status"] = "failure"
        result["error"] = str(e)
        print(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()

    return result


def run_full_benchmark(time_budget: float, device: str = "cuda") -> list[dict]:
    """
    Run all algorithms on all kernels.

    Args:
        time_budget: Time budget per algorithm per kernel (seconds)
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        List of result dictionaries
    """
    if not torch.cuda.is_available() and device == "cuda":
        print("CUDA not available, using CPU")
        device = "cpu"

    kernels = get_kernel_configs()
    algorithms = get_algorithm_configs(time_budget)

    print("\n" + "=" * 70)
    print("Multi-Kernel Autotuner Benchmark")
    print("=" * 70)
    print(f"Kernels: {len(kernels)}")
    print(f"Algorithms: {len(algorithms)}")
    print(f"Time budget per run: {time_budget}s")
    print(f"Device: {device}")
    print(f"Total estimated time: ~{len(kernels) * len(algorithms) * time_budget / 60:.1f} minutes")
    print("=" * 70)

    results = []

    for kernel_config in kernels:
        print(f"\n\n{'#' * 70}")
        print(f"# KERNEL: {kernel_config.description}")
        print(f"{'#' * 70}")

        for algo_name, (algo_class, algo_kwargs) in algorithms.items():
            result = run_single_benchmark(
                kernel_config, algo_name, algo_class, algo_kwargs, device
            )
            results.append(result)

            # Save intermediate results
            with open("multi_kernel_results.json", "w") as f:
                json.dump(results, f, indent=2)

    return results


# ============================================================================
# Results Analysis and Visualization
# ============================================================================


def analyze_results(results: list[dict]) -> None:
    """
    Analyze and visualize benchmark results.

    Creates:
    - Summary tables per kernel
    - Performance comparison plots
    - Rankings across kernels
    """
    import pandas as pd

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Filter only successful runs
    df_success = df[df["status"] == "success"].copy()

    if len(df_success) == 0:
        print("\n⚠ No successful runs to analyze!")
        return

    # Print summary tables for each kernel
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    kernels = df_success["kernel"].unique()

    for kernel in kernels:
        kernel_df = df_success[df_success["kernel"] == kernel].copy()
        kernel_df = kernel_df.sort_values("best_performance")

        print(f"\n{'─' * 70}")
        print(f"Kernel: {kernel}")
        print(f"{'─' * 70}")

        # Find best performance
        best_perf = kernel_df["best_performance"].min()

        print(f"{'Algorithm':<25} {'Time (s)':<12} {'Best (ms)':<12} {'vs Best':<12}")
        print("─" * 70)

        for _, row in kernel_df.iterrows():
            gap = ((row["best_performance"] / best_perf - 1) * 100) if best_perf > 0 else 0
            print(
                f"{row['algorithm']:<25} "
                f"{row['time_elapsed']:<12.1f} "
                f"{row['best_performance']:<12.4f} "
                f"{gap:>+6.1f}%"
            )

    # Overall rankings
    print(f"\n{'=' * 70}")
    print("OVERALL RANKINGS (Average Performance Across Kernels)")
    print(f"{'=' * 70}")

    # Compute average rank for each algorithm
    algorithm_ranks = {}
    for algo in df_success["algorithm"].unique():
        ranks = []
        for kernel in kernels:
            kernel_df = df_success[df_success["kernel"] == kernel].copy()
            kernel_df = kernel_df.sort_values("best_performance").reset_index(drop=True)
            algo_row = kernel_df[kernel_df["algorithm"] == algo]
            if len(algo_row) > 0:
                ranks.append(algo_row.index[0] + 1)  # 1-indexed rank

        if ranks:
            algorithm_ranks[algo] = np.mean(ranks)

    # Sort by average rank
    sorted_algos = sorted(algorithm_ranks.items(), key=lambda x: x[1])

    print(f"{'Rank':<6} {'Algorithm':<25} {'Avg Rank':<12}")
    print("─" * 70)
    for i, (algo, avg_rank) in enumerate(sorted_algos, 1):
        print(f"{i:<6} {algo:<25} {avg_rank:<12.2f}")

    # Create visualization
    create_visualization(df_success)


def create_visualization(df: pd.DataFrame) -> None:
    """Create comprehensive visualization of results."""
    import pandas as pd

    kernels = df["kernel"].unique()
    algorithms = df["algorithm"].unique()

    fig, axes = plt.subplots(1, len(kernels), figsize=(6 * len(kernels), 6))

    if len(kernels) == 1:
        axes = [axes]

    for ax, kernel in zip(axes, kernels):
        kernel_df = df[df["kernel"] == kernel].copy()
        kernel_df = kernel_df.sort_values("best_performance")

        # Normalize to best
        best = kernel_df["best_performance"].min()
        kernel_df["relative_perf"] = (kernel_df["best_performance"] / best - 1) * 100

        ax.barh(kernel_df["algorithm"], kernel_df["relative_perf"], color="steelblue")
        ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Best")
        ax.set_xlabel("Performance Gap from Best (%)")
        ax.set_title(f"{kernel.upper()}\n(Best: {best:.4f}ms)")
        ax.grid(axis="x", alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig("multi_kernel_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\n✓ Visualization saved to multi_kernel_comparison.png")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Multi-kernel autotuner benchmark")
    parser.add_argument(
        "--time-budget",
        type=float,
        default=120,
        help="Time budget per algorithm per kernel (seconds), default: 120",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run on (cuda/cpu)"
    )

    args = parser.parse_args()

    # Run benchmark
    results = run_full_benchmark(args.time_budget, args.device)

    # Analyze results
    analyze_results(results)

    print("\n" + "=" * 70)
    print("✓ Benchmark completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
