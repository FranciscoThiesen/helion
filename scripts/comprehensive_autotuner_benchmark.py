#!/usr/bin/env python3
"""
Comprehensive Autotuner Benchmark - Rigorous Comparison
========================================================

This script provides apples-to-apples comparisons of all Helion autotuner algorithms
with time-equalized budgets across multiple scenarios.

Author: Francisco Geiman Thiesen
Date: 2025-11-04
"""

import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

import helion
import helion.language as lang
from helion.autotuner import (
    DifferentialEvolutionSearch,
    MultiFidelityBayesianSearch,
    MultiFidelityRandomForestSearch,
    PatternSearch,
    RandomSearch,
)


# ============================================================================
# Configuration
# ============================================================================

# Test matrix dimensions
MATRIX_SIZE = 1024
DTYPE = torch.float16

# Scenarios with time budgets (in seconds)
SCENARIOS = {
    "fast": {"budget": 60, "description": "Fast prototyping (<1 min)"},
    "medium": {"budget": 120, "description": "Development iteration (~2 min)"},
    "thorough": {"budget": 300, "description": "Production optimization (~5 min)"},
}

# Output directory
OUTPUT_DIR = Path(".")


# ============================================================================
# Benchmark Kernel
# ============================================================================


@helion.kernel(
    static_shapes=True,
    autotune_config_overrides={
        "range_unroll_factors": [0, 0],
        "range_num_stages": [0, 0],
    },
)
def matmul_benchmark(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication kernel for benchmarking autotuners."""
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


# ============================================================================
# Algorithm Configurations (Time-Equalized)
# ============================================================================


def get_algorithm_configs(time_budget: float) -> dict[str, tuple[type, dict]]:
    """
    Generate algorithm configurations calibrated to the time budget.

    Time calibration is based on empirical measurements:
    - PatternSearch: ~0.15s per config (50 reps)
    - RandomSearch: ~0.1s per config (50 reps)
    - DifferentialEvolution: ~0.15s per config (50 reps)
    - MFBO: Variable (10-50 reps)

    Args:
        time_budget: Target runtime in seconds

    Returns:
        Dictionary mapping algorithm name to (class, kwargs)
    """
    # Empirical time per config (seconds)
    TIME_PER_CONFIG = {
        "PatternSearch": 0.15,
        "RandomSearch": 0.1,
        "DifferentialEvolution": 0.15,
        "MFBO": 0.08,  # Average across all fidelities
        "MFBO-RF": 0.08,  # Similar to MFBO (RF prediction slightly faster than GP)
    }

    # Calculate config counts for each algorithm
    pattern_configs = int(time_budget / TIME_PER_CONFIG["PatternSearch"])
    random_configs = int(time_budget / TIME_PER_CONFIG["RandomSearch"])
    de_configs = int(time_budget / TIME_PER_CONFIG["DifferentialEvolution"])
    mfbo_configs = int(time_budget / TIME_PER_CONFIG["MFBO"])
    mfbo_rf_configs = int(time_budget / TIME_PER_CONFIG["MFBO-RF"])

    # PatternSearch: initial_pop * copies * generations
    # Target: pattern_configs total
    # Use: pop=100, copies=5, then solve for generations
    ps_pop = 100
    ps_copies = 5
    ps_gens = max(1, (pattern_configs - ps_pop) // (ps_pop // ps_copies))

    # DifferentialEvolution: population_size * (max_generations + 1)
    # Target: de_configs total
    # Balance population and generations
    de_pop = min(100, int(np.sqrt(de_configs)))
    de_gens = max(1, de_configs // de_pop - 1)

    # MFBO: Distribute across fidelity levels
    # Use progressive allocation: 40% low, 30% med, 20% high, 10% ultra
    mfbo_low = int(mfbo_configs * 0.4)
    mfbo_med = int(mfbo_configs * 0.3)
    mfbo_high = int(mfbo_configs * 0.2)
    mfbo_ultra = int(mfbo_configs * 0.1)

    return {
        "PatternSearch": (
            PatternSearch,
            {
                "initial_population": ps_pop,
                "copies": ps_copies,
                "max_generations": ps_gens,
            },
        ),
        "RandomSearch": (
            RandomSearch,
            {
                "count": random_configs,
            },
        ),
        "DifferentialEvolution": (
            DifferentialEvolutionSearch,
            {
                "population_size": de_pop,
                "max_generations": de_gens,
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
                "n_low_fidelity": int(mfbo_rf_configs * 0.4),
                "n_medium_fidelity": int(mfbo_rf_configs * 0.3),
                "n_high_fidelity": int(mfbo_rf_configs * 0.2),
                "n_ultra_fidelity": int(mfbo_rf_configs * 0.1),
                "fidelity_low": 10,
                "fidelity_ultra": 50,
                "n_estimators": 100,  # Number of trees
            },
        ),
    }


# ============================================================================
# Benchmark Runner
# ============================================================================


def run_algorithm(
    name: str,
    search_class: type,
    bound_kernel: Any,
    args: tuple,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Run a single algorithm and collect metrics.

    Args:
        name: Algorithm name
        search_class: Search algorithm class
        bound_kernel: Bound kernel to optimize
        args: Kernel arguments
        **kwargs: Algorithm-specific parameters

    Returns:
        Dictionary with results
    """
    print(f"\n{'=' * 70}")
    print(f"Running {name}...")
    print(f"{'=' * 70}\n")

    # Initialize search
    search = search_class(bound_kernel, args, **kwargs)

    # Run autotuning with timing
    start_time = time.time()
    best_config = search.autotune()
    elapsed_time = time.time() - start_time

    # Benchmark best config
    compiled_fn = bound_kernel.compile_config(best_config)
    best_perf = search.benchmark_function(best_config, compiled_fn, fidelity=50)

    print(f"\n✓ {name} completed!")
    print(f"  Best performance: {best_perf:.4f} ms")
    print(f"  Best config: {best_config}")
    print(f"  Time elapsed: {elapsed_time:.2f}s\n")

    return {
        "name": name,
        "best_config": best_config,
        "best_perf": best_perf,
        "elapsed_time": elapsed_time,
        "params": kwargs,
    }


def run_scenario(scenario_name: str, time_budget: float, description: str) -> dict:
    """
    Run all algorithms for a given scenario.

    Args:
        scenario_name: Name of the scenario
        time_budget: Time budget in seconds
        description: Scenario description

    Returns:
        Dictionary with all results
    """
    print(f"\n{'#' * 70}")
    print(f"# SCENARIO: {scenario_name.upper()} ({description})")
    print(f"# Time Budget: {time_budget}s")
    print(f"{'#' * 70}\n")

    # Create test data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(MATRIX_SIZE, MATRIX_SIZE, dtype=DTYPE, device=device)
    y = torch.randn(MATRIX_SIZE, MATRIX_SIZE, dtype=DTYPE, device=device)
    args = (x, y)

    # Get bound kernel
    bound_kernel = matmul_benchmark.bind(args)

    # Get algorithm configurations
    algorithms = get_algorithm_configs(time_budget)

    # Run each algorithm
    results = {}
    for algo_name, (search_class, kwargs) in algorithms.items():
        result = run_algorithm(algo_name, search_class, bound_kernel, args, **kwargs)
        results[algo_name] = result

    return {
        "scenario": scenario_name,
        "time_budget": time_budget,
        "description": description,
        "results": results,
    }


# ============================================================================
# Visualization
# ============================================================================


def create_comparison_plots(all_scenarios: dict[str, dict]) -> None:
    """
    Create comprehensive comparison visualizations.

    Args:
        all_scenarios: Dictionary of scenario results
    """
    # Prepare data
    scenarios = list(all_scenarios.keys())
    algorithms = list(next(iter(all_scenarios.values()))["results"].keys())

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Comprehensive Autotuner Benchmark", fontsize=16, fontweight="bold")

    # Plot 1: Performance by scenario
    ax = axes[0, 0]
    width = 0.15
    x_pos = np.arange(len(scenarios))

    for i, algo in enumerate(algorithms):
        perfs = [
            all_scenarios[s]["results"][algo]["best_perf"] * 1000  # Convert to μs
            for s in scenarios
        ]
        ax.bar(x_pos + i * width, perfs, width, label=algo)

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Best Performance (μs)")
    ax.set_title("Performance Comparison Across Scenarios")
    ax.set_xticks(x_pos + width * (len(algorithms) - 1) / 2)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Time budget usage
    ax = axes[0, 1]
    for i, algo in enumerate(algorithms):
        times = [all_scenarios[s]["results"][algo]["elapsed_time"] for s in scenarios]
        budgets = [all_scenarios[s]["time_budget"] for s in scenarios]
        usage = [t / b * 100 for t, b in zip(times, budgets)]
        ax.plot(scenarios, usage, marker="o", label=algo, linewidth=2)

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Time Budget Usage (%)")
    ax.set_title("Time Budget Utilization")
    ax.axhline(100, color="red", linestyle="--", alpha=0.5, label="Budget Limit")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Quality vs Time
    ax = axes[1, 0]
    for algo in algorithms:
        times = [all_scenarios[s]["results"][algo]["elapsed_time"] for s in scenarios]
        perfs = [
            all_scenarios[s]["results"][algo]["best_perf"] * 1000 for s in scenarios
        ]
        ax.scatter(times, perfs, s=100, label=algo, alpha=0.7)
        # Add trend line
        z = np.polyfit(times, perfs, 1)
        p = np.poly1d(z)
        ax.plot(times, p(times), "--", alpha=0.5)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Best Performance (μs)")
    ax.set_title("Quality vs Time Trade-off")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Relative performance (normalized to best)
    ax = axes[1, 1]
    for i, scenario in enumerate(scenarios):
        best_perf = min(
            all_scenarios[scenario]["results"][algo]["best_perf"]
            for algo in algorithms
        )
        relative = [
            (all_scenarios[scenario]["results"][algo]["best_perf"] / best_perf - 1)
            * 100
            for algo in algorithms
        ]
        x_pos = np.arange(len(algorithms)) + i * 0.2
        ax.bar(x_pos, relative, 0.18, label=scenario)

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Performance Gap from Best (%)")
    ax.set_title("Relative Performance (0% = Best)")
    ax.set_xticks(np.arange(len(algorithms)) + 0.2)
    ax.set_xticklabels(algorithms, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="green", linestyle="-", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comprehensive_comparison.png", dpi=300, bbox_inches="tight")
    print(f"\n✓ Comparison plot saved to {OUTPUT_DIR / 'comprehensive_comparison.png'}")


# ============================================================================
# Results Summary
# ============================================================================


def print_summary_table(all_scenarios: dict[str, dict]) -> None:
    """Print comprehensive summary table."""
    print("\n" + "=" * 100)
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 100)

    for scenario_name, scenario_data in all_scenarios.items():
        print(f"\n{scenario_name.upper()} Scenario ({scenario_data['description']})")
        print(f"Time Budget: {scenario_data['time_budget']}s")
        print("-" * 100)
        print(
            f"{'Algorithm':<25} {'Time (s)':<12} {'Budget %':<12} {'Perf (ms)':<12} {'vs Best':<12}"
        )
        print("-" * 100)

        results = scenario_data["results"]
        best_perf = min(r["best_perf"] for r in results.values())

        for algo_name, result in results.items():
            time_pct = (result["elapsed_time"] / scenario_data["time_budget"]) * 100
            gap_pct = ((result["best_perf"] / best_perf) - 1) * 100
            print(
                f"{algo_name:<25} {result['elapsed_time']:<12.2f} {time_pct:<12.1f} "
                f"{result['best_perf']:<12.4f} {gap_pct:+11.1f}%"
            )

    print("=" * 100)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run comprehensive benchmark across all scenarios."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE AUTOTUNER BENCHMARK")
    print("=" * 70)
    print(f"Matrix size: {MATRIX_SIZE}×{MATRIX_SIZE}")
    print(f"Data type: {DTYPE}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Scenarios: {len(SCENARIOS)}")
    print("=" * 70)

    # Run all scenarios
    all_results = {}
    for scenario_name, scenario_config in SCENARIOS.items():
        result = run_scenario(
            scenario_name,
            scenario_config["budget"],
            scenario_config["description"],
        )
        all_results[scenario_name] = result

    # Generate summary
    print_summary_table(all_results)

    # Create visualizations
    create_comparison_plots(all_results)

    print("\n✓ Comprehensive benchmark complete!")


if __name__ == "__main__":
    main()
