#!/usr/bin/env python3
"""
Convergence analysis comparing MFBO vs PatternSearch vs RandomSearch.
Uses synthetic performance functions that mimic real kernel behavior.
No GPU required.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
import time
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive backend
matplotlib.use("Agg")

# Add autotuner directory to path to avoid triton dependency
autotuner_path = Path(__file__).parent.parent / "helion" / "autotuner"
sys.path.insert(0, str(autotuner_path))

# Direct imports without going through helion package
from config_fragment import BlockSizeFragment
from config_fragment import EnumFragment
from config_fragment import NumWarpsFragment
from config_spec import ConfigSpec
from multifidelity_bo_search import MultiFidelityBOSearch
from pattern_search import PatternSearch
from random_search import RandomSearch


class SyntheticKernelBenchmark:
    """
    Simulates a realistic kernel performance landscape.

    Key properties:
    - Non-convex with multiple local minima
    - Noisy measurements (less noise with higher fidelity)
    - Power-of-2 parameters have non-smooth gradients
    - Realistic performance values (0.1-10ms range)
    """

    def __init__(self, seed: int = 42, dim: int = 5) -> None:
        self.rng = np.random.default_rng(seed)
        self.dim = dim
        self.eval_count = 0
        self.eval_history: list[tuple[dict[str, Any], float, int]] = []

        # Define a complex performance landscape with multiple local optima
        # and one global optimum at specific configuration
        self.global_optimum = {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "num_warps": 8,
            "num_stages": 3,
        }

        # Secondary local optima (suboptimal but good)
        self.local_optima = [
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "num_warps": 4,
                "num_stages": 2,
            },
            {
                "BLOCK_M": 256,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "num_warps": 16,
                "num_stages": 4,
            },
        ]

    def __call__(self, config: dict[str, Any], fidelity: int = 50) -> float:
        """Evaluate config with given fidelity (number of repetitions)."""
        self.eval_count += 1

        # Extract config values
        block_m = config.get("BLOCK_M", 64)
        block_n = config.get("BLOCK_N", 64)
        block_k = config.get("BLOCK_K", 32)
        num_warps = config.get("num_warps", 4)
        num_stages = config.get("num_stages", 3)

        # Base performance (quadratic bowl around global optimum)
        perf = (
            2.0  # baseline
            + 0.002 * (block_m - self.global_optimum["BLOCK_M"]) ** 2
            + 0.002 * (block_n - self.global_optimum["BLOCK_N"]) ** 2
            + 0.003 * (block_k - self.global_optimum["BLOCK_K"]) ** 2
            + 0.15 * (num_warps - self.global_optimum["num_warps"]) ** 2
            + 0.25 * (num_stages - self.global_optimum["num_stages"]) ** 2
        )

        # Add local optima attractions (creates multiple valleys)
        for local_opt in self.local_optima:
            dist_sq = (
                0.001 * (block_m - local_opt["BLOCK_M"]) ** 2
                + 0.001 * (block_n - local_opt["BLOCK_N"]) ** 2
                + 0.002 * (block_k - local_opt["BLOCK_K"]) ** 2
                + 0.1 * (num_warps - local_opt["num_warps"]) ** 2
                + 0.2 * (num_stages - local_opt["num_stages"]) ** 2
            )
            # Gaussian attraction toward local optimum
            perf -= 0.8 * math.exp(-dist_sq / 2.0)

        # Add non-smooth effects (realistic for GPU kernels)
        # Penalize odd block sizes (GPU prefers even)
        if block_m % 128 != 0:
            perf += 0.3
        if block_n % 128 != 0:
            perf += 0.3

        # Penalize mismatched warps/stages
        if num_warps > num_stages * 4:
            perf += 0.5

        # Add measurement noise (inversely proportional to fidelity)
        noise_std = 0.2 / math.sqrt(fidelity)
        noise = self.rng.normal(0, noise_std)
        perf_noisy = max(0.1, perf + noise)

        self.eval_history.append((config.copy(), perf_noisy, fidelity))
        return perf_noisy

    def get_best_so_far(self) -> list[float]:
        """Return best performance observed at each evaluation."""
        best_so_far = []
        best_val = float("inf")
        for _, perf, _ in self.eval_history:
            best_val = min(best_val, perf)
            best_so_far.append(best_val)
        return best_so_far


def create_test_config_spec() -> ConfigSpec:
    """Create a realistic config spec for matmul-like kernel."""
    return ConfigSpec(
        {
            "BLOCK_M": BlockSizeFragment(32, 256, 128),
            "BLOCK_N": BlockSizeFragment(32, 256, 128),
            "BLOCK_K": BlockSizeFragment(16, 128, 64),
            "num_warps": NumWarpsFragment(2, 16, 8),
            "num_stages": EnumFragment([2, 3, 4, 5], 3),
        }
    )


def run_algorithm(
    name: str, SearchClass: type, config_spec: ConfigSpec, seed: int, max_iters: int
) -> dict[str, Any]:
    """Run a single autotuning algorithm and collect results."""
    print(f"\n{'=' * 70}")
    print(f"Running {name}...")
    print(f"{'=' * 70}\n")

    benchmark = SyntheticKernelBenchmark(seed=seed)

    if name == "MultiFidelityBO":
        search = SearchClass(
            config_spec,
            n_low_fidelity=20,
            n_medium_fidelity=10,
            n_high_fidelity=5,
            n_ultra_fidelity=3,
        )
    else:
        search = SearchClass(config_spec)

    # Mock compiled config object
    class MockCompiledConfig:
        pass

    mock_fn = MockCompiledConfig()

    # Override benchmark function
    def mock_benchmark_wrapper(
        config: object,
        fn: object,
        *,
        fidelity: int = 50,
        _benchmark: SyntheticKernelBenchmark = benchmark,
    ) -> float:
        config_dict = {}
        for i, (key, _spec) in enumerate(config_spec.items()):
            config_dict[key] = config[i]  # type: ignore[index]
        return _benchmark(config_dict, fidelity=fidelity)

    search.benchmark_function = mock_benchmark_wrapper  # type: ignore[method-assign]

    # Run search
    start_time = time.time()
    try:
        best_config = search.search(mock_fn, max_iterations=max_iters)
        elapsed = time.time() - start_time

        # Get convergence curve
        convergence = benchmark.get_best_so_far()

        # Convert best_config to dict
        best_config_dict = {}
        for i, (key, _spec) in enumerate(config_spec.items()):
            best_config_dict[key] = best_config[i]

        best_perf = min(perf for _, perf, _ in benchmark.eval_history)

        return {
            "name": name,
            "best_perf": best_perf,
            "best_config": best_config_dict,
            "total_evals": benchmark.eval_count,
            "elapsed_time": elapsed,
            "convergence": convergence,
            "eval_history": benchmark.eval_history,
        }

    except Exception as e:
        print(f"\n✗ {name} failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def plot_convergence(results: dict[str, dict], output_path: Path) -> None:
    """Create convergence plot comparing all algorithms."""
    plt.figure(figsize=(12, 6))

    colors = {
        "MultiFidelityBO": "blue",
        "PatternSearch": "red",
        "RandomSearch": "green",
    }

    for name, result in results.items():
        if result is None:
            continue

        convergence = result["convergence"]
        evals = list(range(1, len(convergence) + 1))

        plt.plot(
            evals,
            convergence,
            label=f"{name} (final: {result['best_perf']:.3f}ms)",
            color=colors.get(name, "gray"),
            linewidth=2,
            alpha=0.8,
        )

    plt.xlabel("Number of Evaluations", fontsize=12)
    plt.ylabel("Best Performance (ms, lower is better)", fontsize=12)
    plt.title(
        "Convergence Comparison: MFBO vs PatternSearch vs RandomSearch", fontsize=14
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Convergence plot saved to {output_path}")
    plt.close()


def print_summary(results: dict[str, dict]) -> None:
    """Print summary comparison table."""
    print(f"\n{'=' * 70}")
    print("CONVERGENCE ANALYSIS SUMMARY")
    print(f"{'=' * 70}\n")

    # Find baseline (PatternSearch)
    baseline = results.get("PatternSearch")
    if not baseline:
        print("Warning: No baseline (PatternSearch) results available")
        return

    print(f"{'Algorithm':<20} {'Best Perf':<15} {'Evaluations':<15} {'Speedup':<15}")
    print("-" * 70)

    for name in ["PatternSearch", "RandomSearch", "MultiFidelityBO"]:
        result = results.get(name)
        if not result:
            continue

        speedup = baseline["total_evals"] / result["total_evals"]

        print(
            f"{name:<20} {result['best_perf']:>10.4f} ms   "
            f"{result['total_evals']:>10}      {speedup:>10.1f}x"
        )

    print(f"\n{'=' * 70}")
    print("Best Configurations Found:")
    print(f"{'=' * 70}\n")

    for name, result in results.items():
        if result:
            print(f"{name}:")
            print(f"  {result['best_config']}")
            print()


def main() -> None:
    """Run full convergence analysis."""
    print("=" * 70)
    print("Multi-Fidelity BO Convergence Analysis")
    print("=" * 70)
    print("\nRunning synthetic benchmarks (no GPU required)...")

    config_spec = create_test_config_spec()
    seed = 42
    max_iters = 100

    algorithms = [
        ("PatternSearch", PatternSearch),
        ("RandomSearch", RandomSearch),
        ("MultiFidelityBO", MultiFidelityBOSearch),
    ]

    results = {}
    for name, SearchClass in algorithms:
        result = run_algorithm(name, SearchClass, config_spec, seed, max_iters)
        if result:
            results[name] = result

    # Generate plots
    output_dir = Path(__file__).parent.parent
    plot_path = output_dir / "convergence_comparison.png"
    plot_convergence(results, plot_path)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
