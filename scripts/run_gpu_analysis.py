#!/usr/bin/env python3
"""
Unified convergence analysis script for GPU machines.
Runs both convergence comparison and best configuration analysis.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive backend for headless servers
matplotlib.use("Agg")

# Add helion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from helion.autotuner.config_fragment import BlockSizeFragment
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_fragment import NumWarpsFragment
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.multifidelity_bo_search import MultiFidelityBOSearch
from helion.autotuner.pattern_search import PatternSearch
from helion.autotuner.random_search import RandomSearch


class SyntheticKernelBenchmark:
    """
    Simulates a realistic kernel performance landscape.

    This creates a challenging optimization problem similar to real GPU kernels:
    - Non-convex with multiple local minima
    - Noisy measurements (noise decreases with higher fidelity)
    - Power-of-2 parameter preferences
    - Realistic performance range (0.5-10ms)
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.eval_count = 0
        self.eval_history: list[tuple[dict[str, Any], float, int]] = []

        # Global optimum (what we want algorithms to find)
        self.global_optimum = {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "num_warps": 8,
            "num_stages": 3,
        }

        # Local optima (suboptimal but decent - traps for naive searchers)
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
        """Evaluate a configuration with given fidelity (number of reps)."""
        self.eval_count += 1

        # Extract config values
        block_m = config.get("BLOCK_M", 64)
        block_n = config.get("BLOCK_N", 64)
        block_k = config.get("BLOCK_K", 32)
        num_warps = config.get("num_warps", 4)
        num_stages = config.get("num_stages", 3)

        # Base performance: quadratic bowl around global optimum
        perf = (
            1.5  # baseline performance
            + 0.001 * (block_m - self.global_optimum["BLOCK_M"]) ** 2
            + 0.001 * (block_n - self.global_optimum["BLOCK_N"]) ** 2
            + 0.002 * (block_k - self.global_optimum["BLOCK_K"]) ** 2
            + 0.12 * (num_warps - self.global_optimum["num_warps"]) ** 2
            + 0.2 * (num_stages - self.global_optimum["num_stages"]) ** 2
        )

        # Add local optima attractions (creates multiple valleys)
        for local_opt in self.local_optima:
            dist_sq = (
                0.0008 * (block_m - local_opt["BLOCK_M"]) ** 2
                + 0.0008 * (block_n - local_opt["BLOCK_N"]) ** 2
                + 0.0015 * (block_k - local_opt["BLOCK_K"]) ** 2
                + 0.08 * (num_warps - local_opt["num_warps"]) ** 2
                + 0.15 * (num_stages - local_opt["num_stages"]) ** 2
            )
            # Gaussian attraction toward local optimum
            perf -= 0.6 * math.exp(-dist_sq / 2.0)

        # Add non-smooth effects (realistic for GPU kernels)
        # GPUs prefer certain block size alignments
        if block_m % 128 != 0:
            perf += 0.25
        if block_n % 128 != 0:
            perf += 0.25

        # Penalize bad warp/stage combinations
        if num_warps > num_stages * 4:
            perf += 0.4

        # Add measurement noise (inversely proportional to fidelity)
        # Higher fidelity = more reps = less noise
        noise_std = 0.15 / math.sqrt(fidelity)
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
    name: str,
    SearchClass: type,
    config_spec: ConfigSpec,
    seed: int,
    max_iters: int,
    verbose: bool = True,
) -> dict[str, Any] | None:
    """Run a single autotuning algorithm and collect results."""
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Running {name}...")
        print(f"{'=' * 70}\n")

    benchmark = SyntheticKernelBenchmark(seed=seed)

    # Create search instance
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

    # Override benchmark function to use our mock
    def mock_benchmark_wrapper(
        config: object,
        fn: object,
        *,
        fidelity: int = 50,
        _benchmark: SyntheticKernelBenchmark = benchmark,
    ) -> float:
        # Convert config to dict for our mock
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

        # Get best performance
        best_perf = min(perf for _, perf, _ in benchmark.eval_history)

        # Get convergence curve
        convergence = benchmark.get_best_so_far()

        # Convert best_config to dict for display
        best_config_dict = {}
        for i, (key, _spec) in enumerate(config_spec.items()):
            best_config_dict[key] = best_config[i]

        if verbose:
            print(f"\n✓ {name} completed!")
            print(f"  Best performance: {best_perf:.4f} ms")
            print(f"  Best config: {best_config_dict}")
            print(f"  Total evaluations: {benchmark.eval_count}")
            print(f"  Time elapsed: {elapsed:.2f}s")

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
        if verbose:
            print(f"\n✗ {name} failed: {e}")
            import traceback

            traceback.print_exc()
        return None


def plot_convergence(
    results: dict[str, dict], output_path: str, title_suffix: str = ""
) -> None:
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
            label=f"{name} (final: {result['best_perf']:.3f}ms, evals: {result['total_evals']})",
            color=colors.get(name, "gray"),
            linewidth=2,
            alpha=0.8,
        )

    plt.xlabel("Number of Evaluations", fontsize=12)
    plt.ylabel("Best Performance (ms, lower is better)", fontsize=12)
    title = "Convergence Comparison: MFBO vs PatternSearch vs RandomSearch"
    if title_suffix:
        title += f" ({title_suffix})"
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Convergence plot saved to {output_path}")
    plt.close()


def print_summary(results: dict[str, dict], show_configs: bool = True) -> None:
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

    if show_configs:
        print(f"\n{'=' * 70}")
        print("Best Configurations Found:")
        print(f"{'=' * 70}\n")

        optimal_config = {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "num_warps": 8,
            "num_stages": 3,
        }

        for name, result in results.items():
            if result:
                config = result["best_config"]
                # Check if found optimal
                is_optimal = all(config.get(k) == v for k, v in optimal_config.items())
                status = " ✓ OPTIMAL" if is_optimal else " (suboptimal)"

                print(f"{name}{status}:")
                print(f"  {config}")
                print()


def run_full_analysis(
    output_dir: Path, seed: int = 42, max_iters: int = 100, verbose: bool = True
) -> dict[str, dict]:
    """Run complete convergence analysis."""
    if verbose:
        print("=" * 70)
        print("Multi-Fidelity Bayesian Optimization - Convergence Analysis")
        print("=" * 70)
        print(f"\nOutput directory: {output_dir}")
        print(f"Random seed: {seed}")
        print(f"Max iterations: {max_iters}\n")

    config_spec = create_test_config_spec()

    algorithms = [
        ("PatternSearch", PatternSearch),
        ("RandomSearch", RandomSearch),
        ("MultiFidelityBO", MultiFidelityBOSearch),
    ]

    results = {}
    for name, SearchClass in algorithms:
        result = run_algorithm(name, SearchClass, config_spec, seed, max_iters, verbose)
        if result:
            results[name] = result

    # Generate plots
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "convergence_comparison.png"
    plot_convergence(results, str(plot_path))

    # Print summary
    if verbose:
        print_summary(results)

    return results


def main() -> None:
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run convergence analysis for MFBO autotuner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("."),
        help="Output directory for plots and results",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max-iters",
        "-i",
        type=int,
        default=100,
        help="Maximum iterations for each algorithm",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    try:
        results = run_full_analysis(
            args.output_dir,
            seed=args.seed,
            max_iters=args.max_iters,
            verbose=not args.quiet,
        )

        if results:
            print("\n" + "=" * 70)
            print("✓ Analysis complete!")
            print("=" * 70)
            return 0
        else:
            print("\n" + "=" * 70)
            print("✗ Analysis failed - no results generated")
            print("=" * 70)
            return 1

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        return 130
    except Exception as e:
        print(f"\n\nError during analysis: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
