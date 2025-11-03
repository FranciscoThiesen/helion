#!/usr/bin/env python3
"""
Compare MFBO vs PatternSearch on synthetic benchmark functions.
No GPU required - uses mock performance functions.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from helion.autotuner.config_fragment import BlockSizeFragment
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_fragment import NumWarpsFragment
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.multifidelity_bo_search import MultiFidelityBOSearch
from helion.autotuner.pattern_search import PatternSearch


class MockBenchmark:
    """Mock benchmark that simulates kernel performance without GPU."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.eval_count = 0
        self.eval_history: list[tuple[dict[str, Any], float, int]] = []

    def __call__(self, config: dict[str, Any], fidelity: int = 50) -> float:
        """
        Synthetic performance function that:
        - Has a clear optimum
        - Includes noise that decreases with fidelity
        - Mimics real kernel behavior
        """
        self.eval_count += 1

        # Extract config values
        block_m = config.get("BLOCK_M", 64)
        block_n = config.get("BLOCK_N", 64)
        block_k = config.get("BLOCK_K", 32)
        num_warps = config.get("num_warps", 4)

        # Optimal values (for matmul-like kernel)
        optimal_block_m = 128
        optimal_block_n = 128
        optimal_block_k = 64
        optimal_warps = 8

        # Performance function (lower is better - represents runtime in ms)
        # Quadratic bowl with optimum at (128, 128, 64, 8)
        perf = (
            0.5
            + 0.001 * (block_m - optimal_block_m) ** 2
            + 0.001 * (block_n - optimal_block_n) ** 2
            + 0.002 * (block_k - optimal_block_k) ** 2
            + 0.1 * (num_warps - optimal_warps) ** 2
        )

        # Add noise inversely proportional to fidelity
        noise_std = 0.1 / math.sqrt(fidelity)
        noise = self.rng.normal(0, noise_std)
        perf_noisy = max(0.01, perf + noise)

        self.eval_history.append((config.copy(), perf_noisy, fidelity))
        return perf_noisy


def create_test_config_spec() -> ConfigSpec:
    """Create a realistic config spec for matmul kernel."""
    return ConfigSpec(
        {
            "BLOCK_M": BlockSizeFragment(64, 256, 128),
            "BLOCK_N": BlockSizeFragment(64, 256, 128),
            "BLOCK_K": BlockSizeFragment(32, 128, 64),
            "num_warps": NumWarpsFragment(2, 16, 8),
            "num_stages": EnumFragment([2, 3, 4, 5], 3),
        }
    )


def run_comparison() -> None:
    """Compare MFBO vs PatternSearch on synthetic benchmark."""
    print("=" * 70)
    print("Autotuner Comparison: MFBO vs PatternSearch")
    print("=" * 70)
    print()

    config_spec = create_test_config_spec()

    # Test both algorithms
    algorithms = [
        ("MultiFidelityBO", MultiFidelityBOSearch),
        ("PatternSearch", PatternSearch),
    ]

    results = {}

    for name, SearchClass in algorithms:
        print(f"\n{'=' * 70}")
        print(f"Running {name}...")
        print(f"{'=' * 70}\n")

        benchmark = MockBenchmark(seed=42)
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
            _benchmark: MockBenchmark = benchmark,
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
            best_config = search.search(mock_fn, max_iterations=100)
            elapsed = time.time() - start_time

            # Get best performance
            best_perf = min(perf for _, perf, _ in benchmark.eval_history)

            # Convert best_config to dict for display
            best_config_dict = {}
            for i, (key, _spec) in enumerate(config_spec.items()):
                best_config_dict[key] = best_config[i]

            results[name] = {
                "best_perf": best_perf,
                "best_config": best_config_dict,
                "total_evals": benchmark.eval_count,
                "elapsed_time": elapsed,
                "eval_history": benchmark.eval_history,
            }

            print(f"\n✓ {name} completed!")
            print(f"  Best performance: {best_perf:.4f} ms")
            print(f"  Best config: {best_config_dict}")
            print(f"  Total evaluations: {benchmark.eval_count}")
            print(f"  Time elapsed: {elapsed:.2f}s")

        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Print comparison
    if len(results) == 2:
        print(f"\n{'=' * 70}")
        print("COMPARISON SUMMARY")
        print(f"{'=' * 70}\n")

        mfbo = results["MultiFidelityBO"]
        pattern = results["PatternSearch"]

        print("Best Performance:")
        print(f"  MFBO:          {mfbo['best_perf']:.4f} ms")
        print(f"  PatternSearch: {pattern['best_perf']:.4f} ms")
        perf_diff = (
            (pattern["best_perf"] - mfbo["best_perf"]) / pattern["best_perf"] * 100
        )
        print(f"  Difference:    {perf_diff:+.1f}%")
        print()

        print("Total Evaluations:")
        print(f"  MFBO:          {mfbo['total_evals']}")
        print(f"  PatternSearch: {pattern['total_evals']}")
        speedup = pattern["total_evals"] / mfbo["total_evals"]
        print(f"  Speedup:       {speedup:.1f}x fewer evaluations")
        print()

        print("Wall Clock Time:")
        print(f"  MFBO:          {mfbo['elapsed_time']:.2f}s")
        print(f"  PatternSearch: {pattern['elapsed_time']:.2f}s")
        time_speedup = pattern["elapsed_time"] / mfbo["elapsed_time"]
        print(f"  Speedup:       {time_speedup:.2f}x faster")
        print()

        print("Best Configurations:")
        print(f"  MFBO:          {mfbo['best_config']}")
        print(f"  PatternSearch: {pattern['best_config']}")
        print()


if __name__ == "__main__":
    run_comparison()
