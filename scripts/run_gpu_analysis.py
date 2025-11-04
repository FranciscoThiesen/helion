#!/usr/bin/env python3
"""
Real GPU convergence analysis script.
Runs convergence comparison using actual GPU kernels and real autotuner search classes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import torch

# Use non-interactive backend for headless servers
matplotlib.use("Agg")

# Add helion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import helion
from helion.autotuner import MultiFidelityBayesianSearch
from helion.autotuner import PatternSearch
from helion.autotuner import RandomSearch
import helion.language as hl


# Define a simple matmul kernel for benchmarking
@helion.kernel(
    static_shapes=True,
    autotune_config_overrides={
        "range_unroll_factors": [0, 0],
        "range_num_stages": [0, 0],
    },
)
def matmul_benchmark(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Simple matrix multiplication kernel for benchmarking autotuners.

    Args:
        x: Left matrix of shape [m, k]
        y: Right matrix of shape [k, n]

    Returns:
        Result matrix of shape [m, n]
    """
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"

    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc

    return out


def run_search_algorithm(
    name: str,
    search_class: type,
    bound_kernel: Any,
    args: tuple,
    verbose: bool = True,
    **search_kwargs: Any,
) -> dict[str, Any]:
    """
    Run a single search algorithm and collect results.

    Args:
        name: Name of the search algorithm
        search_class: The search class to instantiate
        bound_kernel: The bound kernel to tune
        args: Arguments to pass to the kernel
        verbose: Whether to print progress
        **search_kwargs: Additional keyword arguments for the search class

    Returns:
        Dictionary with results including best config, timing, and evaluation count
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Running {name}...")
        print(f"{'=' * 70}\n")

    start_time = time.time()

    try:
        # Create the search instance
        search = search_class(bound_kernel, args, **search_kwargs)

        # Run autotuning
        best_config = search.autotune()

        elapsed = time.time() - start_time

        # Get evaluation count from counters
        total_evals = search.counters.get("total_benchmarks", 0)

        # Benchmark the best config to get actual performance
        compiled_fn = bound_kernel.compile_config(best_config)
        best_perf = search.benchmark_function(best_config, compiled_fn, fidelity=50)

        if verbose:
            print(f"\n✓ {name} completed!")
            print(f"  Best performance: {best_perf:.4f} ms")
            print(f"  Best config: {best_config}")
            print(f"  Total evaluations: {total_evals}")
            print(f"  Time elapsed: {elapsed:.2f}s")

        return {
            "name": name,
            "best_config": best_config,
            "best_perf": best_perf,
            "total_evals": total_evals,
            "elapsed_time": elapsed,
            "success": True,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        if verbose:
            print(f"\n✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()

        return {
            "name": name,
            "best_config": None,
            "best_perf": float("inf"),
            "total_evals": 0,
            "elapsed_time": elapsed,
            "success": False,
            "error": str(e),
        }


def run_full_analysis(
    m: int = 1024,
    k: int = 1024,
    n: int = 1024,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    output_dir: Path = Path("."),
    verbose: bool = True,
) -> dict[str, dict]:
    """
    Run complete convergence analysis on real GPU kernels.

    Args:
        m: Number of rows in left matrix
        k: Number of columns in left matrix / rows in right matrix
        n: Number of columns in right matrix
        dtype: Data type for matrices
        device: Device to run on
        output_dir: Directory to save results
        verbose: Whether to print progress

    Returns:
        Dictionary mapping algorithm name to results
    """
    if verbose:
        print("=" * 70)
        print("Real GPU Kernel Convergence Analysis")
        print("=" * 70)
        print(f"\nMatrix sizes: [{m}, {k}] @ [{k}, {n}]")
        print(f"Data type: {dtype}")
        print(f"Device: {device}")
        print(f"Output directory: {output_dir}\n")

    # Create test tensors
    x = torch.randn([m, k], device=device, dtype=dtype)
    y = torch.randn([k, n], device=device, dtype=dtype)
    args = (x, y)

    # Bind the kernel
    bound_kernel = matmul_benchmark.bind(args)

    # Define algorithms to test
    algorithms = [
        ("PatternSearch", PatternSearch, {}),
        ("RandomSearch", RandomSearch, {"count": 1000}),  # Increased to match MFBO budget
        (
            "MultiFidelityBO",
            MultiFidelityBayesianSearch,
            {
                # Scenario 3: MFBO with EQUAL TIME BUDGET as PatternSearch (~260s)
                # Question: Can MFBO match PatternSearch quality with SAME time?
                # Previous run: 1200/900/600/100 = 179s (need +81s more!)
                # Strategy: Increase all stages to fully utilize PatternSearch's time budget
                "n_low_fidelity": 1500,    # +300 from 1200 (~+10s)
                "n_medium_fidelity": 1200, # +300 from 900 (~+20s)
                "n_high_fidelity": 900,    # +300 from 600 (~+35s)
                "n_ultra_fidelity": 150,   # +50 from 100 (~+16s)
                "fidelity_low": 10,        # Keep 10 reps for good correlation
                "fidelity_ultra": 50,      # Keep 50 reps to match PatternSearch
                # Total: ~3750 configs, targeting ~260s runtime (matching PatternSearch!)
                # Maximum exploration + refinement with equal time!
            },
        ),
    ]

    results = {}
    for name, SearchClass, kwargs in algorithms:
        result = run_search_algorithm(
            name, SearchClass, bound_kernel, args, verbose, **kwargs
        )
        if result["success"]:
            results[name] = result

    # Print summary
    if verbose:
        print_summary(results)

    # Generate comparison plot if we have results
    if len(results) >= 2:
        plot_path = output_dir / "convergence_comparison.png"
        plot_comparison(results, str(plot_path))

    return results


def print_summary(results: dict[str, dict]) -> None:
    """Print summary comparison table."""
    print(f"\n{'=' * 70}")
    print("CONVERGENCE ANALYSIS SUMMARY")
    print(f"{'=' * 70}\n")

    if not results:
        print("No results to display")
        return

    # Find baseline (PatternSearch) for speedup calculation
    baseline = results.get("PatternSearch")

    print(f"{'Algorithm':<25} {'Evaluations':<15} {'Time (s)':<15} {'Speedup':<15}")
    print("-" * 70)

    for name in ["PatternSearch", "RandomSearch", "MultiFidelityBO"]:
        result = results.get(name)
        if not result:
            continue

        speedup_str = "N/A"
        if baseline and baseline["total_evals"] > 0:
            speedup = baseline["total_evals"] / max(result["total_evals"], 1)
            speedup_str = f"{speedup:>10.1f}x"

        print(
            f"{name:<25} {result['total_evals']:>10}      "
            f"{result['elapsed_time']:>10.2f}      {speedup_str}"
        )

    print(f"\n{'=' * 70}")
    print("Best Configurations:")
    print(f"{'=' * 70}\n")

    for name, result in results.items():
        if result.get("best_config"):
            print(f"{name}:")
            print(f"  {result['best_config']}")
            print()


def plot_comparison(results: dict[str, dict], output_path: str) -> None:
    """Create bar plot comparing algorithm performance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    names = list(results.keys())
    evals = [results[name]["total_evals"] for name in names]
    times = [results[name]["elapsed_time"] for name in names]

    colors = {
        "MultiFidelityBO": "blue",
        "PatternSearch": "red",
        "RandomSearch": "green",
    }
    bar_colors = [colors.get(name, "gray") for name in names]

    # Plot evaluations
    ax1.bar(names, evals, color=bar_colors, alpha=0.7)
    ax1.set_ylabel("Number of Evaluations", fontsize=12)
    ax1.set_title("Total Evaluations Required", fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot time
    ax2.bar(names, times, color=bar_colors, alpha=0.7)
    ax2.set_ylabel("Time (seconds)", fontsize=12)
    ax2.set_title("Wall-Clock Time", fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Comparison plot saved to {output_path}")
    plt.close()


def main() -> int:
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run real GPU convergence analysis for autotuner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--m",
        type=int,
        default=1024,
        help="Number of rows in left matrix",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1024,
        help="Shared dimension",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1024,
        help="Number of columns in right matrix",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for matrices",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("."),
        help="Output directory for plots and results",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    # Convert dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    try:
        results = run_full_analysis(
            m=args.m,
            k=args.k,
            n=args.n,
            dtype=dtype,
            device=args.device,
            output_dir=args.output_dir,
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
