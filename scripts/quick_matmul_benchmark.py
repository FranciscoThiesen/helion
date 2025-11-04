#!/usr/bin/env python3
"""Quick matmul benchmark with all algorithms (60s budget)."""

import sys
from pathlib import Path
import time
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

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


@helion.kernel(
    static_shapes=True,
    autotune_config_overrides={
        "range_unroll_factors": [0, 0],
        "range_num_stages": [0, 0],
    },
)
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2

    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )

    for tile_m, tile_n in lang.tile([m, n]):
        acc = lang.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in lang.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc

    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    x = torch.randn([1024, 1024], device=device, dtype=torch.float16)
    y = torch.randn([1024, 1024], device=device, dtype=torch.float16)
    args = (x, y)

    # Quick configs for 60s budget
    algorithms = {
        "PatternSearch": (PatternSearch, {"initial_population": 50, "copies": 5, "max_generations": 3}),
        "RandomSearch": (RandomSearch, {"count": 600}),
        "DifferentialEvolution": (DifferentialEvolutionSearch, {"population_size": 30, "max_generations": 15}),
        "GeneticAlgorithm": (GeneticAlgorithmSearch, {"population_size": 40, "max_generations": 12}),
        "CMA-ES": (CMAESSearch, {"max_generations": 40}),
        "PSO": (ParticleSwarmSearch, {"swarm_size": 25, "max_iterations": 25}),
        "MFBO": (MultiFidelityBayesianSearch, {"n_low_fidelity": 300, "n_medium_fidelity": 200, "n_high_fidelity": 150, "n_ultra_fidelity": 50, "fidelity_low": 10, "fidelity_ultra": 50}),
        "MFBO-RF": (MultiFidelityRandomForestSearch, {"n_low_fidelity": 300, "n_medium_fidelity": 200, "n_high_fidelity": 150, "n_ultra_fidelity": 50, "fidelity_low": 10, "fidelity_ultra": 50}),
    }

    results = []

    for name, (algo_class, kwargs) in algorithms.items():
        print(f"\n{'='*70}")
        print(f"Running {name}")
        print(f"{'='*70}")

        try:
            bound_kernel = matmul.bind(args)
            search = algo_class(bound_kernel, args, **kwargs)

            start = time.time()
            best_config = search.autotune()
            elapsed = time.time() - start

            # Benchmark best config
            final_result = bound_kernel.run_with_config(best_config, args)
            best_perf = final_result.benchmark(fidelity=50) * 1000  # ms

            results.append({
                "algorithm": name,
                "time": elapsed,
                "performance": best_perf,
            })

            print(f"✓ Time: {elapsed:.1f}s, Best: {best_perf:.4f}ms")

        except Exception as e:
            print(f"✗ Failed: {e}")
            results.append({
                "algorithm": name,
                "time": 0,
                "performance": float('inf'),
            })

    # Save results
    with open("quick_matmul_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")

    results_sorted = sorted(results, key=lambda x: x["performance"])
    best = results_sorted[0]["performance"]

    print(f"{'Algorithm':<25} {'Time (s)':<12} {'Best (ms)':<12} {'vs Best':<12}")
    print("-" * 70)

    for r in results_sorted:
        gap = ((r["performance"] / best - 1) * 100) if best > 0 and r["performance"] < float('inf') else float('inf')
        print(f"{r['algorithm']:<25} {r['time']:<12.1f} {r['performance']:<12.4f} {gap:>+6.1f}%")


if __name__ == "__main__":
    main()
