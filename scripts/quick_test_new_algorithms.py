#!/usr/bin/env python3
"""
Quick validation test for new algorithms.
"""

import sys
from pathlib import Path

# Add helion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

import helion
import helion.language as lang
from helion.autotuner import CMAESSearch, GeneticAlgorithmSearch, ParticleSwarmSearch


@helion.kernel(
    static_shapes=True,
    autotune_config_overrides={
        "range_unroll_factors": [0, 0],
        "range_num_stages": [0, 0],
    },
)
def test_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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


def test_algorithm(name: str, algorithm_class: type, kwargs: dict):
    """Test a single algorithm."""
    print(f"\n{'=' * 70}")
    print(f"Testing {name}")
    print(f"{'=' * 70}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    x = torch.randn([512, 512], device=device, dtype=torch.float16)
    y = torch.randn([512, 512], device=device, dtype=torch.float16)
    args = (x, y)

    try:
        bound_kernel = test_matmul.bind(args)
        search = algorithm_class(bound_kernel, args, **kwargs)

        print(f"✓ {name} instantiated successfully")

        best_config = search.autotune()

        print(f"✓ {name} completed successfully")
        print(f"✓ Best config found")

        return True

    except Exception as e:
        print(f"✗ {name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 70)
    print("Quick Validation Test for New Algorithms")
    print("=" * 70)

    algorithms = [
        ("CMA-ES", CMAESSearch, {"max_generations": 5}),
        ("Genetic Algorithm", GeneticAlgorithmSearch, {"population_size": 10, "max_generations": 3}),
        ("PSO", ParticleSwarmSearch, {"swarm_size": 10, "max_iterations": 3}),
    ]

    results = {}
    for name, algo_class, kwargs in algorithms:
        results[name] = test_algorithm(name, algo_class, kwargs)

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{name:<30} {status}")

    all_passed = all(results.values())
    print("=" * 70)

    if all_passed:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
