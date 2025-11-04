#!/usr/bin/env python3
"""
Quick validation test for Random Forest MFBO.
"""

import sys
from pathlib import Path

import torch

# Add helion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import helion
import helion.language as lang
from helion.autotuner import MultiFidelityRandomForestSearch


# Simple test kernel
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


def main():
    print("=" * 70)
    print("Random Forest MFBO Validation Test")
    print("=" * 70)

    # Create small test tensors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    x = torch.randn([512, 512], device=device, dtype=torch.float16)
    y = torch.randn([512, 512], device=device, dtype=torch.float16)
    args = (x, y)

    print("\n1. Testing RF-MFBO instantiation...")
    try:
        bound_kernel = test_matmul.bind(args)
        search = MultiFidelityRandomForestSearch(
            bound_kernel,
            args,
            n_low_fidelity=10,  # Very small for quick test
            n_medium_fidelity=5,
            n_high_fidelity=3,
            n_ultra_fidelity=2,
            fidelity_low=5,
            fidelity_ultra=10,
            n_estimators=50,  # Fewer trees for speed
        )
        print("   ✓ RF-MFBO instantiated successfully")
    except Exception as e:
        print(f"   ✗ Failed to instantiate RF-MFBO: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n2. Testing autotuning (small budget)...")
    try:
        best_config = search.autotune()
        print(f"   ✓ Autotuning completed")
        print(f"   ✓ Best config: {best_config}")
    except Exception as e:
        print(f"   ✗ Autotuning failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n3. Checking results...")
    try:
        assert len(search.evaluated_low) > 0, "No low-fidelity evaluations"
        assert len(search.evaluated_medium) > 0, "No medium-fidelity evaluations"
        assert len(search.evaluated_high) > 0, "No high-fidelity evaluations"
        assert len(search.evaluated_ultra) > 0, "No ultra-fidelity evaluations"
        print(f"   ✓ All stages completed:")
        print(f"      - Low fidelity: {len(search.evaluated_low)} configs")
        print(f"      - Medium fidelity: {len(search.evaluated_medium)} configs")
        print(f"      - High fidelity: {len(search.evaluated_high)} configs")
        print(f"      - Ultra fidelity: {len(search.evaluated_ultra)} configs")
    except AssertionError as e:
        print(f"   ✗ Validation failed: {e}")
        return 1

    print("\n" + "=" * 70)
    print("✓ RF-MFBO validation test PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
