"""
Multi-Fidelity Random Forest Search for Autotuning.

This module implements a Random Forest variant of Multi-Fidelity Bayesian Optimization,
replacing Gaussian Processes with Random Forest Regression for the surrogate model.

Random Forests offer several advantages:
1. Better handling of discrete/categorical parameters
2. More robust to outliers
3. No need for kernel selection
4. Natural uncertainty estimates via tree variance

Author: Francisco Geiman Thiesen
Date: 2025-11-04
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base_search import PopulationBasedSearch, PopulationMember
from .config_encoding import ConfigEncoder

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.kernel import BoundKernel
    from .config_generation import FlatConfig


class MultiFidelityRandomForestSearch(PopulationBasedSearch):
    """
    Multi-fidelity autotuning using Random Forest regression.

    This search strategy uses progressive fidelity levels (low → medium → high → ultra)
    with Random Forest models to guide the search. Unlike Gaussian Processes, Random
    Forests naturally handle discrete parameters and provide robust uncertainty estimates.

    The search progresses through stages:
    1. Low fidelity: Broad exploration with quick evaluations
    2. Medium fidelity: Refine promising regions (mix direct promotion + RF guidance)
    3. High fidelity: Focus on best candidates (mix direct promotion + RF guidance)
    4. Ultra fidelity: Final validation of top configs

    Args:
        kernel: The bound kernel to tune
        args: Arguments for the kernel
        n_low_fidelity: Number of configs to evaluate at low fidelity
        n_medium_fidelity: Number of configs to evaluate at medium fidelity
        n_high_fidelity: Number of configs to evaluate at high fidelity
        n_ultra_fidelity: Number of configs to evaluate at ultra fidelity
        fidelity_low: Number of repetitions for low fidelity (default: 10)
        fidelity_medium: Number of repetitions for medium fidelity (default: 15)
        fidelity_high: Number of repetitions for high fidelity (default: 30)
        fidelity_ultra: Number of repetitions for ultra fidelity (default: 50)
        n_estimators: Number of trees in the random forest (default: 100)
        max_depth: Maximum depth of trees (default: None = unlimited)
        acquisition_fn: Acquisition function ('ei' for Expected Improvement, 'lcb' for Lower Confidence Bound)
    """

    def __init__(
        self,
        kernel: BoundKernel,
        args: Sequence[object],
        n_low_fidelity: int = 300,
        n_medium_fidelity: int = 100,
        n_high_fidelity: int = 30,
        n_ultra_fidelity: int = 10,
        fidelity_low: int = 10,
        fidelity_medium: int = 15,
        fidelity_high: int = 30,
        fidelity_ultra: int = 50,
        n_estimators: int = 100,
        max_depth: int | None = None,
        acquisition_fn: str = "ei",
    ) -> None:
        super().__init__(kernel, args)

        # Stage sizes
        self.n_low = n_low_fidelity
        self.n_medium = n_medium_fidelity
        self.n_high = n_high_fidelity
        self.n_ultra = n_ultra_fidelity

        # Fidelity levels (number of benchmark repetitions)
        self.fid_low = fidelity_low
        self.fid_medium = fidelity_medium
        self.fid_high = fidelity_high
        self.fid_ultra = fidelity_ultra

        # Random Forest hyperparameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth

        # Acquisition function
        self.acquisition_fn = acquisition_fn

        # Config encoder for converting flat configs to numeric arrays
        self.encoder = ConfigEncoder(self.config_gen)

        # Storage for evaluated configs at each fidelity
        self.evaluated_low: list[PopulationMember] = []
        self.evaluated_medium: list[PopulationMember] = []
        self.evaluated_high: list[PopulationMember] = []
        self.evaluated_ultra: list[PopulationMember] = []

    def _autotune(self):
        """
        Run multi-fidelity autotuning with Random Forest guidance.

        Returns:
            Best configuration found
        """
        from ..runtime.config import Config

        self.log("=" * 70)
        self.log("Multi-Fidelity Random Forest Autotuning")
        self.log("=" * 70)
        self.log(f"Stage sizes: {self.n_low} → {self.n_medium} → {self.n_high} → {self.n_ultra}")
        self.log(f"Fidelity levels: {self.fid_low} → {self.fid_medium} → {self.fid_high} → {self.fid_ultra} reps")
        self.log(f"Random Forest: {self.n_estimators} trees, max_depth={self.max_depth}")
        self.log("=" * 70)

        # Stage 1: Low-fidelity exploration
        self._stage_low_fidelity()

        # Stage 2: Medium-fidelity refinement
        self._stage_medium_fidelity()

        # Stage 3: High-fidelity validation
        self._stage_high_fidelity()

        # Stage 4: Ultra-fidelity final selection
        self._stage_ultra_fidelity()

        # Return best config from ultra-fidelity stage
        best = min(self.evaluated_ultra, key=lambda m: m.perf)
        self.log("=" * 70)
        self.log(f"✓ Best configuration: {best.perf:.4f} ms")
        self.log("=" * 70)

        return best.config

    def _stage_low_fidelity(self) -> None:
        """Stage 1: Broad exploration with low-fidelity evaluations."""
        self.log(f"\nStage 1: Low-fidelity exploration ({self.n_low} configs × {self.fid_low} reps)")

        # Generate truly random configurations (no default bias)
        candidates = [self.config_gen.random_flat() for _ in range(self.n_low)]
        members = [self.make_unbenchmarked(flat) for flat in candidates]

        # Benchmark all candidates
        self.evaluated_low = self._benchmark_population_at_fidelity(
            members, fidelity=self.fid_low, desc="Low-fidelity"
        )

        # Show top performers
        top_low = sorted(self.evaluated_low, key=lambda m: m.perf)[:5]
        self.log("\nTop 5 low-fidelity configs:")
        for i, m in enumerate(top_low):
            config = self.config_gen.unflatten(m.flat_values)
            block_sizes = config.config.get("block_sizes", "?")
            num_warps = config.config.get("num_warps", "?")
            self.log(f"  #{i+1}: {m.perf:.4f}ms - block_sizes={block_sizes}, num_warps={num_warps}")

    def _stage_medium_fidelity(self) -> None:
        """Stage 2: Medium-fidelity refinement with direct promotion + RF guidance."""
        self.log(f"\nStage 2: Medium-fidelity validation ({self.n_medium} configs × {self.fid_medium} reps)")

        # Mix direct promotion from low-fid with RF-based selection
        n_promote_direct = self.n_medium // 2
        n_from_rf = self.n_medium - n_promote_direct

        # Direct promotion: Take top performers from low-fidelity
        sorted_low = sorted(self.evaluated_low, key=lambda m: m.perf)
        direct_promote = [m.flat_values for m in sorted_low[:n_promote_direct]]

        # RF-based: Use Random Forest to find promising configs
        candidates_rf = self._select_by_rf_acquisition(
            n_from_rf,
            candidate_pool_size=max(2000, self.n_low * 2),
            use_multifidelity=False,
        )

        # Combine both strategies
        candidates = direct_promote + candidates_rf
        members = [self.make_unbenchmarked(flat) for flat in candidates]

        # Benchmark
        self.evaluated_medium = self._benchmark_population_at_fidelity(
            members, fidelity=self.fid_medium, desc="Medium-fidelity"
        )

        # Show top performers
        top_med = sorted(self.evaluated_medium, key=lambda m: m.perf)[:3]
        self.log("\nTop 3 medium-fidelity configs:")
        for i, m in enumerate(top_med):
            config = self.config_gen.unflatten(m.flat_values)
            block_sizes = config.config.get("block_sizes", "?")
            num_warps = config.config.get("num_warps", "?")
            self.log(f"  #{i+1}: {m.perf:.4f}ms - block_sizes={block_sizes}, num_warps={num_warps}")

    def _stage_high_fidelity(self) -> None:
        """Stage 3: High-fidelity validation with direct promotion + RF guidance."""
        self.log(f"\nStage 3: High-fidelity validation ({self.n_high} configs × {self.fid_high} reps)")

        # Combine low and medium fidelity results for multi-fidelity RF
        source = self.evaluated_low + self.evaluated_medium

        # Mix direct promotion with RF-based selection
        n_promote_direct = self.n_high // 2
        n_from_rf = self.n_high - n_promote_direct

        # Direct promotion: Take top performers from previous stages
        sorted_source = sorted(source, key=lambda m: m.perf)
        direct_promote = [m.flat_values for m in sorted_source[:n_promote_direct]]

        # RF-based: Use multi-fidelity RF
        candidates_rf = self._select_by_rf_acquisition(
            n_from_rf,
            candidate_pool_size=max(1000, len(source) * 2),
            use_multifidelity=True,
        )

        # Combine both strategies
        candidates = direct_promote + candidates_rf
        members = [self.make_unbenchmarked(flat) for flat in candidates]

        # Benchmark
        self.evaluated_high = self._benchmark_population_at_fidelity(
            members, fidelity=self.fid_high, desc="High-fidelity"
        )

        # Show top performers
        top_high = sorted(self.evaluated_high, key=lambda m: m.perf)[:3]
        self.log("\nTop 3 high-fidelity configs:")
        for i, m in enumerate(top_high):
            config = self.config_gen.unflatten(m.flat_values)
            block_sizes = config.config.get("block_sizes", "?")
            num_warps = config.config.get("num_warps", "?")
            self.log(f"  #{i+1}: {m.perf:.4f}ms - block_sizes={block_sizes}, num_warps={num_warps}")

    def _stage_ultra_fidelity(self) -> None:
        """Stage 4: Ultra-fidelity final validation."""
        self.log(f"\nStage 4: Ultra-fidelity validation ({self.n_ultra} configs × {self.fid_ultra} reps)")

        # Combine all previous results for final RF model
        all_source = self.evaluated_low + self.evaluated_medium + self.evaluated_high

        # Take top performers directly (no RF needed at final stage)
        sorted_source = sorted(all_source, key=lambda m: m.perf)
        candidates = [m.flat_values for m in sorted_source[: self.n_ultra]]
        members = [self.make_unbenchmarked(flat) for flat in candidates]

        # Benchmark with highest fidelity
        self.evaluated_ultra = self._benchmark_population_at_fidelity(
            members, fidelity=self.fid_ultra, desc="Ultra-fidelity"
        )

        # Show final results
        top_ultra = sorted(self.evaluated_ultra, key=lambda m: m.perf)[:3]
        self.log("\nTop 3 ultra-fidelity configs:")
        for i, m in enumerate(top_ultra):
            config = self.config_gen.unflatten(m.flat_values)
            block_sizes = config.config.get("block_sizes", "?")
            num_warps = config.config.get("num_warps", "?")
            self.log(f"  #{i+1}: {m.perf:.4f}ms - block_sizes={block_sizes}, num_warps={num_warps}")

    def _benchmark_population_at_fidelity(
        self,
        members: list[PopulationMember],
        fidelity: int,
        *,
        desc: str = "Benchmarking",
    ) -> list[PopulationMember]:
        """
        Benchmark a population at a specific fidelity level.

        Args:
            members: Population members to benchmark.
            fidelity: Number of repetitions.
            desc: Description for progress bar.

        Returns:
            The benchmarked population members.
        """
        # Store fidelity for benchmark_function to use
        self._current_fidelity = fidelity

        configs = [m.config for m in members]
        results = self.parallel_benchmark(list(configs), desc=desc)

        for member, (config_out, fn, perf, status) in zip(
            members, results, strict=True
        ):
            assert config_out is member.config
            member.perfs.append(perf)
            member.fidelities.append(fidelity)
            member.fn = fn
            member.status = status

        return members

    def benchmark_function(self, config: Any, fn: object, *, fidelity: int = 50) -> float:
        """Benchmark with specific fidelity."""
        # Use the fidelity set by _benchmark_population_at_fidelity if available
        actual_fidelity = getattr(self, "_current_fidelity", fidelity)
        return super().benchmark_function(config, fn, fidelity=actual_fidelity)  # type: ignore[no-untyped-call]

    def _select_by_rf_acquisition(
        self,
        n_select: int,
        candidate_pool_size: int = 1000,
        use_multifidelity: bool = False,
    ) -> list[list[int]]:
        """
        Select configurations using Random Forest acquisition function.

        Args:
            n_select: Number of configurations to select
            candidate_pool_size: Size of random candidate pool to score
            use_multifidelity: Whether to use multi-fidelity RF model

        Returns:
            List of selected flat configuration vectors
        """
        # Reserve 15% for pure random exploration (diversity sampling)
        n_random = max(1, int(n_select * 0.15))
        n_from_rf = n_select - n_random

        # Generate candidate pool (truly random, no default bias)
        candidate_pool = [self.config_gen.random_flat() for _ in range(candidate_pool_size)]
        X_candidates = np.array([self.encoder.encode(flat) for flat in candidate_pool])

        # Train Random Forest on evaluated configs
        if use_multifidelity:
            # Multi-fidelity: Train on both low and medium fidelity data
            # Filter out infinity values (failed configs)
            valid_low = [m for m in self.evaluated_low if m.perf != float("inf")]
            valid_medium = [m for m in self.evaluated_medium if m.perf != float("inf")]
            X_train = np.array(
                [self.encoder.encode(m.flat_values) for m in valid_low]
                + [self.encoder.encode(m.flat_values) for m in valid_medium]
            )
            y_train = np.array(
                [m.perf for m in valid_low] + [m.perf for m in valid_medium]
            )
        else:
            # Single-fidelity: Train only on low fidelity
            # Filter out infinity values (failed configs)
            valid_low = [m for m in self.evaluated_low if m.perf != float("inf")]
            X_train = np.array([self.encoder.encode(m.flat_values) for m in valid_low])
            y_train = np.array([m.perf for m in valid_low])

        # Train Random Forest
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
        )
        rf.fit(X_train, y_train)

        # Get predictions and uncertainties
        # For RF, uncertainty = std across trees
        predictions = np.array([tree.predict(X_candidates) for tree in rf.estimators_])
        mu = predictions.mean(axis=0)  # Mean prediction
        sigma = predictions.std(axis=0)  # Uncertainty estimate

        # Compute acquisition scores
        best_so_far = y_train.min()

        if self.acquisition_fn == "ei":
            # Expected Improvement
            scores = expected_improvement(mu, sigma, best_so_far, xi=0.12)
        else:
            # Lower Confidence Bound (for minimization)
            lcb = lower_confidence_bound(mu, sigma, beta=3.5)
            scores = -lcb  # Negate for maximization

        # Select top N from acquisition
        top_indices = np.argsort(scores)[-n_from_rf:][::-1]
        selected_from_rf = [candidate_pool[i] for i in top_indices]

        # Add pure random samples for diversity
        random_samples = [self.config_gen.random_flat() for _ in range(n_random)]

        return selected_from_rf + random_samples


# ============================================================================
# Acquisition Functions
# ============================================================================


def expected_improvement(
    mu: np.ndarray, sigma: np.ndarray, best_so_far: float, xi: float = 0.12
) -> np.ndarray:
    """
    Compute Expected Improvement acquisition function.

    Args:
        mu: Mean predictions
        sigma: Standard deviation predictions
        best_so_far: Best observed value so far
        xi: Exploration parameter (higher = more exploration)

    Returns:
        Expected improvement scores
    """
    from scipy.stats import norm

    # Improvement over current best (with exploration bonus)
    improvement = best_so_far - mu - xi

    # Standardize
    Z = improvement / (sigma + 1e-9)

    # Expected improvement formula
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

    # Set to 0 where sigma is very small
    ei[sigma < 1e-9] = 0.0

    return ei


def lower_confidence_bound(mu: np.ndarray, sigma: np.ndarray, beta: float = 3.5) -> np.ndarray:
    """
    Compute Lower Confidence Bound acquisition function (for minimization).

    Args:
        mu: Mean predictions
        sigma: Standard deviation predictions
        beta: Exploration parameter (higher = more exploration)

    Returns:
        Lower confidence bound scores
    """
    return mu - beta * sigma
