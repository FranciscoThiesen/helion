"""
Tree-structured Parzen Estimator (TPE) Search for Autotuning.

TPE is a sequential model-based optimization algorithm that has shown excellent
performance on discrete/mixed-variable optimization problems. It's the algorithm
behind Optuna and Hyperopt.

Key advantages:
1. Excellent for discrete and categorical parameters
2. No need for distance metrics or kernels
3. Naturally handles mixed-type search spaces
4. Computationally efficient
5. Proven track record in hyperparameter optimization

Reference:
Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011).
"Algorithms for Hyper-Parameter Optimization." NeurIPS.

Author: Francisco Geiman Thiesen
Date: 2025-11-05
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

from .base_search import PopulationBasedSearch, PopulationMember
from .config_encoding import ConfigEncoder

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.kernel import BoundKernel
    from .config_generation import FlatConfig


class TreeStructuredParzenEstimator(PopulationBasedSearch):
    """
    TPE search strategy for GPU kernel autotuning.

    TPE models the distribution of good and bad configurations separately:
    - l(x): Distribution of configurations that led to good performance
    - g(x): Distribution of configurations that led to bad performance

    The acquisition function is: EI(x) = (g(x) - l(x)) / g(x)
    We want to maximize this, which means sampling from l(x).

    Args:
        kernel: The bound kernel to tune
        args: Arguments for the kernel
        n_initial: Number of random configurations to start with
        n_iterations: Number of TPE iterations
        gamma: Quantile to separate good/bad configs (default: 0.25 = top 25%)
        n_ei_candidates: Number of candidates to generate for EI optimization
    """

    def __init__(
        self,
        kernel: BoundKernel,
        args: Sequence[object],
        n_initial: int = 100,
        n_iterations: int = 1500,
        gamma: float = 0.25,
        n_ei_candidates: int = 24,
    ) -> None:
        super().__init__(kernel, args)

        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.gamma = gamma  # Quantile threshold for good configs
        self.n_ei_candidates = n_ei_candidates

        # Config encoder for converting flat configs to numeric arrays
        self.encoder = ConfigEncoder(self.config_gen)

        # Track all evaluated configs
        self.observations: list[PopulationMember] = []

    def _autotune(self):
        """
        Run TPE autotuning.

        Returns:
            Best configuration found
        """
        self.log("=" * 70)
        self.log("Tree-structured Parzen Estimator (TPE) Autotuning")
        self.log("=" * 70)
        self.log(f"Initial random sampling: {self.n_initial} configs")
        self.log(f"TPE iterations: {self.n_iterations}")
        self.log(f"Gamma (good/bad threshold): {self.gamma} (top {self.gamma*100:.0f}%)")
        self.log(f"EI candidates per iteration: {self.n_ei_candidates}")
        self.log("=" * 70)

        # Phase 1: Random exploration
        self._initial_sampling()

        # Phase 2: TPE-guided search
        for iteration in range(self.n_iterations):
            self._tpe_iteration(iteration)

        # Return best config
        best = min(self.observations, key=lambda m: m.perf)
        self.log("=" * 70)
        self.log(f"✓ Best configuration: {best.perf:.4f} ms")
        self.log(f"Total evaluations: {len(self.observations)}")
        self.log("=" * 70)

        return best.config

    def _initial_sampling(self) -> None:
        """Phase 1: Random sampling to seed the model."""
        self.log(f"\nPhase 1: Random sampling ({self.n_initial} configs)")

        configs = [self.config_gen.random_flat() for _ in range(self.n_initial)]
        members = self.parallel_benchmark_flat(configs)

        for member in members:
            if member.perf != float("inf"):
                self.observations.append(member)
                self.population.append(member)

        best_perf = min(m.perf for m in self.observations)
        self.log(f"Initial exploration complete: best={best_perf:.4f} ms, ok={len(self.observations)}")

    def _tpe_iteration(self, iteration: int) -> None:
        """Single TPE iteration: build models and sample next config."""

        # Sort observations by performance
        sorted_obs = sorted(self.observations, key=lambda m: m.perf)

        # Split into good (l) and bad (g) based on gamma quantile
        split_idx = max(1, int(self.gamma * len(sorted_obs)))
        good_configs = [m.flat_values for m in sorted_obs[:split_idx]]
        bad_configs = [m.flat_values for m in sorted_obs[split_idx:]]

        # Build Parzen estimators for good and bad distributions
        l_model = self._build_parzen_estimator(good_configs)
        g_model = self._build_parzen_estimator(bad_configs)

        # Generate candidates by sampling from l(x)
        candidates = []
        for _ in range(self.n_ei_candidates):
            candidate = self._sample_from_model(l_model, good_configs)
            candidates.append(candidate)

        # Evaluate EI for each candidate: EI(x) ∝ g(x) / l(x)
        # We want high g(x) and low l(x), so we maximize g/l ratio
        best_candidate = None
        best_ei = -float("inf")

        for candidate in candidates:
            l_prob = self._evaluate_model(l_model, candidate, good_configs)
            g_prob = self._evaluate_model(g_model, candidate, bad_configs)

            # Expected Improvement: higher g/l is better
            # Add small epsilon to avoid division by zero
            ei = (g_prob + 1e-12) / (l_prob + 1e-12)

            if ei > best_ei:
                best_ei = ei
                best_candidate = candidate

        # Evaluate the best candidate
        if best_candidate is not None:
            try:
                member = self.benchmark_flat(best_candidate)
                if member.perf != float("inf"):
                    self.observations.append(member)
                    self.population.append(member)

                    # Log progress periodically
                    if (iteration + 1) % 100 == 0:
                        best_so_far = min(m.perf for m in self.observations)
                        self.log(
                            f"Iteration {iteration+1}/{self.n_iterations}: "
                            f"best={best_so_far:.4f} ms, evaluated={len(self.observations)}"
                        )
            except Exception:
                pass  # Skip failed configs

    def _make_hashable(self, val):
        """Convert unhashable types (lists) to hashable types (tuples)."""
        if isinstance(val, list):
            return tuple(self._make_hashable(v) for v in val)
        return val

    def _build_parzen_estimator(self, configs: list[FlatConfig]) -> dict:
        """
        Build a simple Parzen estimator for each parameter.

        For discrete parameters, we use categorical distributions (histograms).
        For continuous parameters, we use kernel density estimation.

        Returns:
            Dictionary mapping parameter index to distribution
        """
        model = {}
        n_params = len(self.config_gen.flat_spec)

        for param_idx in range(n_params):
            values = [config[param_idx] for config in configs]

            # Build histogram (categorical distribution)
            unique_values = {}
            for val in values:
                # Convert to hashable type for dictionary key
                hashable_val = self._make_hashable(val)
                if hashable_val not in unique_values:
                    unique_values[hashable_val] = 0
                unique_values[hashable_val] += 1

            # Normalize to probabilities and add smoothing
            total = len(values)
            n_unique = len(unique_values)

            # Laplace smoothing: add pseudo-count to each possibility
            smoothing = 1.0 / max(n_unique, 10)

            probs = {}
            for val, count in unique_values.items():
                probs[val] = (count + smoothing) / (total + smoothing * n_unique)

            model[param_idx] = {
                "type": "categorical",
                "probs": probs,
                "seen_values": list(unique_values.keys()),
            }

        return model

    def _unhashable(self, val):
        """Convert hashable types (tuples) back to unhashable types (lists)."""
        if isinstance(val, tuple):
            return [self._unhashable(v) for v in val]
        return val

    def _sample_from_model(self, model: dict, reference_configs: list[FlatConfig]) -> FlatConfig:
        """
        Sample a configuration from the Parzen estimator.

        Args:
            model: Distribution model for each parameter
            reference_configs: Reference configs for fallback

        Returns:
            Sampled configuration
        """
        config = []

        for param_idx in range(len(self.config_gen.flat_spec)):
            param_model = model[param_idx]

            if param_model["type"] == "categorical":
                # Sample from categorical distribution
                probs = param_model["probs"]
                values = list(probs.keys())
                weights = [probs[v] for v in values]

                # Sample and convert back to original type
                sampled_value = random.choices(values, weights=weights, k=1)[0]
                # Convert tuples back to lists if needed
                sampled_value = self._unhashable(sampled_value)
                config.append(sampled_value)
            else:
                # Fallback: sample from reference configs
                ref_config = random.choice(reference_configs)
                config.append(ref_config[param_idx])

        return config

    def _evaluate_model(self, model: dict, config: FlatConfig, reference_configs: list[FlatConfig]) -> float:
        """
        Evaluate the probability of a configuration under the Parzen estimator.

        Args:
            model: Distribution model
            config: Configuration to evaluate
            reference_configs: Reference configs for smoothing

        Returns:
            Log probability
        """
        log_prob = 0.0

        for param_idx in range(len(config)):
            param_model = model[param_idx]
            value = config[param_idx]

            if param_model["type"] == "categorical":
                probs = param_model["probs"]

                # Convert to hashable for lookup
                hashable_value = self._make_hashable(value)

                if hashable_value in probs:
                    # Known value
                    log_prob += np.log(probs[hashable_value] + 1e-12)
                else:
                    # Unseen value: assign small probability
                    n_unique = len(probs)
                    smoothing = 1.0 / max(n_unique, 10)
                    log_prob += np.log(smoothing + 1e-12)

        return log_prob

    def __repr__(self) -> str:
        return (
            f"TPESearch(n_initial={self.n_initial}, "
            f"n_iterations={self.n_iterations}, "
            f"gamma={self.gamma}, "
            f"n_ei_candidates={self.n_ei_candidates})"
        )
