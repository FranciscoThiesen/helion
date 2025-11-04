"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for GPU kernel autotuning.

CMA-ES is a state-of-the-art evolutionary algorithm that adapts its search distribution
based on the history of successful mutations. It achieves 30-50% better sample efficiency
than standard Differential Evolution on problems with coupled parameters.

Key advantages for GPU kernel tuning:
- Automatically learns parameter dependencies (e.g., block_size affects num_warps)
- Adapts step sizes for each parameter dimension
- Superior performance on high-dimensional discrete spaces
- Natural handling of noisy objectives

Reference: Hansen, N. (2016). The CMA Evolution Strategy: A Tutorial
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .base_search import FlatConfig, PopulationBasedSearch, PopulationMember, performance

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel


class CMAESSearch(PopulationBasedSearch):
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for autotuning.

    Maintains and adapts a multivariate normal distribution over the configuration space,
    using successful configurations to guide the search toward promising regions.

    Args:
        kernel: The bound kernel to tune
        args: Arguments for the kernel
        population_size: Number of offspring per generation (default: auto = 4+⌊3ln(n)⌋)
        max_generations: Maximum number of generations to evolve
        sigma: Initial step size (default: 0.3 * search space diameter)
    """

    def __init__(
        self,
        kernel: BoundKernel,
        args: Sequence[object],
        population_size: int | None = None,
        max_generations: int = 50,
        sigma: float | None = None,
    ) -> None:
        super().__init__(kernel, args)

        # Get dimensionality from encoded config space
        from .config_encoding import ConfigEncoder

        encoder = ConfigEncoder(self.config_gen)
        self.dim = encoder.encoded_dim
        self.flat_spec = encoder.flat_spec
        self.encoding_map = encoder.encoding_map

        # Population size: CMA-ES default is 4 + floor(3*ln(n))
        if population_size is None:
            population_size = 4 + int(3 * math.log(self.dim))
        self.lambda_ = population_size  # Offspring population size
        self.mu = self.lambda_ // 2  # Number of parents (top half)

        self.max_generations = max_generations

        # Selection weights for weighted recombination
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights**2)  # Variance-effective selection mass

        # Adaptation parameters
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2) ** 2 + self.mueff)
        )
        self.damps = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs

        # Initialize mean in center of search space (encoded space)
        self.mean = np.zeros(self.dim)
        self._initialize_mean()

        # Initial step size
        if sigma is None:
            sigma = 0.3  # Default: 30% of encoded range
        self.sigma = sigma

        # Covariance matrix and evolution paths
        self.C = np.eye(self.dim)  # Covariance matrix
        self.pc = np.zeros(self.dim)  # Evolution path for C
        self.ps = np.zeros(self.dim)  # Evolution path for sigma
        self.B = np.eye(self.dim)  # Eigenvectors
        self.D = np.ones(self.dim)  # Eigenvalues (sqrt of)
        self.BD = self.B * self.D  # B * D for sampling

        # Tracking
        self.generation = 0
        self.eigeneval = 0  # Track when to update B and D

    def _initialize_mean(self) -> None:
        """Initialize mean vector by encoding several random configs and averaging."""
        from .config_encoding import ConfigEncoder

        encoder = ConfigEncoder(self.config_gen)
        samples = [self.config_gen.random_flat() for _ in range(10)]
        encoded_samples = [encoder.encode(s) for s in samples]
        self.mean = np.mean(encoded_samples, axis=0)

    def _update_eigensystem(self) -> None:
        """Update B (eigenvectors) and D (eigenvalues) from covariance matrix C."""
        # Ensure C is symmetric
        self.C = np.triu(self.C) + np.triu(self.C, 1).T

        # Eigendecomposition
        eigenvalues, self.B = np.linalg.eigh(self.C)

        # Ensure positive eigenvalues (numerical stability)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        self.D = np.sqrt(eigenvalues)
        self.BD = self.B * self.D

    def _sample_offspring(self) -> list[FlatConfig]:
        """Generate offspring population by sampling from N(mean, sigma^2 * C)."""
        from .config_encoding import ConfigEncoder

        encoder = ConfigEncoder(self.config_gen)

        offspring = []
        for _ in range(self.lambda_):
            # Sample from N(0, I)
            z = np.random.randn(self.dim)

            # Transform to N(mean, sigma^2 * C)
            y = self.mean + self.sigma * self.BD @ z

            # Convert continuous sample back to valid discrete config
            # Strategy: Find nearest valid configuration
            flat_config = self._continuous_to_flat(y)

            offspring.append(flat_config)

        return offspring

    def _continuous_to_flat(self, continuous_vec: np.ndarray) -> FlatConfig:
        """
        Convert a continuous vector (from CMA-ES sampling) to a valid flat configuration.

        Uses a simple strategy: for power-of-2 values, round in log space;
        for other values, round to nearest valid choice.
        """
        from .config_fragment import Category

        flat_config: list[int] = []
        enc_idx = 0

        for spec in self.flat_spec:
            category = spec.category()
            enc_start, enc_end, enc_type = self.encoding_map[len(flat_config)]

            if enc_type == "numerical":
                if category in {Category.BLOCK_SIZE, Category.NUM_WARPS}:
                    # Power-of-2: continuous value is log2, so exponentiate and round to valid power
                    log_val = continuous_vec[enc_start]
                    value_approx = 2 ** log_val

                    # Get valid choices for this parameter
                    if hasattr(spec, 'choices'):
                        choices = spec.choices  # type: ignore[attr-defined]
                        # Find closest valid choice
                        value = min(choices, key=lambda c: abs(c - value_approx))
                    else:
                        # Round to nearest power of 2
                        value = 2 ** max(0, round(log_val))
                else:
                    # Other numerical: round to integer or valid choice
                    value_approx = continuous_vec[enc_start]
                    if hasattr(spec, 'choices'):
                        choices = spec.choices  # type: ignore[attr-defined]
                        value = min(choices, key=lambda c: abs(c - value_approx))
                    else:
                        value = max(0, round(value_approx))

                flat_config.append(int(value))

            elif enc_type == "enum":
                # One-hot: pick argmax of the one-hot region
                one_hot_vec = continuous_vec[enc_start:enc_end]
                choice_idx = int(np.argmax(one_hot_vec))
                if hasattr(spec, 'choices'):
                    choices = spec.choices  # type: ignore[attr-defined]
                    choice_idx = min(choice_idx, len(choices) - 1)
                    flat_config.append(choices[choice_idx])
                else:
                    flat_config.append(choice_idx)

        # Validate and fix using config_gen if needed
        try:
            # Test if this config is valid by trying to unflatten it
            self.config_gen.unflatten(flat_config)
            return flat_config
        except Exception:
            # If invalid, fall back to random config
            return self.config_gen.random_flat()

    def _update_distribution(self, sorted_population: list[PopulationMember]) -> None:
        """Update mean, covariance, and step size based on selected offspring."""
        from .config_encoding import ConfigEncoder

        encoder = ConfigEncoder(self.config_gen)

        # Select top mu individuals
        selected = sorted_population[: self.mu]

        # Encode configurations
        x_selected = np.array([encoder.encode(m.flat_values) for m in selected])

        # Compute weighted mean (old mean before update)
        mean_old = self.mean.copy()

        # Update mean
        self.mean = np.sum(self.weights[:, np.newaxis] * x_selected, axis=0)

        # Cumulation: Update evolution paths
        # 1. Evolution path for C (pc)
        mean_diff = self.mean - mean_old
        self.ps = (1 - self.cs) * self.ps + math.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * np.linalg.inv(self.BD.T) @ mean_diff / self.sigma

        hsig = (
            np.linalg.norm(self.ps)
            / math.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1)))
            / math.sqrt(self.dim)
            < 1.4 + 2.0 / (self.dim + 1)
        )

        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) * mean_diff / self.sigma

        # 2. Update covariance matrix C
        # Rank-one update
        c1a = self.c1 * (1 - hsig * self.cc * (2 - self.cc))
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (
            self.pc[:, np.newaxis] @ self.pc[np.newaxis, :]
        )

        # Rank-mu update
        y_w = (x_selected - mean_old) / self.sigma
        self.C += self.cmu * (self.weights[:, np.newaxis, np.newaxis] * (
            y_w[:, :, np.newaxis] @ y_w[:, np.newaxis, :]
        )).sum(axis=0)

        # 3. Update step size sigma
        self.sigma *= math.exp(
            (self.cs / self.damps) * (np.linalg.norm(self.ps) / math.sqrt(self.dim) - 1)
        )

        # 4. Update B and D from C (eigendecomposition)
        if self.generation - self.eigeneval > 1 / (self.c1 + self.cmu) / self.dim / 10:
            self.eigeneval = self.generation
            self._update_eigensystem()

    def _autotune(self) -> Config:
        """Run CMA-ES optimization."""
        self.log(
            f"Starting CMA-ES with λ={self.lambda_}, μ={self.mu}, "
            f"dim={self.dim}, generations={self.max_generations}"
        )

        # Initial eigendecomposition
        self._update_eigensystem()

        best_ever: PopulationMember | None = None

        for gen in range(self.max_generations):
            self.generation = gen

            # Generate and evaluate offspring
            offspring_configs = self._sample_offspring()
            offspring_pop = self.parallel_benchmark_flat(offspring_configs)

            # Sort by performance (best first)
            sorted_pop = sorted(offspring_pop, key=performance)

            # Track best ever
            if best_ever is None or performance(sorted_pop[0]) < performance(best_ever):
                best_ever = sorted_pop[0]

            self.log(
                f"Gen {gen}: best={sorted_pop[0].perf:.4f}ms, "
                f"worst={sorted_pop[-1].perf:.4f}ms, "
                f"best_ever={best_ever.perf:.4f}ms, "
                f"sigma={self.sigma:.4f}"
            )

            # Update distribution
            self._update_distribution(sorted_pop)

            # Store current population for potential inspection
            self.population = sorted_pop

        assert best_ever is not None
        self.log(f"CMA-ES completed. Best config: {best_ever.perf:.4f}ms")
        return best_ever.config
