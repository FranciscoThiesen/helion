"""
Particle Swarm Optimization (PSO) for GPU kernel autotuning.

PSO is a swarm intelligence algorithm inspired by bird flocking and fish schooling.
Each "particle" represents a configuration that moves through the search space,
influenced by its own best position and the global best position.

Key advantages for GPU kernel tuning:
- Excellent at exploring diverse regions of the search space
- Natural handling of high-dimensional problems
- Good balance between exploration and exploitation
- Can escape local minima through swarm dynamics

Reference: PSO has been successfully applied to GPU kernel autotuning in CLTune
and other frameworks, showing good performance on discrete parameter spaces.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

from .base_search import FlatConfig, PopulationBasedSearch, PopulationMember, performance

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel


class ParticleSwarmSearch(PopulationBasedSearch):
    """
    Particle Swarm Optimization for GPU kernel autotuning.

    Each particle maintains:
    - Current position (configuration)
    - Velocity (direction of movement in parameter space)
    - Personal best position
    - Awareness of global best position

    Args:
        kernel: The bound kernel to tune
        args: Arguments for the kernel
        swarm_size: Number of particles (default: 30)
        max_iterations: Number of iterations to run (default: 40)
        w: Inertia weight for velocity (default: 0.7, decreases over time)
        c1: Cognitive parameter (personal best influence, default: 1.5)
        c2: Social parameter (global best influence, default: 1.5)
        v_max: Maximum velocity as fraction of parameter range (default: 0.2)
    """

    def __init__(
        self,
        kernel: BoundKernel,
        args: Sequence[object],
        swarm_size: int = 30,
        max_iterations: int = 40,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        v_max: float = 0.2,
    ) -> None:
        super().__init__(kernel, args)

        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.w_initial = w
        self.w = w
        self.c1 = c1  # Cognitive (personal best)
        self.c2 = c2  # Social (global best)
        self.v_max = v_max

        # Get flat spec for parameter-aware operations
        self.flat_spec = self.config_gen.flat_spec
        self.dim = len(self.config_gen.random_flat())

        # Initialize encoder for continuous space operations
        from .config_encoding import ConfigEncoder

        self.encoder = ConfigEncoder(self.config_gen)
        self.encoded_dim = self.encoder.encoded_dim

        # Particle state
        self.positions: list[np.ndarray] = []  # Current positions (encoded space)
        self.velocities: list[np.ndarray] = []  # Velocities (encoded space)
        self.personal_best_positions: list[np.ndarray] = []
        self.personal_best_scores: list[float] = []
        self.global_best_position: np.ndarray | None = None
        self.global_best_score: float = float("inf")
        self.global_best_config: FlatConfig | None = None

    def _initialize_swarm(self) -> None:
        """
        Initialize particle positions and velocities.

        Positions are initialized to random valid configurations.
        Velocities are initialized to small random values.
        """
        # Generate and evaluate initial configurations
        configs = self.config_gen.random_population_flat(self.swarm_size)
        members = self.parallel_benchmark_flat(configs)

        # Encode positions
        for member in members:
            encoded_pos = self.encoder.encode(member.flat_values)
            self.positions.append(encoded_pos)

            # Initialize velocity to small random values
            velocity = np.random.uniform(
                -self.v_max, self.v_max, size=self.encoded_dim
            )
            self.velocities.append(velocity)

            # Initialize personal best
            self.personal_best_positions.append(encoded_pos.copy())
            self.personal_best_scores.append(member.perf)

            # Update global best
            if member.perf < self.global_best_score:
                self.global_best_score = member.perf
                self.global_best_position = encoded_pos.copy()
                self.global_best_config = member.flat_values

        self.log(f"Swarm initialized. Global best: {self.global_best_score:.4f}ms")

    def _encoded_to_flat(self, encoded: np.ndarray) -> FlatConfig:
        """
        Convert an encoded position to a valid flat configuration.

        Similar to CMA-ES, we decode and round to nearest valid values.
        """
        from .config_fragment import Category

        flat_config: list[int] = []

        for spec_idx, spec in enumerate(self.flat_spec):
            category = spec.category()
            enc_start, enc_end, enc_type = self.encoder.encoding_map[spec_idx]

            if enc_type == "numerical":
                if category in {Category.BLOCK_SIZE, Category.NUM_WARPS}:
                    # Power-of-2: exponentiate log value
                    log_val = encoded[enc_start]
                    value_approx = 2**log_val

                    if hasattr(spec, "choices"):
                        choices = spec.choices  # type: ignore[attr-defined]
                        value = min(choices, key=lambda c: abs(c - value_approx))
                    else:
                        value = 2 ** max(0, round(log_val))

                    flat_config.append(int(value))
                else:
                    # Other numerical
                    value_approx = encoded[enc_start]
                    if hasattr(spec, "choices"):
                        choices = spec.choices  # type: ignore[attr-defined]
                        value = min(choices, key=lambda c: abs(c - value_approx))
                    else:
                        value = max(0, round(value_approx))
                    flat_config.append(int(value))

            elif enc_type == "enum":
                # One-hot: pick argmax
                one_hot_vec = encoded[enc_start:enc_end]
                choice_idx = int(np.argmax(one_hot_vec))
                if hasattr(spec, "choices"):
                    choices = spec.choices  # type: ignore[attr-defined]
                    choice_idx = min(choice_idx, len(choices) - 1)
                    flat_config.append(choices[choice_idx])
                else:
                    flat_config.append(choice_idx)

        # Validate
        try:
            self.config_gen.unflatten(flat_config)
            return flat_config
        except Exception:
            # Fall back to random if invalid
            return self.config_gen.random_flat()

    def _update_velocities_and_positions(self, iteration: int) -> None:
        """
        Update particle velocities and positions using PSO update rules.

        Velocity update:
        v = w*v + c1*r1*(p_best - x) + c2*r2*(g_best - x)

        Position update:
        x = x + v

        Args:
            iteration: Current iteration number (for inertia weight decay)
        """
        # Adaptive inertia weight: decrease over time for exploitation
        progress = iteration / self.max_iterations
        self.w = self.w_initial * (1 - 0.5 * progress)  # Decay from w_initial to 0.5*w_initial

        assert self.global_best_position is not None

        for i in range(self.swarm_size):
            # Random factors for stochasticity
            r1 = np.random.rand(self.encoded_dim)
            r2 = np.random.rand(self.encoded_dim)

            # Velocity update
            cognitive = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
            social = self.c2 * r2 * (self.global_best_position - self.positions[i])

            self.velocities[i] = self.w * self.velocities[i] + cognitive + social

            # Clamp velocity
            self.velocities[i] = np.clip(
                self.velocities[i],
                -self.v_max * 10,  # Allow larger velocity magnitude in encoded space
                self.v_max * 10,
            )

            # Position update
            self.positions[i] = self.positions[i] + self.velocities[i]

    def _evaluate_swarm(self) -> list[PopulationMember]:
        """
        Evaluate all particle positions.

        Returns:
            List of evaluated population members
        """
        # Convert positions to flat configs
        configs = [self._encoded_to_flat(pos) for pos in self.positions]

        # Benchmark all configs
        members = self.parallel_benchmark_flat(configs)

        return members

    def _update_personal_and_global_best(self, members: list[PopulationMember]) -> None:
        """
        Update personal best and global best based on current evaluations.

        Args:
            members: Evaluated particle configurations
        """
        for i, member in enumerate(members):
            # Update personal best
            if member.perf < self.personal_best_scores[i]:
                self.personal_best_scores[i] = member.perf
                self.personal_best_positions[i] = self.encoder.encode(member.flat_values)

            # Update global best
            if member.perf < self.global_best_score:
                self.global_best_score = member.perf
                self.global_best_position = self.encoder.encode(member.flat_values)
                self.global_best_config = member.flat_values

    def _autotune(self) -> Config:
        """Run Particle Swarm Optimization."""
        self.log(
            f"Starting PSO with swarm_size={self.swarm_size}, "
            f"iterations={self.max_iterations}, w={self.w_initial}, "
            f"c1={self.c1}, c2={self.c2}"
        )

        # Initialize swarm
        self._initialize_swarm()

        # Main PSO loop
        for iteration in range(self.max_iterations):
            # Update velocities and positions
            self._update_velocities_and_positions(iteration)

            # Evaluate new positions
            members = self._evaluate_swarm()

            # Update best positions
            self._update_personal_and_global_best(members)

            # Store population for inspection
            self.population = members

            # Logging
            best_in_iter = min(members, key=performance)
            worst_in_iter = max(members, key=performance)
            avg_perf = np.mean([m.perf for m in members])

            self.log(
                f"Iter {iteration}: global_best={self.global_best_score:.4f}ms, "
                f"iter_best={best_in_iter.perf:.4f}ms, "
                f"avg={avg_perf:.4f}ms, w={self.w:.3f}"
            )

        # Return global best
        assert self.global_best_config is not None
        best_config = self.config_gen.unflatten(self.global_best_config)
        self.log(f"PSO completed. Best config: {self.global_best_score:.4f}ms")
        return best_config.config
