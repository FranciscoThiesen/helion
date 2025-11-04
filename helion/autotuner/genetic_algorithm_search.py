"""
Enhanced Genetic Algorithm for GPU kernel autotuning.

This implementation is specifically designed for mixed discrete-continuous parameter
spaces typical of GPU kernels. Research shows GAs demonstrate superior performance on
such spaces compared to continuous-only optimizers.

Key features:
- Tournament selection with elite preservation
- Smart crossover respecting parameter dependencies
- Adaptive mutation with parameter-aware strategies
- Memetic enhancement: local search on best individuals
- Diversity maintenance to avoid premature convergence

Reference: Genetic algorithms excel on mixed discrete-continuous optimization problems
common in GPU kernel parameter tuning (block sizes, warps, pipeline stages).
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


class GeneticAlgorithmSearch(PopulationBasedSearch):
    """
    Enhanced Genetic Algorithm for GPU kernel autotuning.

    Uses tournament selection, multi-point crossover, adaptive mutation, and
    elite preservation to efficiently search the discrete parameter space.

    Args:
        kernel: The bound kernel to tune
        args: Arguments for the kernel
        population_size: Size of the population (default: 50)
        max_generations: Number of generations to evolve (default: 40)
        tournament_size: Number of individuals in tournament selection (default: 3)
        crossover_rate: Probability of crossover (default: 0.8)
        mutation_rate: Initial probability of mutation per gene (default: 0.1)
        elite_size: Number of elite individuals to preserve (default: 2)
        local_search_freq: How often to apply local search (every N generations, default: 5)
    """

    def __init__(
        self,
        kernel: BoundKernel,
        args: Sequence[object],
        population_size: int = 50,
        max_generations: int = 40,
        tournament_size: int = 3,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elite_size: int = 2,
        local_search_freq: int = 5,
    ) -> None:
        super().__init__(kernel, args)

        self.population_size = population_size
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.initial_mutation_rate = mutation_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.local_search_freq = local_search_freq

        # Get flat spec for parameter-aware operations
        self.flat_spec = self.config_gen.flat_spec
        self.genome_length = len(self.config_gen.random_flat())

        # Track diversity for adaptive parameters
        self.diversity_history: list[float] = []

    def _initialize_population(self) -> list[PopulationMember]:
        """
        Create initial population with diverse configurations.

        Uses Latin Hypercube-like sampling to ensure good coverage.
        """
        # Generate 2x population size and keep the best half (like DE)
        oversized_pop = self.parallel_benchmark_flat(
            self.config_gen.random_population_flat(self.population_size * 2)
        )
        sorted_pop = sorted(oversized_pop, key=performance)
        return sorted_pop[: self.population_size]

    def _tournament_selection(self, population: list[PopulationMember]) -> PopulationMember:
        """
        Select an individual via tournament selection.

        Args:
            population: Current population

        Returns:
            Selected individual
        """
        tournament = random.sample(population, self.tournament_size)
        return min(tournament, key=performance)

    def _crossover(self, parent1: FlatConfig, parent2: FlatConfig) -> tuple[FlatConfig, FlatConfig]:
        """
        Perform multi-point crossover with parameter-aware strategy.

        For GPU parameters, use 2-point crossover to preserve parameter groups
        (e.g., block_sizes often work together).

        Args:
            parent1: First parent configuration
            parent2: Second parent configuration

        Returns:
            Two offspring configurations
        """
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]

        # Use 2-point crossover
        length = len(parent1)
        point1 = random.randint(0, length - 1)
        point2 = random.randint(point1, length)

        offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

        return offspring1, offspring2

    def _mutate(self, config: FlatConfig) -> FlatConfig:
        """
        Perform adaptive mutation on a configuration.

        For each gene (parameter), mutate with probability mutation_rate.
        Mutation strategy depends on parameter type:
        - Power-of-2 values: shift by ±1 or ±2 in log space
        - Enums: random valid choice
        - Integers: ±1 or random

        Args:
            config: Configuration to mutate

        Returns:
            Mutated configuration
        """
        from .config_fragment import Category

        mutated = config[:]

        for i, spec in enumerate(self.flat_spec):
            if random.random() < self.mutation_rate:
                category = spec.category()

                if category in {Category.BLOCK_SIZE, Category.NUM_WARPS}:
                    # Power-of-2: shift in log space
                    current_val = config[i]
                    if hasattr(spec, "choices"):
                        choices = spec.choices  # type: ignore[attr-defined]
                        try:
                            current_idx = choices.index(current_val)
                            # Shift by ±1 or ±2
                            shift = random.choice([-2, -1, 1, 2])
                            new_idx = max(0, min(len(choices) - 1, current_idx + shift))
                            mutated[i] = choices[new_idx]
                        except ValueError:
                            mutated[i] = random.choice(choices)
                    else:
                        # Random power of 2
                        log_val = int(np.log2(max(1, current_val)))
                        new_log = max(0, log_val + random.choice([-1, 0, 1]))
                        mutated[i] = 2**new_log

                elif hasattr(spec, "choices"):
                    # Enum or categorical: random choice
                    choices = spec.choices  # type: ignore[attr-defined]
                    mutated[i] = random.choice(choices)

                else:
                    # Other types (integers, lists, etc.)
                    current_val = config[i]

                    # If it's a list or complex type, just keep it or randomly change via choices
                    if isinstance(current_val, (list, tuple)):
                        if hasattr(spec, "choices"):
                            mutated[i] = random.choice(spec.choices)  # type: ignore[attr-defined]
                        else:
                            mutated[i] = current_val  # Keep as is
                    elif isinstance(current_val, int):
                        # Integer: small perturbation or random
                        if random.random() < 0.7:
                            mutated[i] = max(0, current_val + random.choice([-1, 0, 1]))
                        else:
                            if hasattr(spec, "min") and hasattr(spec, "max"):
                                mutated[i] = random.randint(spec.min, spec.max)  # type: ignore[attr-defined]
                            else:
                                mutated[i] = current_val
                    else:
                        # Other types: keep as is or use choices if available
                        if hasattr(spec, "choices"):
                            mutated[i] = random.choice(spec.choices)  # type: ignore[attr-defined]
                        else:
                            mutated[i] = current_val

        # Validate the mutated configuration
        try:
            self.config_gen.unflatten(mutated)
            return mutated
        except Exception:
            # If invalid, return original
            return config

    def _local_search(self, config: FlatConfig) -> FlatConfig:
        """
        Apply local search (hill climbing) to a configuration.

        Tries single-parameter perturbations and keeps improvements.
        This makes the GA "memetic" by combining global and local search.

        Args:
            config: Starting configuration

        Returns:
            Improved configuration (or original if no improvement)
        """
        current = self.benchmark_flat(config)
        improved = False

        # Try perturbing each parameter
        for i in range(len(config)):
            spec = self.flat_spec[i]
            original_val = config[i]

            # Get neighbor values
            neighbors = []
            if hasattr(spec, "choices"):
                choices = spec.choices  # type: ignore[attr-defined]
                neighbors = [c for c in choices if c != original_val][:3]  # Try up to 3
            else:
                neighbors = [original_val - 1, original_val + 1]

            # Try each neighbor
            for neighbor_val in neighbors:
                if neighbor_val < 0:
                    continue

                test_config = config[:]
                test_config[i] = neighbor_val

                # Validate
                try:
                    self.config_gen.unflatten(test_config)
                except Exception:
                    continue

                # Benchmark
                test_member = self.benchmark_flat(test_config)
                if performance(test_member) < performance(current):
                    current = test_member
                    config = test_config
                    improved = True
                    break  # Found improvement, move to next parameter

            if improved:
                break  # Stop after first improving parameter

        return current.flat_values

    def _compute_diversity(self, population: list[PopulationMember]) -> float:
        """
        Compute population diversity as average pairwise Hamming distance.

        Args:
            population: Current population

        Returns:
            Diversity score (0 = no diversity, 1 = maximum diversity)
        """
        if len(population) < 2:
            return 1.0

        # Sample for efficiency
        sample_size = min(20, len(population))
        sample = random.sample(population, sample_size)

        total_distance = 0
        count = 0

        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                # Hamming distance
                distance = sum(
                    a != b for a, b in zip(sample[i].flat_values, sample[j].flat_values)
                )
                total_distance += distance
                count += 1

        if count == 0:
            return 1.0

        avg_distance = total_distance / count
        max_distance = self.genome_length
        return avg_distance / max_distance

    def _adapt_parameters(self, generation: int, diversity: float) -> None:
        """
        Adapt mutation rate based on diversity and generation.

        If diversity is low, increase mutation to explore more.
        As generations progress, decrease mutation for exploitation.

        Args:
            generation: Current generation number
            diversity: Current population diversity
        """
        # Base decay: reduce mutation rate over time
        progress = generation / self.max_generations
        base_rate = self.initial_mutation_rate * (1 - 0.5 * progress)

        # Adjust based on diversity: if too low, increase mutation
        if diversity < 0.2:
            # Low diversity: boost mutation
            self.mutation_rate = min(0.3, base_rate * 2)
        elif diversity < 0.4:
            self.mutation_rate = base_rate * 1.5
        else:
            self.mutation_rate = base_rate

    def _autotune(self) -> Config:
        """Run the genetic algorithm."""
        self.log(
            f"Starting Enhanced GA with pop={self.population_size}, "
            f"gens={self.max_generations}, crossover={self.crossover_rate}, "
            f"mutation={self.initial_mutation_rate}"
        )

        # Initialize population
        self.population = self._initialize_population()
        self.log(f"Initial best: {self.population[0].perf:.4f}ms")

        # Evolution loop
        for gen in range(self.max_generations):
            # Compute diversity and adapt parameters
            diversity = self._compute_diversity(self.population)
            self.diversity_history.append(diversity)
            self._adapt_parameters(gen, diversity)

            # Preserve elite
            sorted_pop = sorted(self.population, key=performance)
            elite = sorted_pop[: self.elite_size]

            # Generate offspring
            offspring_configs: list[FlatConfig] = []
            while len(offspring_configs) < self.population_size - self.elite_size:
                # Selection
                parent1 = self._tournament_selection(self.population)
                parent2 = self._tournament_selection(self.population)

                # Crossover
                child1, child2 = self._crossover(parent1.flat_values, parent2.flat_values)

                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                offspring_configs.extend([child1, child2])

            # Trim to exact size
            offspring_configs = offspring_configs[: self.population_size - self.elite_size]

            # Evaluate offspring
            offspring = self.parallel_benchmark_flat(offspring_configs)

            # Combine elite + offspring
            self.population = elite + offspring

            # Local search on best individuals (memetic enhancement)
            if gen % self.local_search_freq == 0 and gen > 0:
                self.log(f"Gen {gen}: Applying local search to top 3...")
                for i in range(min(3, len(self.population))):
                    improved_config = self._local_search(self.population[i].flat_values)
                    self.population[i] = self.benchmark_flat(improved_config)

            # Re-sort population
            self.population = sorted(self.population, key=performance)

            self.log(
                f"Gen {gen}: best={self.population[0].perf:.4f}ms, "
                f"worst={self.population[-1].perf:.4f}ms, "
                f"diversity={diversity:.3f}, mutation={self.mutation_rate:.4f}"
            )

        # Return best
        best = min(self.population, key=performance)
        self.log(f"GA completed. Best config: {best.perf:.4f}ms")
        return best.config
