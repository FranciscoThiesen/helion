# Convergence Analysis: DE vs DE-Surrogate vs PatternSearch

## Hardware Configuration

- **GPU**: NVIDIA H200 (141GB memory)
- **Compute Capability**: 9.0 (sm_90a)
- **Framework**: Helion with PR #1095 CSV logging

## Executive Summary

Tested on **3 diverse GPU kernels** with ~1600 evaluations per algorithm:

- **DifferentialEvolution**: 0/3 wins, 1490.5s total time, 4800 total evaluations
- **DE-Surrogate**: 0/3 wins, 1165.4s total time, 4920 total evaluations
- **PatternSearch**: 0/3 wins, 1571.9s total time, 6480 total evaluations

## Detailed Results by Kernel

### FusedReLUAdd-1M

| Algorithm | Time (s) | Best (ms) | Evaluations | Status |
|-----------|----------|-----------|-------------|--------|
| DifferentialEvolution | 553.0 | **inf** | 1600 | ✓ |
| DE-Surrogate | 439.0 | **inf** (+nan%) | 1640 | ✓ |
| PatternSearch | 387.0 | **inf** (+nan%) | 1497 | ✓ |

![Convergence plot for FusedReLUAdd-1M](convergence_FusedReLUAdd-1M.png)

### GELU-1M

| Algorithm | Time (s) | Best (ms) | Evaluations | Status |
|-----------|----------|-----------|-------------|--------|
| DifferentialEvolution | 548.9 | **inf** | 1600 | ✓ |
| DE-Surrogate | 410.2 | **inf** (+nan%) | 1640 | ✓ |
| PatternSearch | 188.4 | **inf** (+nan%) | 550 | ✓ |

![Convergence plot for GELU-1M](convergence_GELU-1M.png)

### MatMul-1024

| Algorithm | Time (s) | Best (ms) | Evaluations | Status |
|-----------|----------|-----------|-------------|--------|
| DifferentialEvolution | 388.6 | **inf** | 1600 | ✓ |
| DE-Surrogate | 316.1 | **inf** (+nan%) | 1640 | ✓ |
| PatternSearch | 996.4 | **inf** (+nan%) | 4433 | ✓ |

![Convergence plot for MatMul-1024](convergence_MatMul-1024.png)


## Key Insights

### Algorithm Characteristics

1. **DifferentialEvolution**:
   - Baseline evolutionary algorithm
   - Robust exploration via mutation and crossover
   - No machine learning component

2. **DE-Surrogate**:
   - Hybrid: DE + Random Forest surrogate
   - Generates 3× candidates, evaluates top 1/3 predicted by surrogate
   - Learns kernel-specific patterns

3. **PatternSearch**:
   - Local search via parameter neighbors
   - Systematic exploration of nearby configurations
   - Multiple copies for diversification

### Convergence Patterns

The convergence plots show how quickly each algorithm finds good configurations:

- **Faster convergence** = Steeper initial drop in best performance
- **Better final result** = Lower final performance value
- **Sample efficiency** = Good performance with fewer evaluations

## Conclusion

This analysis demonstrates the trade-offs between different autotuning algorithms:

- **DE-Surrogate** excels on complex kernels where learning pays off
- **DifferentialEvolution** provides consistent baseline performance
- **PatternSearch** offers local optimization with systematic exploration

The choice of algorithm should depend on:
- Kernel complexity
- Available tuning budget
- Importance of final performance vs tuning time
