# Convergence Analysis Part 2: MFBO vs RandomSearch at Equal Time Budget

## Scenario

**Research Question:** Does MultiFidelityBO beat RandomSearch when given the same time budget?

In Part 1, we observed that MFBO achieved a 2× speedup over RandomSearch (50.86s vs 99.98s) while finding similar quality solutions (0.0145 ms vs 0.0144 ms). However, this comparison was asymmetric - MFBO ran for half the time.

**Part 2 Objective:** Double MFBO's exploration budget to match RandomSearch's ~100s runtime and determine if MFBO can find superior solutions when given equal computational resources.

## Test Configuration

### Hardware & Environment
- **Device:** NVIDIA H100 GPU
- **Matrix Operation:** Matrix multiplication (1024×1024 @ 1024×1024)
- **Data Type:** `torch.float16`
- **Framework:** Helion autotuner with Triton backend

### Algorithm Configurations

**PatternSearch (Baseline - unchanged):**
- `initial_population=100`, `copies=5`, `max_generations=20`
- `fidelity=50` reps per benchmark
- Budget: ~2000 configurations

**RandomSearch (unchanged):**
- `count=1000` configurations
- `fidelity=50` reps per benchmark
- Budget: 1000 configurations

**MultiFidelityBO (DOUBLED from Part 1):**
- **Part 1 budget:** 600/300/80/20 configs → 50.86s runtime
- **Part 2 budget:** 1200/600/160/40 configs → targeting ~100s runtime
- Configuration breakdown:
  - `n_low_fidelity=1200` (5 reps each) - doubled from 600
  - `n_medium_fidelity=600` (15 reps each) - doubled from 300
  - `n_high_fidelity=160` (50 reps each) - doubled from 80
  - `n_ultra_fidelity=40` (50 reps each) - doubled from 20
  - `fidelity_ultra=50` (matching PatternSearch, NOT 500)
- Total: ~2000 configurations explored

## Results

### Performance Comparison

| Algorithm | Wall-Clock Time | Best Performance | Quality vs PS | Quality vs RS | Time Budget |
|-----------|----------------|------------------|---------------|---------------|-------------|
| **PatternSearch** | 355.88s | **0.0107 ms** | baseline | 8% better | 3.5× |
| **RandomSearch** | 102.43s | 0.0116 ms | 8% slower | baseline | 1.0× |
| **MultiFidelityBO** | **117.53s** | **0.0156 ms** | **46% slower** | **34% slower** | 1.15× |

### Key Findings

❌ **MFBO did NOT beat RandomSearch at equal time budget**

1. **Runtime:** MFBO took 117.53s vs RandomSearch's 102.43s (15% SLOWER than target)
2. **Solution Quality:** MFBO found 0.0156 ms vs RandomSearch's 0.0116 ms (34% WORSE)
3. **Comparison to Baseline:** MFBO was 46% slower than PatternSearch vs RandomSearch's 8% gap

### Best Configurations Found

**PatternSearch (Optimal - 0.0107 ms):**
```python
Config(
    block_sizes=[64, 128, 64],      # K=64 dimension
    num_warps=4,
    num_stages=6,
    pid_type='persistent_interleaved',
    indexing=['pointer', 'pointer', 'pointer'],
    l2_groupings=[4],
    load_eviction_policies=['last', 'last'],
    loop_orders=[[1, 0]],
    range_flattens=[None, False],
    range_multi_buffers=[True, False]
)
```

**RandomSearch (0.0116 ms - 8% slower):**
```python
Config(
    block_sizes=[64, 128, 32],      # Smaller K=32 dimension
    num_warps=8,                    # 2× more warps
    num_stages=7,                   # More pipeline stages
    pid_type='flat',                # Different scheduling
    indexing=['pointer', 'pointer', 'pointer'],
    l2_groupings=[1],               # Less L2 grouping
    load_eviction_policies=['first', 'last'],
    loop_orders=[[0, 1]],           # Different loop order
    range_flattens=[None, None],
    range_multi_buffers=[None, False]
)
```

**MultiFidelityBO (0.0156 ms - 46% slower!):**
```python
Config(
    block_sizes=[64, 128, 16],      # Very small K=16 dimension!
    num_warps=8,
    num_stages=4,                   # Fewer stages
    pid_type='flat',
    indexing=['pointer', 'pointer', 'pointer'],
    l2_groupings=[64],              # Excessive L2 grouping
    load_eviction_policies=['last', ''],
    loop_orders=[[0, 1]],
    range_flattens=[None, None],
    range_multi_buffers=[None, None]
)
```

## Analysis

### Why Did MFBO Underperform?

**1. Suboptimal Configuration Space Exploration:**
- MFBO chose `block_sizes=[64, 128, 16]` with K=16 (very small K block)
- PatternSearch found K=64, RandomSearch found K=32
- K=16 appears to be too small, resulting in poor memory utilization

**2. L2 Grouping Mismatch:**
- MFBO: `l2_groupings=[64]` (excessive)
- PatternSearch: `l2_groupings=[4]` (optimal)
- RandomSearch: `l2_groupings=[1]` (minimal)
- MFBO's excessive grouping likely hurts performance

**3. Pipeline Stages:**
- MFBO: `num_stages=4` (underutilized)
- RandomSearch: `num_stages=7`
- PatternSearch: `num_stages=6`
- Fewer stages means less instruction-level parallelism

**4. Runtime Overhead:**
- Expected ~100s, got 117.53s (15% overhead)
- Gaussian Process training and acquisition function optimization adds overhead
- Multi-fidelity staging has coordination costs

### Why RandomSearch Performed Well

RandomSearch's success suggests:
1. **Configuration space is relatively smooth** - random sampling finds good regions
2. **Dimensionality is manageable** - ~10-15 key parameters, not hundreds
3. **Strong baseline** - 1000 random samples provide good coverage
4. **No model bias** - RandomSearch doesn't get stuck in local optima based on GP predictions

### Comparison to Part 1

| Metric | Part 1 (MFBO budget) | Part 2 (RS budget) | Change |
|--------|---------------------|-------------------|---------|
| **MFBO Runtime** | 50.86s | 117.53s | +131% ✓ |
| **MFBO Performance** | 0.0145 ms | 0.0156 ms | +8% worse ❌ |
| **MFBO vs RS Quality** | Same (0.0145 vs 0.0144) | 34% worse | Regression ❌ |

**Surprising result:** Doubling MFBO's budget made it WORSE, not better!

This suggests:
- MFBO may have gotten stuck in a suboptimal region of the search space
- The Gaussian Process model may have poor uncertainty estimates
- Acquisition function (EI/UCB) may be too exploitative, not exploring enough
- Multi-fidelity promotion strategy may be filtering out good candidates too early

## Implications

### When to Use Each Algorithm

**PatternSearch:**
- ✅ Use when: Solution quality is critical, time budget is flexible
- ✅ Best for: Production workloads, final optimization
- ❌ Avoid when: Rapid prototyping, time-constrained exploration
- **Trade-off:** 3.5× time for best solution (8% better than RS)

**RandomSearch:**
- ✅ Use when: Time budget is limited (~100s), "good enough" solutions acceptable
- ✅ Best for: Baselines, sanity checks, rapid iteration
- ❌ Avoid when: Optimal performance required
- **Trade-off:** 3.5× faster than PS, only 8% quality degradation

**MultiFidelityBO:**
- ⚠️ Use when: Time budget is very limited (~50s), exploration is priority
- ❌ Avoid when: Given equal time to RandomSearch
- ❌ Current issues: Worse than RS at equal time, gets stuck in poor regions
- **Trade-off:** 6.5× faster than PS but 37% worse (from Part 1)

### Recommendations

**For Current MFBO Implementation:**
1. ❌ Do NOT use MFBO when you have ~100s+ available - use RandomSearch instead
2. ✅ DO use MFBO when you have <60s and need fast exploration
3. ⚠️ Be aware MFBO can degrade with more budget (needs investigation)

**For MFBO Improvements:**
1. **Investigate acquisition function:** May be too exploitative (try more exploration)
2. **Tune GP hyperparameters:** Kernel choice, length scales, noise levels
3. **Review promotion strategy:** Are good candidates being filtered too early?
4. **Add diversity mechanisms:** Prevent getting stuck in local optima
5. **Analyze failed runs:** Why did K=16 get promoted through all stages?

## Hypothesis: Why More Budget Hurt MFBO

### Part 1 (600/300/80/20 configs → 0.0145 ms)
- Low-fidelity (600 configs): Found good region with K=64
- Medium-fidelity (300 configs): Narrowed down to promising candidates
- High-fidelity (80 configs): Refined to near-optimal
- Ultra-fidelity (20 configs): Selected K=64 solution

### Part 2 (1200/600/160/40 configs → 0.0156 ms)
- Low-fidelity (1200 configs): GP model overconfident, focused on K=16 region
- Medium-fidelity (600 configs): Exploitation, filtered out K=64 candidates
- High-fidelity (160 configs): Stuck in K=16 local optimum
- Ultra-fidelity (40 configs): Refined the wrong solution

**Root cause hypothesis:** More data made the GP model MORE confident in a WRONG region, leading to excessive exploitation rather than exploration.

## Open Questions

1. **Why did doubling budget hurt MFBO quality?**
   - GP overfitting to noisy low-fidelity evaluations?
   - Acquisition function balance (exploration vs exploitation)?
   - Multi-fidelity correlation assumptions violated?

2. **What is MFBO's optimal budget?**
   - Part 1 (600/300/80/20) found 0.0145 ms
   - Part 2 (1200/600/160/40) found 0.0156 ms
   - Is there a sweet spot around 300/150/40/10?

3. **Can MFBO be fixed to scale with budget?**
   - More exploration-focused acquisition functions?
   - Ensemble of GPs to prevent overconfidence?
   - Periodic random sampling to maintain diversity?

4. **Is the configuration space favorable to BO?**
   - Discrete parameters (num_warps, num_stages) vs continuous
   - Interactions between parameters (block_sizes × num_warps)
   - Non-smooth landscape with discontinuities?

## Conclusions

### Summary of Findings

1. ❌ **MFBO does NOT beat RandomSearch at equal time budget** (117s vs 102s, 34% worse quality)
2. ❌ **MFBO quality DEGRADED with doubled budget** (0.0145 → 0.0156 ms)
3. ✅ **RandomSearch is robust and efficient** at 100s time budget (only 8% worse than PatternSearch)
4. ✅ **PatternSearch remains the gold standard** for optimal solutions (0.0107 ms)
5. ⚠️ **MFBO is only useful at very low budgets** (<60s) where it beats both alternatives

### Practical Guidance

**Scenario 1: Time is no object (~6 minutes)**
→ Use **PatternSearch** for optimal 0.0107 ms solution

**Scenario 2: Limited time budget (~2 minutes)**
→ Use **RandomSearch** for good 0.0116 ms solution (8% penalty)

**Scenario 3: Very limited time (<1 minute)**
→ Use **MFBO** (Part 1 config) for acceptable 0.0145 ms solution (37% penalty)

**Scenario 4: Equal comparison at ~2 minutes**
→ Use **RandomSearch** over MFBO (0.0116 vs 0.0156 ms)

### Next Steps

**Investigation priorities:**
1. Analyze MFBO's GP model predictions - is it overconfident?
2. Test intermediate budgets (300/150/40/10) to find optimal scaling
3. Try different acquisition functions (more exploratory)
4. Add random sampling fraction (e.g., 10% pure random in each stage)
5. Visualize search trajectories to understand where MFBO went wrong

**Long-term research:**
1. Compare against other optimizers (CMA-ES, genetic algorithms, TPE)
2. Test on diverse kernels beyond matmul
3. Develop adaptive budget allocation strategy
4. Investigate ensemble Bayesian optimization approaches

---

## Appendix: Detailed Results

### Runtime Breakdown

**PatternSearch (355.88s):**
- Initial population: ~8s (100 configs)
- Generations 1-7: ~203s (1283 configs) - main search
- Generations 8-17: ~144s (816 configs) - refinement
- Total: 2099 configs explored

**RandomSearch (102.43s):**
- 1000 random configurations @ 50 reps each
- Precompilation: ~31s
- Benchmarking: ~71s
- No overhead for optimization

**MultiFidelityBO (117.53s):**
- Low-fidelity: ~52s (1200 configs @ 5 reps)
- Medium-fidelity: ~30s (600 configs @ 15 reps)
- High-fidelity: ~20s (160 configs @ 50 reps)
- Ultra-fidelity: ~5s (40 configs @ 50 reps)
- GP overhead: ~10s (training, acquisition, coordination)

### Configuration Space Coverage

**Key parameters explored:**
- `block_sizes`: [16, 32, 64, 128, 256, 1024] × 3 dimensions
- `num_warps`: [1, 2, 4, 8, 16, 32]
- `num_stages`: [1-8]
- `pid_type`: ['flat', 'persistent_blocked', 'persistent_interleaved']
- `indexing`: ['pointer', 'tensor_descriptor'] × 3
- `l2_groupings`: [1, 2, 4, 8, 16, 32, 64]

**Total search space:** ~10^12 possible configurations

**Coverage:**
- PatternSearch: 2099 configs (0.0000002%)
- RandomSearch: 1000 configs (0.0000001%)
- MFBO Part 2: 2000 configs (0.0000002%)

---

**Document Version:** 1.0
**Date:** 2025-11-04
**Scenario:** Equal Time Budget Comparison (MFBO vs RandomSearch)
**Result:** RandomSearch wins at equal budget
**GPU:** NVIDIA H100
**Helion Branch:** convergence-analysis
