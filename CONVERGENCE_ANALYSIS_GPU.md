# Convergence Analysis - GPU Machine Instructions

This branch contains tools for running convergence analysis of the Multi-Fidelity Bayesian Optimization (MFBO) implementation.

## Quick Start

```bash
# Clone and checkout this branch
git checkout convergence-analysis

# Run the unified analysis script
python3 scripts/run_gpu_analysis.py
```

This will:
1. Run PatternSearch, RandomSearch, and MultiFidelityBO on a synthetic benchmark
2. Generate convergence plots
3. Print comparison tables showing:
   - Best performance found by each algorithm
   - Total evaluations required
   - Speedup vs baseline (PatternSearch)
   - Best configurations found

## Output

The script generates:
- `convergence_comparison.png` - Convergence plot showing all algorithms
- Console output with detailed comparison tables

## Command Line Options

```bash
# Specify output directory
python3 scripts/run_gpu_analysis.py --output-dir ./results

# Change random seed
python3 scripts/run_gpu_analysis.py --seed 123

# Adjust max iterations
python3 scripts/run_gpu_analysis.py --max-iters 200

# Quiet mode (less output)
python3 scripts/run_gpu_analysis.py --quiet

# Get help
python3 scripts/run_gpu_analysis.py --help
```

## What Gets Tested

The analysis uses a synthetic kernel performance function that mimics real GPU behavior:

### Performance Landscape
- **Global optimum**: BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=8, num_stages=3
- **Local optima**: Multiple suboptimal configurations to test exploration
- **Non-convex**: Multiple valleys (like real kernels)
- **Noisy measurements**: Noise inversely proportional to fidelity

### Algorithms Compared

1. **PatternSearch** (Baseline)
   - Strategy: Local search from starting point
   - Pros: Reliable, systematic
   - Cons: Slow, exhaustive

2. **RandomSearch**
   - Strategy: Uniform random sampling
   - Pros: Simple, explores broadly
   - Cons: No learning, inefficient

3. **MultiFidelityBO** (New)
   - Strategy: GP-guided with progressive fidelity
   - Pros: Learns landscape, uses cheap evaluations
   - Expected: 30-40x fewer evaluations

## Expected Results

### Convergence
- **MFBO** should converge fastest (steepest initial descent)
- **PatternSearch** should be slow but steady
- **RandomSearch** should be erratic

### Solution Quality
All algorithms should find the global optimum:
- BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=8, num_stages=3

### Efficiency
- **PatternSearch**: ~300-500 evaluations (baseline)
- **RandomSearch**: ~200-300 evaluations
- **MFBO**: ~40-60 evaluations (**5-10x speedup**)

## Interpreting Results

### Good MFBO Performance
✅ Finds optimal configuration
✅ Uses significantly fewer evaluations than PatternSearch
✅ Converges quickly (steep initial drop in convergence plot)
✅ Smooth convergence curve (not erratic like RandomSearch)

### What to Report
Include in PR:
1. Convergence plot image
2. Final performance comparison table
3. Number of evaluations for each algorithm
4. Speedup factor (PatternSearch evals / MFBO evals)
5. Whether optimal config was found

## Files in This Branch

### Analysis Scripts
- `scripts/run_gpu_analysis.py` - **Main unified script** (use this!)
- `scripts/compare_autotuners.py` - Alternative comparison script
- `scripts/convergence_analysis.py` - Detailed convergence tracking
- `scripts/simple_convergence_demo.py` - Simplified demo

### Documentation
- `CONVERGENCE_ANALYSIS_GPU.md` - This file
- `CONVERGENCE_ANALYSIS.md` - Detailed methodology
- `CONVERGENCE_RESULTS.md` - Expected results and theory
- `PR_DESCRIPTION.md` - Suggested PR description

### Support Files
- `convergence_comparison.png` - Example output plot
- Various .md files with implementation details

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the helion root directory
cd /path/to/helion

# Set PYTHONPATH if needed
export PYTHONPATH=.:$PYTHONPATH
python3 scripts/run_gpu_analysis.py
```

### Missing Dependencies
```bash
# Install required packages
pip install numpy matplotlib scikit-learn scipy
```

### Slow Execution
The analysis should complete in 2-5 minutes. If slower:
- Reduce max iterations: `--max-iters 50`
- Check that GPU is available (though this uses synthetic benchmark, not real kernels)

## Next Steps

After running this analysis:

1. **Save the output**:
   ```bash
   python3 scripts/run_gpu_analysis.py --output-dir ./convergence_results
   ```

2. **Add to PR**: Include the convergence plot and summary table in your PR description

3. **Real kernel testing** (optional): Test MFBO on actual GPU kernels for production validation

## Questions?

See [`CONVERGENCE_ANALYSIS.md`](CONVERGENCE_ANALYSIS.md) for detailed methodology and [`CONVERGENCE_RESULTS.md`](CONVERGENCE_RESULTS.md) for theoretical background.
