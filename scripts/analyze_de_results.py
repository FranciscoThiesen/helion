#!/usr/bin/env python3
"""
Analyze DE vs DE-Surrogate vs PatternSearch results.

Generates comprehensive tables and best configuration analysis from existing results.

Author: Francisco Geiman Thiesen
Date: 2025-11-06
"""

import json
from pathlib import Path


def main():
    # Load existing results
    results_file = 'final_three_kernel_results.json'

    if not Path(results_file).exists():
        print(f"Error: {results_file} not found")
        print("Run the benchmark first to generate results.")
        return

    with open(results_file, 'r') as f:
        all_results = json.load(f)

    # Filter for DE, DE-Surrogate, PatternSearch
    target_algorithms = ['DifferentialEvolution', 'DE-Surrogate', 'PatternSearch']

    print("="*90)
    print("COMPREHENSIVE ANALYSIS: DE vs DE-Surrogate vs PatternSearch")
    print("="*90)
    print()

    # ============================================================================
    # TABLE 1: Time and Number of Evaluations per Method per Kernel
    # ============================================================================
    print("="*90)
    print("TABLE 1: Time and Number of Evaluations per Method per Kernel")
    print("="*90)
    print(f"{'Kernel':<25} {'Method':<25} {'Time (s)':<15} {'Evaluations':<15}")
    print("-"*90)

    kernel_data = {}
    for kernel_result in all_results:
        kernel_name = kernel_result['kernel']
        kernel_data[kernel_name] = {}

        for result in kernel_result['results']:
            algo = result['algorithm']
            if algo in target_algorithms:
                kernel_data[kernel_name][algo] = result
                print(f"{kernel_name:<25} {algo:<25} {result['time']:<15.1f} {result['evaluations']:<15}")

    # ============================================================================
    # TABLE 2: Performance Comparison with Best Configs
    # ============================================================================
    print("\n\n" + "="*90)
    print("TABLE 2: Performance Comparison and Best Configurations")
    print("="*90)

    for kernel_name, algorithms in kernel_data.items():
        print(f"\n{kernel_name}")
        print("-"*90)

        # Find DE baseline
        de_perf = None
        if 'DifferentialEvolution' in algorithms:
            de_perf = algorithms['DifferentialEvolution']['performance']

        # Sort by performance
        sorted_algos = sorted(
            algorithms.items(),
            key=lambda x: x[1]['performance']
        )

        for algo_name, result in sorted_algos:
            perf = result['performance']

            # Calculate vs DE
            if de_perf and de_perf > 0:
                vs_de = (perf - de_perf) / de_perf * 100
                vs_de_str = f"({vs_de:+.1f}% vs DE)"
            else:
                vs_de_str = "(baseline)"

            print(f"  {algo_name:<25} {perf:.4f} ms  {vs_de_str}")

    # ============================================================================
    # TABLE 3: Detailed Time and Evaluation Counts
    # ============================================================================
    print("\n\n" + "="*90)
    print("TABLE 3: Detailed Time and Evaluation Summary")
    print("="*90)

    # Calculate totals
    totals = {}
    for kernel_result in all_results:
        for result in kernel_result['results']:
            algo = result['algorithm']
            if algo in target_algorithms:
                if algo not in totals:
                    totals[algo] = {'time': 0, 'evals': 0, 'count': 0}
                totals[algo]['time'] += result['time']
                totals[algo]['evals'] += result['evaluations']
                totals[algo]['count'] += 1

    print(f"{'Method':<25} {'Total Time':<15} {'Total Evals':<15} {'Avg Time/Kernel':<20}")
    print("-"*90)

    for algo in target_algorithms:
        if algo in totals:
            t = totals[algo]
            avg_time = t['time'] / t['count'] if t['count'] > 0 else 0
            print(f"{algo:<25} {t['time']:<15.1f} {t['evals']:<15} {avg_time:<20.1f}")

    # ============================================================================
    # BEST CONFIGURATION ANALYSIS
    # ============================================================================
    print("\n\n" + "="*90)
    print("BEST CONFIGURATION ANALYSIS")
    print("="*90)

    print("\nNote: Best configurations are available in the JSON file.")
    print("To see detailed configs, check final_three_kernel_results.json")

    # ============================================================================
    # WINNER ANALYSIS
    # ============================================================================
    print("\n\n" + "="*90)
    print("OVERALL WINNER ANALYSIS")
    print("="*90)

    wins = {algo: 0 for algo in target_algorithms}
    performance_gains = {algo: [] for algo in target_algorithms}

    for kernel_name, algorithms in kernel_data.items():
        # Find winner (best performance)
        winner = min(algorithms.items(), key=lambda x: x[1]['performance'])
        wins[winner[0]] += 1

        # Calculate gains vs DE
        if 'DifferentialEvolution' in algorithms:
            de_perf = algorithms['DifferentialEvolution']['performance']
            for algo_name, result in algorithms.items():
                if de_perf > 0:
                    gain = (de_perf - result['performance']) / de_perf * 100
                    performance_gains[algo_name].append(gain)

    print(f"\n{'Method':<25} {'Wins':<10} {'Avg Gain vs DE':<20}")
    print("-"*90)

    for algo in target_algorithms:
        wins_count = wins.get(algo, 0)
        gains = performance_gains.get(algo, [])
        avg_gain = sum(gains) / len(gains) if gains else 0

        print(f"{algo:<25} {wins_count:<10} {avg_gain:+.2f}%")

    # ============================================================================
    # RECOMMENDATION
    # ============================================================================
    print("\n\n" + "="*90)
    print("RECOMMENDATION FOR HELION CONTRIBUTION")
    print("="*90)

    # Find overall best performer
    overall_scores = []
    for algo in target_algorithms:
        gains = performance_gains.get(algo, [])
        if gains:
            avg_gain = sum(gains) / len(gains)
            wins_count = wins.get(algo, 0)
            # Score: average gain + bonus for wins
            score = avg_gain + (wins_count * 2)  # 2% bonus per win
            overall_scores.append((algo, avg_gain, wins_count, score))

    overall_scores.sort(key=lambda x: x[3], reverse=True)

    print("\nRanking:")
    for rank, (algo, avg_gain, wins_count, score) in enumerate(overall_scores, 1):
        print(f"  {rank}. {algo}")
        print(f"     - Average gain vs DE: {avg_gain:+.2f}%")
        print(f"     - Wins: {wins_count}/3 kernels")
        print(f"     - Overall score: {score:.2f}")
        print()

    if overall_scores:
        winner_algo, winner_gain, winner_wins, _ = overall_scores[0]
        print(f"RECOMMENDED: {winner_algo}")
        print(f"  - Consistently outperforms DE by {winner_gain:+.2f}% on average")
        print(f"  - Won on {winner_wins}/3 test kernels")

        # Find if DE-Surrogate has speed advantage
        if 'DE-Surrogate' in totals and 'DifferentialEvolution' in totals:
            de_time = totals['DifferentialEvolution']['time']
            des_time = totals['DE-Surrogate']['time']
            speedup = de_time / des_time

            if speedup > 1:
                print(f"  - Also {speedup:.2f}x faster in wall-clock time than standard DE")

    print("\n" + "="*90)


if __name__ == '__main__':
    main()
