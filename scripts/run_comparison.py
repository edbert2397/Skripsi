#!/usr/bin/env python3
"""
run_comparison.py - Run head-to-head comparison experiments across methods.

Compares our method (orthogonal + EMA) against baselines across benchmarks,
task orders, and seeds.

Methods:
  orthogonal  - Our method: orthogonal gradient subspace selection + EMA
  surprise    - SuRe baseline: surprise-based selection + EMA
  reservoir   - Random baseline: reservoir sampling (no selection bias)

Usage examples:
  # All methods, all benchmarks (full comparison):
  python scripts/run_comparison.py

  # Only compare orthogonal vs surprise (SuRe):
  python scripts/run_comparison.py --methods orthogonal surprise

  # Only on standard_cl benchmark, one seed:
  python scripts/run_comparison.py --benchmarks standard_cl --seeds 42

  # Quick sanity check (1 seed x 1 order):
  python scripts/run_comparison.py --fast
"""

import argparse
import subprocess
import sys
from pathlib import Path
from itertools import product


SCRIPT = Path(__file__).parent / "run_experiment.py"

# ---- Method definitions ----
# Each entry: method_name -> extra CLI args passed to run_experiment.py
METHODS = {
    "orthogonal": ["--selection", "orthogonal"],
    "surprise":   ["--selection", "surprise"],
    "reservoir":  ["--selection", "reservoir", "--no-ema"],
}

ALL_BENCHMARKS = ["standard_cl", "lnt"]
ALL_SEEDS      = [42, 123, 456]
ALL_ORDERS     = [0, 1, 2]


def run_single(method: str, benchmark: str, order: int, seed: int):
    method_args = METHODS[method]
    run_name = f"compare_{method}_{benchmark}_ord{order}_s{seed}"

    cmd = [
        sys.executable, str(SCRIPT),
        "--benchmark", benchmark,
        "--order",     str(order),
        "--seed",      str(seed),
        "--run-name",  run_name,
    ] + method_args

    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[WARNING] Run failed: {run_name}")
    return run_name


def main():
    p = argparse.ArgumentParser(
        description="Head-to-head comparison across methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods:
  orthogonal  Our method: orthogonal gradient subspace selection + EMA
  surprise    SuRe baseline: surprise-based selection + EMA
  reservoir   Random baseline: reservoir sampling, no EMA

Examples:
  python scripts/run_comparison.py
  python scripts/run_comparison.py --methods orthogonal surprise
  python scripts/run_comparison.py --benchmarks lnt --seeds 42 123
  python scripts/run_comparison.py --fast
        """,
    )

    p.add_argument(
        "--methods", nargs="+",
        choices=list(METHODS.keys()),
        default=list(METHODS.keys()),
        help="Methods to compare (default: all)",
    )
    p.add_argument(
        "--benchmarks", nargs="+",
        choices=ALL_BENCHMARKS,
        default=ALL_BENCHMARKS,
        help="Benchmarks to run on (default: all)",
    )
    p.add_argument(
        "--seeds", nargs="+", type=int,
        default=ALL_SEEDS,
        help="Seeds (default: 42 123 456)",
    )
    p.add_argument(
        "--orders", nargs="+", type=int, choices=[0, 1, 2],
        default=ALL_ORDERS,
        help="Task orders (default: 0 1 2)",
    )
    p.add_argument(
        "--fast", action="store_true",
        help="Fast mode: 1 seed x 1 order x standard_cl only (quick VRAM/logic check)",
    )

    cfg = p.parse_args()

    methods    = cfg.methods
    benchmarks = cfg.benchmarks
    seeds      = cfg.seeds
    orders     = cfg.orders

    if cfg.fast:
        seeds      = [seeds[0]]
        orders     = [orders[0]]
        benchmarks = ["standard_cl"]
        print("[Fast mode] Running 1 seed x 1 order on standard_cl only.\n")

    total = len(methods) * len(benchmarks) * len(seeds) * len(orders)
    print(f"[Comparison] Queuing {total} runs:")
    print(f"  Methods    : {methods}")
    print(f"  Benchmarks : {benchmarks}")
    print(f"  Seeds      : {seeds}")
    print(f"  Orders     : {orders}")

    all_runs = []
    for benchmark, method, order, seed in product(benchmarks, methods, orders, seeds):
        run_name = run_single(method, benchmark, order, seed)
        all_runs.append(run_name)

    print(f"\n[Done] Completed {len(all_runs)} comparison runs.")
    print("Runs completed:")
    for r in all_runs:
        print(f"  {r}")


if __name__ == "__main__":
    main()
