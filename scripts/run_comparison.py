#!/usr/bin/env python3
"""
run_comparison.py - Run head-to-head comparison experiments across methods.

Compares orthogonal methods (O-Grad, O-Feat, O-Conflict) against baselines
on the RQ1 benchmark (15-task thesis evaluation, order 0).

Methods:
  orthogonal  - O-Grad: gradient subspace orthogonality + EMA (Method 1)
  feature     - O-Feat: feature subspace orthogonality + EMA (Method 2)
  conflict    - O-Conflict: gradient conflict score + EMA (Method 3)
  surprise    - SuRe baseline: surprise-based selection + EMA
  reservoir   - Random baseline: reservoir sampling (no selection bias)

Usage examples:
  # All methods, all seeds (full comparison):
  python scripts/run_comparison.py

  # Only compare orthogonal vs feature (O-Grad vs O-Feat):
  python scripts/run_comparison.py --methods orthogonal feature

  # Single seed:
  python scripts/run_comparison.py --seeds 42

  # Quick sanity check (1 seed):
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
    "feature":    ["--selection", "feature"],
    "conflict":   ["--selection", "conflict"],
    "surprise":   ["--selection", "surprise"],
    "reservoir":  ["--selection", "reservoir"],
}

ALL_BENCHMARKS = ["rq1"]
ALL_SEEDS      = [42, 123, 456]
ALL_ORDERS     = [0]


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
  orthogonal  O-Grad: gradient subspace orthogonality + EMA (Method 1)
  feature     O-Feat: feature subspace orthogonality + EMA (Method 2)
  conflict    O-Conflict: gradient conflict score + EMA (Method 3)
  surprise    SuRe baseline: surprise-based selection + EMA
  reservoir   Random baseline: reservoir sampling + EMA

Examples:
  python scripts/run_comparison.py
  python scripts/run_comparison.py --methods orthogonal feature
  python scripts/run_comparison.py --seeds 42 123
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
        help="Benchmarks to run on (default: rq1)",
    )
    p.add_argument(
        "--seeds", nargs="+", type=int,
        default=ALL_SEEDS,
        help="Seeds (default: 42 123 456)",
    )
    p.add_argument(
        "--orders", nargs="+", type=int, choices=[0],
        default=ALL_ORDERS,
        help="Task orders (default: 0)",
    )
    p.add_argument(
        "--fast", action="store_true",
        help="Fast mode: 1 seed x rq1 only (quick VRAM/logic check)",
    )

    cfg = p.parse_args()

    methods    = cfg.methods
    benchmarks = cfg.benchmarks
    seeds      = cfg.seeds
    orders     = cfg.orders

    if cfg.fast:
        seeds      = [seeds[0]]
        orders     = [0]
        benchmarks = ["rq1"]
        print("[Fast mode] Running 1 seed x order 0 on rq1 only.\n")

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

    # ---- Auto-generate side-by-side comparison plots ----
    outputs_dir  = Path("outputs")
    plot_script  = Path(__file__).parent / "plot_results.py"
    for bench in benchmarks:
        bench_runs = [
            outputs_dir / r
            for r in all_runs
            if f"_{bench}_" in r and (outputs_dir / r / "final_results.json").exists()
        ]
        if len(bench_runs) < 2:
            continue
        # Build descriptive filename from method names so plots don't overwrite
        methods_tag = "_".join(sorted(methods))
        save_path = outputs_dir / f"comparison_{methods_tag}_{bench}.png"
        cmd = (
            [sys.executable, str(plot_script), "--compare-runs"]
            + [str(p) for p in bench_runs]
            + ["--save-path", str(save_path)]
        )
        print(f"\n[Plot] Generating side-by-side plot for benchmark={bench} ...")
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
