#!/usr/bin/env python3
"""
run_ablation.py - Run individual or all ablation studies.

Each ablation can be run independently:

  python scripts/run_ablation.py --ablation-subspace-rank
  python scripts/run_ablation.py --ablation-buffer-size
  python scripts/run_ablation.py --ablation-replay-ratio
  python scripts/run_ablation.py --ablation-ema-beta
  python scripts/run_ablation.py --ablation-timing
  python scripts/run_ablation.py --ablation-hybrid

  python scripts/run_ablation.py --ablation-all

Without any --ablation-* flag, the script exits with a usage message.

Hardware notes (rented GPU):
  - Each individual ablation run uses the paper-spec batch sizes from
    run_experiment.py (batch_size_current=64, batch_size_replay=32, grad_accum=1, fp16=True).
  - Ablations are run sequentially.
  - Use --fast to run 1 seed × 1 order × reduced candidates for quick checks.
"""

import argparse
import subprocess
import sys
import json
import os
from pathlib import Path
from itertools import product


SCRIPT = Path(__file__).parent / "run_experiment.py"

# ---- Ablation definitions ----
# Each entry: ablation_name -> list of CLI arg strings to sweep

ABLATION_SUBSPACE_RANK = {
    "name": "subspace_rank",
    "description": "Ablation 7.1: Subspace rank k ∈ {5, 10, 20, 50}",
    "grid": [
        ["--subspace-rank-k", "5"],
        ["--subspace-rank-k", "10"],
        ["--subspace-rank-k", "20"],
        ["--subspace-rank-k", "50"],
    ],
    "fixed": ["--benchmark", "lnt", "--selection", "orthogonal"],
}

ABLATION_BUFFER_SIZE = {
    "name": "buffer_size",
    "description": "Ablation 7.2: Buffer size ∈ {150, 300, 500, 1500}",
    "grid": [
        ["--buffer-size", "150"],
        ["--buffer-size", "300"],
        ["--buffer-size", "500"],
        ["--buffer-size", "1500"],
    ],
    "fixed": ["--benchmark", "lnt", "--selection", "orthogonal", "--replay-ratio", "1:4"],
}

ABLATION_REPLAY_RATIO = {
    "name": "replay_ratio",
    "description": "Ablation 7.3: Replay ratio ∈ {1:2, 1:4, 1:8, 1:16}",
    "grid": [
        ["--replay-ratio", "1:2"],
        ["--replay-ratio", "1:4"],
        ["--replay-ratio", "1:8"],
        ["--replay-ratio", "1:16"],
    ],
    "fixed": ["--benchmark", "lnt", "--selection", "orthogonal", "--buffer-size", "300"],
}

ABLATION_EMA_BETA = {
    "name": "ema_beta",
    "description": "Ablation 7.4: EMA β ∈ {0.985, 0.99, 0.995, 0.999}",
    "grid": [
        ["--beta", "0.985"],
        ["--beta", "0.99"],
        ["--beta", "0.995"],
        ["--beta", "0.999"],
    ],
    "fixed": ["--benchmark", "lnt", "--selection", "orthogonal"],
}

ABLATION_TIMING = {
    "name": "timing",
    "description": "Ablation 7.5: Buffer update timing (Before vs After)",
    "grid": [
        [],                            # After (default)
        ["--update-buffer-before"],    # Before
    ],
    "grid_names": ["after", "before"],
    "fixed": ["--benchmark", "lnt", "--selection", "orthogonal"],
}

ABLATION_HYBRID = {
    "name": "hybrid",
    "description": "Ablation 7.6: Hybrid alpha ∈ {0.0, 0.25, 0.5, 0.75, 1.0}",
    "grid": [
        ["--selection", "hybrid", "--hybrid-alpha", "0.0"],
        ["--selection", "hybrid", "--hybrid-alpha", "0.25"],
        ["--selection", "hybrid", "--hybrid-alpha", "0.5"],
        ["--selection", "hybrid", "--hybrid-alpha", "0.75"],
        ["--selection", "hybrid", "--hybrid-alpha", "1.0"],
    ],
    "fixed": ["--benchmark", "lnt"],
}

ALL_ABLATIONS = [
    ABLATION_SUBSPACE_RANK,
    ABLATION_BUFFER_SIZE,
    ABLATION_REPLAY_RATIO,
    ABLATION_EMA_BETA,
    ABLATION_TIMING,
    ABLATION_HYBRID,
]


def run_single(
    ablation: dict,
    grid_args: list,
    seed: int,
    order: int,
    grid_name: str = None,
    extra_args: list = None,
):
    """Run one cell in the ablation grid."""
    fixed = ablation.get("fixed", [])
    grid_tag = grid_name or "_".join(str(a).lstrip("-").replace("-", "_") for a in grid_args if a)
    run_name = f"ablation_{ablation['name']}_{grid_tag}_ord{order}_s{seed}"

    cmd = [
        sys.executable, str(SCRIPT),
        "--seed", str(seed),
        "--order", str(order),
        "--run-name", run_name,
    ] + fixed + grid_args + (extra_args or [])

    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[WARNING] Run failed: {run_name}")
    return run_name


def run_ablation(
    ablation: dict,
    seeds: list,
    orders: list,
    extra_args: list = None,
):
    print(f"\n{'='*70}")
    print(f"[Ablation] {ablation['description']}")
    print(f"{'='*70}")

    grid = ablation["grid"]
    grid_names = ablation.get("grid_names", [None] * len(grid))

    all_runs = []
    for (seed, order), (g_args, g_name) in product(
        product(seeds, orders),
        zip(grid, grid_names)
    ):
        run_name = run_single(ablation, g_args, seed, order, g_name, extra_args)
        all_runs.append(run_name)

    print(f"\n[Ablation '{ablation['name']}'] Completed {len(all_runs)} runs.")
    return all_runs


def parse_args():
    p = argparse.ArgumentParser(
        description="Run ablation studies for Orthogonal Replay thesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ablation flags (pick one or --ablation-all):
  --ablation-subspace-rank    Subspace rank k ∈ {5, 10, 20, 50}       [Section 7.1]
  --ablation-buffer-size      Buffer size ∈ {150, 300, 500, 1500}     [Section 7.2]
  --ablation-replay-ratio     Replay ratio ∈ {1:2, 1:4, 1:8, 1:16}   [Section 7.3]
  --ablation-ema-beta         EMA β ∈ {0.985, 0.99, 0.995, 0.999}    [Section 7.4]
  --ablation-timing           Buffer update: Before vs After            [Section 7.5]
  --ablation-hybrid           Hybrid α ∈ {0.0, 0.25, 0.5, 0.75, 1.0} [Section 7.6]
  --ablation-all              Run ALL ablations sequentially

Without any --ablation-* flag: exits and shows this message.

Examples:
  python scripts/run_ablation.py --ablation-subspace-rank
  python scripts/run_ablation.py --ablation-subspace-rank --fast
  python scripts/run_ablation.py --ablation-all --seeds 42 --orders 0
        """,
    )

    # Individual ablation flags
    p.add_argument("--ablation-subspace-rank", action="store_true")
    p.add_argument("--ablation-buffer-size", action="store_true")
    p.add_argument("--ablation-replay-ratio", action="store_true")
    p.add_argument("--ablation-ema-beta", action="store_true")
    p.add_argument("--ablation-timing", action="store_true")
    p.add_argument("--ablation-hybrid", action="store_true")
    p.add_argument("--ablation-all", action="store_true",
                   help="Run all ablations sequentially")

    # Scope control
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456],
                   help="Seeds to run (default: 42 123 456)")
    p.add_argument("--orders", nargs="+", type=int, choices=[0, 1, 2], default=[0, 1, 2],
                   help="Task orders to run (default: 0 1 2)")

    # Quick sanity-check mode for 6GB GPU
    p.add_argument("--fast", action="store_true",
                   help="Fast mode: 1 seed × 1 order, reduced data (for quick VRAM/logic checks)")

    return p.parse_args()


def main():
    cfg = parse_args()

    # Check if any ablation flag is set
    ablation_flags = [
        cfg.ablation_subspace_rank,
        cfg.ablation_buffer_size,
        cfg.ablation_replay_ratio,
        cfg.ablation_ema_beta,
        cfg.ablation_timing,
        cfg.ablation_hybrid,
        cfg.ablation_all,
    ]
    if not any(ablation_flags):
        print(__doc__)
        print("\nNo ablation flag specified. Use --ablation-<name> or --ablation-all.")
        print("Run with --help for full usage.")
        sys.exit(0)

    seeds = cfg.seeds
    orders = cfg.orders
    extra_args = []

    if cfg.fast:
        seeds = [seeds[0]]
        orders = [orders[0]]
        # Reduce to standard_cl for speed and use fewer samples via smaller batch
        extra_args = ["--benchmark", "standard_cl"]
        print("[Fast mode] Running 1 seed × 1 order on standard_cl benchmark.")

    # Map flags to ablation configs
    flag_map = [
        (cfg.ablation_subspace_rank, ABLATION_SUBSPACE_RANK),
        (cfg.ablation_buffer_size, ABLATION_BUFFER_SIZE),
        (cfg.ablation_replay_ratio, ABLATION_REPLAY_RATIO),
        (cfg.ablation_ema_beta, ABLATION_EMA_BETA),
        (cfg.ablation_timing, ABLATION_TIMING),
        (cfg.ablation_hybrid, ABLATION_HYBRID),
    ]

    if cfg.ablation_all:
        to_run = ALL_ABLATIONS
    else:
        to_run = [abl for flag, abl in flag_map if flag]

    for ablation in to_run:
        run_ablation(ablation, seeds=seeds, orders=orders, extra_args=extra_args)

    print("\n[Done] All requested ablations completed.")


if __name__ == "__main__":
    main()
