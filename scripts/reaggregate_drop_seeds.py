#!/usr/bin/env python3
"""
reaggregate_drop_seeds.py - Re-aggregate an existing method+benchmark+beta
run from its per-seed folders, optionally dropping seeds.

Reuses aggregate_method() from run_comparison.py so the output folder is
format-identical to the originals: final_results.json (mean + std), merged
log.jsonl, config.json with the new seeds list, and re-rendered plots.

Examples:
  # Drop seed 42 from the 8-seed surprise run:
  python scripts/reaggregate_drop_seeds.py --method surprise --drop 42

  # Keep only an explicit subset:
  python scripts/reaggregate_drop_seeds.py --method surprise --keep 43 44 45 46 47 48 49

  # Drop two seeds at once, custom beta:
  python scripts/reaggregate_drop_seeds.py --method orthogonal --beta 0.995 --drop 42 49
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_comparison import aggregate_method, DEFAULT_K, DEFAULT_N


# Default seed pool for re-aggregation. The original 8-seed runs used
# 42..49; expose this as a CLI default so the common case is a one-liner.
DEFAULT_SEED_POOL = [42, 43, 44, 45, 46, 47, 48, 49]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True,
                   help="Method name (e.g. surprise, orthogonal, feature, conflict, reservoir, no_replay)")
    p.add_argument("--benchmark", default="rq1")
    p.add_argument("--order", type=int, default=0)
    p.add_argument("--beta", type=float, default=0.995)
    p.add_argument("--subspace-rank-k", type=int, default=DEFAULT_K)
    p.add_argument("--n-estimation-batches", type=int, default=DEFAULT_N)
    p.add_argument("--pool", nargs="+", type=int, default=DEFAULT_SEED_POOL,
                   help=f"Seed pool to start from (default: {DEFAULT_SEED_POOL})")

    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--drop", nargs="+", type=int,
                     help="Seeds to remove from the pool")
    grp.add_argument("--keep", nargs="+", type=int,
                     help="Explicit seed subset to keep (overrides --pool)")

    p.add_argument("--outputs-dir", default=None,
                   help="Outputs directory (default: <repo>/outputs)")
    args = p.parse_args()

    if args.keep:
        seeds = list(args.keep)
    else:
        drop = set(args.drop)
        seeds = [s for s in args.pool if s not in drop]

    if not seeds:
        sys.exit("[reaggregate] No seeds left after filtering — refusing to aggregate.")

    outputs_dir = Path(args.outputs_dir) if args.outputs_dir \
        else Path(__file__).parent.parent / "outputs"

    print(f"[reaggregate] method={args.method} benchmark={args.benchmark} "
          f"order={args.order} beta={args.beta}")
    print(f"[reaggregate] seeds to aggregate: {seeds}")
    print(f"[reaggregate] outputs_dir: {outputs_dir}")

    agg_dir = aggregate_method(
        method=args.method,
        benchmark=args.benchmark,
        order=args.order,
        seeds=seeds,
        beta=args.beta,
        outputs_dir=outputs_dir,
        subspace_rank_k=args.subspace_rank_k,
        n_estimation_batches=args.n_estimation_batches,
    )

    if agg_dir is None:
        sys.exit("[reaggregate] Aggregation produced no output (no loadable per-seed runs).")
    print(f"[reaggregate] Done -> {agg_dir}")


if __name__ == "__main__":
    main()
