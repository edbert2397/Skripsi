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
import json
import statistics
import subprocess
import sys
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from plot_results import plot_run
from src.data.task_orders import get_task_order


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


def _beta_tag(beta: float) -> str:
    return f"_b{beta:g}".replace(".", "p")


def _seeds_tag(seeds: list) -> str:
    """Folder fragment for a seed list. Single seed keeps the legacy `s42` form;
    multi-seed becomes `seed_42_43_44_45_46`."""
    if len(seeds) == 1:
        return f"s{seeds[0]}"
    return "seed_" + "_".join(str(s) for s in seeds)


def aggregate_method(method: str, benchmark: str, order: int,
                     seeds: list, beta: float, outputs_dir: Path):
    """Average per-seed results for one method into a single folder.

    Reads each seed's `compare_{method}_{benchmark}_ord{order}_s{seed}{beta_tag}/`
    and writes `compare_{method}_{benchmark}_ord{order}_seed_42_43_..._{beta_tag}/`
    containing mean (+ std) summary metrics, mean accuracy matrix, merged
    log.jsonl (each line tagged with its source seed), a config.json with the
    seeds list, and re-rendered plots from plot_run().

    Returns the aggregated dir Path, or None if no per-seed run is loadable.
    Falls back to the lone seed's dir when only one seed is present.
    """
    beta_tag = _beta_tag(beta)

    per_seed = []
    for seed in seeds:
        d = outputs_dir / f"compare_{method}_{benchmark}_ord{order}_s{seed}{beta_tag}"
        if not (d / "final_results.json").exists():
            print(f"[Aggregate] {method}: skipping seed {seed} (no final_results.json at {d})")
            continue
        per_seed.append((seed, d))

    if not per_seed:
        print(f"[Aggregate] {method}: no completed runs, skipping.")
        return None
    if len(per_seed) == 1:
        # Nothing to average; the single per-seed folder is already the result.
        return per_seed[0][1]

    payloads, configs = [], []
    for seed, d in per_seed:
        with open(d / "final_results.json") as f:
            payloads.append((seed, json.load(f)))
        cfg_path = d / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                configs.append((seed, json.load(f)))

    # Mean / std of the three scalar summary metrics
    summary_mean, summary_std = {}, {}
    for k in ("FP", "AP", "Forgetting"):
        vals = [p["summary"][k] for _, p in payloads if k in p.get("summary", {})]
        if not vals:
            continue
        summary_mean[k] = sum(vals) / len(vals)
        summary_std[k] = statistics.pstdev(vals) if len(vals) > 1 else 0.0

    # Element-wise mean of the (trained_task -> eval_task -> accuracy) matrix.
    # Keys are JSON strings, sorted numerically.
    trained_keys = set()
    for _, p in payloads:
        trained_keys.update(p.get("all_results", {}).keys())

    avg_all, std_all = {}, {}
    for tk in sorted(trained_keys, key=int):
        eval_keys = set()
        for _, p in payloads:
            eval_keys.update(p.get("all_results", {}).get(tk, {}).keys())
        avg_all[tk], std_all[tk] = {}, {}
        for ek in sorted(eval_keys, key=int):
            cell = [p["all_results"][tk][ek]
                    for _, p in payloads
                    if ek in p.get("all_results", {}).get(tk, {})]
            if not cell:
                continue
            avg_all[tk][ek] = sum(cell) / len(cell)
            std_all[tk][ek] = statistics.pstdev(cell) if len(cell) > 1 else 0.0

    # task_sizes is identical across seeds (same data loader + same n_train/n_test);
    # taking the first seed's copy is correct, not lossy.
    task_sizes = payloads[0][1].get("task_sizes", {})

    # Compute-time aggregation. Older runs predate the timing instrumentation
    # in run_experiment.py, so these fields are optional — only seeds that
    # actually have them contribute to the average.
    total_secs = [p["compute_seconds_total"]
                  for _, p in payloads if "compute_seconds_total" in p]
    compute_total_mean = sum(total_secs) / len(total_secs) if total_secs else None
    compute_total_std = (statistics.pstdev(total_secs)
                        if len(total_secs) > 1 else (0.0 if total_secs else None))

    # Per-task seconds: dict task_id_str -> seconds. Mean each task across seeds.
    per_task_keys = set()
    for _, p in payloads:
        per_task_keys.update(p.get("compute_seconds_per_task", {}).keys())
    compute_per_task_mean, compute_per_task_std = {}, {}
    for tk in sorted(per_task_keys, key=int):
        cell = [p["compute_seconds_per_task"][tk]
                for _, p in payloads
                if tk in p.get("compute_seconds_per_task", {})]
        if not cell:
            continue
        compute_per_task_mean[tk] = sum(cell) / len(cell)
        compute_per_task_std[tk] = statistics.pstdev(cell) if len(cell) > 1 else 0.0

    seeds_used = [s for s, _ in per_seed]
    seeds_tag = _seeds_tag(seeds_used)
    agg_dir = outputs_dir / f"compare_{method}_{benchmark}_ord{order}_{seeds_tag}{beta_tag}"
    agg_dir.mkdir(parents=True, exist_ok=True)

    out_payload = {
        "summary": summary_mean,
        "summary_std": summary_std,
        "task_sizes": task_sizes,
        "all_results": avg_all,
        "all_results_std": std_all,
        "n_seeds": len(seeds_used),
        "seeds": seeds_used,
    }
    if compute_total_mean is not None:
        out_payload["compute_seconds_total_mean"] = compute_total_mean
        out_payload["compute_seconds_total_std"] = compute_total_std
        out_payload["compute_seconds_total_n"] = len(total_secs)
    if compute_per_task_mean:
        out_payload["compute_seconds_per_task_mean"] = compute_per_task_mean
        out_payload["compute_seconds_per_task_std"] = compute_per_task_std
    with open(agg_dir / "final_results.json", "w") as f:
        json.dump(out_payload, f, indent=2)

    if configs:
        cfg_out = dict(configs[0][1])
        cfg_out.pop("seed", None)
        cfg_out["seeds"] = seeds_used
        with open(agg_dir / "config.json", "w") as f:
            json.dump(cfg_out, f, indent=2)

    # Merge per-seed log.jsonl, tagging each record with its source seed.
    with open(agg_dir / "log.jsonl", "w") as f_out:
        for seed, d in per_seed:
            log_path = d / "log.jsonl"
            if not log_path.exists():
                continue
            with open(log_path) as f_in:
                for line in f_in:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rec["seed"] = seed
                    f_out.write(json.dumps(rec) + "\n")

    # Plots: plot_run only reads summary[FP|AP|Forgetting] + all_results, both of
    # which are means in the averaged JSON, so the rendered plots represent the
    # mean across seeds without any plot_run code change.
    task_order = get_task_order(benchmark, order)
    plot_run(agg_dir, task_order)

    if compute_total_mean is not None:
        std_str = f" ± {compute_total_std:.1f}s" if len(total_secs) > 1 else ""
        print(f"[Aggregate] {method}: avg compute = {compute_total_mean:.1f}s"
              f"{std_str} ({compute_total_mean/60:.1f} min)"
              f" over {len(total_secs)}/{len(seeds_used)} seed(s) with timing data")
    else:
        print(f"[Aggregate] {method}: no compute_seconds_total fields found "
              f"(per-seed runs predate timing instrumentation; re-run to populate)")
    print(f"[Aggregate] {method}: averaged {len(seeds_used)} seeds -> {agg_dir}")
    return agg_dir


def run_single(method: str, benchmark: str, order: int, seed: int, beta: float):
    method_args = METHODS[method]
    beta_tag = _beta_tag(beta)
    run_name = f"compare_{method}_{benchmark}_ord{order}_s{seed}{beta_tag}"

    cmd = [
        sys.executable, str(SCRIPT),
        "--benchmark", benchmark,
        "--order",     str(order),
        "--seed",      str(seed),
        "--beta",      str(beta),
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
        "--beta", type=float, default=0.995,
        help="EMA decay for slow-LoRA (default: 0.995; try 0.975 for tighter memory window)",
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
    print(f"  Beta       : {cfg.beta}")

    all_runs = []
    for benchmark, method, order, seed in product(benchmarks, methods, orders, seeds):
        run_name = run_single(method, benchmark, order, seed, cfg.beta)
        all_runs.append(run_name)

    print(f"\n[Done] Completed {len(all_runs)} comparison runs.")
    print("Runs completed:")
    for r in all_runs:
        print(f"  {r}")

    outputs_dir = Path("outputs")

    # ---- Aggregate across seeds (one folder per method per benchmark) ----
    # For each (benchmark, method, order), collapse the per-seed run dirs into
    # a single seed-averaged folder named e.g. `compare_orthogonal_rq1_ord0_seed_42_43_44_45_46_b0p975`.
    # When only one seed was run, aggregate_method short-circuits to that seed's dir.
    aggregated = {}  # (benchmark, method, order) -> Path
    for benchmark, method, order in product(benchmarks, methods, orders):
        agg_dir = aggregate_method(method, benchmark, order, seeds, cfg.beta, outputs_dir)
        if agg_dir is not None:
            aggregated[(benchmark, method, order)] = agg_dir

    # ---- Auto-generate side-by-side comparison plots ----
    # Use the seed-averaged dirs (one bar per method) instead of per-seed dirs,
    # so the plot doesn't show n_methods * n_seeds bars when seeds > 1.
    plot_script = Path(__file__).parent / "plot_results.py"
    seeds_tag = _seeds_tag(seeds)
    beta_tag = _beta_tag(cfg.beta)
    for bench in benchmarks:
        bench_dirs = [
            aggregated[(bench, m, o)]
            for m in methods for o in orders
            if (bench, m, o) in aggregated
        ]
        if len(bench_dirs) < 2:
            continue
        methods_tag = "_".join(sorted(methods))
        save_path = outputs_dir / f"comparison_{methods_tag}_{bench}_{seeds_tag}{beta_tag}.png"
        cmd = (
            [sys.executable, str(plot_script), "--compare-runs"]
            + [str(p) for p in bench_dirs]
            + ["--save-path", str(save_path)]
        )
        print(f"\n[Plot] Generating side-by-side plot for benchmark={bench} ...")
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
