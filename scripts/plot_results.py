#!/usr/bin/env python3
"""
plot_results.py - Aggregate outputs/ and generate thesis tables + heatmaps.

Usage:
  python scripts/plot_results.py --results-dir outputs/
  python scripts/plot_results.py --results-dir outputs/ --ablation subspace_rank
  python scripts/plot_results.py --results-dir outputs/ --comparison
  python scripts/plot_results.py --results-dir outputs/ --comparison --benchmark lnt
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_all_results(results_dir: Path):
    runs = {}
    for run_dir in sorted(results_dir.iterdir()):
        final = run_dir / "final_results.json"
        config = run_dir / "config.json"
        if not final.exists() or not config.exists():
            continue
        with open(final) as f:
            res = json.load(f)
        with open(config) as f:
            cfg = json.load(f)
        runs[run_dir.name] = {"results": res, "config": cfg}
    return runs


def aggregate_by_method(runs: dict):
    """
    Group runs by (selection, use_ema, use_orthogonal_lora, benchmark).
    Compute mean ± std over seeds/orders.
    """
    grouped = defaultdict(list)
    for run_name, data in runs.items():
        cfg = data["config"]
        key = (
            cfg.get("selection", "?"),
            cfg.get("use_ema", True),
            cfg.get("use_orthogonal_lora", False),
            cfg.get("benchmark", "?"),
        )
        summary = data["results"].get("summary", {})
        grouped[key].append(summary)

    rows = []
    for (sel, ema, olora, bench), summaries in sorted(grouped.items()):
        fps = [s.get("FP", float("nan")) for s in summaries]
        aps = [s.get("AP", float("nan")) for s in summaries]
        fgs = [s.get("Forgetting", float("nan")) for s in summaries]

        def _mean(lst): return sum(lst) / len(lst) if lst else float("nan")
        def _std(lst):
            m = _mean(lst)
            return (sum((x - m) ** 2 for x in lst) / max(len(lst) - 1, 1)) ** 0.5

        method_name = sel
        if ema:
            method_name = "Slow " + method_name.capitalize()
        else:
            method_name = method_name.capitalize()
        if olora:
            method_name += " + O-LoRA"

        rows.append({
            "Method": method_name,
            "Benchmark": bench,
            "FP_mean": _mean(fps),
            "FP_std": _std(fps),
            "AP_mean": _mean(aps),
            "AP_std": _std(aps),
            "F_mean": _mean(fgs),
            "F_std": _std(fgs),
            "n_runs": len(summaries),
        })

    return pd.DataFrame(rows)


def print_main_table(df: pd.DataFrame, benchmark: str):
    sub = df[df["Benchmark"] == benchmark].copy()
    print(f"\n{'='*70}")
    print(f"Main Results Table — {benchmark}")
    print(f"{'='*70}")
    print(f"{'Method':<35} {'FP':>8} {'AP':>8} {'Forgetting':>12}  n")
    print("-" * 70)
    for _, row in sub.iterrows():
        print(
            f"{row['Method']:<35} "
            f"{row['FP_mean']:>7.3f}±{row['FP_std']:.3f} "
            f"{row['AP_mean']:>7.3f}±{row['AP_std']:.3f} "
            f"{row['F_mean']:>11.3f}±{row['F_std']:.3f}  {row['n_runs']}"
        )


def filter_comparison_runs(runs: dict) -> dict:
    """Return only runs produced by run_comparison.py (prefix: compare_)."""
    return {k: v for k, v in runs.items() if k.startswith("compare_")}


# ---- Canonical display names for comparison methods ----
_METHOD_LABELS = {
    ("orthogonal", True,  False): "Ours (Orthogonal + EMA)",
    ("surprise",   True,  False): "SuRe (Surprise + EMA)",
    ("reservoir",  False, False): "Reservoir (Random)",
}

_METHOD_ORDER = [
    "Ours (Orthogonal + EMA)",
    "SuRe (Surprise + EMA)",
    "Reservoir (Random)",
]

_METHOD_COLORS = {
    "Ours (Orthogonal + EMA)": "steelblue",
    "SuRe (Surprise + EMA)":   "darkorange",
    "Reservoir (Random)":      "slategray",
}


def aggregate_comparison(runs: dict) -> pd.DataFrame:
    """
    Group comparison runs by (selection, use_ema, benchmark).
    Returns a DataFrame with mean ± std per method per benchmark,
    using the canonical display labels from _METHOD_LABELS.
    """
    comp_runs = filter_comparison_runs(runs)
    if not comp_runs:
        return pd.DataFrame()

    grouped = defaultdict(list)
    for run_name, data in comp_runs.items():
        cfg = data["config"]
        key = (
            cfg.get("selection", "?"),
            cfg.get("use_ema", True),
            cfg.get("use_orthogonal_lora", False),
            cfg.get("benchmark", "?"),
        )
        summary = data["results"].get("summary", {})
        grouped[key].append(summary)

    def _mean(lst): return sum(lst) / len(lst) if lst else float("nan")
    def _std(lst):
        m = _mean(lst)
        return (sum((x - m) ** 2 for x in lst) / max(len(lst) - 1, 1)) ** 0.5

    rows = []
    for (sel, ema, olora, bench), summaries in sorted(grouped.items()):
        label = _METHOD_LABELS.get((sel, ema, olora), f"{sel} ema={ema}")
        fps   = [s.get("FP",        float("nan")) for s in summaries]
        aps   = [s.get("AP",        float("nan")) for s in summaries]
        fgs   = [s.get("Forgetting", float("nan")) for s in summaries]
        rows.append({
            "Method":    label,
            "Benchmark": bench,
            "FP_mean":   _mean(fps), "FP_std":  _std(fps),
            "AP_mean":   _mean(aps), "AP_std":  _std(aps),
            "F_mean":    _mean(fgs), "F_std":   _std(fgs),
            "n_runs":    len(summaries),
        })

    return pd.DataFrame(rows)


def plot_comparison_bars(df: pd.DataFrame, benchmark: str, save_dir: Path):
    """
    Grouped bar chart: FP / AP / Forgetting for each comparison method.
    One PNG per benchmark.
    """
    import numpy as np

    sub = df[df["Benchmark"] == benchmark].copy()
    if sub.empty:
        print(f"[plot_comparison] No comparison runs found for benchmark={benchmark}")
        return

    # Sort by canonical order; unknown methods go to end
    sub["_order"] = sub["Method"].apply(
        lambda m: _METHOD_ORDER.index(m) if m in _METHOD_ORDER else 99
    )
    sub = sub.sort_values("_order").reset_index(drop=True)

    metrics      = ["FP", "AP", "Forgetting"]
    metric_means = ["FP_mean", "AP_mean", "F_mean"]
    metric_stds  = ["FP_std",  "AP_std",  "F_std"]
    metric_labels = ["Final Performance (FP)", "Average Performance (AP)", "Forgetting (↓ better)"]

    n_methods = len(sub)
    n_metrics = len(metrics)
    x         = np.arange(n_metrics)
    bar_w     = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (_, row) in enumerate(sub.iterrows()):
        color  = _METHOD_COLORS.get(row["Method"], f"C{i}")
        offset = (i - n_methods / 2 + 0.5) * bar_w
        means  = [row[m] for m in metric_means]
        stds   = [row[s] for s in metric_stds]
        bars   = ax.bar(
            x + offset, means, bar_w,
            label=f"{row['Method']} (n={row['n_runs']})",
            color=color, alpha=0.85,
            yerr=stds, capsize=4, error_kw={"elinewidth": 1.2},
        )
        for bar, val in zip(bars, means):
            if not (val != val):   # skip NaN
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(stds) * 0.05 + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"Method Comparison — {benchmark}", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=min(0, sub["F_mean"].min() - 0.05))

    plt.tight_layout()
    out = save_dir / f"comparison_{benchmark}.png"
    plt.savefig(out, dpi=150)
    print(f"[Saved] {out}")
    plt.close()


def plot_comparison_per_benchmark(df: pd.DataFrame, save_dir: Path):
    """
    Side-by-side subplots: one column per benchmark, rows = FP / AP / Forgetting.
    Gives a single figure summarising all benchmarks at once.
    """
    import numpy as np

    benchmarks = df["Benchmark"].unique().tolist()
    if not benchmarks:
        return

    fig, axes = plt.subplots(3, len(benchmarks), figsize=(7 * len(benchmarks), 12))
    if len(benchmarks) == 1:
        axes = [[ax] for ax in axes]   # normalise to 2-D list

    metrics = [
        ("FP_mean",  "FP_std",  "Final Performance (FP)"),
        ("AP_mean",  "AP_std",  "Average Performance (AP)"),
        ("F_mean",   "F_std",   "Forgetting (↓ better)"),
    ]

    for col, bench in enumerate(benchmarks):
        sub = df[df["Benchmark"] == bench].copy()
        sub["_order"] = sub["Method"].apply(
            lambda m: _METHOD_ORDER.index(m) if m in _METHOD_ORDER else 99
        )
        sub = sub.sort_values("_order").reset_index(drop=True)

        for row, (mean_col, std_col, ylabel) in enumerate(metrics):
            ax      = axes[row][col]
            methods = sub["Method"].tolist()
            means   = sub[mean_col].tolist()
            stds    = sub[std_col].tolist()
            colors  = [_METHOD_COLORS.get(m, f"C{i}") for i, m in enumerate(methods)]

            bars = ax.bar(
                range(len(methods)), means, color=colors, alpha=0.85,
                yerr=stds, capsize=5, error_kw={"elinewidth": 1.3},
            )
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(
                [m.replace(" (", "\n(") for m in methods],
                fontsize=8, ha="center",
            )
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(f"{bench}", fontsize=10)
            ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
            ax.grid(axis="y", alpha=0.3)
            for bar, val in zip(bars, means):
                if val == val:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                    )

    plt.suptitle("Method Comparison Across Benchmarks", fontsize=14, y=1.01)
    plt.tight_layout()
    out = save_dir / "comparison_all_benchmarks.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[Saved] {out}")
    plt.close()


def print_comparison_table(df: pd.DataFrame):
    """Print a clean side-by-side comparison table to stdout."""
    benchmarks = sorted(df["Benchmark"].unique())
    for bench in benchmarks:
        sub = df[df["Benchmark"] == bench].copy()
        sub["_order"] = sub["Method"].apply(
            lambda m: _METHOD_ORDER.index(m) if m in _METHOD_ORDER else 99
        )
        sub = sub.sort_values("_order")

        print(f"\n{'='*75}")
        print(f"Comparison Table — {bench}")
        print(f"{'='*75}")
        print(f"{'Method':<30} {'FP':>12} {'AP':>12} {'Forgetting':>12}  n")
        print("-" * 75)
        for _, row in sub.iterrows():
            print(
                f"{row['Method']:<30} "
                f"{row['FP_mean']:>6.3f}±{row['FP_std']:.3f}  "
                f"{row['AP_mean']:>6.3f}±{row['AP_std']:.3f}  "
                f"{row['F_mean']:>6.3f}±{row['F_std']:.3f}  "
                f"{row['n_runs']}"
            )


def plot_runs_side_by_side(run_dirs: list, save_path: Path = None):
    """
    Given a list of run directories (each with final_results.json + config.json),
    plot FP / AP / Forgetting as a grouped bar chart side-by-side.

    Args:
        run_dirs: list of Path objects pointing to individual run directories.
        save_path: where to save the PNG (default: first run_dir's parent /
                   comparison_side_by_side.png).
    """
    import numpy as np

    records = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        final_path  = run_dir / "final_results.json"
        config_path = run_dir / "config.json"

        if not final_path.exists():
            print(f"[plot_runs_side_by_side] Skipping {run_dir}: no final_results.json")
            continue

        with open(final_path) as f:
            results = json.load(f)
        cfg = {}
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)

        summary = results.get("summary", {})
        key = (cfg.get("selection", "?"), cfg.get("use_ema", True), cfg.get("use_orthogonal_lora", False))
        label = _METHOD_LABELS.get(key, run_dir.name)

        records.append({
            "label":      label,
            "run_name":   run_dir.name,
            "FP":         summary.get("FP",        float("nan")),
            "AP":         summary.get("AP",        float("nan")),
            "Forgetting": summary.get("Forgetting", float("nan")),
        })

    if not records:
        print("[plot_runs_side_by_side] No valid runs to plot.")
        return

    metrics = ["FP", "AP", "Forgetting"]
    metric_labels = ["Final Performance (FP ↑)", "Average Performance (AP ↑)", "Forgetting (↓ better)"]
    palette = ["steelblue", "darkorange", "seagreen", "tomato", "mediumpurple", "slategray"]

    n_methods = len(records)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    bar_w = 0.7 / n_methods

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, rec in enumerate(records):
        color  = _METHOD_COLORS.get(rec["label"], palette[i % len(palette)])
        offset = (i - n_methods / 2 + 0.5) * bar_w
        values = [rec[m] for m in metrics]

        bars = ax.bar(
            x + offset, values, bar_w,
            label=f"{rec['label']}\n({rec['run_name']})",
            color=color, alpha=0.88,
        )
        for bar, val in zip(bars, values):
            if val == val:   # skip NaN
                va  = "bottom" if val >= 0 else "top"
                adj = 0.008 if val >= 0 else -0.008
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + adj,
                    f"{val:.4f}", ha="center", va=va, fontsize=8,
                )

    # Highlight winning bar per metric with a star
    for mi, metric in enumerate(metrics):
    # For Forgetting: highest (least negative) wins; for FP/AP: highest wins
        best_val = max((r[metric] for r in records if r[metric] == r[metric]), default=None)
        if best_val is None:
            continue
        for i, rec in enumerate(records):
            if rec[metric] == best_val:
                offset = (i - n_methods / 2 + 0.5) * bar_w
                ax.text(
                    x[mi] + offset,
                    best_val + (0.025 if best_val >= 0 else -0.04),
                    "★", ha="center", va="bottom", fontsize=12,
                    color=_METHOD_COLORS.get(rec["label"], palette[i % len(palette)]),
                )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Side-by-Side Method Comparison", fontsize=13)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    all_vals = [rec[m] for rec in records for m in metrics if rec[m] == rec[m]]
    ax.set_ylim(bottom=min(0, min(all_vals) - 0.05), top=max(all_vals) + 0.12)

    plt.tight_layout()

    if save_path is None:
        save_path = Path(run_dirs[0]).parent / "comparison_side_by_side.png"
    save_path = Path(save_path)
    plt.savefig(save_path, dpi=150)
    print(f"[Saved] {save_path}")
    plt.close()


def plot_ablation_line(runs: dict, ablation_key: str, x_param: str, x_values: list,
                       method_label: str, save_dir: Path):
    """Plot an ablation curve (x_param vs FP/AP/Forgetting)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = ["FP", "AP", "Forgetting"]

    for ax, metric in zip(axes, metrics):
        xs, ys, yerrs = [], [], []
        for x_val in x_values:
            vals = []
            for run_name, data in runs.items():
                cfg = data["config"]
                if str(cfg.get(ablation_key, "")) == str(x_val):
                    vals.append(data["results"]["summary"].get(metric, float("nan")))
            if vals:
                mean = sum(vals) / len(vals)
                std = (sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
                xs.append(x_val)
                ys.append(mean)
                yerrs.append(std)

        ax.errorbar(xs, ys, yerr=yerrs, marker="o", capsize=4)
        ax.set_xlabel(x_param)
        ax.set_ylabel(metric)
        ax.set_title(f"{method_label}: {metric} vs {x_param}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = save_dir / f"ablation_{ablation_key}.png"
    plt.savefig(out, dpi=150)
    print(f"[Saved] {out}")
    plt.close()


def plot_run(run_dir: Path, task_names: list = None):
    """
    Plot results for a single completed run.
    Generates three files in run_dir:
      - accuracy_matrix.png  : heatmap of a[trained_up_to][eval_task]
      - accuracy_over_time.png : per-task accuracy line chart
      - summary.png          : FP / AP / Forgetting bar chart
    """
    final_path = run_dir / "final_results.json"
    if not final_path.exists():
        print(f"[plot_run] No final_results.json in {run_dir}")
        return

    with open(final_path) as f:
        data = json.load(f)

    summary = data["summary"]
    # all_results keys are strings from JSON
    all_results = {
        int(k): {int(kk): vv for kk, vv in v.items()}
        for k, v in data["all_results"].items()
    }

    n_tasks = max(all_results.keys()) + 1
    labels = task_names if task_names and len(task_names) == n_tasks else [f"T{i}" for i in range(n_tasks)]

    # ---- Accuracy matrix ----
    matrix = [[float("nan")] * n_tasks for _ in range(n_tasks)]
    for trained, evals in all_results.items():
        for evaled, acc in evals.items():
            matrix[trained][evaled] = acc

    import numpy as np
    mat = np.array(matrix)

    fig, ax = plt.subplots(figsize=(max(5, n_tasks), max(4, n_tasks - 1)))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Accuracy")
    ax.set_xticks(range(n_tasks))
    ax.set_yticks(range(n_tasks))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Task evaluated")
    ax.set_ylabel("Trained up to task")
    ax.set_title(f"Accuracy matrix — {run_dir.name}")
    for i in range(n_tasks):
        for j in range(n_tasks):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if mat[i, j] > 0.4 else "white")
    plt.tight_layout()
    plt.savefig(run_dir / "accuracy_matrix.png", dpi=150)
    plt.close()
    print(f"[plot_run] Saved accuracy_matrix.png")

    # ---- Accuracy over time ----
    fig, ax = plt.subplots(figsize=(8, 4))
    for task_id in range(n_tasks):
        xs, ys = [], []
        for trained in sorted(all_results.keys()):
            if task_id in all_results[trained]:
                xs.append(trained)
                ys.append(all_results[trained][task_id])
        if xs:
            ax.plot(xs, ys, marker="o", label=labels[task_id])
    ax.set_xlabel("Tasks learned so far")
    ax.set_ylabel("Accuracy (slow model)")
    ax.set_title(f"Per-task accuracy over time — {run_dir.name}")
    ax.set_xticks(range(n_tasks))
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "accuracy_over_time.png", dpi=150)
    plt.close()
    print(f"[plot_run] Saved accuracy_over_time.png")

    # ---- Summary bar chart ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: final per-task accuracy
    last = all_results[max(all_results.keys())]
    task_ids = sorted(last.keys())
    accs = [last[t] for t in task_ids]
    axes[0].bar([labels[t] for t in task_ids], accs, color="steelblue")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Final accuracy per task (FP)")
    axes[0].tick_params(axis="x", rotation=45)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    # Right: FP / AP / Forgetting
    metrics = ["FP", "AP", "Forgetting"]
    values = [summary[m] for m in metrics]
    colors = ["steelblue", "darkorange", "tomato"]
    bars = axes[1].bar(metrics, values, color=colors)
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_title("Summary metrics")
    axes[1].set_ylabel("Value")
    for bar, v in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     v + (0.005 if v >= 0 else -0.015),
                     f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle(run_dir.name, fontsize=10)
    plt.tight_layout()
    plt.savefig(run_dir / "summary.png", dpi=150)
    plt.close()
    print(f"[plot_run] Saved summary.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="outputs", type=Path)
    p.add_argument("--run", type=Path, default=None,
                   help="Plot a single run directory (e.g. outputs/standard_cl_ord0_s42_surprise_ema/)")
    p.add_argument("--ablation", type=str, default=None,
                   help="Plot a specific ablation: subspace_rank | buffer_size | replay_ratio | ema_beta | hybrid")
    p.add_argument("--comparison", action="store_true",
                   help="Plot comparison across methods (orthogonal vs surprise vs reservoir)")
    p.add_argument("--compare-runs", nargs="+", type=Path, default=None, metavar="RUN_DIR",
                   help="Plot specific run directories side by side (e.g. outputs/compare_orthogonal_lnt_ord0_s42 outputs/compare_surprise_lnt_ord0_s42)")
    p.add_argument("--save-path", type=Path, default=None,
                   help="Output path for --compare-runs plot (default: outputs/comparison_side_by_side.png)")
    p.add_argument("--benchmark", default="lnt", choices=["standard_cl", "lnt"])
    cfg = p.parse_args()

    if cfg.run is not None:
        plot_run(cfg.run)
        return

    if cfg.compare_runs is not None:
        plot_runs_side_by_side(cfg.compare_runs, save_path=cfg.save_path)
        return

    results_dir = cfg.results_dir
    if not results_dir.exists():
        print(f"[Error] {results_dir} does not exist.")
        return

    runs = load_all_results(results_dir)
    if not runs:
        print("[Error] No completed runs found.")
        return

    print(f"[Loaded] {len(runs)} runs from {results_dir}")

    # ---- Comparison mode ----
    if cfg.comparison:
        comp_df = aggregate_comparison(runs)
        if comp_df.empty:
            print("[Error] No comparison runs found (expected prefix 'compare_' in run names).")
            print("        Run: python scripts/run_comparison.py first.")
            return
        print_comparison_table(comp_df)
        plot_comparison_bars(comp_df, cfg.benchmark, results_dir)
        plot_comparison_bars(comp_df, "standard_cl" if cfg.benchmark == "lnt" else "lnt", results_dir)
        plot_comparison_per_benchmark(comp_df, results_dir)
        out_csv = results_dir / "comparison_table.csv"
        comp_df.to_csv(out_csv, index=False)
        print(f"\n[Saved] {out_csv}")
        return

    df = aggregate_by_method(runs)
    print_main_table(df, cfg.benchmark)
    print_main_table(df, "standard_cl")

    # Save table as CSV
    out_csv = results_dir / "main_table.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[Saved] {out_csv}")

    # Ablation plots
    ablation_specs = {
        "subspace_rank": ("subspace_rank_k", [5, 10, 20, 50]),
        "buffer_size": ("buffer_size", [150, 300, 500, 1500]),
        "replay_ratio": ("replay_ratio", ["1:2", "1:4", "1:8", "1:16"]),
        "ema_beta": ("beta", [0.985, 0.99, 0.995, 0.999]),
        "hybrid": ("hybrid_alpha", [0.0, 0.25, 0.5, 0.75, 1.0]),
    }

    if cfg.ablation:
        if cfg.ablation not in ablation_specs:
            print(f"Unknown ablation '{cfg.ablation}'. Choose from {list(ablation_specs)}")
        else:
            key, x_values = ablation_specs[cfg.ablation]
            plot_ablation_line(runs, key, cfg.ablation, x_values, "Orthogonal", results_dir)
    else:
        for abl_name, (key, x_values) in ablation_specs.items():
            plot_ablation_line(runs, key, abl_name, x_values, "Orthogonal", results_dir)


if __name__ == "__main__":
    main()
