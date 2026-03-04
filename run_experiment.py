"""
DLOG Experiment Runner — main entry point.

Usage:
    python run_experiment.py                  # Full experiment
    python run_experiment.py --smoke-test     # Quick sanity check (5 steps)
    python run_experiment.py --ablation       # Run 3 orth ablations (parameter projection)
"""
import os
import sys
import json
import argparse
import random
import gc
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from transformers import AutoTokenizer
from typing import Optional

from config import DLOGConfig
from dlog_model import DLOGModel, BaselineModel
from trainer import DLOGTrainer, BaselineTrainer, SuReTrainer
from data_pipeline import build_task_dataloaders
from metrics import compute_all_subspace_overlaps


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# Plotting utilities
# ======================================================================
def plot_results(dlog_results: dict, baseline_results: dict, sure_results: Optional[dict], output_dir: str):
    """Generate comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.2)

    # --- 1. Forgetting Comparison Bar Chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    methods = ["DLOG", "Baseline\n(Single LoRA)"]
    if sure_results:
        methods.append("SuRe-Style")

    fp_vals = [
        dlog_results["cl_metrics"]["Final Performance (FP)"],
        baseline_results["cl_metrics"]["Final Performance (FP)"],
    ]
    ft_vals = [
        dlog_results["cl_metrics"]["Forgetting (FT)"],
        baseline_results["cl_metrics"]["Forgetting (FT)"],
    ]
    if sure_results:
        fp_vals.append(sure_results["cl_metrics"]["Final Performance (FP)"])
        ft_vals.append(sure_results["cl_metrics"]["Forgetting (FT)"])

    x = np.arange(len(methods))
    width = 0.35
    bars1 = ax.bar(x - width/2, fp_vals, width, label="Final Perf (FP) ↑", color="#4CAF50")
    bars2 = ax.bar(x + width/2, ft_vals, width, label="Forgetting (FT) ↓", color="#F44336")
    ax.set_ylabel("Score")
    ax.set_title("DLOG vs Baselines: Performance & Forgetting")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", padding=3)
    ax.bar_label(bars2, fmt="%.3f", padding=3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "forgetting_comparison.png"), dpi=150)
    plt.close()

    # --- 2. Training Loss over Steps ---
    num_plots = 3 if sure_results else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(18 if sure_results else 14, 5))

    # DLOG loss
    dlog_steps = [e["step"] for e in dlog_results["train_log"]]
    dlog_losses = [e["task_loss"] for e in dlog_results["train_log"]]
    axes[0].plot(dlog_steps, dlog_losses, label="Task Loss", color="#2196F3")
    if any("orth_loss" in e for e in dlog_results["train_log"]):
        orth_losses = [e.get("orth_loss", 0) for e in dlog_results["train_log"]]
        axes[0].plot(dlog_steps, orth_losses, label="Orth Loss", color="#FF9800", alpha=0.7)
    axes[0].set_title("DLOG Training")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Baseline loss
    bl_steps = [e["step"] for e in baseline_results["train_log"]]
    bl_losses = [e.get("loss", e.get("task_loss", 0)) for e in baseline_results["train_log"]]
    axes[1].plot(bl_steps, bl_losses, label="Loss", color="#9C27B0")
    axes[1].set_title("Baseline Training")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    # SuRe loss
    if sure_results:
        sr_steps = [e["step"] for e in sure_results["train_log"]]
        sr_losses = [e.get("loss", e.get("task_loss", 0)) for e in sure_results["train_log"]]
        axes[2].plot(sr_steps, sr_losses, label="Loss", color="#E91E63")
        axes[2].set_title("SuRe-Style Training")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Loss")
        axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_loss.png"), dpi=150)
    plt.close()

    # --- 3. Leakage over steps (if available) ---
    leakage_entries = [e for e in dlog_results["train_log"] if "leakage" in e]
    if leakage_entries:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            [e["step"] for e in leakage_entries],
            [e["leakage"] for e in leakage_entries],
            color="#E91E63", marker="o", markersize=3,
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("Leakage Ratio")
        ax.set_title("DLOG: Leakage into Protected Subspace (lower = better)")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "leakage.png"), dpi=150)
        plt.close()

    # --- 4. Efficiency Comparison Table ---
    fig, ax = plt.subplots(figsize=(10, 3 if not sure_results else 4))
    ax.axis("off")
    
    headers = ["Metric", "DLOG", "Baseline"]
    if sure_results: headers.append("SuRe")

    table_data = [headers]
    table_data.append([
         "Wall-clock (s)",
         str(dlog_results["efficiency"]["total_time_sec"]),
         str(baseline_results["efficiency"]["total_time_sec"])
    ] + ([str(sure_results["efficiency"]["total_time_sec"])] if sure_results else []))
    
    table_data.append([
         "Forward Passes",
         str(dlog_results["efficiency"]["forward_passes"]),
         str(baseline_results["efficiency"]["forward_passes"])
    ] + ([str(sure_results["efficiency"]["forward_passes"])] if sure_results else []))
    
    table_data.append([
         "Projection Overhead %",
         str(dlog_results["efficiency"]["projection_overhead_pct"]),
         "N/A"
    ] + (["N/A"] if sure_results else []))
    
    table_data.append([
         "Peak Memory (MB)",
         str(dlog_results["efficiency"]["peak_memory_mb"]),
         str(baseline_results["efficiency"]["peak_memory_mb"])
    ] + ([str(sure_results["efficiency"]["peak_memory_mb"])] if sure_results else []))
    
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    # Header styling
    for j in range(len(headers)):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", fontweight="bold")
    plt.title("Efficiency Comparison", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "efficiency_table.png"), dpi=150)
    plt.close()

    # --- 5. Performance Matrix Heatmaps ---
    fig, axes = plt.subplots(1, num_plots, figsize=(18 if sure_results else 14, 5))
    task_names_short = [t.replace("_", "\n") for t in dlog_results.get("config", {}).get("task_order", ["T1", "T2", "T3", "T4"])]
    if len(task_names_short) != len(dlog_results["cl_metrics"]["Performance Matrix"]):
        task_names_short = [f"T{i+1}" for i in range(len(dlog_results["cl_metrics"]["Performance Matrix"]))]

    plot_configs = [(dlog_results, "DLOG"), (baseline_results, "Baseline")]
    if sure_results:
        plot_configs.append((sure_results, "SuRe-Style"))

    for ax_idx, (results, title) in enumerate(plot_configs):
        R = np.array(results["cl_metrics"]["Performance Matrix"])
        sns.heatmap(
            R, annot=True, fmt=".2f", cmap="YlGn",
            xticklabels=task_names_short, yticklabels=task_names_short,
            ax=axes[ax_idx], vmin=0, vmax=1,
        )
        axes[ax_idx].set_title(f"{title} — Performance Matrix")
        axes[ax_idx].set_xlabel("Eval Task")
        axes[ax_idx].set_ylabel("After Training Task")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_matrix.png"), dpi=150)
    plt.close()

    print(f"  Plots saved to {output_dir}/")


# ======================================================================
# Main experiment
# ======================================================================
def run_experiment(config: DLOGConfig, smoke_test: bool = False):
    """Run the full DLOG vs Baseline experiment."""
    set_seed(config.seed)

    max_steps = config.smoke_test_steps if smoke_test else config.num_train_steps_per_task
    max_train_samples = config.smoke_test_samples if smoke_test else None
    max_eval_samples = 20 if smoke_test else 500

    print(f"\n{'#'*60}")
    print(f"  DLOG Experiment {'(SMOKE TEST)' if smoke_test else ''}")
    print(f"  Model: {config.model_name}")
    print(f"  Tasks: {config.task_order}")
    print(f"  Steps/task: {max_steps}")
    print(f"  LoRA rank: {config.lora_rank}")
    print(f"  Projection: {config.projection_type}")
    print(f"  Soft: {config.use_soft_constraint}, Hard: {config.use_hard_constraint}")
    print(f"  Replay: {'DISABLED' if config.no_replay else 'enabled'}")
    print(f"{'#'*60}\n")

    # --- Load tokenizer ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # Causal LMs (e.g. Qwen2.5, LLaMA) lack a dedicated pad token;
    # set pad=eos and right-padding so attention_mask is computed correctly.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # --- Build dataloaders ---
    print("Building dataloaders...")
    train_loaders, eval_loaders = build_task_dataloaders(
        config, tokenizer,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )

    # =============================================
    # Experiment 1: DLOG
    # =============================================
    print("\n" + "="*60)
    print("  EXPERIMENT 1: DLOG (Dual-LoRA Orthogonal Gating)")
    print("="*60)

    dlog_model = DLOGModel(config)
    dlog_trainer = DLOGTrainer(config, dlog_model, tokenizer)
    dlog_trainer.train_all_tasks(train_loaders, eval_loaders, max_steps_override=max_steps)
    dlog_results = dlog_trainer.get_results()
    dlog_results["config"]["task_order"] = config.task_order

    # Cleanup GPU memory
    del dlog_model, dlog_trainer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # =============================================
    # Experiment 2: Baseline (Single LoRA + Replay)
    # =============================================
    replay_label = "No Replay" if config.no_replay else "Random Replay"
    print("\n" + "="*60)
    print(f"  EXPERIMENT 2: Baseline (Single LoRA + {replay_label})")
    print("="*60)

    set_seed(config.seed)  # Reset seed for fair comparison

    print("Starting Experiment 2 loading...")
    baseline_model = BaselineModel(config)
    baseline_trainer = BaselineTrainer(config, baseline_model, tokenizer)
    baseline_trainer.train_all_tasks(train_loaders, eval_loaders, max_steps_override=max_steps)
    baseline_results = baseline_trainer.get_results()

    del baseline_model, baseline_trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # =============================================
    # Experiment 3: SuRe-Style (SuRe)
    # =============================================
    sure_results = None
    # print("\n" + "="*60)
    # print("  EXPERIMENT 3: SuRe-Style (SOTA/Upper Bound)")
    # print("="*60)

    # set_seed(config.seed)

    # print("Starting Experiment 3 loading...")
    # sure_model = DLOGModel(config) # SuRe uses Dual-LoRA+EMA
    # sure_trainer = SuReTrainer(config, sure_model, tokenizer)
    # sure_trainer.train_all_tasks(train_loaders, eval_loaders, max_steps_override=max_steps)
    # sure_results = sure_trainer.get_results()

    # del sure_model, sure_trainer
    # gc.collect()
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    #     torch.cuda.synchronize()


    # =============================================
    # Results
    # =============================================
    print("\n" + "#"*60)
    print("  RESULTS SUMMARY")
    print("#"*60)

    for name, res in [("DLOG", dlog_results), ("Baseline", baseline_results)]:
        print(f"\n  {name}:")
        print(f"    Final Performance (FP): {res['cl_metrics']['Final Performance (FP)']:.4f}")
        print(f"    Average Performance (AP): {res['cl_metrics']['Average Performance (AP)']:.4f}")
        print(f"    Forgetting (FT): {res['cl_metrics']['Forgetting (FT)']:.4f}")
        print(f"    Backward Transfer (BWT): {res['cl_metrics']['Backward Transfer (BWT)']:.4f}")
        print(f"    Learning Accuracy (LA): {res['cl_metrics']['Learning Accuracy (LA)']:.4f}")
        if "efficiency" in res:
            print(f"    Wall-clock: {res['efficiency']['total_time_sec']}s")
            print(f"    Forward passes: {res['efficiency']['forward_passes']}")

    if "subspace_overlap" in dlog_results:
        print(f"\n  DLOG Subspace Separation:")
        print(f"    A matrices: {dlog_results['subspace_overlap']['mean_A_separation']:.4f}")
        print(f"    B matrices: {dlog_results['subspace_overlap']['mean_B_separation']:.4f}")

    # Save results
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "dlog_results.json"), "w") as f:
        json.dump(dlog_results, f, indent=2, default=str)
    with open(os.path.join(output_dir, "baseline_results.json"), "w") as f:
        json.dump(baseline_results, f, indent=2, default=str)
    # with open(os.path.join(output_dir, "sure_results.json"), "w") as f:
    #     json.dump(sure_results, f, indent=2, default=str)

    # Generate plots
    plot_results(dlog_results, baseline_results, sure_results, output_dir)

    # Save summary table
    with open(os.path.join(output_dir, "results_table.txt"), "w") as f:
        f.write("DLOG Experiment Results\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Model: {config.model_name}\n")
        f.write(f"Tasks: {config.task_order}\n\n")
        f.write(f"{'Metric':<30} {'DLOG':>12} {'Baseline':>12}\n")
        f.write("-" * 56 + "\n")
        f.write(f"{'Final Performance (FP)':<30} "
                f"{dlog_results['cl_metrics']['Final Performance (FP)']:>12.4f} "
                f"{baseline_results['cl_metrics']['Final Performance (FP)']:>12.4f}\n")
        f.write(f"{'Average Performance (AP)':<30} "
                f"{dlog_results['cl_metrics']['Average Performance (AP)']:>12.4f} "
                f"{baseline_results['cl_metrics']['Average Performance (AP)']:>12.4f}\n")
        f.write(f"{'Forgetting (FT)':<30} "
                f"{dlog_results['cl_metrics']['Forgetting (FT)']:>12.4f} "
                f"{baseline_results['cl_metrics']['Forgetting (FT)']:>12.4f}\n")
        f.write(f"{'Backward Transfer (BWT)':<30} "
                f"{dlog_results['cl_metrics']['Backward Transfer (BWT)']:>12.4f} "
                f"{baseline_results['cl_metrics']['Backward Transfer (BWT)']:>12.4f}\n")
        f.write(f"{'Learning Accuracy (LA)':<30} "
                f"{dlog_results['cl_metrics']['Learning Accuracy (LA)']:>12.4f} "
                f"{baseline_results['cl_metrics']['Learning Accuracy (LA)']:>12.4f}\n")
        f.write(f"{'Wall-clock (s)':<30} "
                f"{dlog_results['efficiency']['total_time_sec']:>12} "
                f"{baseline_results['efficiency']['total_time_sec']:>12}\n")
        f.write(f"{'Forward Passes':<30} "
                f"{dlog_results['efficiency']['forward_passes']:>12} "
                f"{baseline_results['efficiency']['forward_passes']:>12}\n")

    print(f"\n  Results saved to {output_dir}/")
    return dlog_results, baseline_results


# ======================================================================
# Ablation runner
# ======================================================================
def run_ablation(config: DLOGConfig, smoke_test: bool = False):
    """Run ablation study: proposal-aligned orthogonal-gating variants (parameter projection only)."""
    ablation_configs = [
        ("Soft Only", {"use_soft_constraint": True, "use_hard_constraint": False, "projection_type": "parameter"}),
        ("Hard Only (Parameter)", {"use_soft_constraint": False, "use_hard_constraint": True, "projection_type": "parameter"}),
        ("Soft + Hard (Parameter)", {"use_soft_constraint": True, "use_hard_constraint": True, "projection_type": "parameter"}),
    ]

    max_steps = config.smoke_test_steps if smoke_test else config.num_train_steps_per_task
    max_train_samples = config.smoke_test_samples if smoke_test else None
    max_eval_samples = 20 if smoke_test else 500

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    train_loaders, eval_loaders = build_task_dataloaders(
        config, tokenizer,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )

    all_results = {}

    for name, overrides in ablation_configs:
        print(f"\n{'='*60}")
        print(f"  Ablation: {name}")
        print(f"{'='*60}")

        set_seed(config.seed)

        abl_config = DLOGConfig(**{
            **config.__dict__,
            **overrides,
        })

        model = DLOGModel(abl_config)
        trainer = DLOGTrainer(abl_config, model, tokenizer)
        trainer.train_all_tasks(train_loaders, eval_loaders, max_steps_override=max_steps)
        results = trainer.get_results()
        results["ablation_name"] = name
        all_results[name] = results

        del model, trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save ablation results
    output_dir = os.path.join(config.output_dir, "ablation_orth_parameter")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "ablation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Ablation summary table
    print(f"\n{'='*60}")
    print(f"  ABLATION RESULTS")
    print(f"{'='*60}")
    print(f"{'Variant':<30} {'FP':>8} {'AP':>8} {'FT':>8} {'BWT':>8} {'LA':>8}")
    print("-" * 74)
    for name, res in all_results.items():
        fp = res["cl_metrics"]["Final Performance (FP)"]
        ap = res["cl_metrics"]["Average Performance (AP)"]
        ft = res["cl_metrics"]["Forgetting (FT)"]
        bwt = res["cl_metrics"]["Backward Transfer (BWT)"]
        la = res["cl_metrics"]["Learning Accuracy (LA)"]
        print(f"{name:<30} {fp:>8.4f} {ap:>8.4f} {ft:>8.4f} {bwt:>8.4f} {la:>8.4f}")

    has_premerge = all("cl_metrics_premerge" in res for res in all_results.values())
    if has_premerge:
        print(f"\n{'Variant':<30} {'FP(pre)':>10} {'AP(pre)':>10} {'FT(pre)':>10} {'BWT(pre)':>10} {'LA(pre)':>10}")
        print("-" * 88)
        for name, res in all_results.items():
            pre = res["cl_metrics_premerge"]
            print(
                f"{name:<30} "
                f"{pre['Final Performance (FP)']:>10.4f} "
                f"{pre['Average Performance (AP)']:>10.4f} "
                f"{pre['Forgetting (FT)']:>10.4f} "
                f"{pre['Backward Transfer (BWT)']:>10.4f} "
                f"{pre['Learning Accuracy (LA)']:>10.4f}"
            )

    with open(os.path.join(output_dir, "ablation_table.txt"), "w") as f:
        f.write(f"{'Variant':<30} {'FP':>8} {'AP':>8} {'FT':>8} {'BWT':>8} {'LA':>8}\n")
        f.write("-" * 74 + "\n")
        for name, res in all_results.items():
            fp = res["cl_metrics"]["Final Performance (FP)"]
            ap = res["cl_metrics"]["Average Performance (AP)"]
            ft = res["cl_metrics"]["Forgetting (FT)"]
            bwt = res["cl_metrics"]["Backward Transfer (BWT)"]
            la = res["cl_metrics"]["Learning Accuracy (LA)"]
            f.write(f"{name:<30} {fp:>8.4f} {ap:>8.4f} {ft:>8.4f} {bwt:>8.4f} {la:>8.4f}\n")

        if has_premerge:
            f.write("\n")
            f.write(f"{'Variant':<30} {'FP(pre)':>10} {'AP(pre)':>10} {'FT(pre)':>10} {'BWT(pre)':>10} {'LA(pre)':>10}\n")
            f.write("-" * 88 + "\n")
            for name, res in all_results.items():
                pre = res["cl_metrics_premerge"]
                f.write(
                    f"{name:<30} "
                    f"{pre['Final Performance (FP)']:>10.4f} "
                    f"{pre['Average Performance (AP)']:>10.4f} "
                    f"{pre['Forgetting (FT)']:>10.4f} "
                    f"{pre['Backward Transfer (BWT)']:>10.4f} "
                    f"{pre['Learning Accuracy (LA)']:>10.4f}\n"
                )

    print(f"\n  Ablation results saved to {output_dir}/")
    return all_results


# ======================================================================
# CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="DLOG Experiment Runner")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick sanity check with minimal steps")
    parser.add_argument("--ablation", action="store_true",
                        help="Run 3 orth ablations (parameter projection only)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B",
                        help="Model name (default: Qwen/Qwen2.5-1.5B). "
                             "Example: --model Qwen/Qwen2.5-0.5B")
    parser.add_argument("--rank", type=int, default=8,
                        help="LoRA rank (default: 8)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override training steps per task")
    parser.add_argument("--projection", type=str, default="parameter",
                        choices=["parameter", "memory_gradient"],
                        help="Projection type for hard constraint")
    parser.add_argument("--lambda-orth", type=float, default=None,
                        help="Soft constraint weight (default: auto — 0.01 with hard, 0.10 without)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--no-soft", action="store_true",
                        help="Disable soft constraint")
    parser.add_argument("--no-hard", action="store_true",
                        help="Disable hard constraint")
    parser.add_argument("--no-replay", action="store_true",
                        help="Disable replay for ALL methods (pure sequential fine-tuning)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU (for testing)")

    args = parser.parse_args()

    config = DLOGConfig(
        model_name=args.model,
        lora_rank=args.rank,
        **(dict(lambda_orth=args.lambda_orth, auto_lambda_orth=False) if args.lambda_orth is not None else {}),
        projection_type=args.projection,
        use_soft_constraint=not args.no_soft,
        use_hard_constraint=not args.no_hard,
        no_replay=args.no_replay,
        output_dir=args.output_dir,
        device="cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"),
        fp16=False,  # Disabled: multi-backward loop incompatible with GradScaler
    )

    if args.steps:
        config.num_train_steps_per_task = args.steps

    if args.ablation:
        run_ablation(config, smoke_test=args.smoke_test)
    else:
        run_experiment(config, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
