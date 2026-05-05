#!/usr/bin/env python3
"""
run_experiment.py - Main entry point for RQ1 continual learning experiments.

Usage examples:
  # Default (orthogonal + EMA, rq1, order 0, seed 42):
  python scripts/run_experiment.py

  # SuRe baseline:
  python scripts/run_experiment.py --selection surprise

  # Our method, no EMA (ablation):
  python scripts/run_experiment.py --selection orthogonal --no-ema

  # Triple Defense extension:
  python scripts/run_experiment.py --selection orthogonal --use-orthogonal-lora

  # Hybrid with custom alpha:
  python scripts/run_experiment.py --selection hybrid --hybrid-alpha 0.5
"""

import argparse
import sys
import os
import json
import torch
import random
import numpy as np
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from plot_results import plot_run

from src.models.dual_lora import DualLoRAModel
from src.models.orthogonal_lora import OrthogonalLoRAModel
from src.selection import build_selector
from src.buffer.replay_buffer import ReplayBuffer
from src.training.trainer import ContinualTrainer
from src.training.evaluator import Evaluator
from src.data.datasets import load_task
from src.data.task_orders import get_task_order
from src.utils.logging import Logger


# ---- RTX 4050 6GB defaults (paper-aligned) ----
# buffer_size is None here — resolved in main() based on benchmark:
#   rq1: 600  (2% of 15 * 2,000 = 30,000)
DEFAULTS = dict(
    benchmark="rq1",
    order=0,
    seed=42,
    selection="orthogonal",
    use_ema=True,
    use_orthogonal_lora=False,
    beta=0.995,                  # paper Appendix G; EMA memory window = 200 steps
    buffer_size=None,            # resolved per benchmark (see above)
    replay_ratio="1:2",
    subspace_rank_k=10,
    n_estimation_batches=10,
    grad_batch_size=4,           # mini-batch for per-sample grad estimation
    hybrid_alpha=0.5,
    lr=1e-3,
    batch_size_current=8,        # micro-batch; effective = 8*8 accum = 64 (paper spec)
    batch_size_replay=8,         # micro-batch; 4 triggers/step * 8 = 32 total (paper spec)
    grad_accum_steps=8,          # effective current batch = 64
    fp16=True,
    gradient_checkpointing=True,
    update_buffer_before=False,  # False = Orthogonal-After (best ablation variant)
    use_wandb=False,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Orthogonal Replay CL experiments")

    p.add_argument("--benchmark", choices=["rq1"], default=DEFAULTS["benchmark"])
    p.add_argument("--order", type=int, choices=[0], default=DEFAULTS["order"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])

    p.add_argument("--selection", choices=["orthogonal", "feature", "conflict", "surprise", "reservoir", "hybrid"],
                   default=DEFAULTS["selection"])
    p.add_argument("--no-ema", dest="use_ema", action="store_false", default=DEFAULTS["use_ema"])
    p.add_argument("--use-orthogonal-lora", action="store_true", default=DEFAULTS["use_orthogonal_lora"])

    p.add_argument("--beta", type=float, default=DEFAULTS["beta"])
    p.add_argument("--buffer-size", type=int, default=None,
                   help="Buffer size (default: 600 for rq1)")
    p.add_argument("--replay-ratio", choices=["1:2", "1:4", "1:8", "1:16"], default=DEFAULTS["replay_ratio"])

    p.add_argument("--subspace-rank-k", type=int, default=DEFAULTS["subspace_rank_k"])
    p.add_argument("--n-estimation-batches", type=int, default=DEFAULTS["n_estimation_batches"])
    p.add_argument("--grad-batch-size", type=int, default=DEFAULTS["grad_batch_size"])
    p.add_argument("--hybrid-alpha", type=float, default=DEFAULTS["hybrid_alpha"])

    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--batch-size-current", type=int, default=DEFAULTS["batch_size_current"])
    p.add_argument("--batch-size-replay", type=int, default=DEFAULTS["batch_size_replay"])
    p.add_argument("--grad-accum-steps", type=int, default=DEFAULTS["grad_accum_steps"])
    p.add_argument("--no-fp16", dest="fp16", action="store_false", default=DEFAULTS["fp16"])
    p.add_argument("--no-grad-checkpointing", dest="gradient_checkpointing", action="store_false",
                   default=DEFAULTS["gradient_checkpointing"])

    p.add_argument("--update-buffer-before", action="store_true",
                   default=DEFAULTS["update_buffer_before"],
                   help="Orthogonal-Before variant (default: After)")

    p.add_argument("--use-wandb", action="store_true", default=DEFAULTS["use_wandb"])
    p.add_argument("--run-name", type=str, default=None)

    return p.parse_args()


def build_run_name(cfg) -> str:
    sel = cfg.selection
    ema = "ema" if cfg.use_ema else "noema"
    olora = "_olora" if cfg.use_orthogonal_lora else ""
    timing = "_before" if cfg.update_buffer_before else ""
    return f"{cfg.benchmark}_ord{cfg.order}_s{cfg.seed}_{sel}_{ema}{olora}{timing}"


def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    # Resolve buffer size from benchmark if not explicitly overridden
    # rq1: 2% of 15 tasks * 2,000 train = 30,000 total
    if cfg.buffer_size is None:
        cfg.buffer_size = 600

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        print(f"[GPU] {props.name}, {props.total_memory / 1e9:.1f} GB VRAM")

    run_name = cfg.run_name or build_run_name(cfg)
    logger = Logger(cfg, run_name=run_name, use_wandb=cfg.use_wandb)

    # ---- Parse replay ratio ----
    ratio_num, ratio_den = map(int, cfg.replay_ratio.split(":"))
    cfg.replay_interval = ratio_den  # replay every ratio_den steps

    # ---- Load model ----
    print("[Model] Loading T5-Large...")
    base_model = T5ForConditionalGeneration.from_pretrained("t5-large", local_files_only=True)
    tokenizer = T5Tokenizer.from_pretrained("t5-large", legacy=False, local_files_only=True)

    ModelClass = OrthogonalLoRAModel if cfg.use_orthogonal_lora else DualLoRAModel
    model = ModelClass(
        base_model=base_model,
        lora_rank=8,
        lora_alpha=32,
        lora_target_modules=["q", "v"],
        beta=cfg.beta,
        use_gradient_checkpointing=cfg.gradient_checkpointing,
    ).to(device)

    # ---- Build selector ----
    cfg.selection_method = cfg.selection
    selector = build_selector(cfg)

    # ---- Build buffer ----
    buffer = ReplayBuffer(max_size=cfg.buffer_size, seed=cfg.seed)

    # ---- Benchmark setup ----
    # rq1: 2,000 balanced train / 1,000 balanced test per task
    task_order = get_task_order(cfg.benchmark, cfg.order)
    n_train, n_test = 2000, 1000

    # ---- Evaluator ----
    evaluator = Evaluator(device=device, use_fp16=cfg.fp16)

    # ---- Trainer ----
    trainer = ContinualTrainer(
        model=model,
        buffer=buffer,
        selector=selector,
        evaluator=evaluator,
        config=cfg,
        device=device,
        logger=logger,
    )

    # ---- Load all test loaders upfront ----
    all_test_loaders = {}
    print("[Data] Loading all tasks...")
    all_data = {}
    task_sizes = {}
    for task_name in task_order:
        train_samples, train_loader, test_loader = load_task(
            task_name, tokenizer, n_train, n_test,
            batch_size=cfg.batch_size_current,
            seed=cfg.seed,
        )
        n_tr = len(train_samples)
        n_te = len(test_loader.dataset)
        task_sizes[task_name] = {"train": n_tr, "test": n_te}
        print(f"  [Load] {task_name}: train={n_tr}, test={n_te}")
        all_data[task_name] = (train_samples, train_loader, test_loader)

    # Summary table so it's easy to scan against the requested n_train/n_test
    print(f"\n[Data] Sample counts per task (requested train={n_train}, test={n_test}):")
    print(f"  {'task':<16} {'train':>7} {'test':>7}")
    for tn in task_order:
        s = task_sizes[tn]
        print(f"  {tn:<16} {s['train']:>7} {s['test']:>7}")
    print()

    # ---- Main CL loop ----
    for task_id, task_name in enumerate(task_order):
        print(f"\n{'='*60}")
        print(f"[Task {task_id}/{len(task_order)-1}] {task_name}")
        print(f"{'='*60}")

        train_samples, train_loader, _ = all_data[task_name]

        # Register test loader for evaluation
        for i, tn in enumerate(task_order[:task_id + 1]):
            all_test_loaders[i] = all_data[tn][2]

        # Prepare O-LoRA for new task (no-op for DualLoRA)
        if hasattr(model, "prepare_for_new_task"):
            model.prepare_for_new_task()

        trainer.train_on_task(
            task_id=task_id,
            dataset=train_samples,
            dataloader=train_loader,
        )

        results = evaluator.evaluate_all_tasks(model, all_test_loaders, task_id)
        logger.log_task_result(task_id, {task_order[k]: v for k, v in results.items()})

    # ---- Final metrics ----
    summary = evaluator.summary()
    print(f"\n{'='*60}")
    print(f"[FINAL] FP={summary['FP']:.4f}  AP={summary['AP']:.4f}  Forgetting={summary['Forgetting']:.4f}")
    print(f"{'='*60}")

    out_path = logger.log_dir / "final_results.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "task_sizes": task_sizes, "all_results": {
            str(k): {str(kk): vv for kk, vv in v.items()}
            for k, v in evaluator.all_results.items()
        }}, f, indent=2)
    print(f"[Saved] {out_path}")

    plot_run(logger.log_dir, task_order)

    logger.finish()


if __name__ == "__main__":
    main()
