#!/usr/bin/env python3
"""
smoke_test_comparison.py - End-to-end sanity check for run_comparison.py methods.

Runs the EXACT same pipeline as run_comparison.py (orthogonal, surprise, reservoir).

Benchmark modes:
  standard_cl  - Tiny smoke test (2 tasks, 10 train, 1 test, 5-slot buffer)
  lnt          - Full LNT benchmark (15 tasks, 1000 train, 500 test, 300-slot buffer)
                 This is the paper's primary setting where surprise-based selection
                 shows the largest gains over random replay.

EMA settings mirror run_comparison.py exactly:
  orthogonal  -> EMA ON   (our method)
  surprise    -> EMA ON   (SuRe baseline)
  reservoir   -> EMA OFF  (random baseline, no scores to smooth)

Usage:
  python scripts/smoke_test_comparison.py
  python scripts/smoke_test_comparison.py --methods orthogonal surprise
  python scripts/smoke_test_comparison.py --benchmark lnt --methods orthogonal surprise
  python scripts/smoke_test_comparison.py --no-fp16
"""

import sys
import time
import copy
import argparse
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dual_lora import DualLoRAModel
from src.selection import build_selector
from src.buffer.replay_buffer import ReplayBuffer
from src.training.trainer import ContinualTrainer
from src.training.evaluator import Evaluator
from src.data.datasets import load_task

# ---- Benchmark configurations ----
BENCHMARK_CONFIGS = {
    "standard_cl": {
        "tasks":       ["ag_news", "amazon_reviews"],   # first 2 tasks, order 0
        "n_train":     10,
        "n_test":      1,
        "buffer_size": 5,
        "description": "Smoke test (2 tasks, tiny data)",
    },
    "lnt": {
        # All 15 LNT tasks, but tiny data — verifies every dataset loads and
        # the 15-task pipeline runs end-to-end without crashing.
        # For the real experiment use run_experiment.py / run_comparison.py.
        "tasks": [
            "ag_news", "amazon_reviews", "dbpedia", "yahoo_answers",
            "mnli", "qqp", "rte", "sst2",
            "wic", "cb", "copa", "boolq", "multirc", "imdb", "sst2_v2",
        ],
        "n_train":     10,    # tiny — just enough to exercise the training loop
        "n_test":      1,     # one sample to exercise the evaluator
        "buffer_size": 15,    # 1 slot per task (exercises rebalancing logic)
        "description": "LNT smoke test (15 tasks, tiny data — pipeline check only)",
    },
}

# Mirror run_comparison.py's EMA settings per method
COMPARISON_METHODS = {
    "orthogonal": {"selection": "orthogonal", "use_ema": True},
    "surprise":   {"selection": "surprise",   "use_ema": True},
    "reservoir":  {"selection": "reservoir",  "use_ema": False},
}


def make_cfg(selection: str, use_ema: bool, fp16: bool) -> SimpleNamespace:
    # Both benchmarks are smoke tests — use the same tiny settings.
    # Full-scale paper config (batch=64, accum=8, replay=32/step) lives in
    # run_experiment.py / run_comparison.py.
    return SimpleNamespace(
        selection_method=selection,
        use_ema=use_ema,
        update_buffer_before=False,
        replay_interval=2,
        grad_accum_steps=1,
        fp16=fp16,
        lr=1e-3,
        batch_size_current=2,
        batch_size_replay=2,
        subspace_rank_k=2,        # tiny subspace — still exercises SVD path
        n_estimation_batches=2,
        grad_batch_size=2,
        hybrid_alpha=0.5,
        log_every_n_steps=9999,   # suppress mid-step logging noise
    )


def run_one(
    method_name: str,
    base_t5,
    tokenizer,
    device: torch.device,
    fp16: bool,
    bench_cfg: dict,
) -> float:
    """
    Run the full CL pipeline for one comparison method on the given benchmark config.
    Returns Final Performance (FP) accuracy.
    """
    tasks       = bench_cfg["tasks"]
    n_train     = bench_cfg["n_train"]
    n_test      = bench_cfg["n_test"]
    buffer_size = bench_cfg["buffer_size"]

    settings  = COMPARISON_METHODS[method_name]
    selection = settings["selection"]
    use_ema   = settings["use_ema"]
    cfg       = make_cfg(selection, use_ema, fp16)

    model = DualLoRAModel(
        base_model=copy.deepcopy(base_t5),
        lora_rank=8,
        lora_alpha=32,
        lora_target_modules=["q", "v"],
        beta=0.995,
        use_gradient_checkpointing=True,
    ).to(device)

    selector  = build_selector(cfg)
    buffer    = ReplayBuffer(max_size=buffer_size)
    evaluator = Evaluator(device=device, use_fp16=fp16)
    trainer   = ContinualTrainer(
        model=model,
        buffer=buffer,
        selector=selector,
        evaluator=evaluator,
        config=cfg,
        device=device,
        logger=None,
    )

    # Load data for all tasks upfront
    print(f"  [Data] Loading {len(tasks)} tasks (n_train={n_train}, n_test={n_test})...")
    all_data = {}
    for task_name in tasks:
        train_samples, train_loader, test_loader = load_task(
            task_name, tokenizer, n_train, n_test,
            batch_size=cfg.batch_size_current, seed=42,
            num_workers=0,  # avoid worker spawning issues on Windows
        )
        all_data[task_name] = (train_samples, train_loader, test_loader)

    # CL loop — same structure as run_experiment.py
    all_test_loaders = {}
    for task_id, task_name in enumerate(tasks):
        train_samples, train_loader, _ = all_data[task_name]

        for i, tn in enumerate(tasks[: task_id + 1]):
            all_test_loaders[i] = all_data[tn][2]

        print(f"  [Task {task_id+1:2d}/{len(tasks)}] {task_name}")

        if hasattr(model, "prepare_for_new_task"):
            model.prepare_for_new_task()

        trainer.train_on_task(
            task_id=task_id,
            dataset=train_samples,
            dataloader=train_loader,
        )
        evaluator.evaluate_all_tasks(model, all_test_loaders, task_id)

    del model
    torch.cuda.empty_cache()

    return evaluator.summary()["FP"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Comparison pipeline for orthogonal vs surprise vs reservoir"
    )
    p.add_argument(
        "--methods", nargs="+",
        choices=list(COMPARISON_METHODS.keys()),
        default=list(COMPARISON_METHODS.keys()),
        help="Which method(s) to run (default: all)",
    )
    p.add_argument(
        "--benchmark",
        choices=list(BENCHMARK_CONFIGS.keys()),
        default="standard_cl",
        help="Benchmark setting: 'standard_cl' (tiny smoke test) or 'lnt' (full 15-task LNT run)",
    )
    p.add_argument("--no-fp16", dest="fp16", action="store_false", default=True)
    return p.parse_args()


def main():
    args      = parse_args()
    methods   = args.methods
    bench_cfg = BENCHMARK_CONFIGS[args.benchmark]
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Run] Benchmark        : {args.benchmark}  — {bench_cfg['description']}")
    print(f"[Run] Device           : {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        print(f"[Run] GPU              : {props.name}, {props.total_memory / 1e9:.1f} GB VRAM")
    print(f"[Run] Tasks ({len(bench_cfg['tasks']):2d})      : {bench_cfg['tasks']}")
    print(f"[Run] n_train/n_test   : {bench_cfg['n_train']} / {bench_cfg['n_test']}")
    print(f"[Run] Buffer size      : {bench_cfg['buffer_size']}  (rebalanced to floor({bench_cfg['buffer_size']}/d) per task)")
    print(f"[Run] Methods          : {methods}")
    print(f"[Run] fp16             : {args.fp16}")
    print()
    for m in methods:
        s = COMPARISON_METHODS[m]
        print(f"  {m:12s}  selection={s['selection']:12s}  EMA={s['use_ema']}")

    print("\n[Run] Loading T5-Large (once)...")
    base_t5   = T5ForConditionalGeneration.from_pretrained("t5-large")
    tokenizer = T5Tokenizer.from_pretrained("t5-large", legacy=False)

    results = {}
    for method in methods:
        print(f"\n{'='*60}")
        print(f"[Run] Method: {method}  [{args.benchmark}]")
        s = COMPARISON_METHODS[method]
        print(f"      selection={s['selection']}  EMA={s['use_ema']}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            fp      = run_one(method, base_t5, tokenizer, device, args.fp16, bench_cfg)
            elapsed = time.time() - t0
            results[method] = ("PASS", fp, elapsed)
            print(f"[Run] {method}: PASS  FP={fp:.4f}  ({elapsed:.1f}s)")
        except Exception:
            elapsed = time.time() - t0
            results[method] = ("FAIL", None, elapsed)
            print(f"[Run] {method}: FAIL  ({elapsed:.1f}s)")
            traceback.print_exc()

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"[Run] {args.benchmark.upper()} Comparison Summary")
    print(f"{'='*60}")
    print(f"  {'Method':12s}  {'Status':6s}  {'FP':>8s}  {'EMA':>5s}  Time")
    print(f"  {'-'*12}  {'-'*6}  {'-'*8}  {'-'*5}  ----")

    all_pass = True
    for method, (status, fp, elapsed) in results.items():
        fp_str  = f"{fp:.4f}" if fp is not None else "   N/A"
        ema_str = str(COMPARISON_METHODS[method]["use_ema"])
        print(f"  {method:12s}  {status:6s}  {fp_str:>8s}  {ema_str:>5s}  {elapsed:.1f}s")
        if status != "PASS":
            all_pass = False

    print()
    if all_pass:
        print(f"[Run] ALL PASSED  ({args.benchmark})")
        sys.exit(0)
    else:
        print(f"[Run] SOME TESTS FAILED  ({args.benchmark})")
        sys.exit(1)


if __name__ == "__main__":
    main()
