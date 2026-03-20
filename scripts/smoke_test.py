#!/usr/bin/env python3
"""
smoke_test.py - End-to-end sanity check on tiny data.

Runs the EXACT same pipeline as run_experiment.py but with:
  - 10 training samples, 5 test samples per task
  - 2 tasks only (ag_news -> amazon_reviews)
  - Tiny buffer (5 slots)
  - 1 gradient accumulation step (no accumulation overhead)
  - All (or one specified) selector(s), each with a fresh model

Usage:
  python scripts/smoke_test.py                      # test all 4 selectors
  python scripts/smoke_test.py --selection surprise
  python scripts/smoke_test.py --no-ema
  python scripts/smoke_test.py --use-orthogonal-lora
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
from src.models.orthogonal_lora import OrthogonalLoRAModel
from src.selection import build_selector
from src.buffer.replay_buffer import ReplayBuffer
from src.training.trainer import ContinualTrainer
from src.training.evaluator import Evaluator
from src.data.datasets import load_task

# ---- Fixed smoke-test scale ----
SMOKE_TASKS   = ["ag_news", "amazon_reviews"]  # first 2 of standard_cl order 0
N_TRAIN       = 10
N_TEST        = 5
BUFFER_SIZE   = 5
ALL_SELECTORS = ["reservoir", "surprise", "orthogonal", "feature", "hybrid"]


def make_cfg(selection: str, use_ema: bool, fp16: bool) -> SimpleNamespace:
    return SimpleNamespace(
        selection_method=selection,
        use_ema=use_ema,
        update_buffer_before=False,
        replay_interval=2,
        grad_accum_steps=1,       # no accumulation — keeps smoke run fast
        fp16=fp16,
        lr=1e-3,
        batch_size_current=2,
        batch_size_replay=2,
        subspace_rank_k=2,        # tiny subspace; still exercises SVD path
        n_estimation_batches=2,
        grad_batch_size=2,
        hybrid_alpha=0.5,
        log_every_n_steps=9999,   # suppress mid-step logging noise
    )


def run_one(
    selection: str,
    base_t5,
    tokenizer,
    device: torch.device,
    use_ema: bool,
    use_orthogonal_lora: bool,
    fp16: bool,
) -> float:
    """
    Run the full 2-task CL pipeline for one selector.
    Returns Final Performance (FP) accuracy.

    NOTE: deepcopy(base_t5) is required because DualLoRAModel / OrthogonalLoRAModel
    call get_peft_model(base_model, ...) which wraps base_model in-place,
    making it a PeftModel. Without deepcopy the second selector run would fail.
    """
    cfg = make_cfg(selection, use_ema, fp16)

    ModelClass = OrthogonalLoRAModel if use_orthogonal_lora else DualLoRAModel
    model = ModelClass(
        base_model=copy.deepcopy(base_t5),
        lora_rank=8,
        lora_alpha=32,
        lora_target_modules=["q", "v"],
        beta=0.995,
        use_gradient_checkpointing=True,
    ).to(device)

    selector  = build_selector(cfg)
    buffer    = ReplayBuffer(max_size=BUFFER_SIZE)
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

    # Load tiny data for both tasks upfront
    all_data = {}
    for task_name in SMOKE_TASKS:
        train_samples, train_loader, test_loader = load_task(
            task_name, tokenizer, N_TRAIN, N_TEST,
            batch_size=cfg.batch_size_current, seed=42,
            num_workers=0,  # avoid worker process spawning on Windows
        )
        all_data[task_name] = (train_samples, train_loader, test_loader)

    # CL loop — identical structure to run_experiment.py main()
    all_test_loaders = {}
    for task_id, task_name in enumerate(SMOKE_TASKS):
        train_samples, train_loader, _ = all_data[task_name]

        for i, tn in enumerate(SMOKE_TASKS[: task_id + 1]):
            all_test_loaders[i] = all_data[tn][2]

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
    p = argparse.ArgumentParser(description="Smoke test — tiny end-to-end run")
    p.add_argument(
        "--selection",
        choices=ALL_SELECTORS + ["all"],
        default="all",
        help="Which selector(s) to test (default: all)",
    )
    p.add_argument("--no-ema", dest="use_ema", action="store_false", default=True)
    p.add_argument("--use-orthogonal-lora", action="store_true", default=False)
    p.add_argument("--no-fp16", dest="fp16", action="store_false", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    selectors = ALL_SELECTORS if args.selection == "all" else [args.selection]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Smoke] Device          : {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        print(f"[Smoke] GPU             : {props.name}, {props.total_memory / 1e9:.1f} GB VRAM")
    print(f"[Smoke] Tasks           : {SMOKE_TASKS}")
    print(f"[Smoke] n_train/n_test  : {N_TRAIN} / {N_TEST}")
    print(f"[Smoke] Selectors       : {selectors}")
    print(f"[Smoke] EMA={args.use_ema}  OrthogonalLoRA={args.use_orthogonal_lora}  fp16={args.fp16}")

    print("\n[Smoke] Loading T5-Large (once)...")
    base_t5   = T5ForConditionalGeneration.from_pretrained("t5-large")
    tokenizer = T5Tokenizer.from_pretrained("t5-large", legacy=False)

    results = {}
    for sel in selectors:
        print(f"\n{'='*55}")
        print(f"[Smoke] Selector: {sel}")
        print(f"{'='*55}")
        t0 = time.time()
        try:
            fp = run_one(
                selection=sel,
                base_t5=base_t5,
                tokenizer=tokenizer,
                device=device,
                use_ema=args.use_ema,
                use_orthogonal_lora=args.use_orthogonal_lora,
                fp16=args.fp16,
            )
            elapsed = time.time() - t0
            results[sel] = ("PASS", fp, elapsed)
            print(f"[Smoke] {sel}: PASS  FP={fp:.4f}  ({elapsed:.1f}s)")
        except Exception:
            elapsed = time.time() - t0
            results[sel] = ("FAIL", None, elapsed)
            print(f"[Smoke] {sel}: FAIL  ({elapsed:.1f}s)")
            traceback.print_exc()

    # ---- Summary ----
    print(f"\n{'='*55}")
    print("[Smoke] Summary")
    print(f"{'='*55}")
    all_pass = True
    for sel, (status, fp, elapsed) in results.items():
        fp_str = f"FP={fp:.4f}" if fp is not None else "FP=N/A "
        print(f"  {sel:12s}  {status}  {fp_str}  ({elapsed:.1f}s)")
        if status != "PASS":
            all_pass = False

    print()
    if all_pass:
        print("[Smoke] ALL PASSED")
        sys.exit(0)
    else:
        print("[Smoke] SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
