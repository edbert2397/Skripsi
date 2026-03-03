"""
DLOG Trainer — orchestrates the continual learning loop.

Training procedure for each task t:
  1. (Phase 1) If first task: train Fast LoRA only (Slow is zero-init, silent).
     Slow stays at zero throughout Task 1 (gradient chain algebraically broken).
  2. (Phase 2) Subsequent tasks:
       - Freeze Slow LoRA completely — NO EMA updates during training.
         This is critical: P_slow = I - (A_s A_s^T)/(||A_s||^2 + λ) must be
         FIXED throughout the task so the null-space projection is stable.
       - Train Fast LoRA on mixed batches (task + replay)
       - Apply soft constraint (L_orth) + hard constraint (gradient projection)
  3. (End of each task) consolidate_after_task(): hard copy Fast → Slow, reset Fast.
     This is the ONLY time Slow is ever updated (per Section 3.1).
"""
import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from typing import Dict, Optional, List
from tqdm import tqdm

from config import DLOGConfig
from dlog_model import DLOGModel, BaselineModel
from orthogonal_gating import (
    compute_orth_loss,
    apply_hard_projection,
    MemoryGradientProjector,
)
from data_pipeline import (
    ReservoirReplayBuffer,
    SurpriseReplayBuffer,
    create_mixed_batch,
    build_task_dataloaders,
)
from metrics import (
    CLMetricsTracker,
    GradientConflictTracker,
    EfficiencyTracker,
    evaluate_accuracy,
    compute_all_subspace_overlaps,
)
from ipc_freeze import ImportanceTracker, FreezeManager


# ======================================================================
# DLOG Trainer
# ======================================================================
class DLOGTrainer:
    """
    Full DLOG continual learning trainer.
    """

    def __init__(self, config: DLOGConfig, model: DLOGModel, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = config.device

        # Move model to device
        self.model.to(self.device)

        # Replay buffer
        self.replay_buffer = ReservoirReplayBuffer(config.replay_buffer_size)

        # Memory-gradient projector (for Option 2)
        self.projector = MemoryGradientProjector(
            buffer_size=config.memory_grad_buffer_size
        )

        # Metrics
        self.cl_metrics = CLMetricsTracker(config.task_order)
        self.cl_metrics_premerge = CLMetricsTracker(config.task_order)
        self.grad_tracker = GradientConflictTracker()
        self.efficiency = EfficiencyTracker()

        # Logging
        self.train_log: List[Dict] = []

        # IPC: Important Module Freezing (arXiv:2504.13407 adaptation)
        if config.ipc_enabled:
            self.ipc_tracker = ImportanceTracker(config, model)
            self.ipc_manager = FreezeManager(config)
        else:
            self.ipc_tracker = None
            self.ipc_manager = None

    def train_all_tasks(
        self,
        train_loaders: Dict,
        eval_loaders: Dict,
        max_steps_override: Optional[int] = None,
    ):
        """Train sequentially over all tasks in config.task_order."""
        max_steps = max_steps_override or self.config.num_train_steps_per_task

        self.efficiency.start_timer()

        for task_idx, task_name in enumerate(self.config.task_order):
            print(f"\n{'='*60}")
            print(f"  Task {task_idx+1}/{len(self.config.task_order)}: {task_name}")
            print(f"{'='*60}")

            self._train_single_task(
                task_idx=task_idx,
                task_name=task_name,
                train_loader=train_loaders[task_name],
                eval_loaders=eval_loaders,
                max_steps=max_steps,
            )

            # Class-IL: number of classes seen so far (cumulative, used for eval masking).
            from config import DATASET_CONFIGS as _DC
            num_seen_classes = sum(
                _DC[t]["num_classes"] for t in self.config.task_order[:task_idx + 1]
            )

            # Evaluate before consolidation (training-time mode: base + slow + fast).
            print(f"\n  Evaluating after task {task_name} (pre-merge: base+slow+fast, seen classes={num_seen_classes})...")
            for eval_idx, eval_name in enumerate(self.config.task_order):
                acc = evaluate_accuracy(
                    self.model, eval_loaders[eval_name],
                    self.tokenizer, device=self.device,
                    num_seen_classes=num_seen_classes,
                )
                self.cl_metrics_premerge.record(task_idx, eval_idx, acc)
                print(f"    {eval_name}: {acc:.4f}")

            # Consolidate Fast → Slow and reset Fast so deployed evaluation uses base + slow.
            # IPC-frozen modules are skipped (their Slow slot is preserved, Fast is only reset).
            ipc_frozen = self.ipc_manager.get_frozen_keys() if self.ipc_manager else set()
            self.model.consolidate_after_task(skip_keys=ipc_frozen)
            print(f"  Consolidated Fast → Slow for task {task_name}. Fast LoRA reset for next task.")

            # --- IPC task-boundary: score → select → freeze → reset stats ---
            if self.config.ipc_enabled and self.ipc_tracker is not None:
                self.ipc_manager.select_and_freeze(
                    self.ipc_tracker, self.model, task_idx
                )
                # full reset: treat next task independently
                self.ipc_tracker.reset_stats(decay=False)
                # Re-apply frozen requires_grad after consolidation reset Fast params.
                self.ipc_manager.apply_freezing_to_model(self.model)

            # Evaluate in deployed mode (base + slow) after consolidation.
            print(f"\n  Evaluating after task {task_name} (deployed: base+slow, seen classes={num_seen_classes})...")
            for eval_idx, eval_name in enumerate(self.config.task_order):
                acc = evaluate_accuracy(
                    self.model, eval_loaders[eval_name],
                    self.tokenizer, device=self.device,
                    num_seen_classes=num_seen_classes,
                )
                self.cl_metrics.record(task_idx, eval_idx, acc)
                print(f"    {eval_name}: {acc:.4f}")

        self.efficiency.stop_timer()
        self.efficiency.update_memory()

    def _train_single_task(
        self,
        task_idx: int,
        task_name: str,
        train_loader,
        eval_loaders: Dict,
        max_steps: int,
    ):
        is_first_task = (task_idx == 0)

        # --- Setup optimizer (DLOG: train Fast LoRA + classification head every task) ---
        self.model.freeze_slow()
        # First, ensure all fast params are trainable (then re-freeze IPC frozen ones below).
        for p in self.model.get_fast_params():
            p.requires_grad = True
        # IPC: exclude frozen modules from optimizer and set their requires_grad=False.
        if self.config.ipc_enabled and self.ipc_manager is not None:
            self.ipc_manager.apply_freezing_to_model(self.model)
            fast_params = self.ipc_manager.get_trainable_fast_params(self.model)
        else:
            fast_params = self.model.get_fast_params()

        head_params = []
        for name, p in self.model.base_model.named_parameters():
            if "score" in name:
                p.requires_grad = True
                head_params.append(p)

        # Remove duplicates while preserving order.
        seen = set()
        params = []
        for p in fast_params + head_params:
            pid = id(p)
            if pid not in seen:
                seen.add(pid)
                params.append(p)

        # Debug print once per task to verify classification head is trainable.
        score_module = getattr(self.model.base_model, "score", None)
        if score_module is not None and hasattr(score_module, "weight"):
            print(f"  [DLOG] score.weight.requires_grad={score_module.weight.requires_grad}")
        else:
            score_param = next(
                ((name, p) for name, p in self.model.base_model.named_parameters() if "score" in name),
                None,
            )
            if score_param is not None:
                print(f"  [DLOG] {score_param[0]}.requires_grad={score_param[1].requires_grad}")
            else:
                print("  [DLOG] score.* params not found.")

        if not is_first_task:
            # Reset projector for new task
            self.projector.reset()

        optimizer = AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(self.config.warmup_steps, max_steps // 10),
            num_training_steps=max_steps,
        )
        scaler = GradScaler(enabled=self.config.fp16)

        # --- Training loop ---
        self.model.train()
        step = 0
        epoch = 0
        dual_layers = self.model.get_dual_lora_layers()

        pbar = tqdm(total=max_steps, desc=f"Training {task_name}")

        while step < max_steps:
            epoch += 1
            for batch in train_loader:
                if step >= max_steps:
                    break

                # --- 1. Prepare batch (with replay if not first task) ---
                if is_first_task:
                    # No replay for first task, but fill buffer
                    combined_batch = {k: v.to(self.device) for k, v in batch.items()}
                    replay_batch = None
                    self.replay_buffer.add_batch(batch)
                else:
                    combined_batch, replay_batch = create_mixed_batch(
                        batch, self.replay_buffer,
                        replay_ratio=self.config.replay_ratio,
                        device=self.device,
                    )

                # --- 2. Pass 1: Memory-gradient basis update (Only if using memory_gradient) ---
                is_projection_step = (step % self.config.project_every_k == 0)
                if (
                    not is_first_task
                    and self.config.use_hard_constraint
                    and self.config.projection_type == "memory_gradient"
                    and replay_batch is not None
                    and is_projection_step
                ):
                    optimizer.zero_grad()
                    with autocast(device_type="cuda", enabled=self.config.fp16):
                        mem_outputs = self.model(
                            input_ids=replay_batch["input_ids"],
                            attention_mask=replay_batch["attention_mask"],
                            labels=replay_batch["labels"],
                        )
                    scaler.scale(mem_outputs.loss).backward()
                    scaler.unscale_(optimizer)
                    self.efficiency.record_forward()
                    
                    # Accumulate memory-gradient basis before main task backward
                    fast_params = self.model.get_fast_params()
                    self.projector.accumulate(fast_params)

                # --- 3. Pass 2: Main Task Forward + Backward ---
                optimizer.zero_grad()

                with autocast(device_type="cuda", enabled=self.config.fp16):
                    outputs = self.model(
                        input_ids=combined_batch["input_ids"],
                        attention_mask=combined_batch["attention_mask"],
                        labels=combined_batch["labels"],
                    )
                    task_loss = outputs.loss

                    # Soft orthogonality constraint
                    if not is_first_task and self.config.use_soft_constraint:
                        orth_loss = compute_orth_loss(dual_layers)
                        total_loss = task_loss + self.config.lambda_orth * orth_loss
                    else:
                        orth_loss = torch.tensor(0.0)
                        total_loss = task_loss

                self.efficiency.record_forward()

                # --- 4. Backward & Gradients ready for projection ---
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)

                # --- 4b. IPC: update importance stats from current gradients ---
                # Must run after unscale_ so grads are in the original scale,
                # and before zero_grad() which happens at the start of the next iteration.
                if (
                    self.config.ipc_enabled
                    and self.ipc_tracker is not None
                    and step % self.config.ipc_update_every_n_steps == 0
                ):
                    self.ipc_tracker.update_from_grads()

                # --- 5. Hard orthogonal projection ---
                if (
                    not is_first_task
                    and self.config.use_hard_constraint
                    and step % self.config.project_every_k == 0
                ):
                    self.efficiency.start_projection_timer()
                    apply_hard_projection(
                        dual_layers,
                        projector=self.projector,
                        projection_type=self.config.projection_type,
                    )
                    self.efficiency.stop_projection_timer()

                # --- 6. Gradient clipping and optimizer step ---
                nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # --- 6b. IPC: replay-only scoring microbatch (optional) ---
                # Runs a separate scoring-only backward on pure replay data
                # WITHOUT calling optimizer.step(), then clears grads.
                # This enriches bar_I / bar_U with signal from old tasks, combating
                # the bias toward the current task in the mixed-batch statistics.
                if (
                    self.config.ipc_enabled
                    and self.ipc_tracker is not None
                    and self.config.ipc_scoring_microbatch_enabled
                    and not is_first_task
                    and step % self.config.ipc_scoring_microbatch_every_k_steps == 0
                    and len(self.replay_buffer) > 0
                ):
                    self._run_ipc_scoring_microbatch(optimizer)

                # --- 7. NO EMA UPDATE (Section 3.1 compliance) ---
                # Slow LoRA is ONLY updated at the end of each task via
                # consolidate_after_task(). Keeping Slow frozen during training
                # ensures P_slow = I - (A_s A_s^T)/(||A_s||^2 + λ) is constant,
                # so every gradient step projects Fast into the SAME null-space.
                # EMA would make the null-space a moving target, violating the
                # stable orthogonal guarantee of Section 3.1.

                # --- 8. Logging ---
                step += 1
                if step % self.config.log_every == 0:
                    log_entry = {
                        "task": task_name,
                        "step": step,
                        "task_loss": task_loss.item(),
                        "orth_loss": orth_loss.item() if isinstance(orth_loss, torch.Tensor) else orth_loss,
                    }

                    # Compute leakage if using memory-gradient projection
                    if (
                        not is_first_task
                        and self.config.use_hard_constraint
                        and self.config.projection_type == "memory_gradient"
                    ):
                        leakage = self.projector.compute_leakage(
                            self.model.get_fast_params()
                        )
                        log_entry["leakage"] = leakage

                    self.train_log.append(log_entry)
                    pbar.set_postfix(
                        loss=f"{task_loss.item():.4f}",
                        orth=f"{orth_loss.item() if isinstance(orth_loss, torch.Tensor) else 0:.4f}",
                    )

                pbar.update(1)

        pbar.close()

    # ------------------------------------------------------------------
    # IPC replay-only scoring microbatch
    # ------------------------------------------------------------------

    def _run_ipc_scoring_microbatch(self, optimizer) -> None:
        """
        Score-only forward+backward on a pure replay microbatch.

        - Does NOT call optimizer.step()  →  parameters unchanged.
        - Calls ipc_tracker.update_from_grads() to enrich stats.
        - Clears gradients immediately after so the next training step
          starts from a clean slate.
        """
        micro_bsz = self.config.ipc_scoring_microbatch_bsz
        micro_batch = self.replay_buffer.sample(micro_bsz)
        if micro_batch is None:
            return

        micro_batch = {k: v.to(self.device) for k, v in micro_batch.items()}

        # Scoring forward+backward (no mixed precision needed — already unscaled).
        with torch.autocast(device_type="cuda", enabled=self.config.fp16):
            outputs = self.model(
                input_ids=micro_batch["input_ids"],
                attention_mask=micro_batch["attention_mask"],
                labels=micro_batch["labels"],
            )
        outputs.loss.backward()

        # Update EMA stats from the fresh replay-only gradients.
        self.ipc_tracker.update_from_grads()

        # Clear gradients — do NOT call optimizer.step().
        optimizer.zero_grad()

    def get_results(self) -> Dict:
        """Gather all results into a single dict."""
        subspace = compute_all_subspace_overlaps(self.model.get_dual_lora_layers())
        return {
            "method": "DLOG",
            "config": {
                "projection_type": self.config.projection_type,
                "use_soft": self.config.use_soft_constraint,
                "use_hard": self.config.use_hard_constraint,
                "lambda_orth": self.config.lambda_orth,
                "lora_rank": self.config.lora_rank,
                "ema_decay": self.config.ema_decay,
            },
            "cl_metrics": self.cl_metrics.summary(),
            "cl_metrics_premerge": self.cl_metrics_premerge.summary(),
            "efficiency": self.efficiency.summary(),
            "gradient_conflict": self.grad_tracker.history,
            "subspace_overlap": {
                "mean_A_separation": subspace["mean_A_separation"],
                "mean_B_separation": subspace["mean_B_separation"],
            },
            "train_log": self.train_log,
        }


# ======================================================================
# Baseline Trainer (Single LoRA, no gating)
# ======================================================================
class BaselineTrainer:
    """
    Baseline: Single LoRA, sequential finetuning with optional replay.
    No orthogonal gating.
    """

    def __init__(self, config: DLOGConfig, model: BaselineModel, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = config.device

        self.model.to(self.device)
        self.replay_buffer = ReservoirReplayBuffer(config.replay_buffer_size)
        self.cl_metrics = CLMetricsTracker(config.task_order)
        self.efficiency = EfficiencyTracker()
        self.train_log: List[Dict] = []

    def train_all_tasks(
        self,
        train_loaders: Dict,
        eval_loaders: Dict,
        max_steps_override: Optional[int] = None,
    ):
        max_steps = max_steps_override or self.config.num_train_steps_per_task

        self.efficiency.start_timer()

        for task_idx, task_name in enumerate(self.config.task_order):
            print(f"\n{'='*60}")
            print(f"  [Baseline] Task {task_idx+1}/{len(self.config.task_order)}: {task_name}")
            print(f"{'='*60}")

            self._train_single_task(
                task_idx, task_name, train_loaders[task_name],
                eval_loaders, max_steps,
            )

            # Class-IL: number of classes seen so far (cumulative, used for eval masking).
            from config import DATASET_CONFIGS as _DC
            num_seen_classes = sum(
                _DC[t]["num_classes"] for t in self.config.task_order[:task_idx + 1]
            )
            print(f"\n  Evaluating after task {task_name} (seen classes={num_seen_classes})...")
            for eval_idx, eval_name in enumerate(self.config.task_order):
                acc = evaluate_accuracy(
                    self.model, eval_loaders[eval_name],
                    self.tokenizer, device=self.device,
                    num_seen_classes=num_seen_classes,
                )
                self.cl_metrics.record(task_idx, eval_idx, acc)
                print(f"    {eval_name}: {acc:.4f}")

        self.efficiency.stop_timer()
        self.efficiency.update_memory()

    def _train_single_task(
        self, task_idx, task_name, train_loader, eval_loaders, max_steps
    ):
        params = self.model.get_lora_params()
        for p in params:
            p.requires_grad = True

        optimizer = AdamW(
            params, lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, self.config.warmup_steps, max_steps,
        )
        scaler = GradScaler(enabled=self.config.fp16)

        self.model.train()
        step = 0
        pbar = tqdm(total=max_steps, desc=f"[Baseline] {task_name}")

        while step < max_steps:
            for batch in train_loader:
                if step >= max_steps:
                    break

                if task_idx == 0:
                    combined_batch = {k: v.to(self.device) for k, v in batch.items()}
                    self.replay_buffer.add_batch(batch)
                else:
                    combined_batch, _ = create_mixed_batch(
                        batch, self.replay_buffer,
                        replay_ratio=self.config.replay_ratio,
                        device=self.device,
                    )

                optimizer.zero_grad()
                with autocast(device_type="cuda", enabled=self.config.fp16):
                    outputs = self.model(
                        input_ids=combined_batch["input_ids"],
                        attention_mask=combined_batch["attention_mask"],
                        labels=combined_batch["labels"],
                    )
                    loss = outputs.loss

                self.efficiency.record_forward()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                step += 1
                if step % self.config.log_every == 0:
                    self.train_log.append({
                        "task": task_name, "step": step, "loss": loss.item(),
                    })
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update(1)

        pbar.close()

    def get_results(self) -> Dict:
        return {
            "method": "Baseline (Single LoRA + Replay)",
            "cl_metrics": self.cl_metrics.summary(),
            "efficiency": self.efficiency.summary(),
            "train_log": self.train_log,
        }


# ======================================================================
# SuReTrainer (SuRe-Style)
# ======================================================================
class SuReTrainer:
    """
    SuRe (Surprise-prioritised Replay + EMA integration).
    Uses DLOGModel for Dual-LoRA and EMA, but no orthogonal gating.
    """

    def __init__(self, config: DLOGConfig, model: DLOGModel, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = config.device

        self.model.to(self.device)
        self.replay_buffer = SurpriseReplayBuffer(config.replay_buffer_size)

        self.cl_metrics = CLMetricsTracker(config.task_order)
        self.efficiency = EfficiencyTracker()
        self.train_log: List[Dict] = []

    def train_all_tasks(
        self,
        train_loaders: Dict,
        eval_loaders: Dict,
        max_steps_override: Optional[int] = None,
    ):
        max_steps = max_steps_override or self.config.num_train_steps_per_task
        self.efficiency.start_timer()

        for task_idx, task_name in enumerate(self.config.task_order):
            print(f"\n{'='*60}")
            print(f"  [SuRe-Style] Task {task_idx+1}/{len(self.config.task_order)}: {task_name}")
            print(f"{'='*60}")

            self._train_single_task(task_idx, task_name, train_loaders[task_name], max_steps)

            # Class-IL: number of classes seen so far (cumulative, used for eval masking).
            from config import DATASET_CONFIGS as _DC
            num_seen_classes = sum(
                _DC[t]["num_classes"] for t in self.config.task_order[:task_idx + 1]
            )
            print(f"\n  Evaluating after task {task_name} (seen classes={num_seen_classes})...")
            for eval_idx, eval_name in enumerate(self.config.task_order):
                acc = evaluate_accuracy(
                    self.model, eval_loaders[eval_name],
                    self.tokenizer, device=self.device,
                    num_seen_classes=num_seen_classes,
                )
                self.cl_metrics.record(task_idx, eval_idx, acc)
                print(f"    {eval_name}: {acc:.4f}")

        self.efficiency.stop_timer()
        self.efficiency.update_memory()

    def _train_single_task(self, task_idx, task_name, train_loader, max_steps):
        is_first_task = (task_idx == 0)

        if is_first_task:
            self.model.unfreeze_all_lora()
            params = self.model.get_all_lora_params()
        else:
            self.model.freeze_slow()
            params = self.model.get_fast_params()

        optimizer = AdamW(
            params, lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=min(self.config.warmup_steps, max_steps // 10),
            num_training_steps=max_steps,
        )
        scaler = GradScaler(enabled=self.config.fp16)

        self.model.train()
        step = 0
        pbar = tqdm(total=max_steps, desc=f"[SuRe] {task_name}")

        while step < max_steps:
            for batch in train_loader:
                if step >= max_steps:
                    break

                batch_device = {k: v.to(self.device) for k, v in batch.items()}

                # --- Compute NLL (Surprise) ---
                with torch.no_grad():
                    outputs_for_nll = self.model(
                        input_ids=batch_device["input_ids"],
                        attention_mask=batch_device["attention_mask"],
                    )
                    loss_fct = nn.CrossEntropyLoss(reduction='none')
                    per_sample_nll = loss_fct(outputs_for_nll.logits, batch_device["labels"]).cpu().tolist()
                self.efficiency.record_forward()

                # Add to Surprise Replay Buffer
                self.replay_buffer.add_batch(batch, per_sample_nll)

                if is_first_task:
                    combined_batch = batch_device
                else:
                    combined_batch, _ = create_mixed_batch(
                        batch, self.replay_buffer,
                        replay_ratio=self.config.replay_ratio,
                        device=self.device,
                    )

                optimizer.zero_grad()
                with autocast(device_type="cuda", enabled=self.config.fp16):
                    outputs = self.model(
                        input_ids=combined_batch["input_ids"],
                        attention_mask=combined_batch["attention_mask"],
                        labels=combined_batch["labels"],
                    )
                    loss = outputs.loss

                self.efficiency.record_forward()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                if not is_first_task:
                    self.model.ema_update_slow(self.config.ema_decay)

                step += 1
                if step % self.config.log_every == 0:
                    self.train_log.append({
                        "task": task_name, "step": step, "loss": loss.item(),
                    })
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update(1)

        pbar.close()

    def get_results(self) -> Dict:
        return {
            "method": "SuRe-Style (EMA + Surprise Replay)",
            "cl_metrics": self.cl_metrics.summary(),
            "efficiency": self.efficiency.summary(),
            "train_log": self.train_log,
        }
