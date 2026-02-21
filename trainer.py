"""
DLOG Trainer — orchestrates the continual learning loop.

Training procedure for each task t:
  1. (Phase 1) If first task: train both Slow+Fast LoRA, fill replay buffer
  2. (Phase 2) Subsequent tasks:
       - Freeze Slow LoRA (or use EMA consolidation)
       - Train Fast LoRA on mixed batches (task + replay)
       - Apply soft constraint (L_orth) + hard constraint (gradient projection)
       - EMA update Slow LoRA after each step
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
        self.grad_tracker = GradientConflictTracker()
        self.efficiency = EfficiencyTracker()

        # Logging
        self.train_log: List[Dict] = []

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

            # Evaluate on all tasks seen so far
            print(f"\n  Evaluating after task {task_name}...")
            for eval_idx, eval_name in enumerate(self.config.task_order):
                acc = evaluate_accuracy(
                    self.model, eval_loaders[eval_name],
                    self.tokenizer, device=self.device,
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

        # --- Setup optimizer ---
        if is_first_task:
            # Train all LoRA params
            self.model.unfreeze_all_lora()
            params = self.model.get_all_lora_params()
        else:
            # Freeze Slow, train only Fast
            self.model.freeze_slow()
            params = self.model.get_fast_params()
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

                # --- 7. EMA consolidation (Slow ← Fast) ---
                if not is_first_task:
                    self.model.ema_update_slow(self.config.ema_decay)

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

            print(f"\n  Evaluating after task {task_name}...")
            for eval_idx, eval_name in enumerate(self.config.task_order):
                acc = evaluate_accuracy(
                    self.model, eval_loaders[eval_name],
                    self.tokenizer, device=self.device,
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

            print(f"\n  Evaluating after task {task_name}...")
            for eval_idx, eval_name in enumerate(self.config.task_order):
                acc = evaluate_accuracy(
                    self.model, eval_loaders[eval_name],
                    self.tokenizer, device=self.device,
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
