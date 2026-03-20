"""
trainer.py - Main continual learning training loop.

Phases per task:
  1. Gradient subspace estimation (selector.prepare_for_task)
  2. (Buffer update happens AFTER training: Orthogonal-After variant)
  3. Dual-learner training with replay
  4. EMA consolidation (after every optimizer step)
  5. Buffer update with selection

To do Orthogonal-Before, pass update_buffer_before=True.
"""

import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from typing import Dict, List, Optional
from tqdm import tqdm

from ..buffer.replay_buffer import ReplayBuffer
from ..selection.base import BaseSelector
from ..training.ema import ema_step
from ..training.evaluator import Evaluator


class ContinualTrainer:
    def __init__(
        self,
        model,
        buffer: ReplayBuffer,
        selector: BaseSelector,
        evaluator: Evaluator,
        config,
        device: torch.device,
        logger=None,
    ):
        self.model = model
        self.buffer = buffer
        self.selector = selector
        self.evaluator = evaluator
        self.config = config
        self.device = device
        self.logger = logger

        self.use_ema: bool = getattr(config, "use_ema", True)
        self.update_buffer_before: bool = getattr(config, "update_buffer_before", False)
        self.replay_interval: int = getattr(config, "replay_interval", 2)  # 1:2 ratio
        self.grad_accum_steps: int = getattr(config, "grad_accum_steps", 8)
        self.fp16: bool = getattr(config, "fp16", True)
        self.amp_dtype = torch.bfloat16 if (self.fp16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

        from torch.amp import GradScaler
        self._scaler = GradScaler("cuda") if (self.fp16 and self.amp_dtype == torch.float16) else None
        self._global_step = 0

    # ------------------------------------------------------------------
    def _build_optimizer(self):
        return AdamW(
            self.model.fast_lora_parameters(),
            lr=self.config.lr,
            weight_decay=0.01,
        )

    def _merge_batches(self, batch_cur: dict, batch_rep: List[dict]) -> dict:
        """Collate current batch with a list of replay samples."""
        if not batch_rep:
            return batch_cur

        def _stack(key, items):
            tensors = [b[key] for b in items if key in b]
            if not tensors:
                return None
            return torch.stack(tensors, dim=0).to(self.device)

        keys = set(batch_rep[0].keys())
        rep_batch = {k: _stack(k, batch_rep) for k in keys}

        merged = {}
        for k in batch_cur:
            cur_v = batch_cur[k]
            rep_v = rep_batch.get(k)
            if isinstance(cur_v, torch.Tensor) and rep_v is not None:
                # Pad to same length along seq dim
                max_len = max(cur_v.size(1), rep_v.size(1))
                pad_id = 0 if k != "labels" else -100
                if cur_v.size(1) < max_len:
                    cur_v = torch.nn.functional.pad(cur_v, (0, max_len - cur_v.size(1)), value=pad_id)
                if rep_v.size(1) < max_len:
                    rep_v = torch.nn.functional.pad(rep_v, (0, max_len - rep_v.size(1)), value=pad_id)
                merged[k] = torch.cat([cur_v, rep_v], dim=0)
            else:
                merged[k] = cur_v
        return merged

    # ------------------------------------------------------------------
    def train_on_task(
        self,
        task_id: int,
        dataset: list,
        dataloader,
        candidate_dataset: Optional[list] = None,
    ):
        """
        Full training procedure for one task.

        Args:
            task_id:           0-indexed task number
            dataset:           list of tokenised samples (for buffer selection)
            dataloader:        DataLoader for the current task
            candidate_dataset: if None, uses dataset for buffer selection
        """
        if candidate_dataset is None:
            candidate_dataset = dataset

        cfg = self.config
        optimizer = self._build_optimizer()

        # ---- Orthogonal-Before variant ----
        if self.update_buffer_before:
            self.selector.prepare_for_task(self.model, dataloader, self.device)
            self._update_buffer(task_id, candidate_dataset)

        # ---- Phase 3: Train with replay ----
        self.model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        n_steps = 0

        for step, batch in enumerate(tqdm(dataloader, desc=f"Task {task_id}", leave=False)):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Sample replay batch (1:replay_interval ratio)
            if step % self.replay_interval == 0 and len(self.buffer) > 0:
                rep_samples = self.buffer.sample(cfg.batch_size_replay)
                batch = self._merge_batches(batch, rep_samples)

            # Forward + backward
            if self.fp16:
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    loss = self.model.compute_loss(batch)
                loss = loss / self.grad_accum_steps
                if self._scaler is not None:
                    self._scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                loss = self.model.compute_loss(batch) / self.grad_accum_steps
                loss.backward()

            total_loss += loss.item() * self.grad_accum_steps

            if (step + 1) % self.grad_accum_steps == 0:
                if self.fp16:
                    if self._scaler is not None:
                        self._scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.fast_lora_parameters(), 1.0)
                        self._scaler.step(optimizer)
                        self._scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.fast_lora_parameters(), 1.0)
                        optimizer.step()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.fast_lora_parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad()

                # ---- Phase 4: EMA consolidation ----
                if self.use_ema:
                    ema_step(self.model)

                if self.logger and n_steps % getattr(cfg, "log_every_n_steps", 50) == 0:
                    self.logger.log({"train/loss": total_loss / max(1, n_steps)}, step=self._global_step)

                n_steps += 1
                self._global_step += 1

        # ---- Phase 2: Update buffer (Orthogonal-After, default) ----
        if not self.update_buffer_before:
            self.selector.prepare_for_task(self.model, dataloader, self.device)
            self._update_buffer(task_id, candidate_dataset)

        avg_loss = total_loss / max(1, n_steps)
        print(f"[Task {task_id}] avg_loss={avg_loss:.4f} | buffer={len(self.buffer)}")

        # Notify model about task completion (for O-LoRA)
        if hasattr(self.model, "accumulate_task_subspace"):
            self.model.accumulate_task_subspace()

    def _update_buffer(self, task_id: int, candidate_dataset: list):
        """Run selection and update buffer for current task."""
        n_tasks_after = self.buffer.n_tasks() + (0 if task_id in self.buffer._store else 1)
        quota = max(1, self.buffer.max_size // n_tasks_after)
        print(f"[Buffer] Selecting {quota} samples for task {task_id} from {len(candidate_dataset)} candidates...")
        selected = self.selector.select_for_buffer(
            model=self.model,
            candidates=candidate_dataset,
            quota=quota,
            device=self.device,
        )
        self.buffer.update(task_id, selected)
        print(f"[Buffer] Updated: {self.buffer.task_counts()}")
