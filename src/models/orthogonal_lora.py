"""
orthogonal_lora.py - O-LoRA subspace allocation for Triple Defense (Insight 6).

After each task, we accumulate the used LoRA column space and project
the next task's A-matrix into the remaining null space, ensuring
Task t's updates are orthogonal to all prior tasks.
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import T5ForConditionalGeneration
from copy import deepcopy
from typing import List, Optional


class OrthogonalLoRAModel(nn.Module):
    """
    Maintains a single PEFT model but re-initialises the LoRA A matrices
    for each new task so they lie in the null space of all prior tasks.

    The EMA slow/fast structure is preserved on top of this.
    """

    def __init__(
        self,
        base_model: T5ForConditionalGeneration,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_target_modules: list = None,
        beta: float = 0.995,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()

        if lora_target_modules is None:
            lora_target_modules = ["q", "v"]

        for param in base_model.parameters():
            param.requires_grad_(False)

        if use_gradient_checkpointing:
            base_model.gradient_checkpointing_enable()

        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=0.1,
            bias="none",
        )
        self.fast_model = get_peft_model(base_model, lora_cfg)

        slow_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=0.0,
            bias="none",
        )
        self.slow_model = get_peft_model(deepcopy(base_model), slow_cfg)
        for param in self.slow_model.parameters():
            param.requires_grad_(False)

        self.beta = beta
        self.lora_rank = lora_rank

        # Accumulated column spaces: list of [d_in, k] tensors per layer
        # keyed by lora_A parameter name
        self._used_subspaces: dict = {}

        self._sync_slow_to_fast()

    # ------------------------------------------------------------------
    def _lora_A_params(self, model):
        for name, param in model.named_parameters():
            if "lora_A" in name:
                yield name, param

    def _lora_params(self, model):
        for name, param in model.named_parameters():
            if "lora_" in name:
                yield name, param

    def _sync_slow_to_fast(self):
        with torch.no_grad():
            for (_, pf), (_, ps) in zip(
                self._lora_params(self.fast_model),
                self._lora_params(self.slow_model),
            ):
                ps.data.copy_(pf.data)

    # ------------------------------------------------------------------
    def prepare_for_new_task(self):
        """
        Called BEFORE training on a new task.
        Re-initialises lora_A matrices so they are orthogonal to all
        previously used subspaces.
        """
        with torch.no_grad():
            for name, param in self._lora_A_params(self.fast_model):
                d_out, d_in = param.shape  # lora_A: [r, d_in]
                # param is [r, d_in]; we work in d_in space

                if name not in self._used_subspaces:
                    # First task: plain random init (following Hu et al.)
                    nn.init.kaiming_uniform_(param)
                else:
                    U_used = self._used_subspaces[name]  # [d_in, k_used]
                    # Project random init into null space of U_used
                    rand = torch.randn_like(param)  # [r, d_in]
                    # Orthogonal projection: rand - rand @ U @ U^T
                    # rand.T shape: [d_in, r]
                    proj = rand @ (U_used @ U_used.T)  # [r, d_in]
                    null_rand = rand - proj
                    # Normalise rows
                    norms = null_rand.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    param.data.copy_(null_rand / norms * 0.01)

                # Reset lora_B to zero (standard LoRA init)
                # Find corresponding B matrix
                b_name = name.replace("lora_A", "lora_B")
                for n2, p2 in self.fast_model.named_parameters():
                    if n2 == b_name:
                        nn.init.zeros_(p2)
                        break

    def accumulate_task_subspace(self):
        """
        Called AFTER training on a task.
        Adds the column space of each lora_A to _used_subspaces.
        """
        with torch.no_grad():
            for name, param in self._lora_A_params(self.fast_model):
                # param: [r, d_in] -> compute column space of A^T = row space of A
                A = param.data  # [r, d_in]
                # SVD to get column space basis
                try:
                    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
                    # Vh: [r, d_in]; rows are right singular vectors
                    new_basis = Vh.T  # [d_in, r]
                except Exception:
                    new_basis = A.T / (A.norm() + 1e-8)

                if name not in self._used_subspaces:
                    self._used_subspaces[name] = new_basis
                else:
                    existing = self._used_subspaces[name]  # [d_in, k]
                    combined = torch.cat([existing, new_basis], dim=1)
                    # Re-orthogonalise to avoid numerical drift
                    Q, _ = torch.linalg.qr(combined)
                    self._used_subspaces[name] = Q

    # ------------------------------------------------------------------
    def forward(self, input_ids, attention_mask, labels=None, use_slow=False):
        model = self.slow_model if use_slow else self.fast_model
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def compute_loss(self, batch: dict) -> torch.Tensor:
        return self.forward(**batch, use_slow=False).loss

    @torch.no_grad()
    def ema_update(self):
        for (_, pf), (_, ps) in zip(
            self._lora_params(self.fast_model),
            self._lora_params(self.slow_model),
        ):
            ps.data.mul_(self.beta).add_((1.0 - self.beta) * pf.data)

    def generate(self, **kwargs):
        """Generate using slow LoRA weights."""
        return self.slow_model.generate(**kwargs)

    def fast_lora_parameters(self):
        return [p for _, p in self._lora_params(self.fast_model)]

    def named_fast_lora_parameters(self):
        return list(self._lora_params(self.fast_model))
