"""
conflict.py - O-Conflict: Gradient conflict score selection (Method 3).

Samples are scored by how much their gradient direction conflicts with
the current task's mean gradient:

    score(z) = -cos(g(z), g_ref)

Key differences from O-Grad:
  - No SVD/subspace estimation needed (only a mean gradient vector)
  - Measures directional conflict, not subspace residual magnitude
  - Same backward-pass cost per sample, but cheaper preparation phase

Inspired by A-GEM, GSS, and CNL's finding that negative gradient similarity
is the causal mechanism of catastrophic forgetting.
"""

import torch
from typing import List, Optional
from tqdm import tqdm

from .base import BaseSelector
from ..utils.conflict_score import estimate_reference_gradient, gradient_conflict_score
from ..utils.gradient_utils import extract_lora_grad_vector


class ConflictSelector(BaseSelector):
    def __init__(
        self,
        n_estimation_batches: int = 10,
        grad_batch_size: int = 8,
        use_fp16: bool = True,
    ):
        self.n_estimation_batches = n_estimation_batches
        self.grad_batch_size = grad_batch_size
        self.use_fp16 = use_fp16
        self.g_ref: Optional[torch.Tensor] = None  # [d_lora] CPU fp32

    def prepare_for_task(self, model, dataloader, device: torch.device):
        """Phase 1C: compute mean gradient of current task (no SVD needed)."""
        print(f"[ConflictSelector] Computing reference gradient "
              f"(avg over {self.n_estimation_batches} batches)...")
        self.g_ref = estimate_reference_gradient(
            model=model,
            dataloader=dataloader,
            device=device,
            n_estimation_batches=self.n_estimation_batches,
            use_fp16=self.use_fp16,
        )
        print(f"[ConflictSelector] Reference gradient computed. "
              f"Shape: {self.g_ref.shape}, norm: {self.g_ref.norm().item():.4f}")

    def score_samples(
        self,
        model,
        candidates: List[dict],
        device: torch.device,
    ) -> List[float]:
        """Phase 2C: score each candidate by -cos(g(z), g_ref).

        Like O-Grad, requires a backward pass per sample to get g(z).
        Unlike O-Grad, the scoring itself is a simple dot product (no SVD).
        """
        if self.g_ref is None:
            raise RuntimeError("Call prepare_for_task() before score_samples().")

        print(f"[ConflictSelector] Scoring {len(candidates)} candidates...")
        # Must stay in training mode for gradient checkpointing to work.
        # In eval mode, T5 enables use_cache=True which breaks gradient flow.
        was_training = model.training
        model.train()
        m = getattr(model, "fast_model", model)
        for name, param in m.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)

        scores = []
        for i, sample in enumerate(tqdm(candidates, desc="Scoring candidates (conflict)", leave=False)):
            batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in sample.items()}
            m.zero_grad()
            if self.use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = model.compute_loss(batch)
            else:
                loss = model.compute_loss(batch)
            loss.backward()

            grad_vec = extract_lora_grad_vector(model)  # CPU fp32, [d_lora]
            grad_vec = torch.nan_to_num(grad_vec, nan=0.0, posinf=0.0, neginf=0.0)
            scores.append(gradient_conflict_score(grad_vec, self.g_ref))
            del grad_vec

            if (i + 1) % self.grad_batch_size == 0:
                torch.cuda.empty_cache()

        if not was_training:
            model.eval()
        m.zero_grad()
        torch.cuda.empty_cache()
        return scores
