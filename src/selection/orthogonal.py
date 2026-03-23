"""
orthogonal.py - Orthogonal gradient selection (our core contribution).

Samples are scored by how orthogonal their gradient is to the current
task's gradient subspace: score(z) = ||g(z)_⊥||.
"""

import torch
from typing import List, Optional
from tqdm import tqdm

from .base import BaseSelector
from ..utils.subspace import estimate_gradient_subspace, orthogonal_residual_norm
from ..utils.gradient_utils import extract_lora_grad_vector


class OrthogonalSelector(BaseSelector):
    def __init__(
        self,
        subspace_rank_k: int = 10,
        n_estimation_batches: int = 10,
        grad_batch_size: int = 8,
        use_fp16: bool = True,
    ):
        self.k = subspace_rank_k
        self.n_estimation_batches = n_estimation_batches
        self.grad_batch_size = grad_batch_size
        self.use_fp16 = use_fp16
        self.projection_matrix: Optional[torch.Tensor] = None  # V_k: [d_lora, k] CPU fp32

    def prepare_for_task(self, model, dataloader, device: torch.device):
        """Phase 1: estimate gradient subspace from current task batches."""
        print(f"[OrthogonalSelector] Estimating gradient subspace (k={self.k})...")
        self.projection_matrix = estimate_gradient_subspace(
            model=model,
            dataloader=dataloader,
            device=device,
            subspace_rank_k=self.k,
            n_estimation_batches=self.n_estimation_batches,
            use_fp16=self.use_fp16,
        )
        print(f"[OrthogonalSelector] Subspace estimated. V_k shape: {self.projection_matrix.shape}")

    def score_samples(
        self,
        model,
        candidates: List[dict],
        device: torch.device,
    ) -> List[float]:
        """Phase 2: score each candidate by ||g(z)_⊥||.

        Gradients are computed and scored one at a time to avoid accumulating
        1000 x d_lora tensors in CPU RAM (would be ~9 GB for T5-Large LoRA).
        """
        if self.projection_matrix is None:
            raise RuntimeError("Call prepare_for_task() before score_samples().")

        print(f"[OrthogonalSelector] Scoring {len(candidates)} candidates...")
        # Must stay in training mode for gradient checkpointing compatibility.
        was_training = model.training
        model.train()
        m = getattr(model, "fast_model", model)
        for _, param in m.named_parameters():
            if "lora_" in _:
                param.requires_grad_(True)

        amp_dtype = (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                     else torch.float16)
        use_scaler = self.use_fp16 and (amp_dtype == torch.float16)
        scaler = torch.amp.GradScaler("cuda") if use_scaler else None
        scores = []
        for i, sample in enumerate(tqdm(candidates, desc="Scoring candidates", leave=False)):
            batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in sample.items()}
            m.zero_grad()
            if self.use_fp16:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    loss = model.compute_loss(batch)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    inv_scale = 1.0 / scaler.get_scale()
                    for name, param in m.named_parameters():
                        if "lora_" in name and param.grad is not None:
                            param.grad.data.mul_(inv_scale)
                else:
                    loss.backward()
            else:
                loss = model.compute_loss(batch)
                loss.backward()

            grad_vec = extract_lora_grad_vector(model)  # CPU fp32, [d_lora]
            grad_vec = torch.nan_to_num(grad_vec, nan=0.0, posinf=0.0, neginf=0.0)
            scores.append(orthogonal_residual_norm(grad_vec, self.projection_matrix))
            del grad_vec

            if (i + 1) % self.grad_batch_size == 0:
                torch.cuda.empty_cache()

        if not was_training:
            model.eval()
        m.zero_grad()
        torch.cuda.empty_cache()
        return scores
