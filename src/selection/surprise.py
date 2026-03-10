"""
surprise.py - SuRe's NLL-based surprise selection (baseline).

score(z) = NLL(model; z) = -log p(y | x; θ)
Higher NLL = more surprising = stored in buffer.
"""

import torch
import torch.nn.functional as F
from typing import List

from .base import BaseSelector


class SurpriseSelector(BaseSelector):
    def __init__(self, use_fp16: bool = True, batch_size: int = 32):
        self.use_fp16 = use_fp16
        self.batch_size = batch_size

    def score_samples(
        self,
        model,
        candidates: List[dict],
        device: torch.device,
    ) -> List[float]:
        """Score each sample by its NLL under the current model."""
        model.eval()
        scores = []

        for start in range(0, len(candidates), self.batch_size):
            chunk = candidates[start : start + self.batch_size]

            # Stack tensor fields into a batch; skip non-tensor fields
            batch = {}
            for key in chunk[0]:
                vals = [s[key] for s in chunk]
                if isinstance(vals[0], torch.Tensor):
                    batch[key] = torch.stack(vals).to(device)

            with torch.no_grad():
                if self.use_fp16:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model.forward(**batch, use_slow=False)
                else:
                    outputs = model.forward(**batch, use_slow=False)

            # Compute per-sample NLL from logits (shape [B, T, V])
            logits = outputs.logits.float()          # [B, T, V]
            labels = batch["labels"]                 # [B, T]
            B, T, V = logits.shape

            per_token_loss = F.cross_entropy(
                logits.view(B * T, V),
                labels.view(B * T),
                ignore_index=-100,
                reduction="none",
            ).view(B, T)                             # [B, T]

            mask = labels != -100                    # [B, T]
            per_sample_nll = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            scores.extend(per_sample_nll.tolist())

        model.train()
        return scores
