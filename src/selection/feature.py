"""
feature.py - O-Feat: Feature subspace orthogonality selection (Method 2).

Samples are scored by how orthogonal their feature representation is to
the current task's feature subspace:

    score(z) = ||h(z)_⊥||

where h(z) is the mean-pooled encoder hidden state of sample z.

Key advantage over O-Grad: NO backward pass required — only forward passes.
This makes O-Feat significantly cheaper per candidate sample.
"""

import torch
from typing import List, Optional
from tqdm import tqdm

from .base import BaseSelector
from ..utils.feature_subspace import estimate_feature_subspace, feature_orthogonal_residual_norm


class FeatureSelector(BaseSelector):
    def __init__(
        self,
        subspace_rank_k: int = 10,
        n_estimation_batches: int = 10,
        score_batch_size: int = 32,
        use_fp16: bool = True,
    ):
        self.k = subspace_rank_k
        self.n_estimation_batches = n_estimation_batches
        self.score_batch_size = score_batch_size
        self.use_fp16 = use_fp16
        self.projection_matrix: Optional[torch.Tensor] = None  # V_k: [d_hidden, k] CPU fp32

    def prepare_for_task(self, model, dataloader, device: torch.device):
        """Phase 1B: estimate feature subspace from current task encoder outputs."""
        print(f"[FeatureSelector] Estimating feature subspace (k={self.k})...")
        self.projection_matrix = estimate_feature_subspace(
            model=model,
            dataloader=dataloader,
            device=device,
            subspace_rank_k=self.k,
            n_estimation_batches=self.n_estimation_batches,
            use_fp16=self.use_fp16,
        )
        print(f"[FeatureSelector] Subspace estimated. V_k shape: {self.projection_matrix.shape}")

    def _extract_mean_pooled(
        self,
        model,
        samples: List[dict],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Forward-pass a batch of samples through the encoder and return
        mean-pooled hidden states on CPU.

        Args:
            model:   DualLoRAModel
            samples: list of tokenised dicts
            device:  CUDA device

        Returns:
            Tensor of shape [len(samples), d_hidden] on CPU fp32.
        """
        m = getattr(model, "fast_model", model)

        input_ids = torch.stack([s["input_ids"] for s in samples]).to(device)
        attention_mask = torch.stack([s["attention_mask"] for s in samples]).to(device)
        labels = torch.stack([s["labels"] for s in samples]).to(device)

        with torch.no_grad():
            if self.use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = m(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        output_hidden_states=True,
                    )
            else:
                outputs = m(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=True,
                )

        # encoder_last_hidden_state: [B, seq_len, d_hidden]
        hidden = outputs.encoder_last_hidden_state.float()
        mask = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]

        # Mean pooling over non-padding tokens
        h = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, d_hidden]

        return h.cpu()

    def score_samples(
        self,
        model,
        candidates: List[dict],
        device: torch.device,
    ) -> List[float]:
        """Phase 2B: score each candidate by ||h(z)_⊥||.

        Unlike O-Grad, this only requires forward passes (no backward),
        so we can process in larger batches for efficiency.
        """
        if self.projection_matrix is None:
            raise RuntimeError("Call prepare_for_task() before score_samples().")

        print(f"[FeatureSelector] Scoring {len(candidates)} candidates (forward-only)...")
        model.eval()

        scores = []
        for start in range(0, len(candidates), self.score_batch_size):
            end = min(start + self.score_batch_size, len(candidates))
            chunk = candidates[start:end]

            h_batch = self._extract_mean_pooled(model, chunk, device)  # [B, d_hidden]
            h_batch = torch.nan_to_num(h_batch, nan=0.0, posinf=0.0, neginf=0.0)

            for i in range(h_batch.size(0)):
                score = feature_orthogonal_residual_norm(
                    h_batch[i], self.projection_matrix
                )
                scores.append(score)

            if (start // self.score_batch_size + 1) % 10 == 0:
                torch.cuda.empty_cache()

        model.train()
        torch.cuda.empty_cache()
        return scores
