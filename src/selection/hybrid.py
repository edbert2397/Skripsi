"""
hybrid.py - Combined orthogonal + surprise selection.

score(z) = alpha * ||g(z)_⊥|| + (1 - alpha) * NLL(z)

alpha=0 -> pure surprise (SuRe)
alpha=1 -> pure orthogonal (ours)
"""

import torch
from typing import List

from .base import BaseSelector
from .orthogonal import OrthogonalSelector
from .surprise import SurpriseSelector


class HybridSelector(BaseSelector):
    def __init__(
        self,
        alpha: float = 0.5,
        subspace_rank_k: int = 10,
        n_estimation_batches: int = 10,
        grad_batch_size: int = 8,
        use_fp16: bool = True,
    ):
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"
        self.alpha = alpha
        self._orth = OrthogonalSelector(
            subspace_rank_k=subspace_rank_k,
            n_estimation_batches=n_estimation_batches,
            grad_batch_size=grad_batch_size,
            use_fp16=use_fp16,
        )
        self._surp = SurpriseSelector(use_fp16=use_fp16)

    def prepare_for_task(self, model, dataloader, device: torch.device):
        """Delegate subspace estimation to OrthogonalSelector."""
        self._orth.prepare_for_task(model, dataloader, device)

    @property
    def singular_values(self):
        """Expose inner OrthogonalSelector's spectrum for spectral logging."""
        return getattr(self._orth, "singular_values", None)

    def score_samples(
        self,
        model,
        candidates: List[dict],
        device: torch.device,
    ) -> List[float]:
        if self.alpha == 0.0:
            return self._surp.score_samples(model, candidates, device)
        if self.alpha == 1.0:
            return self._orth.score_samples(model, candidates, device)

        orth_scores = self._orth.score_samples(model, candidates, device)
        surp_scores = self._surp.score_samples(model, candidates, device)

        # Normalise both to [0, 1] range before combining
        def _norm(lst):
            lo, hi = min(lst), max(lst)
            rng = hi - lo + 1e-8
            return [(x - lo) / rng for x in lst]

        orth_n = _norm(orth_scores)
        surp_n = _norm(surp_scores)

        return [
            self.alpha * o + (1.0 - self.alpha) * s
            for o, s in zip(orth_n, surp_n)
        ]
