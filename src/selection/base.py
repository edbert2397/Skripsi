"""base.py - Abstract interface for all buffer selectors."""

from abc import ABC, abstractmethod
from typing import List, Optional
import torch


class BaseSelector(ABC):
    """
    All selectors implement score_samples() and select_for_buffer().
    """

    @abstractmethod
    def score_samples(
        self,
        model,
        candidates: List[dict],
        device: torch.device,
    ) -> List[float]:
        """Return a scalar score per candidate (higher = more valuable to store)."""
        ...

    def select_for_buffer(
        self,
        model,
        candidates: List[dict],
        quota: int,
        device: torch.device,
    ) -> List[dict]:
        """Select top-quota candidates by score."""
        if quota <= 0 or not candidates:
            return []
        scores = self.score_samples(model, candidates, device)
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [candidates[i] for i in indices[:quota]]

    def prepare_for_task(self, model, dataloader, device: torch.device):
        """
        Hook called before buffer update for current task.
        Default: no-op. Subclasses (e.g. OrthogonalSelector) override.
        """
        pass
