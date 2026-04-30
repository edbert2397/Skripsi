"""
reservoir.py - RandomBalancedSelector (SuRe's Random-A baseline).

Task-balanced random selection: per-task quota = S/d is enforced by
ReplayBuffer.update; this selector contributes the within-task uniform
random draw. NOT classical Vitter (1985) reservoir sampling, which would
be online and task-agnostic.

Corresponds to "Random Buffer Update After" (Random-A) in Hazard et al.,
2025, Tables 4-5.
"""

from typing import List

import numpy as np
import torch

from .base import BaseSelector


class RandomBalancedSelector(BaseSelector):
    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)

    def score_samples(
        self,
        model,
        candidates: List[dict],
        device: torch.device,
    ) -> List[float]:
        """Assign uniform random scores in [0, 1)."""
        return self._rng.random(len(candidates)).tolist()

    def select_for_buffer(
        self,
        model,
        candidates: List[dict],
        quota: int,
        device: torch.device,
    ) -> tuple[List[dict], None]:
        """Uniform random subsample of `candidates` of size <= quota.

        Returns scores=None so that ReplayBuffer.update keeps trimming
        within-task uniformly at random (the Random-A semantics).
        """
        if quota <= 0 or not candidates:
            return [], None
        k = min(quota, len(candidates))
        idx = self._rng.choice(len(candidates), size=k, replace=False)
        selected = [candidates[i] for i in idx]
        return selected, None


# DeprecationWarning: use RandomBalancedSelector. Kept so existing configs
# (`selection_method: reservoir`) and pickled run-name strings still resolve.
ReservoirSelector = RandomBalancedSelector
