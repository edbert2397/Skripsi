"""
reservoir.py - Random reservoir sampling baseline.

Uses reservoir sampling (Vitter 1985) so each sample is equally likely
to be in the buffer regardless of stream length.
"""

import random
from typing import List
import torch

from .base import BaseSelector


class ReservoirSelector(BaseSelector):
    def score_samples(
        self,
        model,
        candidates: List[dict],
        device: torch.device,
    ) -> List[float]:
        """Assign random scores (equivalent to reservoir sampling)."""
        return [random.random() for _ in candidates]
