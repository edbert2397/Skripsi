"""
no_replay.py - Naive sequential fine-tune baseline (no buffer, no replay).

Acts as a lower bound: the buffer stays empty, so the trainer's
`len(self.buffer) > 0` guard skips the replay step entirely on every
optimiser step. Training reduces to plain sequential fine-tuning of the
fast LoRA on each task in turn.
"""

from typing import List, Optional, Tuple

import torch

from .base import BaseSelector


class NoReplaySelector(BaseSelector):
    """Selects nothing. Buffer stays empty; replay never fires."""

    def score_samples(self, model, candidates: List[dict], device: torch.device) -> List[float]:
        # Unused (select_for_buffer short-circuits), but BaseSelector requires it.
        return [0.0] * len(candidates)

    def select_for_buffer(
        self,
        model,
        candidates: List[dict],
        quota: int,
        device: torch.device,
    ) -> Tuple[List[dict], Optional[List[float]]]:
        return [], []
