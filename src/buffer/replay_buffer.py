"""
replay_buffer.py - Fixed-size replay buffer with per-task equal quotas.

When a new task is added, buffer is rebalanced: quota = max_size // n_tasks.
Old tasks are trimmed to the new (smaller) quota by KEEPING the highest-scored
samples, preserving the careful selection done by O-Feat, O-Grad, O-Conflict,
and Surprise selectors.

Internal storage is Dict[int, List[Tuple[dict, float]]] — (sample, score) pairs.
All public methods that return samples still return plain List[dict] so the rest
of the codebase does not need to change.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np


class ReplayBuffer:
    def __init__(self, max_size: int = 300, seed: Optional[int] = None):
        self.max_size = max_size
        # task_id -> list of (sample, score) pairs
        self._store: Dict[int, List[Tuple[dict, float]]] = {}
        # Local rng used only for the None-score fallback in update(); keeps
        # Random-A trimming reproducible across machines. sample()/sample_by_task()
        # intentionally still use the global `random` (already seeded in run script).
        self._rng = np.random.default_rng(seed) if seed is not None else None

    def __len__(self) -> int:
        return sum(len(v) for v in self._store.values())

    def n_tasks(self) -> int:
        return len(self._store)

    def update(self, task_id: int, samples: List[dict], scores: Optional[List[float]] = None):
        """
        Add selected samples for task_id and rebalance all tasks to equal quota.

        Args:
            task_id: The task whose samples are being added.
            samples: Tokenised sample dicts selected by the selector.
            scores:  Corresponding selector scores (higher = more valuable).
                     When None (e.g. Reservoir), random scores are assigned so
                     that trimming remains effectively random for that method.
        """
        if not samples:
            return

        # Assign scores if not provided (random fallback for Random-A / Reservoir).
        # Use the seeded local rng when available so trimming order is reproducible.
        if scores is None:
            if self._rng is not None:
                scores = self._rng.random(len(samples)).tolist()
            else:
                scores = [random.random() for _ in samples]

        # Compute new quota
        n_tasks = len(self._store) + (0 if task_id in self._store else 1)
        quota = max(1, self.max_size // n_tasks)

        # Trim existing tasks — keep the highest-scored survivors
        for tid in list(self._store.keys()):
            pairs = self._store[tid]
            if len(pairs) > quota:
                # Sort descending by score, keep top-quota
                pairs.sort(key=lambda p: p[1], reverse=True)
                self._store[tid] = pairs[:quota]

        # Pair incoming samples with their scores and keep top-quota
        incoming = list(zip(samples, scores))
        incoming.sort(key=lambda p: p[1], reverse=True)
        self._store[task_id] = incoming[:quota]

    def sample(self, n: int) -> List[dict]:
        """
        Sample n items uniformly at random across all stored tasks.
        Returns fewer than n if buffer doesn't have enough samples.
        """
        all_pairs = [p for task_pairs in self._store.values() for p in task_pairs]
        if not all_pairs:
            return []
        n = min(n, len(all_pairs))
        return [p[0] for p in random.sample(all_pairs, n)]

    def sample_by_task(self, n_per_task: int) -> List[dict]:
        """Sample n_per_task from each task (returns plain dicts)."""
        result = []
        for task_pairs in self._store.values():
            k = min(n_per_task, len(task_pairs))
            result.extend(p[0] for p in random.sample(task_pairs, k))
        return result

    def get_all(self) -> List[dict]:
        """Return all stored samples as plain dicts."""
        return [p[0] for task_pairs in self._store.values() for p in task_pairs]

    def task_counts(self) -> Dict[int, int]:
        return {tid: len(pairs) for tid, pairs in self._store.items()}
