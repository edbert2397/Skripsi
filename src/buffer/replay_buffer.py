"""
replay_buffer.py - Fixed-size replay buffer with per-task equal quotas.

When a new task is added, buffer is rebalanced: quota = max_size // n_tasks.
Old tasks are truncated to the new (smaller) quota.
"""

import random
from typing import Dict, List, Optional


class ReplayBuffer:
    def __init__(self, max_size: int = 300):
        self.max_size = max_size
        self._store: Dict[int, List[dict]] = {}   # task_id -> list of samples

    def __len__(self) -> int:
        return sum(len(v) for v in self._store.values())

    def n_tasks(self) -> int:
        return len(self._store)

    def update(self, task_id: int, samples: List[dict]):
        """
        Add selected samples for task_id and rebalance all tasks to equal quota.
        """
        if not samples:
            return

        # Compute new quota
        n_tasks = len(self._store) + (0 if task_id in self._store else 1)
        quota = max(1, self.max_size // n_tasks)

        # Trim existing tasks
        for tid in list(self._store.keys()):
            if len(self._store[tid]) > quota:
                self._store[tid] = random.sample(self._store[tid], quota)

        # Add new task samples
        self._store[task_id] = samples[:quota]

    def sample(self, n: int) -> List[dict]:
        """
        Sample n items uniformly at random across all stored tasks.
        Returns fewer than n if buffer doesn't have enough samples.
        """
        all_samples = [s for task_samples in self._store.values() for s in task_samples]
        if not all_samples:
            return []
        n = min(n, len(all_samples))
        return random.sample(all_samples, n)

    def sample_by_task(self, n_per_task: int) -> List[dict]:
        """Sample n_per_task from each task."""
        result = []
        for task_samples in self._store.values():
            k = min(n_per_task, len(task_samples))
            result.extend(random.sample(task_samples, k))
        return result

    def get_all(self) -> List[dict]:
        return [s for task_samples in self._store.values() for s in task_samples]

    def task_counts(self) -> Dict[int, int]:
        return {tid: len(samples) for tid, samples in self._store.items()}
