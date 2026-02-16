"""
Evaluation Metrics for DLOG.

Covers:
  - Standard CL: Final Performance (FP), Average Performance (AP), Forgetting (FT)
  - Mechanistic: Gradient Cosine Similarity, Leakage Ratio
  - Geometric: Subspace Overlap (principal angles via SVD)
  - Efficiency: wall-clock time, forward passes, memory
"""
import time
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


# ======================================================================
# 1. Standard CL Metrics
# ======================================================================
class CLMetricsTracker:
    """
    Tracks performance matrix R[i][j] = accuracy on task j after training task i.

    From this we compute:
      - Final Performance (FP): average of R[T][j] for all j
      - Average Performance (AP): average of R[i][i] for all i
      - Forgetting (FT): average of max_{l<=i} R[l][j] - R[T][j] for j < T
    """

    def __init__(self, task_names: List[str]):
        self.task_names = task_names
        self.n_tasks = len(task_names)
        # R[i][j] = accuracy on task j after training on task i (0-indexed)
        self.R = np.zeros((self.n_tasks, self.n_tasks))

    def record(self, after_task_idx: int, eval_task_idx: int, accuracy: float):
        """Record accuracy on eval_task after training through after_task."""
        self.R[after_task_idx, eval_task_idx] = accuracy

    def final_performance(self) -> float:
        """FP = (1/T) Σ_j R[T-1][j]"""
        return float(np.mean(self.R[-1, :]))

    def average_performance(self) -> float:
        """AP = (1/T) Σ_i R[i][i]"""
        return float(np.mean(np.diag(self.R)))

    def forgetting(self) -> float:
        """
        FT = (1/(T-1)) Σ_{j<T} (max_{l in 0..T-1} R[l][j] - R[T-1][j])
        Lower is better.
        """
        if self.n_tasks <= 1:
            return 0.0
        fgt = []
        for j in range(self.n_tasks - 1):
            best = np.max(self.R[:, j])
            final = self.R[-1, j]
            fgt.append(best - final)
        return float(np.mean(fgt))

    def summary(self) -> Dict:
        return {
            "Final Performance (FP)": self.final_performance(),
            "Average Performance (AP)": self.average_performance(),
            "Forgetting (FT)": self.forgetting(),
            "Performance Matrix": self.R.tolist(),
        }


# ======================================================================
# 2. Gradient Cosine Similarity
# ======================================================================
class GradientConflictTracker:
    """
    Tracks gradient cosine similarity between old-task and new-task gradients.

    Usage:
        tracker.snapshot_gradients(params)  # after backward on old-task data
        ...
        tracker.compute_cosine_sim(params)  # after backward on new-task data
    """

    def __init__(self):
        self._snapshots: Dict[int, torch.Tensor] = {}
        self.history: List[float] = []

    @torch.no_grad()
    def snapshot_gradients(self, params: List[nn.Parameter]):
        """Capture current .grad as the 'old task' reference."""
        self._snapshots.clear()
        for p in params:
            if p.grad is not None:
                self._snapshots[id(p)] = p.grad.detach().clone().flatten()

    @torch.no_grad()
    def compute_cosine_sim(self, params: List[nn.Parameter]) -> float:
        """
        Compute CosSim = <g_old, g_new> / (||g_old|| * ||g_new||)
        averaged across all params that have both old and new grads.
        """
        cos_sims = []
        for p in params:
            pid = id(p)
            if p.grad is None or pid not in self._snapshots:
                continue
            g_new = p.grad.detach().flatten()
            g_old = self._snapshots[pid]
            dot = torch.dot(g_old, g_new)
            norm_old = torch.norm(g_old)
            norm_new = torch.norm(g_new)
            if norm_old > 1e-8 and norm_new > 1e-8:
                cos_sims.append((dot / (norm_old * norm_new)).item())

        avg = sum(cos_sims) / max(len(cos_sims), 1)
        self.history.append(avg)
        return avg


# ======================================================================
# 3. Subspace Overlap (Principal Angles via SVD)
# ======================================================================
@torch.no_grad()
def subspace_overlap(A: torch.Tensor, B: torch.Tensor) -> Dict:
    """
    Compute the overlap between column spaces of A and B using principal angles.

    A: [d, r1], B: [d, r2]

    Returns:
        principal_angles: list of angles (radians)
        mean_angle: mean principal angle
        overlap_score: 1 - mean(cos(angles))  (0 = identical, 1 = orthogonal)
    """
    # QR to get orthonormal bases
    Q_A, _ = torch.linalg.qr(A.float(), mode="reduced")
    Q_B, _ = torch.linalg.qr(B.float(), mode="reduced")

    # SVD of Q_A^T @ Q_B
    M = Q_A.T @ Q_B
    _, S, _ = torch.linalg.svd(M)

    # Clamp singular values to [0, 1] for numerical stability
    S = torch.clamp(S, 0.0, 1.0)
    angles = torch.acos(S)  # principal angles

    mean_angle = angles.mean().item()
    overlap = S.mean().item()  # mean cosine of principal angles

    return {
        "principal_angles": angles.tolist(),
        "mean_angle_rad": mean_angle,
        "mean_angle_deg": np.degrees(mean_angle),
        "overlap_score": overlap,           # 1 = identical, 0 = orthogonal
        "separation_score": 1.0 - overlap,  # 1 = orthogonal, 0 = identical
    }


def compute_all_subspace_overlaps(dual_layers) -> Dict:
    """
    Compute subspace overlap between Slow and Fast LoRA
    across all DualLoRALinear layers.
    """
    results = {"per_layer_A": [], "per_layer_B": []}
    for i, layer in enumerate(dual_layers):
        A_s, B_s = layer.get_slow_params()
        A_f, B_f = layer.get_fast_params()

        # A matrices: [r, d] -> transpose to [d, r] for column-space analysis
        overlap_A = subspace_overlap(A_s.T, A_f.T)
        overlap_B = subspace_overlap(B_s, B_f)

        results["per_layer_A"].append(overlap_A)
        results["per_layer_B"].append(overlap_B)

    # Averages
    results["mean_A_separation"] = np.mean(
        [o["separation_score"] for o in results["per_layer_A"]]
    )
    results["mean_B_separation"] = np.mean(
        [o["separation_score"] for o in results["per_layer_B"]]
    )
    return results


# ======================================================================
# 4. Efficiency Tracking
# ======================================================================
class EfficiencyTracker:
    """Track wall-clock time, forward passes, and GPU memory."""

    def __init__(self):
        self.total_time = 0.0
        self.forward_passes = 0
        self.projection_time = 0.0
        self.peak_memory_mb = 0.0
        self._start = None

    def start_timer(self):
        self._start = time.time()

    def stop_timer(self):
        if self._start:
            self.total_time += time.time() - self._start
            self._start = None

    def record_forward(self, n: int = 1):
        self.forward_passes += n

    def start_projection_timer(self):
        self._proj_start = time.time()

    def stop_projection_timer(self):
        self.projection_time += time.time() - self._proj_start

    def update_memory(self):
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self.peak_memory_mb = max(self.peak_memory_mb, mem)

    def summary(self) -> Dict:
        return {
            "total_time_sec": round(self.total_time, 2),
            "forward_passes": self.forward_passes,
            "projection_time_sec": round(self.projection_time, 4),
            "projection_overhead_pct": round(
                100 * self.projection_time / max(self.total_time, 1e-8), 2
            ),
            "peak_memory_mb": round(self.peak_memory_mb, 1),
        }


# ======================================================================
# 5. Accuracy Evaluation (generation-based)
# ======================================================================
@torch.no_grad()
def evaluate_accuracy(model, eval_loader, tokenizer, device="cuda", max_gen=16) -> float:
    """
    Evaluate classification accuracy by generating from T5 and
    matching against the true label text.
    """
    model.eval()
    correct = 0
    total = 0

    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label_texts = batch["label_text"]  # list of strings

        # Generate
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_gen,
            do_sample=False,
        )
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for pred, label in zip(preds, label_texts):
            if pred.strip().lower() == label.strip().lower():
                correct += 1
            total += 1

    model.train()
    return correct / max(total, 1)
