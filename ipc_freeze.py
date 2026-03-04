"""
IPC-style Important Module Freezing for DLOG.

Adapted from arXiv:2504.13407 to operate on PEFT Dual-LoRA modules
identified by (layer_index, target_name) pairs extracted from PEFT keys.

Algorithm per training step:
  1. I(θ)   = |θ ⊙ ∇θ|                       elementwise sensitivity
  2. bar_I ← β1·bar_I + (1-β1)·I             EMA-smoothed sensitivity
  3. U       = |I − bar_I|                    elementwise uncertainty
  4. bar_U ← β2·bar_U + (1-β2)·U             EMA-smoothed uncertainty
  5. S(θ)   = bar_I ⊙ bar_U                  elementwise final score
  6. S_module(m) = mean(S over A_fast ∪ B_fast)   module-level score

At each task boundary:
  - Rank modules by S_module, select top ipc_freeze_p fraction.
  - Enforce global cap ipc_max_frozen_fraction.
  - Apply requires_grad=False to frozen modules' Fast LoRA params.
  - Skip frozen modules during consolidate_after_task().

NOTE: EMA here updates only importance statistics (bar_I, bar_U).
      It does NOT touch Slow LoRA weights — those are managed exclusively
      by DLOGModel.consolidate_after_task() per Section 3.1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Utility: parse (layer_index, target_name) from a dotted module key.
# For Qwen2.5: 'model.layers.3.self_attn.q_proj' → (3, 'q_proj')
# ---------------------------------------------------------------------------

def _parse_layer_target(key: str) -> Tuple[int, str]:
    """
    Given a dotted module key, return (layer_index, target_name).
    layer_index is the first numeric token found; target_name is the last token.
    """
    parts = key.split(".")
    layer_idx = -1
    for p in parts:
        if p.isdigit():
            layer_idx = int(p)
            break
    target_name = parts[-1]
    return layer_idx, target_name


# ---------------------------------------------------------------------------
# ImportanceTracker
# ---------------------------------------------------------------------------

class ImportanceTracker:
    """
    Maintains EMA sensitivity (bar_I) and EMA uncertainty (bar_U) statistics
    for each Fast LoRA module (A_fast, B_fast) in the DLOGModel.

    Usage
    -----
    After every loss.backward() (before optimizer.zero_grad()):
        tracker.update_from_grads()

    After each task:
        scores = tracker.get_module_scores()
        tracker.reset_stats()
    """

    def __init__(self, config, model) -> None:
        """
        Parameters
        ----------
        config : DLOGConfig
        model  : DLOGModel  (already moved to device; dual_lora_keys populated)
        """
        self.config = config
        self.model = model
        self.beta1 = config.ipc_beta1
        self.beta2 = config.ipc_beta2

        # Per-module EMA state.  Keys = model._dual_lora_keys strings.
        # Each value dict holds tensors on the same device as the params,
        # or None before the first gradient has been seen.
        self._stats: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}
        for key in model._dual_lora_keys:
            self._stats[key] = {
                "bar_I_A": None,   # EMA sensitivity for A_fast (same shape)
                "bar_I_B": None,   # EMA sensitivity for B_fast
                "bar_U_A": None,   # EMA uncertainty for A_fast
                "bar_U_B": None,   # EMA uncertainty for B_fast
                "n_updates": 0,    # how many times we've updated this entry
            }

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_from_grads(self) -> None:
        """
        Read .grad from every non-frozen Fast LoRA parameter and update
        the EMA sensitivity / uncertainty accumulators.

        Must be called *after* loss.backward() and *before* optimizer.zero_grad().
        """
        for key, layer in zip(self.model._dual_lora_keys,
                               self.model.get_dual_lora_layers()):
            st = self._stats[key]

            A = layer.A_fast
            B = layer.B_fast

            # Skip modules whose grads are absent (frozen or first init).
            if A.grad is None or B.grad is None:
                continue

            # 1. Elementwise sensitivity: I = |∇θ| (gradient norm, NOT weight×grad).
            #
            # Why NOT |θ * ∇θ|: LoRA standard init sets B_fast=0, so
            # I_B = |0 * grad| = 0 always at the start, and grad(A_fast) =
            # B_fast.T @ upstream = 0 too, so I_A = 0 as well.  Both EMA
            # accumulators (bar_I, bar_U) start anchored at zero, and their
            # product S = bar_I * bar_U ≈ 1e-10 → printed as 0.0000000.
            #
            # Using |∇θ| avoids this because grad_B ≠ 0 even when B=0
            # (upstream gradient flows through A_fast which is Kaiming-init),
            # and grad_A receives signal once B_fast starts growing.
            I_A = A.grad.abs()
            I_B = B.grad.abs()

            # 2. EMA sensitivity: bar_I ← β1·bar_I + (1-β1)·I
            if st["bar_I_A"] is None:
                st["bar_I_A"] = I_A.clone()
                st["bar_I_B"] = I_B.clone()
            else:
                st["bar_I_A"].mul_(self.beta1).add_(I_A, alpha=1.0 - self.beta1)
                st["bar_I_B"].mul_(self.beta1).add_(I_B, alpha=1.0 - self.beta1)

            # 3. Elementwise uncertainty: U = |I − bar_I|
            U_A = (I_A - st["bar_I_A"]).abs()
            U_B = (I_B - st["bar_I_B"]).abs()

            # 4. EMA uncertainty: bar_U ← β2·bar_U + (1-β2)·U
            if st["bar_U_A"] is None:
                st["bar_U_A"] = U_A.clone()
                st["bar_U_B"] = U_B.clone()
            else:
                st["bar_U_A"].mul_(self.beta2).add_(U_A, alpha=1.0 - self.beta2)
                st["bar_U_B"].mul_(self.beta2).add_(U_B, alpha=1.0 - self.beta2)

            st["n_updates"] += 1

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def get_module_scores(self) -> Dict[str, float]:
        """
        Compute S_module(m) = mean(bar_I ⊙ bar_U) averaged over A_fast and B_fast.

        Returns
        -------
        dict mapping module key → scalar score (0.0 if no updates seen yet).
        """
        scores: Dict[str, float] = {}
        with torch.no_grad():
            for key, st in self._stats.items():
                if st["bar_I_A"] is None or st["bar_U_A"] is None:
                    scores[key] = 0.0
                    continue
                # Score = mean(bar_I).  We intentionally drop the bar_U
                # multiplicative term: S = bar_I * bar_U would square a
                # near-zero quantity, collapsing all scores to ~1e-10.
                # bar_I alone (EMA-smoothed gradient norm) is a well-established
                # proxy for parameter importance (cf. Fisher-diag, EWC).
                s_A = st["bar_I_A"].mean().item()
                s_B = st["bar_I_B"].mean().item()
                scores[key] = (s_A + s_B) / 2.0
        return scores

    # ------------------------------------------------------------------
    # Reset / decay between tasks
    # ------------------------------------------------------------------

    def reset_stats(self, decay: bool = False, decay_factor: float = 0.5) -> None:
        """
        Prepare statistics for the next task.

        Parameters
        ----------
        decay : bool
            False → full reset (treat each task independently; recommended default).
            True  → multiply accumulators by decay_factor to soft-carry signal
                    from the previous task.  Useful if tasks share structure.
        decay_factor : float
            Scale applied when decay=True.
        """
        for st in self._stats.values():
            if not decay or st["bar_I_A"] is None:
                st["bar_I_A"] = None
                st["bar_I_B"] = None
                st["bar_U_A"] = None
                st["bar_U_B"] = None
                st["n_updates"] = 0
            else:
                st["bar_I_A"].mul_(decay_factor)
                st["bar_I_B"].mul_(decay_factor)
                st["bar_U_A"].mul_(decay_factor)
                st["bar_U_B"].mul_(decay_factor)
                st["n_updates"] = 0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def n_modules_with_data(self) -> int:
        """Return how many modules have received at least one gradient update."""
        return sum(1 for st in self._stats.values() if st["bar_I_A"] is not None)


# ---------------------------------------------------------------------------
# FreezeManager
# ---------------------------------------------------------------------------

class FreezeManager:
    """
    Manages the cumulative set of frozen LoRA modules across tasks.

    "Frozen module" means:
      (a) A_fast.requires_grad = False (excluded from optimizer).
      (b) consolidate_after_task() is told to skip it (Slow slot protected).
    """

    def __init__(self, config) -> None:
        self.config = config
        self.frozen_keys: Set[str] = set()         # cumulative across all tasks

    # ------------------------------------------------------------------
    # Task-boundary decision
    # ------------------------------------------------------------------

    def select_and_freeze(
        self,
        tracker: ImportanceTracker,
        model,
        task_idx: int,
    ) -> None:
        """
        Score all modules, add top-p% to the frozen set (respecting global cap),
        and log the results.

        Call this *after* consolidate_after_task() for task task_idx.

        Parameters
        ----------
        tracker  : the ImportanceTracker that accumulated stats during this task.
        model    : DLOGModel
        task_idx : 0-based task index (for logging).
        """
        scores = tracker.get_module_scores()
        all_keys: List[str] = list(scores.keys())
        n_total = len(all_keys)

        if n_total == 0:
            return

        # Rank descending by score.
        ranked: List[str] = sorted(all_keys, key=lambda k: scores[k], reverse=True)

        # How many new modules to freeze this task.
        n_freeze_this_task = max(1, round(self.config.ipc_freeze_p * n_total))
        # Global hard cap.
        n_max_frozen = max(1, round(self.config.ipc_max_frozen_fraction * n_total))

        newly_frozen: List[str] = []
        for key in ranked:
            if len(self.frozen_keys) >= n_max_frozen:
                break
            if key not in self.frozen_keys:
                self.frozen_keys.add(key)
                newly_frozen.append(key)
                if len(newly_frozen) >= n_freeze_this_task:
                    break

        # Enforce cap: if somehow over, keep the highest-scoring n_max_frozen.
        if len(self.frozen_keys) > n_max_frozen:
            top_set: Set[str] = set(ranked[:n_max_frozen])
            self.frozen_keys = self.frozen_keys & top_set

        # ------ Logging -----------------------------------------------
        print(f"\n  [IPC] === Task {task_idx + 1} boundary — module importance scores ===")
        print(f"  [IPC] Updates seen per module (sample): "
              f"{tracker._stats[ranked[0]]['n_updates'] if ranked else 0}")
        header = f"  {'Module key':<55} {'L':>3} {'Target':<10} {'Score':>12}  Status"
        print(header)
        print(f"  {'-' * (len(header) - 2)}")
        top_n = min(10, len(ranked))
        for key in ranked[:top_n]:
            layer_idx, target = _parse_layer_target(key)
            if key in newly_frozen:
                status = "** NEWLY FROZEN **"
            elif key in self.frozen_keys:
                status = "frozen"
            else:
                status = "active"
            print(f"  {key:<55} {layer_idx:>3} {target:<10} {scores[key]:>14.4e}  {status}")

        print(f"\n  [IPC] Total frozen modules after task {task_idx + 1}: "
              f"{len(self.frozen_keys)}/{n_total}  "
              f"(newly frozen: {len(newly_frozen)}, cap: {n_max_frozen})")
        if self.frozen_keys:
            frozen_summary = ", ".join(
                f"L{_parse_layer_target(k)[0]}.{_parse_layer_target(k)[1]}"
                for k in sorted(self.frozen_keys)
            )
            print(f"  [IPC] Frozen set: {frozen_summary}")

    # ------------------------------------------------------------------
    # Apply freezing to model parameters
    # ------------------------------------------------------------------

    def apply_freezing_to_model(self, model) -> None:
        """
        For every frozen module, set A_fast.requires_grad = B_fast.requires_grad = False.
        For every non-frozen module, restore requires_grad = True (so re-calling this
        after the trainer's own requires_grad=True pass is safe).

        Call this *after* the trainer has set requires_grad=True on all fast params,
        so that frozen modules get re-frozen.
        """
        for key, layer in zip(model._dual_lora_keys,
                               model.get_dual_lora_layers()):
            if key in self.frozen_keys:
                layer.A_fast.requires_grad_(False)
                layer.B_fast.requires_grad_(False)
            else:
                # Ensure non-frozen params are trainable (defensive).
                layer.A_fast.requires_grad_(True)
                layer.B_fast.requires_grad_(True)

    # ------------------------------------------------------------------
    # Optimizer param filtering
    # ------------------------------------------------------------------

    def get_trainable_fast_params(self, model) -> List[nn.Parameter]:
        """
        Return Fast LoRA parameters *excluding* frozen modules.
        Use this instead of model.get_fast_params() when building the optimizer.
        """
        params: List[nn.Parameter] = []
        for key, layer in zip(model._dual_lora_keys,
                               model.get_dual_lora_layers()):
            if key not in self.frozen_keys:
                params.append(layer.A_fast)
                params.append(layer.B_fast)
        return params

    # ------------------------------------------------------------------
    # Consolidation protection
    # ------------------------------------------------------------------

    def get_frozen_keys(self) -> Set[str]:
        """Return a copy of the currently frozen key set."""
        return set(self.frozen_keys)

    # ------------------------------------------------------------------
    # Correctness assertions (call in tests or debug mode)
    # ------------------------------------------------------------------

    def assert_frozen_params_excluded(self, model, optimizer) -> None:
        """
        Assert that:
          1. All frozen modules' A_fast / B_fast have requires_grad=False.
          2. None of those parameters appear in any optimizer param group.
        """
        opt_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        for key, layer in zip(model._dual_lora_keys,
                               model.get_dual_lora_layers()):
            if key in self.frozen_keys:
                assert not layer.A_fast.requires_grad, \
                    f"[IPC] FAIL: {key}.A_fast still has requires_grad=True"
                assert not layer.B_fast.requires_grad, \
                    f"[IPC] FAIL: {key}.B_fast still has requires_grad=True"
                assert id(layer.A_fast) not in opt_param_ids, \
                    f"[IPC] FAIL: {key}.A_fast is in optimizer param groups"
                assert id(layer.B_fast) not in opt_param_ids, \
                    f"[IPC] FAIL: {key}.B_fast is in optimizer param groups"

    def assert_frozen_grads_absent(self, model) -> None:
        """
        Assert that frozen modules' A_fast / B_fast have .grad == None
        (or all-zero) after a backward pass.
        Call after backward() when testing.
        """
        for key, layer in zip(model._dual_lora_keys,
                               model.get_dual_lora_layers()):
            if key in self.frozen_keys:
                if layer.A_fast.grad is not None:
                    assert layer.A_fast.grad.abs().max().item() == 0.0, \
                        f"[IPC] FAIL: {key}.A_fast has non-zero grad despite being frozen"
                if layer.B_fast.grad is not None:
                    assert layer.B_fast.grad.abs().max().item() == 0.0, \
                        f"[IPC] FAIL: {key}.B_fast has non-zero grad despite being frozen"
