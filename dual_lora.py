"""
DualLoRALinear — the core building block of DLOG.

Replaces a single nn.Linear with:
    W_total = W_frozen + B_s @ A_s  (Slow/Memory)  +  B_f @ A_f  (Fast/Task)

Slow LoRA consolidates learned knowledge; Fast LoRA adapts to new tasks.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DualLoRALinear(nn.Module):
    """
    A linear layer augmented with two LoRA branches (Slow + Fast).

    Parameters
    ----------
    in_features : int
        Input dimension of the original linear layer.
    out_features : int
        Output dimension of the original linear layer.
    rank : int
        Low-rank dimension for both LoRA branches.
    alpha : float
        LoRA scaling factor  (scaling = alpha / rank).
    bias : bool
        Whether the original layer has bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # --- Frozen base weight (loaded later) ---
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

        # --- Slow LoRA (Memory) ---
        self.A_slow = nn.Parameter(torch.empty(rank, in_features))
        self.B_slow = nn.Parameter(torch.empty(out_features, rank))

        # --- Fast LoRA (Task) ---
        self.A_fast = nn.Parameter(torch.empty(rank, in_features))
        self.B_fast = nn.Parameter(torch.empty(out_features, rank))

        self.reset_lora_parameters()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def reset_lora_parameters(self):
        """
        Slow LoRA: ZERO init — completely silent until explicit consolidation.

        Why: If A_slow starts as Kaiming, the slow branch trains on Task 1 data
        alongside Fast (since both are in the optimizer), causing them to converge
        to almost-identical solutions. When Task 2 starts, A_fast ≈ A_slow, so
        the orthogonal loss is immediately enormous and destabilises training.

        With zero init, the gradient chain through Slow is broken:
            d(B_slow @ A_slow @ x)/d(A_slow) = B_slow^T @ upstream = 0  (B_slow=0)
            d(B_slow @ A_slow @ x)/d(B_slow) = A_slow @ x      = 0  (A_slow=0)
        So Slow stays at exactly zero throughout Task 1, and weight decay cannot
        pull it away from zero (it is already at zero). Only Fast learns Task 1.
        After Task 1 finishes, consolidate_after_task() hard-copies Fast → Slow.

        Fast LoRA: standard LoRA init — B=0 so output is 0 initially, but A=Kaiming
        so B receives non-zero upstream gradients and training commences immediately.
        """
        # Slow — completely silent until consolidation
        nn.init.zeros_(self.A_slow)
        nn.init.zeros_(self.B_slow)

        # Fast — standard LoRA init
        nn.init.kaiming_uniform_(self.A_fast, a=math.sqrt(5))
        nn.init.zeros_(self.B_fast)

    def reset_fast(self):
        """
        Reset Fast LoRA to standard init — call between tasks via consolidate_after_task().

        This ensures Fast starts orthogonally fresh for each new task instead of
        inheriting the previous task's values, which would cause the orthogonal loss
        to be huge initially (A_fast ≈ A_slow at the start of every new task).
        """
        nn.init.kaiming_uniform_(self.A_fast, a=math.sqrt(5))
        nn.init.zeros_(self.B_fast)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        y = W_frozen @ x  +  scaling * (B_s @ A_s @ x)  +  scaling * (B_f @ A_f @ x)
        """
        # Base
        out = F.linear(x, self.weight, self.bias)

        # Slow LoRA
        slow = F.linear(F.linear(x, self.A_slow), self.B_slow) * self.scaling

        # Fast LoRA
        fast = F.linear(F.linear(x, self.A_fast), self.B_fast) * self.scaling

        return out + slow + fast

    # ------------------------------------------------------------------
    # Helpers for gating
    # ------------------------------------------------------------------
    def get_slow_params(self):
        """Return (A_slow, B_slow) for orthogonal gating computations."""
        return self.A_slow, self.B_slow

    def get_fast_params(self):
        """Return (A_fast, B_fast) for orthogonal gating computations."""
        return self.A_fast, self.B_fast

    # ------------------------------------------------------------------
    # NO EMA — Slow is ONLY updated via consolidate_after_task()
    # ------------------------------------------------------------------
    # Per Section 3.1: P_slow = I - (A_s A_s^T)/(||A_s||^2 + λ) must be
    # FIXED during task training. EMA would shift A_slow every step, making
    # the null-space a moving target and violating the stable orthogonal guarantee.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}"
        )


class SingleLoRALinear(nn.Module):
    """
    Baseline: a single LoRA branch (no Slow/Fast separation).
    Used for the SeqFT + Single LoRA baseline comparison.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.empty(out_features, rank))

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        lora = F.linear(F.linear(x, self.A), self.B) * self.scaling
        return out + lora

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}"
        )
